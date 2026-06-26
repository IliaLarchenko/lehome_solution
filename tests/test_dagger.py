"""Tests for DAgger recovery data collection system."""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import the modules under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dagger_collect import (
    EpisodeRecorder,
    FailureQueue,
    build_dagger_frame,
    convert_bimanual_to_action,
    decode_observation,
    get_next_session_dir,
    get_solved_npz_names,
    load_pipeline_config,
)


# ---------------------------------------------------------------------------
# Joint conversion tests
# ---------------------------------------------------------------------------

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def _reading(left: dict | None = None, right: dict | None = None) -> dict:
    base = {n: 0.0 for n in JOINT_NAMES}
    return {
        "left": {**base, **(left or {})},
        "right": {**base, **(right or {})},
    }


class TestJointConversion:
    """Test SO101 degree-mode reading -> sim-radian conversion."""

    def test_zero_degrees_maps_to_zero_radians(self):
        """0° on the arm joints is the sim zero pose (gripper is the exception)."""
        result = convert_bimanual_to_action(_reading())
        for i in (0, 1, 2, 3, 4, 6, 7, 8, 9, 10):
            assert abs(result[i]) < 1e-6

    def test_arm_joint_is_plain_deg_to_rad(self):
        """Arm joints convert with the identity deg→rad mapping."""
        result = convert_bimanual_to_action(
            _reading(left={"shoulder_pan": 110.0}, right={"wrist_roll": -180.0})
        )
        assert abs(result[0] - np.radians(110.0)) < 1e-5
        assert abs(result[10] - np.radians(-180.0)) < 1e-5

    def test_gripper_range(self):
        """Gripper has asymmetric limits: 0-100 → USD (-10°, 100°)."""
        result = convert_bimanual_to_action(_reading())
        # gripper at 0 (closed) → USD lower = -10 deg
        expected = -10.0 * np.pi / 180.0
        assert abs(result[5] - expected) < 0.01
        result = convert_bimanual_to_action(_reading(left={"gripper": 100.0}))
        assert abs(result[5] - np.radians(100.0)) < 1e-4

    def test_bimanual_conversion_shape_and_dtype(self):
        """Full bimanual conversion produces a float32 12-dim output."""
        result = convert_bimanual_to_action(_reading())
        assert result.shape == (12,)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Failure queue tests
# ---------------------------------------------------------------------------

class TestFailureQueue:
    """Test failure queue management."""

    @pytest.fixture
    def queue_dir(self, tmp_path):
        """Create temp dir with mock NPZ files."""
        for i, garment in enumerate(["Top_Long_Seen_0", "Top_Long_Seen_0",
                                      "Pant_Short_Seen_1"]):
            meta = {
                "garment": garment,
                "garment_type": "top-long-sleeve" if "Top" in garment else "short-pant",
                "seed": 42 + i,
                "failure_frame": 100 + i,
            }
            npz_path = tmp_path / f"{garment}_seed{42 + i}_ep0.npz"
            np.savez(
                npz_path,
                garment_points=np.zeros((10, 3), dtype=np.float32),
                garment_velocities=np.zeros((10, 3), dtype=np.float32),
                left_joint_pos=np.zeros(6, dtype=np.float32),
                right_joint_pos=np.zeros(6, dtype=np.float32),
                metadata=json.dumps(meta),
            )
        return tmp_path

    def test_load(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        assert q.total == 3
        assert q.remaining == 3
        assert q.done_count == 0

    def test_current_and_next(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        first = q.current()
        assert first is not None
        assert "garment" in first
        assert "garment_points" in first

        second = q.next()
        assert second is not None

    def test_skip(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        q.skip()
        assert q.done_count == 1
        assert q.remaining == 2

    def test_mark_done(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        q.mark_done(True)
        assert q.success_count == 1
        assert q.done_count == 1

    def test_retry_doesnt_advance(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        first = q.current()
        q.retry()
        assert q.current() is first

    def test_garment_grouping(self, queue_dir):
        """With FR-proportional ordering, items for the same type should be grouped."""
        # Without success_rates, items are shuffled randomly.
        # With success_rates, FR-proportional ordering groups by type.
        sr = {"by_type": {"top_long": 0.5, "short_pant": 0.5}}
        q = FailureQueue(str(queue_dir), success_rates=sr)
        garments = []
        while q.current():
            garments.append(q.current()["garment"])
            q.skip()
        # All 3 items should be present
        assert len(garments) == 3
        top_indices = [i for i, g in enumerate(garments) if g == "Top_Long_Seen_0"]
        assert len(top_indices) == 2
        assert top_indices[1] - top_indices[0] == 1  # contiguous

    def test_empty_dir(self, tmp_path):
        q = FailureQueue(str(tmp_path))
        assert q.total == 0
        assert q.current() is None

    def test_position_str(self, queue_dir):
        q = FailureQueue(str(queue_dir))
        assert q.position_str() == "1/3"
        q.skip()
        assert q.position_str() == "2/3"


# ---------------------------------------------------------------------------
# Episode recorder tests
# ---------------------------------------------------------------------------

class TestEpisodeRecorder:
    """Test episode recording and PKL writing."""

    def test_add_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = EpisodeRecorder("Top_Long_Seen_0", "top-long-sleeve", 42,
                                   Path(tmpdir), session_idx=0)
            obs = {
                "observation.state": np.zeros(12, dtype=np.float32),
                "observation.images.top_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
                "observation.images.left_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
                "observation.images.right_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
                "check_status": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            }
            action = np.ones(12, dtype=np.float32)
            rec.add_frame(obs, action)

            assert len(rec.frames) == 1
            f = rec.frames[0]
            assert f["task"] == "Top_Long_Seen_0"
            assert f["is_keyframe"] is True
            assert np.isnan(f["success_pred"])
            np.testing.assert_array_equal(f["action"], action)

    def test_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = EpisodeRecorder("G", "t", 0, Path(tmpdir))
            obs = {
                "observation.state": np.zeros(12, dtype=np.float32),
                "observation.images.top_rgb": np.zeros((10, 10, 3), dtype=np.uint8),
                "observation.images.left_rgb": np.zeros((10, 10, 3), dtype=np.uint8),
                "observation.images.right_rgb": np.zeros((10, 10, 3), dtype=np.uint8),
            }
            for _ in range(30):
                rec.add_frame(obs, np.zeros(12))
            assert abs(rec.frames[29]["timestamp"] - 29 / 30.0) < 0.001

    def test_pkl_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rec = EpisodeRecorder("Top_Long_Seen_0", "top-long-sleeve", 42,
                                   Path(tmpdir), session_idx=5)
            obs = {
                "observation.state": np.arange(12, dtype=np.float32),
                "observation.images.top_rgb": np.ones((10, 10, 3), dtype=np.uint8) * 128,
                "observation.images.left_rgb": np.ones((10, 10, 3), dtype=np.uint8) * 64,
                "observation.images.right_rgb": np.ones((10, 10, 3), dtype=np.uint8) * 32,
                "check_status": np.array([1.0, 0.0], dtype=np.float32),
            }
            action = np.ones(12, dtype=np.float32) * 0.5
            rec.add_frame(obs, action)
            rec.add_frame(obs, action)

            pkl_path, _video_path = rec.save(success=True)
            assert pkl_path is not None
            assert pkl_path.exists()
            assert "dagger" in pkl_path.name

            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)

            assert "frames" in payload
            assert len(payload["frames"]) == 2
            assert payload["garment_info"]["garment_name"] == "Top_Long_Seen_0"
            assert payload["dagger_metadata"]["session_idx"] == 5

            # Verify frame contents survived roundtrip
            f0 = payload["frames"][0]
            np.testing.assert_array_equal(f0["action"], action)
            np.testing.assert_array_equal(
                f0["observation.images.top_rgb"],
                np.ones((10, 10, 3), dtype=np.uint8) * 128,
            )


# ---------------------------------------------------------------------------
# Observation decoding tests
# ---------------------------------------------------------------------------

class TestDecodeObservation:
    """Test WebSocket observation decoding."""

    def test_decode_base64_image(self):
        import base64
        img = np.ones((10, 20, 3), dtype=np.uint8) * 42
        encoded = {
            "observation.images.top_rgb": {
                "base64": base64.b64encode(img.tobytes()).decode("ascii"),
                "shape": [10, 20, 3],
                "dtype": "uint8",
            },
            "frame_idx": 5,
        }
        decoded = decode_observation(encoded)
        np.testing.assert_array_equal(decoded["observation.images.top_rgb"], img)
        assert decoded["frame_idx"] == 5

    def test_decode_list_to_array(self):
        encoded = {"observation.state": [1.0, 2.0, 3.0]}
        decoded = decode_observation(encoded)
        np.testing.assert_array_equal(decoded["observation.state"], [1.0, 2.0, 3.0])

    def test_passthrough_scalars(self):
        encoded = {"success": True, "frame_idx": 42}
        decoded = decode_observation(encoded)
        assert decoded["success"] is True
        assert decoded["frame_idx"] == 42


# ---------------------------------------------------------------------------
# UI tests (minimal — no actual window)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# HF integration tests
# ---------------------------------------------------------------------------

class TestDaggerManifest:
    """Test dagger manifest and HF integration helpers."""

    def test_get_solved_npz_names_empty(self):
        assert get_solved_npz_names({}) == set()

    def test_get_solved_npz_names(self):
        manifest = {
            "a.npz": {"outcome": "success", "session": "s1"},
            "b.npz": {"outcome": "failure", "session": "s1"},
            "c.npz": {"outcome": "success", "session": "s2"},
            "d.npz": {"outcome": "semi_success", "session": "s2"},
            "e.npz": {"outcome": "discarded", "session": "s2"},
        }
        solved = get_solved_npz_names(manifest)
        assert solved == {"a.npz", "c.npz", "d.npz", "e.npz"}

    def test_failure_queue_exclude(self, tmp_path):
        """FailureQueue skips excluded NPZ files."""
        for name, garment in [("G1_seed1_ep0", "G1"), ("G2_seed2_ep0", "G2")]:
            meta = {"garment": garment, "garment_type": "top", "seed": 1}
            np.savez(
                tmp_path / f"{name}.npz",
                garment_points=np.zeros((5, 3), dtype=np.float32),
                garment_velocities=np.zeros((5, 3), dtype=np.float32),
                left_joint_pos=np.zeros(6, dtype=np.float32),
                right_joint_pos=np.zeros(6, dtype=np.float32),
                metadata=json.dumps(meta),
            )
        q = FailureQueue(str(tmp_path), exclude_npzs={"G1_seed1_ep0.npz"})
        assert q.total == 1
        assert q.current()["garment"] == "G2"

    def test_failure_queue_npz_filename_tracked(self, tmp_path):
        """FailureQueue items include npz_filename."""
        meta = {"garment": "G1", "garment_type": "top", "seed": 1}
        np.savez(
            tmp_path / "G1_seed1_ep0.npz",
            garment_points=np.zeros((5, 3), dtype=np.float32),
            garment_velocities=np.zeros((5, 3), dtype=np.float32),
            left_joint_pos=np.zeros(6, dtype=np.float32),
            right_joint_pos=np.zeros(6, dtype=np.float32),
            metadata=json.dumps(meta),
        )
        q = FailureQueue(str(tmp_path))
        assert q.current()["npz_filename"] == "G1_seed1_ep0.npz"

    def test_episode_recorder_npz_filename(self, tmp_path):
        """EpisodeRecorder stores npz_filename in dagger_metadata."""
        rec = EpisodeRecorder("G1", "top", 42, tmp_path, 0, npz_filename="G1_seed42_ep0.npz")
        rec.add_frame({
            "observation.state": np.zeros(12),
            "observation.images.top_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.images.left_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
            "observation.images.right_rgb": np.zeros((240, 320, 3), dtype=np.uint8),
        }, np.zeros(12))
        pkl_path, _ = rec.save(success=True)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        assert data["dagger_metadata"]["npz_filename"] == "G1_seed42_ep0.npz"

    def test_load_pipeline_config(self, tmp_path):
        """load_pipeline_config reads YAML correctly."""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text("hf_dataset_repo: test/repo\nconfig_name: test\n")
        cfg = load_pipeline_config(str(cfg_path))
        assert cfg["hf_dataset_repo"] == "test/repo"

    def test_get_next_session_dir(self, tmp_path, monkeypatch):
        """get_next_session_dir auto-increments."""
        import dagger_collect
        monkeypatch.setattr(dagger_collect, "DAGGER_BASE_DIR", tmp_path)
        # No existing sessions
        assert get_next_session_dir() == tmp_path / "session_001"
        # Create session_001
        (tmp_path / "session_001").mkdir()
        assert get_next_session_dir() == tmp_path / "session_002"
        # Create session_005 (gap)
        (tmp_path / "session_005").mkdir()
        assert get_next_session_dir() == tmp_path / "session_006"


# ---------------------------------------------------------------------------
# Dataset schema regression
# ---------------------------------------------------------------------------

class TestDaggerFrameSchema:
    """Regression guard: dagger must keep producing frames that the current
    ``EvalDatasetWriter(use_value=True)`` accepts. When a new required column
    is added to the schema (as happened with ``completion_pred`` + ``ttc_pred``
    in commit c9926a0, which silently broke dagger for ~110 commits), this
    test should fail with a clear ``Feature mismatch`` message.
    """

    @staticmethod
    def _raw_frame(image_shape):
        h, w, _ = image_shape
        return {
            "observation.state": np.zeros(12, dtype=np.float32),
            "action": np.zeros(12, dtype=np.float32),
            "observation.images.top_rgb": np.zeros((h, w, 3), dtype=np.uint8),
            "observation.images.left_rgb": np.zeros((h, w, 3), dtype=np.uint8),
            "observation.images.right_rgb": np.zeros((h, w, 3), dtype=np.uint8),
            "task": "G1",
        }

    def test_dagger_frame_keys_match_writer_features(self):
        """Key set of dagger frame must equal EvalDatasetWriter(use_value=True) features."""
        from lehome_solution.eval.dataset_writer import EvalDatasetWriter
        image_shape = (240, 320, 3)
        writer = EvalDatasetWriter(
            root=Path("/nonexistent"),  # never created
            repo_id="schema-check",
            use_value=True,
            image_shape=image_shape,
        )
        frame = build_dagger_frame(
            raw=self._raw_frame(image_shape),
            writer_image_shape=image_shape,
            default_task="G1",
            binary_success=1.0,
            garment_type_id=0,
            dense_reward=0.0,
            checkpoint_held=0.0,
        )
        # "task" is a LeRobot-special add_frame arg (maps to task_index), not a
        # feature column — exclude it from the schema comparison.
        expected = set(writer.features.keys())
        actual = set(frame.keys()) - {"task"}
        missing = expected - actual
        extra = actual - expected
        assert not missing and not extra, (
            f"dagger frame schema drift. missing={sorted(missing)} extra={sorted(extra)}"
        )

    def test_dagger_frame_accepted_by_lerobot(self, tmp_path):
        """End-to-end: LeRobot add_frame must accept the dagger frame dict."""
        from lehome_solution.eval.dataset_writer import EvalDatasetWriter
        image_shape = (240, 320, 3)
        writer = EvalDatasetWriter(
            root=tmp_path / "ds",
            repo_id="schema-check",
            use_value=True,
            image_shape=image_shape,
        ).create()
        frame = build_dagger_frame(
            raw=self._raw_frame(image_shape),
            writer_image_shape=image_shape,
            default_task="G1",
            binary_success=1.0,
            garment_type_id=0,
            dense_reward=0.5,
            checkpoint_held=1.0,
        )
        # Raises ValueError("Feature mismatch ...") on any schema drift.
        writer.dataset.add_frame(frame)
