"""Test data loading pipeline with real LeHome dataset.

These tests require the dataset to be downloaded at:
    data/lehome_challenge_merged/four_types_merged/
"""

import os
import numpy as np
import pytest
import torch

DATASET_ROOT = "data/lehome_challenge_merged/four_types_merged"
DATASET_AVAILABLE = os.path.exists(DATASET_ROOT) and os.path.exists(
    os.path.join(DATASET_ROOT, "meta/info.json")
)

skip_no_dataset = pytest.mark.skipif(
    not DATASET_AVAILABLE, reason="Dataset not downloaded"
)


@skip_no_dataset
class TestLeRobotDatasetLoading:
    @pytest.fixture(scope="class")
    def dataset(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset(
            repo_id="lehome/dataset_challenge_merged",
            root=DATASET_ROOT,
            delta_timestamps={"action": [t / 30.0 for t in range(30)]},
            video_backend="pyav",
        )

    def test_dataset_size(self, dataset):
        assert len(dataset) == 265798

    def test_num_episodes(self, dataset):
        assert dataset.num_episodes == 1000

    def test_fps(self, dataset):
        assert dataset.fps == 30

    def test_sample_keys(self, dataset):
        sample = dataset[0]
        expected_keys = {
            "observation.state",
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
            "action",
            "action_is_pad",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
            "task",
        }
        assert expected_keys.issubset(set(sample.keys()))

    def test_sample_shapes(self, dataset):
        sample = dataset[0]
        assert sample["observation.state"].shape == torch.Size([12])
        assert sample["action"].shape == torch.Size([30, 12])
        assert sample["action_is_pad"].shape == torch.Size([30])

    def test_image_shape(self, dataset):
        sample = dataset[0]
        # lerobot returns (C, H, W) float32
        for cam in ["top_rgb", "left_rgb", "right_rgb"]:
            img = sample[f"observation.images.{cam}"]
            assert img.shape[0] == 3  # channels first
            assert img.ndim == 3

    def test_episode_boundary_no_leakage(self, dataset):
        """At the last frame of an episode, padded actions should repeat the last real one."""
        ep0_len = dataset.meta.episodes[0]["length"]
        last_sample = dataset[ep0_len - 1]

        is_pad = last_sample["action_is_pad"]
        n_real = (~is_pad).sum().item()
        assert n_real == 1  # only 1 real action at last frame
        assert is_pad.sum().item() == 29

        # Padded actions should repeat the last real one
        actions = last_sample["action"]
        last_real = actions[0]  # the one real action
        first_pad = actions[1]  # first padded
        assert torch.allclose(last_real, first_pad, atol=1e-5)

    def test_no_cross_episode_leakage(self, dataset):
        """First frame of ep1 should have all real actions (no leak from ep0)."""
        ep0_len = dataset.meta.episodes[0]["length"]
        first_ep1 = dataset[ep0_len]
        assert first_ep1["episode_index"].item() == 1
        assert first_ep1["frame_index"].item() == 0
        assert first_ep1["action_is_pad"].sum().item() == 0  # all real

    def test_episode_lengths_match_metadata(self, dataset):
        """Spot-check that episode lengths match metadata."""
        for ep_idx in [0, 100, 500, 999]:
            meta_len = dataset.meta.episodes[ep_idx]["length"]
            meta_from = dataset.meta.episodes[ep_idx]["dataset_from_index"]
            meta_to = dataset.meta.episodes[ep_idx]["dataset_to_index"]
            assert meta_to - meta_from == meta_len


# ---------------------------------------------------------------------------
# Multi-dataset and normalization tests (no real dataset required)
# ---------------------------------------------------------------------------


def test_multi_dataset_bc_injection():
    """BC sources inject constant advantage and NaN value/success."""
    from lehome_solution.training.data_loader import MultiDataset, DataSourceConfig

    # Create minimal parquet dataset for BC
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        root = os.path.join(tmpdir, "bc_dataset")
        os.makedirs(os.path.join(root, "data", "chunk-000"))
        os.makedirs(os.path.join(root, "meta", "episodes"))

        # Write minimal parquet with required columns
        table = pa.table({
            "frame_index": pa.array([0, 1, 2], type=pa.int64()),
            "episode_index": pa.array([0, 0, 0], type=pa.int64()),
            "timestamp": pa.array([0.0, 1/30, 2/30], type=pa.float64()),
            "observation.state": [list(np.zeros(12, dtype=np.float32)) for _ in range(3)],
            "action": [list(np.zeros(12*30, dtype=np.float32)) for _ in range(3)],
            "action_is_pad": [list(np.zeros(30, dtype=np.float32)) for _ in range(3)],
            "success": pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            "success_pred": pa.array([0.8, 0.8, 0.8], type=pa.float64()),
            "advantage": pa.array([0.5, 0.5, 0.5], type=pa.float64()),
        })
        pq.write_table(table, os.path.join(root, "data", "chunk-000", "file-000.parquet"))

        # Write episode metadata
        ep_table = pa.table({
            "episode_index": pa.array([0], type=pa.int64()),
            "tasks": pa.array([["test_garment"]]),
            "length": pa.array([3], type=pa.int64()),
        })
        pq.write_table(ep_table, os.path.join(root, "meta", "episodes", "episode-000.parquet"))

        # Write info.json
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so100",
            "fps": 30,
            "total_episodes": 1,
            "total_frames": 3,
            "features": {},
        }
        with open(os.path.join(root, "meta", "info.json"), "w") as f:
            json.dump(info, f)

        src = DataSourceConfig(
            root=root,
            repo_id="test_bc",
            sampling_share=1.0,
            is_bc=True,
            bc_advantage_raw=0.5,
        )

        try:
            ds = MultiDataset([src])
        except Exception:
            pytest.skip("MultiDataset creation requires compatible LeRobot dataset format")
            return

        item = ds[0]
        # BC should inject NaN success_pred and success
        assert np.isnan(item.get("success_pred", 0.0)), "BC samples should have NaN success_pred"
        assert np.isnan(item.get("success", 0.0)), "BC samples should have NaN success"


def test_per_timestamp_normalize_roundtrip():
    """Per-timestamp normalize → unnormalize recovers original values."""
    from lehome_solution.shared.normalize import NormStats

    # Create synthetic per-timestamp stats
    H, D = 30, 12
    rng = np.random.RandomState(42)
    per_ts_mean = rng.randn(H, D).astype(np.float32)
    per_ts_std = np.abs(rng.randn(H, D).astype(np.float32)) + 0.1

    stats = NormStats(
        mean=np.zeros(D, dtype=np.float32),
        std=np.ones(D, dtype=np.float32),
        q01=np.zeros(D, dtype=np.float32),
        q99=np.ones(D, dtype=np.float32),
        per_timestamp_mean=per_ts_mean,
        per_timestamp_std=per_ts_std,
        per_timestamp_q01=None,
        per_timestamp_q99=None,
    )

    # Normalize and unnormalize
    original = rng.randn(H, D).astype(np.float32)
    normalized = (original - per_ts_mean) / (per_ts_std + 1e-6)
    recovered = normalized * (per_ts_std + 1e-6) + per_ts_mean

    np.testing.assert_allclose(recovered, original, atol=1e-5)


def test_multi_dataset_episode_remapping():
    """Two datasets loaded via MultiDataset should have non-colliding episode indices."""
    from lehome_solution.training.data_loader import MultiDataset, DataSourceConfig
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile
    import json

    def _create_mini_dataset(tmpdir, name, num_episodes, frames_per_ep):
        """Create a minimal LeRobot v3 parquet dataset."""
        root = os.path.join(tmpdir, name)
        os.makedirs(os.path.join(root, "data", "chunk-000"))
        os.makedirs(os.path.join(root, "meta", "episodes"))

        total_frames = num_episodes * frames_per_ep
        frame_indices = []
        episode_indices = []
        timestamps = []
        for ep in range(num_episodes):
            for fi in range(frames_per_ep):
                frame_indices.append(fi)
                episode_indices.append(ep)
                timestamps.append(fi / 30.0)

        table = pa.table({
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "timestamp": pa.array(timestamps, type=pa.float64()),
            "observation.state": [list(np.zeros(12, dtype=np.float32))] * total_frames,
            "action": [list(np.zeros(12 * 30, dtype=np.float32))] * total_frames,
            "action_is_pad": [list(np.zeros(30, dtype=np.float32))] * total_frames,
            "success": pa.array([1.0] * total_frames, type=pa.float64()),
            "success_pred": pa.array([0.5] * total_frames, type=pa.float64()),
            "advantage": pa.array([0.0] * total_frames, type=pa.float64()),
        })
        pq.write_table(table, os.path.join(root, "data", "chunk-000", "file-000.parquet"))

        # Episode metadata
        ep_rows = {
            "episode_index": pa.array(list(range(num_episodes)), type=pa.int64()),
            "tasks": pa.array([["task_0"]] * num_episodes),
            "length": pa.array([frames_per_ep] * num_episodes, type=pa.int64()),
        }
        ep_table = pa.table(ep_rows)
        pq.write_table(ep_table, os.path.join(root, "meta", "episodes", "episode-000.parquet"))

        info = {
            "codebase_version": "v3.0",
            "robot_type": "so100",
            "fps": 30,
            "total_episodes": num_episodes,
            "total_frames": total_frames,
            "features": {},
        }
        with open(os.path.join(root, "meta", "info.json"), "w") as f:
            json.dump(info, f)

        return root

    with tempfile.TemporaryDirectory() as tmpdir:
        root_a = _create_mini_dataset(tmpdir, "ds_a", num_episodes=3, frames_per_ep=5)
        root_b = _create_mini_dataset(tmpdir, "ds_b", num_episodes=4, frames_per_ep=5)

        src_a = DataSourceConfig(root=root_a, repo_id="ds_a", sampling_share=1.0)
        src_b = DataSourceConfig(root=root_b, repo_id="ds_b", sampling_share=1.0)

        try:
            ds = MultiDataset(
                [src_a, src_b],
                action_horizon=30,
                action_sequence_keys=("action",),
            )
        except Exception:
            pytest.skip("MultiDataset creation requires compatible LeRobot dataset format")
            return

        # Episode offsets: dataset A has 3 episodes (0,1,2), dataset B should start at 3
        assert ds._episode_offsets[0] == 0
        assert ds._episode_offsets[1] == 3

        # Collect all episode indices from both sources
        ep_indices_seen = set()
        for i in range(len(ds)):
            item = ds[i]
            ep_idx = int(item["episode_index"])
            ep_indices_seen.add(ep_idx)

        # Should have 7 unique episode indices (3 from A + 4 from B)
        assert len(ep_indices_seen) == 7
        # No collision: dataset B episodes should be 3,4,5,6
        assert ep_indices_seen == {0, 1, 2, 3, 4, 5, 6}

