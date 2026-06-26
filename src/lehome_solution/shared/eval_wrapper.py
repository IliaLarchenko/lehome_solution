"""LeHome policy wrapper: stateless chunk inference with rolling inpainting."""

import logging
import threading
import numpy as np
import cv2
import dataclasses

logger = logging.getLogger(__name__)

def resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize an image to (target_h, target_w) with letterbox padding."""
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


@dataclasses.dataclass
class LeHomeWrapperConfig:
    """Configuration for LeHome policy wrapper execution parameters."""
    actions_to_execute: int = 15
    actions_to_keep: int = 2
    execute_in_n_steps: int = 15
    num_steps: int = 10
    # Best-of-N rollout sampling. When > 1, each worker's single inference
    # request is expanded into N candidates sharing the same prefix KV cache
    # but with independent flow-matching noise. Candidates are scored by the
    # WM-flow Δsuccess head (already a residual over the V-head baseline),
    # averaged across the CFG cond/uncond passes:
    #   ``0.5 * (s_cond + s_uncond)``
    # and the argmax candidate's chunk is returned. Adds roughly N×
    # action-expert cost (the VLM prefix is shared), no change to the batch
    # protocol seen by workers.
    num_rollout_candidates: int = 1
    # Realtime inference: when True, skip the unconditional CFG forward pass
    # entirely. Only the conditional (advantage>0) branch runs per denoising
    # step, ~2× faster. Loses CFG amplification — set this only when cfg_scale
    # is implicitly 1.0 (the conditional pass is the final answer). Used by the
    # real-robot server path (scripts/serve.py).
    cfg_disabled: bool = False


class LeHomePolicyWrapper:
    """Stateless policy wrapper for LeHome models.

    ``infer_chunk_batched`` maps observations to action chunks; all per-episode
    state (chunk caching, inpaint anchors) is owned by the caller (the sim
    worker proxy / the real-robot client).
    """

    def __init__(
        self,
        policy,
        action_horizon: int = 30,
        config: LeHomeWrapperConfig | None = None,
    ) -> None:
        self.policy = policy
        self.action_horizon = action_horizon
        self.config = config if config is not None else LeHomeWrapperConfig()

        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError("actions_to_execute + actions_to_keep exceeds action_horizon")

        # Lock for policy.infer() — the base policy mutates self._rng on each
        # call, so concurrent calls would corrupt it. Only one GPU inference
        # runs at a time anyway.
        self._infer_lock = threading.Lock()

    # --- Core logic ---

    def _process_obs(self, obs: dict) -> dict:
        state = np.asarray(obs["observation.state"], dtype=np.float32)
        # Do NOT resize here. The policy's own transform pipeline applies ResizeImages(224, 224)
        # via JAX (same method as training). Pre-resizing with cv2 would degrade image quality
        # and make results resolution-dependent even though they should be identical.
        result = {
            "observation/top_rgb": np.asarray(obs["observation.images.top_rgb"])[..., :3],
            "observation/left_rgb": np.asarray(obs["observation.images.left_rgb"])[..., :3],
            "observation/right_rgb": np.asarray(obs["observation.images.right_rgb"])[..., :3],
            "observation/state": state,
        }
        if "garment_type_id" in obs:
            result["garment_type_id"] = np.int32(obs["garment_type_id"])
        return result

    def _interpolate_actions(self, actions: np.ndarray, target_steps: int) -> np.ndarray:
        from scipy.interpolate import interp1d
        original_indices = np.linspace(0, len(actions) - 1, len(actions))
        target_indices = np.linspace(0, len(actions) - 1, target_steps)
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind="cubic")
            interpolated[:, dim] = f(target_indices)
        return interpolated

    def infer_chunk_batched(
        self,
        requests: list[tuple[dict, np.ndarray | None, dict | None]],
    ) -> list[tuple[np.ndarray, np.ndarray | None, float | None, float | None, dict]]:
        """Batched inference: process multiple (obs, initial_actions, config) requests at once.

        Each request is independently post-processed (interpolation, inpainting extraction).
        The GPU forward pass is batched.

        Args:
            requests: list of (obs, initial_actions, inference_config) tuples.

        Returns:
            list of (chunk, next_initial, value_pred, checkpoint_pred, effective_config) tuples.
        """
        B = len(requests)
        if B == 0:
            return []

        # Resolve per-request wrapper-level params and collect model inputs
        obs_list = []
        ia_list = []
        overrides_list = []
        ate_list = []
        atk_list = []
        eins_list = []
        n_list: list[int] = []  # per-request num_rollout_candidates

        for obs, initial_actions, inference_config in requests:
            if inference_config:
                ate = inference_config.get("actions_to_execute", self.config.actions_to_execute)
                atk = inference_config.get("actions_to_keep", self.config.actions_to_keep)
                eins = inference_config.get("execute_in_n_steps", self.config.execute_in_n_steps)
                n_i = inference_config.get(
                    "num_rollout_candidates", self.config.num_rollout_candidates
                )
                model_overrides = {}
                if "time_threshold_inpaint" in inference_config:
                    model_overrides["time_threshold_inpaint"] = inference_config["time_threshold_inpaint"]
                if "noise_temperature" in inference_config:
                    model_overrides["noise_temperature"] = inference_config["noise_temperature"]
                if "cfg_scale" in inference_config:
                    model_overrides["cfg_scale"] = inference_config["cfg_scale"]
                if "explore_noise_scale" in inference_config:
                    model_overrides["explore_noise_scale"] = inference_config["explore_noise_scale"]
            else:
                ate = self.config.actions_to_execute
                atk = self.config.actions_to_keep
                eins = self.config.execute_in_n_steps
                n_i = self.config.num_rollout_candidates
                model_overrides = {}

            obs_list.append(self._process_obs(obs))
            ia_list.append(initial_actions)
            overrides_list.append(model_overrides if model_overrides else None)
            ate_list.append(ate)
            atk_list.append(atk)
            eins_list.append(eins)
            n_list.append(max(1, int(n_i)))

        # Best-of-N: candidates share the SAME VLM prefix output (cheap part:
        # tile kv_cache) but each gets independent flow-matching noise so
        # trajectories diverge. We then score by the WM-flow Δsuccess
        # ``0.5 * (s_cond + s_uncond)`` and pick the argmax candidate.
        # N=1 means "no best-of-N" (original path).
        #
        # N is per-request: Thompson sampling can pick a different N per
        # episode. We pass a UNIFORM N_batch = max(n_list) into the policy
        # so the policy server's total batch size stays predictable
        # (otherwise every distinct total size triggers a JAX recompile and
        # the orchestrator's 300s no-output killer fires). The policy runs
        # the VLM prefix on B unique requests and tiles to B*N_batch for
        # the flow loop. Selection below is bounded to each request's own
        # sampled n_list[i] so the bandit reward stays unbiased — the
        # bandit gets exactly best-of-N_i quality, never a free upgrade
        # from extras that exist only for compile-shape uniformity.
        N_batch = max(n_list) if n_list else 1
        any_best_of_n = N_batch > 1

        # Single batched GPU call. obs_list is B unique requests; the policy
        # internally tiles its kv_cache to B*N_batch and returns B*N_batch
        # outputs (row k → request k//N_batch, candidate k%N_batch).
        if not self._infer_lock.acquire(timeout=300):
            raise TimeoutError("Could not acquire inference lock within 300s")
        try:
            outputs = self.policy.infer_batched(
                obs_list, ia_list, overrides_list, num_candidates=N_batch,
                cfg_disabled=self.config.cfg_disabled,
            )
        finally:
            self._infer_lock.release()

        # Best-of-N selection: pick the argmax candidate per original request.
        # The WM-flow success head now predicts Δsuccess directly
        # (= true_success − V̂(s_t), action-conditional residual), so the
        # score simplifies to the average Δ across the CFG cond/uncond
        # passes. Positive = this chunk is expected to improve over the
        # V-head baseline; negative = chunk makes things worse. Completion
        # is no longer multiplied in — its sigmoid-bounded semantics differ
        # from Δsuccess and mixing them corrupts the ranking.
        # Candidates missing either WM-flow success scalar get -inf score.
        # When all candidates are missing scores, spread stats are NaN.
        #
        # RETRY: if every finite-scored candidate in a group predicts
        # negative Δsuccess (i.e. the model thinks all N options make
        # things worse), we draw 2×N MORE candidates for that group and
        # pick the best from the full 3×N pool. Groups with no finite
        # scores at all are NOT retried — more samples won't rescue a
        # missing signal. ``best_of_n_n_valid`` reflects the full pool
        # size, so ``n_valid > N`` downstream implies retry fired.
        if any_best_of_n:
            def _score(g: dict) -> float:
                sc = g.get("wm_flow_success_cond")
                su = g.get("wm_flow_success_uncond")
                if sc is None or su is None:
                    return float("-inf")
                # Pure Δsuccess average (cond/uncond). Already a delta over V̂.
                return 0.5 * (float(sc) + float(su))

            # Group the first-pass outputs per original request, using the
            # uniform stride N_batch (= max sampled N). Then slice each group
            # down to its own sampled n_list[i] so the bandit reward reflects
            # exactly best-of-N_i (extras computed for compile-shape uniformity
            # are discarded).
            groups: list[list[dict]] = []
            for i in range(B):
                full_grp = list(outputs[i * N_batch : (i + 1) * N_batch])
                groups.append(full_grp[: n_list[i]])

            # Identify groups whose best finite score is still negative
            # (only meaningful when N_i > 1; N_i == 1 groups are skipped —
            # one sample is not "all-negative best-of-N", it's just the
            # baseline path).
            retry_needed: list[int] = []
            for i, grp in enumerate(groups):
                if n_list[i] <= 1:
                    continue
                fin = [s for s in (_score(g) for g in grp) if np.isfinite(s)]
                if fin and max(fin) < 0.0:
                    retry_needed.append(i)

            if retry_needed:
                # Uniform retry stride R = 2 × max(N_i over retried groups)
                # so the retry call reuses the policy's compiled shape for
                # batch size len(retry_needed) * R. Per-group selection still
                # truncates extras down to the group's own 2 × n_list[i].
                # The policy now does the candidate replication internally
                # (VLM prefix runs once on the unique retry inputs) — pass
                # num_candidates=R so the layout matches.
                R_per_group = [2 * n_list[i] for i in retry_needed]
                R = max(R_per_group)
                retry_obs = [obs_list[i] for i in retry_needed]
                retry_ia = [ia_list[i] for i in retry_needed]
                retry_overrides = [overrides_list[i] for i in retry_needed]
                if not self._infer_lock.acquire(timeout=300):
                    raise TimeoutError(
                        "Could not acquire inference lock within 300s (retry pass)"
                    )
                try:
                    retry_out = self.policy.infer_batched(
                        retry_obs, retry_ia, retry_overrides, num_candidates=R,
                        cfg_disabled=self.config.cfg_disabled,
                    )
                finally:
                    self._infer_lock.release()
                for j, i in enumerate(retry_needed):
                    full_extra = list(retry_out[j * R : (j + 1) * R])
                    groups[i].extend(full_extra[: R_per_group[j]])
                logger.debug(
                    "best-of-N retry: %d/%d groups had all-negative first pass",
                    len(retry_needed), B,
                )

            selected: list[dict] = []
            chosen_indices: list[int] = []
            for grp in groups:
                arr = np.asarray([_score(g) for g in grp], dtype=np.float64)
                finite = np.isfinite(arr)
                best = int(arr.argmax()) if finite.any() else 0
                chosen = dict(grp[best])  # shallow-copy so we don't mutate the original
                if finite.any():
                    fin = arr[finite]
                    chosen["best_of_n_score_chosen"] = float(arr[best])
                    chosen["best_of_n_score_mean"] = float(fin.mean())
                    chosen["best_of_n_score_min"] = float(fin.min())
                    chosen["best_of_n_score_max"] = float(fin.max())
                    chosen["best_of_n_score_std"] = float(fin.std())
                    chosen["best_of_n_score_spread"] = float(fin.max() - fin.min())
                    chosen["best_of_n_n_valid"] = int(fin.size)
                else:
                    nan = float("nan")
                    chosen.update({
                        "best_of_n_score_chosen": nan,
                        "best_of_n_score_mean": nan,
                        "best_of_n_score_min": nan,
                        "best_of_n_score_max": nan,
                        "best_of_n_score_std": nan,
                        "best_of_n_score_spread": nan,
                        "best_of_n_n_valid": 0,
                    })
                selected.append(chosen)
                chosen_indices.append(best)
            outputs = selected
            logger.debug(
                "best-of-N selection for %d requests (N_batch=%d, N per req=%s): chosen indices = %s",
                B, N_batch, n_list, chosen_indices,
            )

        # Post-process each result independently
        results = []
        for i in range(B):
            actions = outputs[i]["actions"]
            if not isinstance(actions, np.ndarray):
                actions = np.asarray(actions)
            if len(actions.shape) == 3:
                actions = actions[0]

            ate = min(ate_list[i], len(actions))
            atk = atk_list[i]
            eins = eins_list[i]

            # Inpainting: trailing actions for next chunk
            inpainting_end = ate + atk
            if atk > 0 and len(actions) >= inpainting_end:
                next_initial = actions[ate:inpainting_end].copy()
            else:
                next_initial = None

            # Extract and interpolate chunk
            chunk = actions[:ate].copy()
            if eins != ate:
                chunk = self._interpolate_actions(chunk, eins)

            raw_value = outputs[i].get("success_pred")
            value_pred = float(raw_value) if raw_value is not None else None
            raw_cp = outputs[i].get("checkpoint_pred")
            checkpoint_pred = float(raw_cp) if raw_cp is not None else None
            raw_gt = outputs[i].get("garment_type_pred")
            garment_type_pred = int(raw_gt) if raw_gt is not None else None
            raw_comp = outputs[i].get("completion_pred")
            completion_pred = float(raw_comp) if raw_comp is not None else None
            raw_ttc = outputs[i].get("ttc_pred")
            ttc_pred = float(raw_ttc) if raw_ttc is not None else None

            eff_i = {
                "execute_in_n_steps": len(chunk),
                "actions_to_keep": atk,
                "garment_type_pred": garment_type_pred,
                "completion_pred": completion_pred,
                "ttc_pred": ttc_pred,
            }
            raw_kpt = outputs[i].get("keypoint_distances")
            if raw_kpt is not None:
                eff_i["keypoint_distances"] = np.asarray(raw_kpt, dtype=np.float32).tolist()
            for k in (
                "success_cond", "completion_cond", "keypoint_cond",
                "success_uncond", "completion_uncond", "keypoint_uncond",
            ):
                raw = outputs[i].get(f"wm_flow_{k}")
                if raw is not None:
                    arr = np.asarray(raw, dtype=np.float32)
                    eff_i[f"wm_flow_{k}"] = float(arr) if arr.ndim == 0 else arr.tolist()

            # Best-of-N selection stats (only populated when N > 1). These let
            # the terminal log and wandb answer "were the candidates actually
            # diverse?" — spread near 0 means best-of-N is futile.
            for k in (
                "best_of_n_score_chosen", "best_of_n_score_mean",
                "best_of_n_score_min", "best_of_n_score_max",
                "best_of_n_score_std", "best_of_n_score_spread",
                "best_of_n_n_valid",
            ):
                v = outputs[i].get(k)
                if v is not None:
                    eff_i[k] = v

            results.append((
                chunk,
                next_initial,
                value_pred,
                checkpoint_pred,
                eff_i,
            ))

        return results
