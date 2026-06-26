# Learning to Fold — My Prizewinning Solution for the LeHome Challenge 2026

**[LeHome Challenge 2026](https://github.com/lehome-official/lehome-challenge)** (ICRA 2026) is a competition on bimanual garment folding with a [LeRobot SO-ARM101](https://github.com/huggingface/lerobot) setup. This system placed **1st of 62 teams in the online (simulation) round** and **2nd in the real-world final**.

[![Watch the video](https://img.youtube.com/vi/_LBU3ue0CpU/maxresdefault.jpg)](https://youtu.be/_LBU3ue0CpU)

📄 **Tech report:** [arXiv:2606.27163](https://arxiv.org/abs/2606.27163) · ✍️ **Blog post:** [ilialarchenko.com/projects/lehome2026](https://ilialarchenko.com/projects/lehome2026) · 🤗 **Checkpoints & data:** [`lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim) (first-round policy) · [`lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real) (second-round policy)

---

## What this is

You can use this code to reproduce my solution: the whole asynchronous distributed RL cycle, model training, autonomous rollout collection, DAgger and manual teleop collection, and evaluation of the trained policies in simulation and on the real robot.

The full explanation of the logic is in the tech report and the blog post — please refer to them for the details.

This code was written under competition pressure, so it is not production-ready; I would rather recommend using it as a reference alongside the tech report. There are also some naming inconsistencies relative to the tech report — I wrote the report afterwards and tried to present everything more clearly, while the code still carries a lot of legacy. 

I did a big refactoring of the code before releasing, and still running some tests. All the main logic should work well, but if you are trying to reproduce the results and something doesn't work as expected, please raise an issue.

## Important dependencies

This repo uses my fork of the official LeHome Challenge repository: https://github.com/IliaLarchenko/lehome-challenge — you need it to run rollout collection. The fork contains extra logic that adds flexibility and lets me collect privileged simulation data during rollouts (used only for training). The actual submission runs against the *official* LeHome Challenge repository — see the [submission process I proposed and used](https://github.com/lehome-official/lehome-challenge/pull/65).

My policy is built on top of Pi0.5 and I actively use the [openpi](https://github.com/Physical-Intelligence/openpi) repository. I reference back to openpi whenever I build on top of logic from there.

Finally, I reused a lot of ideas from my team's winning solution of the [BEHAVIOR-1K Challenge 2025](https://github.com/IliaLarchenko/behavior-1k-solution) — I recommend you check it out too.

---

## Setup

Requires an NVIDIA GPU (training was done on an H200, most of rollouts on RTX PRO 6000; 500+ GB disk for data + checkpoints).

```bash
# Clone with submodules (OpenPI + LeHome Challenge)
git clone --recurse-submodules https://github.com/IliaLarchenko/lehome_solution
cd lehome_solution

# Installs system deps, venvs, Isaac Sim, and downloads assets
bash setup.sh
# This script will install all the dependencies (while resolving some issues I have faced during my experiments)
# It will also download the assets. You can skip the data download by using the flag --no-data.
# Setup is pretty extensive - if you want lighter version for you sytem - you can inspect it and run only the steps you need.
```

Python is invoked via `uv run python ...`. Authenticate with Hugging Face once to pull data and checkpoints:

```bash
uv run huggingface-cli login
uv run python scripts/download_hf_assets.py  # sim + real BC datasets, can download separately if you want
```

Both BC datasets ship with a single generic task string (`"Fold the Garment"`) and no per-episode garment label. So you must enrich the task labels once after downloading:

```bash
uv run python scripts/enrich_garment_type.py --domain sim   # data/lehome_challenge_merged/four_types_merged
uv run python scripts/enrich_garment_type.py --domain real  # data/lehome_real/four_types_merged
```

(Run only the domain(s) you downloaded; pass `--dry_run` first to preview the episode → garment mapping. This rewrites the dataset parquets in place.)

Authenticate to wandb if you wnat to use it for logging:
```bash
uv run wandb login
```

Otherwise disable it in the config files.
```yaml
wandb_enabled: false
```

---

## Main entry points

Three workflows cover almost everything:

1. **The RL pipeline** (`run_rl_pipeline.py`) — distributed BC + RL training in sim.
2. **The rollout worker / eval** (`run_eval.py`, or the pipeline's `--rollout_worker`) — run a checkpoint in Isaac Sim to evaluate it and collect data.
3. **The real-robot runner** (`record_real_dagger.py`) — the one you'll use with the real robot: autonomous rollout + DAgger teleop corrections, also the recorder for all real datasets.

---

## 1. Distributed RL pipeline (sim training)

![RL flywheel](media/rl_flywheel.svg)

The sim track is an automated loop — **BC warmup → rollout → advantage recomputation → RL training → rollout …** — orchestrated by `run_rl_pipeline.py` and configured entirely in `configs/rl_pipeline_sim.yaml`.

It runs **distributed**, with Hugging Face Hub as the message bus between two roles:

- **Trainer** — trains on the latest rollout datasets, recomputes advantages, and uploads new checkpoints to HF.
- **Rollout worker(s)** — poll HF for the newest checkpoint, collect Isaac Sim episodes (random / curriculum / hard-mining / success-replay strategies), and upload the datasets back to HF.

Trainer and workers can run on the same machine or on separate machines; you can attach many rollout workers to one trainer to scale collection. An HF sync daemon handles all uploads/downloads asynchronously in the background.

### Config

First, create or edit `configs/rl_pipeline_sim.yaml` to point at your own Hugging Face model and dataset repos:

```yaml
hf_model_repo:   <your-hf-username>/<model-repo>      # checkpoints bus
hf_dataset_repo: <your-hf-username>/<dataset-repo>    # rollout datasets bus
```

Then point at the initial BC training dataset, and at any rollout/DAgger datasets you want to seed the mix with when continuing from a previous run. Each entry has a `sampling_share` that decays per iteration toward the per-group `min_sampling_share` as fresh rollouts accumulate:

```yaml
bc_dataset:
  root: data/lehome_challenge_merged/four_types_merged
  initial_sampling_share: 1.0
  min_sampling_share: 0.9

initial_rl_datasets:        # autonomous rollouts to start from (optional)
  - root: <path to a rollouts dataset>
    sampling_share: 1.0

initial_dagger_datasets:    # human DAgger recoveries to start from (optional)
  - root: <path to a dagger dataset>
    sampling_share: 1.0
```

Most other knobs (rollout strategies, augmentation, advantage/GAE, success-replay, precision boost, LR schedule) are documented inline in the YAML. Then run the training:

```bash
# Compute norm stats and fast tokenizer.
uv run python scripts/compute_norm_stats.py   --config-name pi_modified_bc_rl --extra_roots data/lehome_challenge_merged/four_types_merged
uv run python scripts/train_fast_tokenizer.py --config-name pi_modified_bc_rl --extra_roots data/lehome_challenge_merged/four_types_merged
# Trainer (one process): trains, recomputes advantages, uploads checkpoints
uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --trainer
```

I recommend waiting until the full warmup is done, but technically you can start rollout collection as soon as the first checkpoint is uploaded to Hugging Face. Just run the rollout worker on the same or a different machine:

```bash
# Rollout worker(s): poll HF for checkpoints, collect episodes, upload datasets
uv run python scripts/run_rl_pipeline.py --config configs/rl_pipeline_sim.yaml --rollout_worker
```

That's it — everything is synchronized between the trainer and the rollout workers through the Hugging Face Hub: new rollouts are added to the training mix and the latest checkpoints are used for the next rollouts.

Most scripts have many more options and flags to control training and rollout collection in detail; see the code and the inline comments.

---

## 2. Rollout worker / evaluation

To evaluate a checkpoint (or just watch it fold) without the full training loop, `run_eval.py` starts a policy server itself and launches N Isaac Sim workers, each cycling through garments via `env.switch_garment()`.

Grab my final sim-round submission policy [`lehome_sim`](https://huggingface.co/IliaLarchenko/lehome_sim) and evaluate it:

```bash
# Download the submission checkpoint
uv run hf download IliaLarchenko/lehome_sim --local-dir outputs/checkpoints/lehome_sim

# Evaluate it across all garments with 4 parallel Isaac Sim workers
uv run python scripts/run_eval.py \
  --checkpoint_dir outputs/checkpoints/lehome_sim \
  --config_name pi_modified_bc_rl \
  --num_workers 4 --all
```

Videos, success metrics, and a LeRobot eval dataset land under `outputs/eval_videos/`. The garment subset defaults to **unseen**; pass `--all` for every garment or `--seen_only` for the seen set, `--garment_types top_long` to restrict to specific garments, and `--no_wandb` to skip logging. (Inside the pipeline, the `--rollout_worker` role does the same collection on a loop, feeding the trainer.)

**Just want success rates? Use `--metrics_only`** — it skips the heavy, disk-hungry artifacts (trajectory pkls, the LeRobot dataset, debug videos), so it runs faster and writes almost nothing. Only `eval_summary.txt` / `eval_results.json` (and logs) are still produced:

```bash
# Fast benchmark: success rates only, 2 episodes per garment, no videos/dataset/pkls
uv run python scripts/run_eval.py \
  --checkpoint_dir outputs/checkpoints/lehome_sim \
  --config_name pi_modified_bc_rl \
  --num_workers 4 --all \
  --num_episodes 2 --metrics_only --no_wandb
```

(`--metrics_only` is shorthand for `--no_save_pkl --no_save_dataset --no_save_debug_video`; `--no_save_pkl` alone implies the other two, since the dataset and videos are rendered from the pkls.)

This is the checkpoint I used for the final sim-round submission; if everything works you should see 80%+ average success rate on seen garments, and slightly lower on the released unseen garments.

---

## 3. Real robot — DAgger runner

`record_real_dagger.py` is the primary real-robot tool: it runs the policy autonomously and lets a human take over with teleop leaders to correct failures (DAgger), recording everything as a training dataset. It runs in the **lehome-challenge venv** (needs lerobot + the hardware stack) and reads all hardware mapping (arm ports, leader ports, cameras) from `configs/real_robot.yaml`.

To run it you will need a physical setup: two leader arms, two follower arms (46 cm apart), and three cameras — one overhead (~65 cm above the table) and two wrist cameras. Refer to the [official real dataset](https://huggingface.co/datasets/lehome/dataset_challenge_real) to align the camera placement. I also provide a calibration script to simplify the alignment: `scripts/real_camera_align.py`.

First, **edit `configs/real_robot.yaml` to match your rig** — the follower/leader serial ports and the camera device IDs are machine-specific, so replace the committed values with your own (`ls /dev/serial/by-id/` for the arms, `ls /dev/v4l/by-path/` for the wrist cameras):

```yaml
arms:
  left:
    port:        /dev/serial/by-id/<your-left-follower>
    leader_port: /dev/serial/by-id/<your-left-leader>
  right:
    port:        /dev/serial/by-id/<your-right-follower>
    leader_port: /dev/serial/by-id/<your-right-leader>
cameras:
  top:          { serial: "<your-realsense-serial>" }   # I used RealSense top camera, it requires slightly different setup
  # But you can use the same config as for the wrist cameras if you use different camera.
  left_wrist:   { device: /dev/v4l/by-path/<your-left-wrist> }
  right_wrist:  { device: /dev/v4l/by-path/<your-right-wrist> }
```

There are separate ways to run autonomous rollouts or manual data collection, but since DAgger combines both I recommend simply using it for everything.

For (semi-)autonomous rollouts you will need a policy trained for the real robot. You can use my final submission policy [`lehome_real`](https://huggingface.co/IliaLarchenko/lehome_real) or train your own.

My policy is fairly robust — I expect it to work relatively well if your real-life setup is similar to the official competition setup.

**DAgger mode (default)** needs a policy server running in another terminal (main venv):

```bash
# Terminal 1 — policy server (main venv). Download the real policy first:
#   uv run hf download IliaLarchenko/lehome_real --local-dir outputs/checkpoints/lehome_real
uv run python scripts/serve.py \
  --actions_to_execute 5 --actions_to_keep 1 --execute_in_n_steps 5 --num_steps 10 \
  policy:checkpoint --policy.config pi_modified_real_bc --policy.dir outputs/checkpoints/lehome_real

# Terminal 2 — robot runner (lehome-challenge venv)
source lehome-challenge/.venv/bin/activate
python scripts/record_real_dagger.py --config configs/real_robot.yaml \
  --garment top_long --server ws://localhost:8000
```

Each episode starts as an autonomous rollout; `SPACE` toggles into manual teleop correction (and back), `→` saves and advances, `←` discards and re-records, `Esc` stops. Per-frame `task_is_policy` labels (human vs. policy) are stored so training can up-weight the human corrections. The policy is trained in the robot's native degree-mode units and camera names, so nothing is converted on the inference path.

For convenience I use a foot pedal bound to the `SPACE`, `→` and `←` keys — I highly recommend it and find it very useful for teleop DAgger data collection.

**Manual-only mode** is pure teleop recording (BC data), no server needed:

```bash
python scripts/record_real_dagger.py --config configs/real_robot.yaml --garment top_long --manual_only
```

### DAgger in simulation

The sim equivalent is `dagger_collect.py`: it restores saved failure states in Isaac Sim and lets you teleoperate recoveries with the bimanual SO101 leaders. It can run config-driven (pulling failure states from HF and uploading recoveries) or fully local:

```bash
uv run python scripts/dagger_collect.py --config configs/rl_pipeline_sim.yaml
# or local:
uv run python scripts/dagger_collect.py \
  --failure_dir outputs/eval_videos/rl_XXX/physics_states/failure \
  --output_dir outputs/dagger_episodes/session_001
```

I encourage you to try it, but in practice teleop in simulation is pretty hard.

---

## Training on real data

All real-training parameters live in one file: `configs/train_real_bc.yaml`. The mix is three buckets — organizer BC, home teleop + DAgger recoveries, and sim success-replays.

Unlike the sim track, it is not wrapped in a single automated RL pipeline, so you run each step manually. First set `init_checkpoint` (the sim policy to transfer from — e.g. the downloaded `lehome_sim`) and `hf_model_repo` (your own repo for checkpoint uploads) in the YAML.

```bash
# 1. Recompute norm stats + FAST tokenizer over the full union of sources
uv run python scripts/compute_norm_stats.py   --config-name configs/train_real_bc.yaml --all-sources
uv run python scripts/train_fast_tokenizer.py --config-name configs/train_real_bc.yaml --all-sources

# 2. Train (resume from latest checkpoint). You can also --overwrite the checkpoint and start from scratch.
uv run python scripts/train.py --yaml configs/train_real_bc.yaml --resume
```

## Citation

If you find this work useful, please cite it as:

```bibtex
@misc{larchenko2026lehome,
      title         = {{Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)}},
      author        = {Larchenko, Ilia},
      year          = {2026},
      eprint        = {2606.27163},
      archivePrefix = {arXiv},
      primaryClass  = {cs.RO},
      url           = {https://arxiv.org/abs/2606.27163}
}
```

