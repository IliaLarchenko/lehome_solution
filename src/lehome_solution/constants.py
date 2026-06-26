"""Canonical constants for the LeHome project.

All garment types, action dimensions, image keys, and other project-wide
constants live here. Other modules should import from here instead of
defining their own copies.
"""

# Garment type definitions (canonical ordering — do not reorder)
GARMENT_TYPES = ("top_long", "top_short", "pant_long", "pant_short")
GARMENT_TYPE_TO_ID: dict[str, int] = {name: i for i, name in enumerate(GARMENT_TYPES)}
NUM_GARMENT_TYPES = len(GARMENT_TYPES)

# Action space
ACTION_DIM = 12  # 6 joints per arm × 2 arms
ACTION_HORIZON = 30  # 1 second at 30 FPS

# Garment type -> directory prefix under Challenge_Garment/Release/
GARMENT_TYPE_PREFIX = {
    "top_long": "Top_Long",
    "top_short": "Top_Short",
    "pant_long": "Pant_Long",
    "pant_short": "Pant_Short",
}

# Value function / advantage constants
EMA_ALPHA = 0.2  # EMA smoothing for P(success) prediction
SUCCESS_TAIL_K = 20  # Frames to correct at episode end (value tail correction)
SUCCESS_TAIL_BOOST = 20.0  # Multiplier for success-head BCE loss on success tail frames

# Stuck detection constants
FPS = 30
STUCK_WINDOW_SECONDS = 4  # Rolling window for stuck detection (seconds)
STUCK_WINDOW = STUCK_WINDOW_SECONDS * FPS  # 120 frames
STUCK_THRESHOLD = 0.01  # Std threshold below which state/action is considered stuck
STUCK_KEEP_SECONDS = 1  # Keep 1 second into the stuck window
STUCK_KEEP = STUCK_KEEP_SECONDS * FPS  # 30 frames

# BC/DAgger FR-proportional sampling
BC_FR_SCALE = 3.0  # exp(BC_FR_SCALE * (1 - SR_garment)) sampling weight for BC/DAgger frames
DAGGER_GARMENT_MAX_BOOST = 1.0  # DAgger per-episode boost cap: a rare-garment episode is sampled at most this multiple of the average episode's probability (guard = avg_eps_per_garment / DAGGER_GARMENT_MAX_BOOST; dividing per-frame weight by max(n_ep_g, guard) equalizes per-garment mass for n_ep_g ≥ guard)

# World-modeling / keypoint-distance auxiliary head layout.
# Raw per-frame check_distances has width up to 7 (long_pant uses all 7; tops
# use 5; pant_short uses 4). The keypoint_distance head produces
# KEYPOINT_HEAD_WIDTH = 21 outputs with per-garment-type slices — aligned to
# GARMENT_TYPES = ("top_long", "top_short", "pant_long", "pant_short"):
#   top_long   [0:5]   (5 conditions, uses raw[0:5])
#   top_short  [5:10]  (5 conditions, uses raw[0:5])
#   pant_long  [10:17] (7 conditions, uses raw[0:7])
#   pant_short [17:21] (4 conditions, uses raw[0:4])
# Rationale: per-garment slices so the head can learn garment-conditioned semantics,
# even though tops happen to share check_top_sleeve internally.
KEYPOINT_HEAD_WIDTH = 21

# Per-distance comparator for check_distances, keyed by garment_type.
# "<=" means the success condition is "ratio ≤ 1" (keypoints close together);
# ">=" means "ratio ≥ 1" (keypoints apart). Pant_long has 3 extra "<="
# conditions appended by eval/launcher.py PATCH 5 (threshold = 20cm × scale).
DISTANCE_CMPS: dict[str, tuple[str, ...]] = {
    "top_long":   ("<=", "<=", "<=", ">=", ">="),
    "top_short":  ("<=", "<=", "<=", ">=", ">="),
    "pant_long":  ("<=", ">=", ">=", "<=", "<=", "<=", "<="),
    "pant_short": ("<=", "<=", ">=", ">="),
}
# (slice_start, slice_width) keyed by garment_type_id (0..3).
KEYPOINT_SLICES: dict[int, tuple[int, int]] = {
    0: (0, 5),    # top_long
    1: (5, 5),    # top_short
    2: (10, 7),   # pant_long
    3: (17, 4),   # pant_short
}
# Clip cap applied to keypoint distance TARGETS (not saved raw values) during
# both training loss and rollout-metric aggregation. Distances are stored as
# ``value / threshold``; anything materially above 1.0 is "far" and any
# fine-grained structure above ~3 is noise we don't want the MSE to chase.
# We deliberately DO NOT clip the raw column in the parquet — only the target
# the model is trained against and the ground truth used for metric MAE.
KEYPOINT_DISTANCE_CAP = 3.0
