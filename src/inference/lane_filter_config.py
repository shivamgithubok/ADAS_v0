"""
Lane robustness quick-win knobs.

Tune these, restart the app, and compare lane clutter vs miss rate.
Start with the defaults below, then adjust one knob at a time.

Suggested sweeps:
  CONF_THRESHOLD : 0.35 → 0.45 → 0.55   (higher = fewer lanes)
  NMS_THRES_M    : 2.5  → 2.0  → 1.5    (lower  = stronger de-dupe)
  MAX_LANES      : 6 → 4                (hard cap after NMS)
  ENABLE_FILL_MISSING_LANES : False unless ego corridor looks incomplete
"""

# --- Lateral scale correction ---
# The warped model image is geometrically correct: measured against the paint
# mask under P, an ego lane in it spans 3.67 m. The network reports 4.53 m for
# the same lane, so every regressed X is 1.235x too wide. Y is not regressed
# (it is the fixed anchor grid), so one lateral divisor is the whole fix.
# Measured on GRMN6694_540 by scripts/debug/lane_scale_probe.py.
# All metre gates below are in true metres after this divisor.
LANE_X_SCALE = 1.235

# --- Detection scoring / NMS (postprocess) ---
# Aggressive + shorter range on GRMN6694_360 (stride2, N=902):
#   y50 + max_lanes 3 + |x|<=5.5  vs  mild y100/L4/x7:
#   avgL 2.85→2.45, >=4 33%→0%, CONF 71.0%→73.4%, corr ~91% held.
CONF_THRESHOLD = 0.43          # was 0.40; mild raise drops weak extras
NMS_THRES_M = 1.4              # true m; was 2.0 in inflated network m
MAX_LANES = 4                  # ego L/R + one adjacent each side
MIN_VISIBLE_POINTS = 4         # was effectively 2; drop short/noisy segments
MIN_ABS_MEAN_X_M = 0.0         # >0 rejects near-center ghosts but hurts ego pair
MAX_ABS_MEAN_X_M = 6.5         # true m; 1.85+3.7 adjacent paint now in range
MAX_LATERAL_JUMP_M = 2.02      # 2.5 / 1.235
MAX_ABS_SLOPE = 0.35           # max |Δx / Δy| over visible span (filters diagonals)
# Only use / plot / track anchors with Y <= this (meters). Model still predicts to 100.
MAX_LANE_Y_M = 50.0            # tracking / ego pair; far anchors add noise more than signal
DRAW_LANE_Y_M = 40.0           # front-view polylines; p90 draw error grows past 40 m

# --- Synthetic lane interpolation (draw path) ---
ENABLE_FILL_MISSING_LANES = False  # was always on; invents fake lines
FILL_MIN_GAP_M = 6.0               # only used if ENABLE_FILL_MISSING_LANES=True
FILL_MIN_SCORE = 0.50              # both neighbors must be >= this score if scores exist

# Shared temporal hold: EKF coast, ego-pair, and onesided reconstruct
# must use the same budget so tracks and the drivable fill drop together.
LANE_HOLD_FRAMES = 15          # ~0.5s at 30fps

# --- Temporal EKF tracking ---
EKF_MAX_MISSED_FRAMES = LANE_HOLD_FRAMES
EKF_DIST_THRESHOLD_M = 1.2     # true m; was 1.8 in inflated network m
EKF_CONFIRM_HITS = 2           # show lane after 2 consecutive detections
EKF_REQUIRE_CONFIRMED = True   # drop one-frame ghosts; show after EKF_CONFIRM_HITS

# --- Ego corridor / P0 lane-pair (ADAS) ---
EGO_LANE_WIDTH_MIN_M = 2.9     # reject pairs narrower than a real lane
# True-metre window around a 3.7 m highway lane. The unscaled 2.8–4.8
# band was matching the network's 4.54 m ego gap, not paint.
EGO_LANE_WIDTH_MAX_M = 4.1     # reject pairs that span 2+ lanes
# If no pair sits in 2.9–4.1 m but both ego paints exist a bit too wide
# (network leftover after LANE_X_SCALE), still lock them. Two-lane gaps
# are ~7.4 m, so 5.1 stays one inflated lane. 3D / CIPO still clamps to 3.7.
EGO_LANE_WIDTH_FALLBACK_MAX_M = 5.1
EGO_LANE_WIDTH_TARGET_M = 3.7  # prefer pairs near standard lane width
EGO_CORRIDOR_MARGIN_M = 0.24   # 3D / CIPO inset; front fill uses DRAW_CORRIDOR_WIDTH_M instead
CORRIDOR_WIDTH_MAX_M = 3.9     # if ego pair is wider, shrink to target around center
CORRIDOR_WIDTH_CLAMP_M = 3.7   # 3D occupancy width (real lane)
# Front-view fill is a fixed band around the ego centre. Ego paint only
# tells where that band sits; it does not set the fill width.
CORRIDOR_FORCE_FIXED_WIDTH = True
DRAW_CORRIDOR_WIDTH_M = 2.5
# Front fill starts just above the hood. Lane polylines begin at the 5 m anchor,
# so the corridor uses that same near point instead of starting farther up the road.
CORRIDOR_IMAGE_HOOD_FRAC = 0.06
CORRIDOR_Y_START_M = 10.0      # 5 m is the bonnet; first reliable paint is ~10 m
EGO_PAIR_HOLD_FRAMES = LANE_HOLD_FRAMES
EGO_PAIR_MATCH_X_M = 1.01      # 1.25 / 1.235; rematch held lanes by |Δmean_x|
# Camera X is the optical axis, not the vehicle centerline. Subtract this
# (meters, + = camera right of center) before occupancy / scoring. Do not
# bake it into P — 3D outputs stay in the model camera frame.
CAMERA_LATERAL_OFFSET_M = 0.0
# Ego association / occupancy uses only this near Y band (m). Anchor grid is
# 5 m steps, so 15 m = three samples; 8 m would be a single point.
EGO_PAIR_NEAR_Y_M = 15.0
# Occupancy-first ego pair: kill adjacent-lane latch.
# A pair "occupies" ego when each boundary is at least INNER meters from X=0.
# Center weight dominates; width is only a tie-break. Do not use |center|<1.2
# as a hard cap — ADAS clips often have ~1.5 m camera-frame bias.
EGO_OCCUPANCY_INNER_M = 0.32    # 0.40 / 1.235
EGO_CENTER_SCORE_W = 3.0
EGO_WIDTH_SCORE_W = 0.25
EGO_REQUIRE_CONTAINS_0 = True  # never pick a pair that does not contain X=0
# If no occupying pair exists, allow a weaker contains-0 pair (left<0<right)
# whose |center| is still below FALLBACK_MAX_CENTER (neighbour latch is ~1.6–1.9 m).
EGO_OCCUPANCY_FALLBACK_CONTAINS0 = True
EGO_FALLBACK_MAX_CENTER_M = 1.29  # 1.59 / 1.235
# closest-left + closest-right re-locks the adjacent lane; keep off.
EGO_LEGACY_FALLBACK = False
# Sticky −1/+1 roles: rematch last ego paint; find_ego_lanes only on cold
# start or after a dwelled lane-change (center jump).
EGO_STICKY_INDEX = True
LANE_CHANGE_CENTER_M = 0.97    # 1.2 / 1.235; |Δcorridor center| to start re-index
LANE_CHANGE_DWELL_FRAMES = 10  # ~0.33 s at 30 fps; 3 frames was detector noise

# --- One-sided ego reconstruct (P1) ---
# If only one ego paint line is measured, rebuild the missing side from a
# locked width. Visualization / BEV use it; CIPO treats this as probable
# (PREDICTED), not a hard blank.
ENABLE_ONESIDED_RECONSTRUCT = True
ONESIDED_MAX_Y_M = 40.0        # only invent the missing side in the near field
ONESIDED_HOLD_FRAMES = LANE_HOLD_FRAMES
ONESIDED_MIN_LOCK_FRAMES = 3   # confirmed pairs required before trusting W
ONESIDED_W_EMA_ALPHA = 0.20    # slow width lock (higher = follow new gaps more)
ONESIDED_MATCH_X_M = 1.22      # 1.50 / 1.235; rematch the live side to last ego X

# --- Corridor temporal EMA (P2) ---
CORRIDOR_EMA_ALPHA = 0.35      # higher = trust new frame more
CORRIDOR_EMA_MAX_JUMP_M = 1.46  # 1.8 / 1.235; reject / hard-switch if jump exceeds this

# --- P3 dark / low-light (CLAHE + adaptive conf) ---
# Garmin A/B (GRMN6694_540_nohud, N=1803):
#   CLAHE (gated or always) *hurts* corridor — OpenLane engine domain shift.
#   Lower conf on dark frames only *helps* (+46 corridor, both-gone 43%→64%).
ENABLE_DARK_CLAHE = False
ENABLE_CLAHE_ALWAYS = False
ENABLE_ADAPTIVE_CONF = True
DARK_LUMA_MAX = 125.0           # mean HSV-V of lower 2/3 of the model image
CLAHE_CLIP = 2.5
CLAHE_TILE = 8
DARK_CONF_THRESHOLD = 0.38      # slightly below day conf for dark frames

# --- CIPO occupancy / hysteresis ---
# Score = vehicle-interval overlap with the ego corridor (0–1).
CIPO_SCORE_ENTER_HIGH = 0.60   # enter in-path on this frame
CIPO_SCORE_ENTER = 0.45        # enter after CIPO_ENTER_HITS
CIPO_SCORE_HOLD = 0.25         # stay in-path while score stays above this
CIPO_ENTER_HITS = 2            # medium-score frames before marking in_path
CIPO_EXIT_MISS = 8             # low-score frames before clearing in_path
CIPO_U_MARGIN_PX = 8.0         # 2D fallback slack around projected ego lines
CIPO_X_MARGIN_M = 0.40         # base meters beyond corridor edges
CIPO_X_MARGIN_PER_Z = 0.02     # extra meters per meter of range
CIPO_STICK_MARGIN_M = 5.0      # challenger must be this much closer to steal CIPO
CIPO_DANGER_ENTER_M = 14.0
CIPO_DANGER_EXIT_M = 17.0
CIPO_WARN_ENTER_M = 28.0
CIPO_WARN_EXIT_M = 32.0
