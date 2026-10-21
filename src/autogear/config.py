"""Every constant of the study, in one place.

The values below are the ones used to produce the numbers of the report. Changing one of them
changes the results; the scripts never override them silently.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------------------------
# Signals and dataset
# ---------------------------------------------------------------------------------------------
PEDAL_MIN_CADENCE_RPM = 40.0      # cadence above which the speed-to-cadence ratio is trusted
PEDAL_MIN_SPEED_MS = 2.0          # speed above which the speed-to-cadence ratio is trusted
COASTING_MAX_CADENCE_RPM = 15.0   # below this cadence the rider is coasting
CONTROL_MIN_CADENCE_RPM = 20.0    # the controller only shifts above this cadence (chain must turn)

SEGMENT_MAX_GAP_S = 5             # a timestamp gap above this cuts a ride into segments
SEGMENT_MAX_STOP_S = 60           # a stop this long cuts a ride into segments
SEGMENT_MIN_LEN_S = 120           # shorter segments are discarded

ALTITUDE_MEDIAN_WINDOW_S = 61     # rolling median used to detect altitude spikes
ALTITUDE_SPIKE_M = 20.0           # distance from the rolling median that flags a spike
SLOPE_SMOOTHING_S = 10            # altitude and distance are smoothed over this window
SLOPE_POST_SMOOTHING_S = 5        # the slope itself is smoothed again over this window
SLOPE_CLIP = 0.25                 # |slope| is clipped at 25 %
ACC_CLIP = 3.0                    # |acceleration| is clipped at 3 m/s^2

# ---------------------------------------------------------------------------------------------
# Gear reconstruction (Viterbi on the log-development)
# ---------------------------------------------------------------------------------------------
LEVEL_MIN_SEPARATION = 0.045      # minimum distance between two gear levels, in log-development
LEVEL_MIN_FRACTION = 0.01         # a level must hold at least 1 % of the pedalling seconds
VITERBI_SIGMA = 0.05              # emission standard deviation, in log-development
VITERBI_SWITCH_COST = 15.0        # cost of a gear change (lambda)
VITERBI_JUMP_COST = 2.0           # extra cost per additional step in a multi-gear jump
VITERBI_EMISSION_CAP = 25.0       # emission costs are capped for robustness to outliers

# ---------------------------------------------------------------------------------------------
# Controller and selection protocol
# ---------------------------------------------------------------------------------------------
CONTROLLER_GRID = [dict(h=h, k=k, d=d)
                   for h in (0.3, 0.6, 1.0, 1.5)     # hysteresis margin, in steps
                   for k in (1, 3, 5, 8)             # persistence, in seconds
                   for d in (0, 8)]                  # anti-reversal delay, in seconds
HORIZONS_S = (1, 3)               # prediction horizons H tried by the nested selection
MAX_RATE_RATIO = 1.2              # a setting may shift at most 1.2 x as often as the rider
MAX_REVERSAL_SHARE = 0.08         # ... and reverse at most 8 % of its shifts
REVERSAL_WINDOW_S = 4             # a shift undone within 4 s counts as a reversal

# ---------------------------------------------------------------------------------------------
# Retained model: gradient-boosted regression trees
# ---------------------------------------------------------------------------------------------
GBDT_PARAMS = dict(max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.9,
                   min_child_weight=20, objective="reg:squarederror", tree_method="hist",
                   nthread=2, seed=0)
GBDT_ROUNDS = 100

# ---------------------------------------------------------------------------------------------
# First approach: two-stage event classifier on a raw 30 x 5 window
# ---------------------------------------------------------------------------------------------
EVENT_WINDOW_S = 30
EVENT_PARAMS = dict(max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=3, reg_lambda=2.0, tree_method="hist", nthread=1)
EVENT_MAX_ROUNDS = 500
EVENT_EARLY_STOPPING = 40
EVENT_DIRECTION_ROUNDS = 200
EVENT_VALIDATION_SHARE = 0.15
EVENT_GRID = [dict(th=th, cool=cl) for th in (0.5, 0.6, 0.7, 0.8, 0.9) for cl in (2, 4, 8)]
EVENT_F1_TOLERANCE_S = 2

# ---------------------------------------------------------------------------------------------
# Storage convention used to compare model sizes (Table 3 of the report)
# ---------------------------------------------------------------------------------------------
BYTES_PER_TREE_NODE = 8
BYTES_PER_FLOAT = 4
