"""
Central configuration for the STGCN emergency rerouting project.
Import constants from here instead of hardcoding them in each script.
"""

# --- Data ---
NUM_TIMESTEPS_IN = 12
NUM_TIMESTEPS_OUT = 12
TRAIN_RATIO = 0.8          # fraction of full dataset used for train+val (rest is test)
TRAIN_VAL_SPLIT = 0.85     # fraction of train+val used for train (rest is val)

# CHANGED: which of the 12 future steps to train/evaluate against.
# METR-LA snapshots are 5-minute intervals, so:
#   index 0  -> 5 min ahead  (persistence baseline nearly unbeatable here)
#   index 2  -> 15 min ahead
#   index 5  -> 30 min ahead (recommended starting point -- standard reported
#                              horizon in traffic forecasting papers, and more
#                              operationally meaningful for emergency rerouting
#                              than a 5-min forecast)
#   index 11 -> 60 min ahead
PREDICTION_HORIZON_INDEX = 5

# --- Model architecture ---
NODE_FEATURES = 2
HIDDEN_DIMENSIONS = 32

# --- Training ---
SEED = 42
LEARNING_RATE = 0.001
ACCUMULATION_STEPS = 32
MAX_EPOCHS = 150
EARLY_STOP_PATIENCE = 12
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

# --- Paths ---
CHECKPOINT_PATH = "output/stgcn_la_weights.pt"

# --- Hardware ---
# Auto-detects GPU if available (e.g. on Colab), falls back to CPU otherwise.
DEVICE = "cuda"  # overridden to "cpu" automatically in training script if no GPU found

# --- Fast local iteration ---
# Set to a value < 1.0 (e.g. 0.05) to train on a small fraction of the data,
# just to verify the pipeline runs correctly before committing to a full run
# on Colab. Set to 1.0 for the real full-dataset run.
SUBSET_FRACTION = 1.0

# --- Wardrop Equilibrium (real METR-LA network) ---
# METR-LA's edge_attr is a graph-diffusion similarity weight (higher = more
# strongly connected / "closer" in the sensor graph), NOT a real road
# distance or free-flow travel time. There is no ground-truth capacity or
# road-length data bundled with this dataset. The constants below are
# DOCUMENTED MODELING ASSUMPTIONS -- state them explicitly in the paper's
# methodology section rather than presenting them as measured values.
FREE_FLOW_TIME_SCALE = 10.0   # scales inverse edge_attr into a travel-time-like unit
DEFAULT_EDGE_CAPACITY = 1000.0  # uniform capacity assumption across all edges (flow units)

# Synthetic OD (origin-destination) demand generation, since METR-LA does
# not ship a real OD matrix. Demand is sampled between a random subset of
# node pairs -- using all ~207*206 possible pairs would be computationally
# unnecessary for a first working version and isn't representative of real
# sparse urban travel demand anyway.
NUM_SYNTHETIC_OD_PAIRS = 30
OD_DEMAND_MIN = 50.0
OD_DEMAND_MAX = 300.0
OD_GENERATION_SEED = 42  # separate from SEED so OD demand stays fixed even
                          # if training reproducibility settings change

# --- Emergency Vehicle Routing ---
# When the emergency vehicle is granted priority (e.g. signal preemption,
# a reserved lane) on the edges along its route, this reduces the
# EFFECTIVE capacity available to background traffic on those edges.
# This is what makes the emergency vehicle a Stackelberg LEADER rather than
# just another routed agent: its presence changes the game background
# traffic has to re-solve, rather than passively routing through a fixed
# equilibrium. 0.2 = a 20% capacity reduction on the emergency route's edges.
EMERGENCY_LANE_CAPACITY_REDUCTION = 0.2