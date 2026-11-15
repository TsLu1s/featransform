from typing import Final

# Pipeline constants — single source of truth for defaults reused across modules.
# Keep these in sync with the values referenced by OptimizationConfig (models.py).
DEFAULT_VALIDATION_SPLIT: Final[float] = 0.25
DEFAULT_OPTIMIZATION_ITERATIONS: Final[int] = 10
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_N_JOBS: Final[int] = -1

# Validation bounds
MIN_VALIDATION_SPLIT: Final[float] = 0.05
MAX_VALIDATION_SPLIT: Final[float] = 0.45
MAX_OPTIMIZATION_ITERATIONS: Final[int] = 10
MIN_SAMPLES_REQUIRED: Final[int] = 5

# An integer target with more than this many distinct values is treated as a
# regression target rather than a multiclass classification target when the
# task type is not stated explicitly. Ambiguous by nature — set `task_type`
# explicitly to override.
MAX_CLASSIFICATION_CLASSES: Final[int] = 20

# Model defaults
DEFAULT_CONTAMINATION: Final[float] = 0.05
DEFAULT_N_CLUSTERS: Final[int] = 5
DEFAULT_N_COMPONENTS: Final[int] = 4
DEFAULT_N_ESTIMATORS: Final[int] = 100

# DBSCAN ``eps="auto"``: eps is the given quantile of each point's distance to
# its ``min_samples``-th nearest neighbour (the classic k-distance heuristic),
# estimated on at most this many rows.
DBSCAN_AUTO_EPS_QUANTILE: Final[float] = 0.9
DBSCAN_AUTO_EPS_MAX_SAMPLES: Final[int] = 10_000

# Suffix appended to an engineered feature whose name collides with a column
# already present in the data.
ENGINEERED_COLLISION_SUFFIX: Final[str] = "_ft"
