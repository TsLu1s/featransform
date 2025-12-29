from typing import Final

# Pipeline constants
DEFAULT_VALIDATION_SPLIT: Final[float] = 0.15
DEFAULT_OPTIMIZATION_ITERATIONS: Final[int] = 6
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_N_JOBS: Final[int] = -1

# Validation bounds
MIN_VALIDATION_SPLIT: Final[float] = 0.05
MAX_VALIDATION_SPLIT: Final[float] = 0.45
MAX_OPTIMIZATION_ITERATIONS: Final[int] = 10
MIN_SAMPLES_REQUIRED: Final[int] = 5

# Model defaults
DEFAULT_CONTAMINATION: Final[float] = 0.002
DEFAULT_N_CLUSTERS: Final[int] = 5
DEFAULT_N_COMPONENTS: Final[int] = 4
DEFAULT_N_ESTIMATORS: Final[int] = 100

# Feature naming patterns
ANOMALY_PREFIX: Final[str] = "anomaly_"
CLUSTER_PREFIX: Final[str] = "cluster_"
COMPONENT_PREFIX: Final[str] = "comp_"

# File formats
SUPPORTED_FORMATS: Final[tuple] = ("pickle", "joblib", "json", "yaml")

