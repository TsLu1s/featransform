# Version
__version__ = "2.0.0"

# Configuration
from featransform.configs.baseline import FTconfig

# Enums
from featransform.core.enums import (
    EncodingStrategy,
    FeatureType,
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    PipelineStage,
    SelectionStrategy,
    TaskType,
)

# Exceptions
from featransform.core.exceptions import (
    ConfigurationError,
    FeatureTransformError,
    NotFittedError,
    TransformationError,
)
from featransform.core.models import (
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    ProcessingConfig,
)

# Main pipeline
from featransform.pipeline import Featransform
from featransform.utils.data_generator import DatasetGenerator, make_dataset

# Utilities
from featransform.utils.serializer import PipelineSerializer

__all__ = [
    # Main classes
    "Featransform",
    # Configuration
    "FTconfig",
    "PipelineConfig",
    "ProcessingConfig",
    "OptimizationConfig",
    "ModelConfig",
    # Enums
    "FeatureType",
    "ModelFamily",
    "TaskType",
    "ImputationStrategy",
    "EncodingStrategy",
    "SelectionStrategy",
    "OptimizationMetric",
    "PipelineStage",
    # Processing components
    "CoreImputer",
    "CategoricalEncoder",
    # Feature engineering components
    "AnomalyEnsemble",
    "ClusteringEnsemble",
    "DimensionalityEnsemble",
    # Feature selection
    "FeatureSelector",
    "BaseEvaluator",
    # Utilities
    "PipelineSerializer",
    "DatasetGenerator",
    "make_dataset",
    # Builder
    "BuilderPipeline",
    # Exceptions
    "FeatureTransformError",
    "ConfigurationError",
    "NotFittedError",
    "TransformationError",
]
