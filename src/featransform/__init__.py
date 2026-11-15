from featransform._version import __version__
from featransform.builder import BuilderPipeline
from featransform.configs.baseline import FTconfig
from featransform.core.enums import (
    EncodingStrategy,
    FeatureType,
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    PipelineStage,
    ScalingStrategy,
    SelectionStrategy,
    TaskType,
)
from featransform.core.exceptions import (
    ConfigurationError,
    FeatureTransformError,
    NotFittedError,
    PipelineError,
    TransformationError,
    ValidationError,
)
from featransform.core.models import (
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    ProcessingConfig,
)
from featransform.features.anomaly import AnomalyEnsemble
from featransform.features.clustering import ClusteringEnsemble
from featransform.features.dimensionality import DimensionalityEnsemble
from featransform.optimization.evaluator import BaseEvaluator
from featransform.optimization.selector import FeatureSelector
from featransform.pipeline import Featransform
from featransform.processing.datetime_features import DatetimeExpander
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.imputation import CoreImputer
from featransform.processing.scaling import FeatureScaler
from featransform.utils.data_generator import DatasetGenerator, make_dataset
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
    "ScalingStrategy",
    # Processing components
    "CoreImputer",
    "CategoricalEncoder",
    "DatetimeExpander",
    "FeatureScaler",
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
    "ValidationError",
    "PipelineError",
    # Version
    "__version__",
]
