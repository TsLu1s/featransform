# Version
__version__ = "2.0.0"

# Main pipeline
from featransform.pipeline import Featransform

# Configuration
from featransform.configs.baseline import FTconfig
from featransform.core.models import PipelineConfig, ProcessingConfig, OptimizationConfig, ModelConfig

# Enums
from featransform.core.enums import (
    FeatureType,
    ModelFamily,
    TaskType,
    ImputationStrategy,
    EncodingStrategy,
    SelectionStrategy,
    OptimizationMetric,
    PipelineStage
)

# Utilities
from featransform.utils.serializer import PipelineSerializer
from featransform.utils.data_generator import DatasetGenerator, make_dataset

# Exceptions
from featransform.core.exceptions import (
    FeatureTransformError,
    ConfigurationError,
    NotFittedError,
    TransformationError,
)

__all__ = [
    # Main classes
    'Featransform',
    
    # Configuration
    'FTconfig',
    'PipelineConfig', 
    'ProcessingConfig',
    'OptimizationConfig',
    'ModelConfig',
    
    # Enums
    'FeatureType',
    'ModelFamily',
    'TaskType',
    'ImputationStrategy',
    'EncodingStrategy',
    'SelectionStrategy',
    'OptimizationMetric',
    'PipelineStage',
    
    # Processing components
    'CoreImputer',
    'CategoricalEncoder',
    
    # Feature engineering components
    'AnomalyEnsemble',
    'ClusteringEnsemble', 
    'DimensionalityEnsemble',
    
    # Feature selection
    'FeatureSelector',
    'BaseEvaluator',
    
    # Utilities
    'PipelineSerializer',
    'DatasetGenerator',
    'make_dataset',
    
    # Builder
    'BuilderPipeline',
    
    # Exceptions
    'FeatureTransformError',
    'ConfigurationError',
    'NotFittedError',
    'TransformationError',
]