from featransform.core.models import PipelineConfig, ProcessingConfig, OptimizationConfig, ModelConfig
from featransform.core.enums import (
    ImputationStrategy, EncodingStrategy, SelectionStrategy, ModelFamily, TaskType
)
from typing import Optional


class FTconfig:
    """Preset configurations for quick pipeline setup."""
    
    @staticmethod
    def _get_available():  # Private method with single underscore
        """Internal method to get available configurations."""
        return [m for m in dir(FTconfig) 
                if not m.startswith('_') and 
                callable(getattr(FTconfig, m))]

    @staticmethod
    def _get(name: str, task_type: Optional[TaskType] = TaskType.REGRESSION):
        """Get configuration by name."""
        configs = {
            'minimal': FTconfig.minimal,
            'standard': FTconfig.standard,
            'optimized': FTconfig.optimized,
            'complete': FTconfig.complete
        }
        
        if name not in configs:
            print(f"Unknown config '{name}', using 'standard'")
            name = 'standard'
        
        return configs[name](task_type=task_type)
    
    @staticmethod
    def minimal(task_type: Optional[TaskType] = TaskType.REGRESSION) -> PipelineConfig:
        """Minimal configuration for quick testing."""
        return PipelineConfig(
            task_type=task_type,
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                encoding_strategy=EncodingStrategy.LABEL,
                handle_datetime=True,
                drop_constant=True,
                drop_duplicates=False
            ),
            anomaly_models=[
                ModelConfig(
                    model_family=ModelFamily.ISOLATION_FOREST,
                    parameters={'n_estimators': 100}
                )
            ],
            verbose=False,
            n_jobs=1
        )
    
    @staticmethod
    def standard(task_type: Optional[TaskType] = TaskType.REGRESSION) -> PipelineConfig:
        """Standard configuration for general use."""
        return PipelineConfig(
            task_type=task_type,
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.ITERATIVE,
                encoding_strategy=EncodingStrategy.LABEL,
                handle_datetime=True,
                drop_constant=True,
                drop_duplicates=True
            ),
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST),
                ModelConfig(model_family=ModelFamily.LOCAL_OUTLIER_FACTOR)
            ],
            clustering_models=[
                ModelConfig(
                    model_family=ModelFamily.KMEANS,
                    parameters={'n_clusters': 5}
                )
            ],
            dimensionality_models=[
                ModelConfig(
                    model_family=ModelFamily.PCA,
                    parameters={'n_components': 0.95}
                )
            ],
            optimization=OptimizationConfig(
                selection_strategy=SelectionStrategy.IMPORTANCE,
                n_iterations=10,
                validation_split=0.25,
                min_features=10
            ),
            verbose=False,
        )
    
    @staticmethod
    def optimized(task_type: Optional[TaskType] = TaskType.REGRESSION) -> PipelineConfig:
        """Configuration with feature selection optimization."""
        return PipelineConfig(
            task_type=task_type,
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.ITERATIVE,
                encoding_strategy=EncodingStrategy.LABEL,
                handle_datetime=True,
                drop_constant=True,
                drop_duplicates=True
            ),
    
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST),
                ModelConfig(model_family=ModelFamily.ONE_CLASS_SVM)
            ],
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS),
                ModelConfig(model_family=ModelFamily.GAUSSIAN_MIXTURE)
            ],
            dimensionality_models=[
                ModelConfig(model_family=ModelFamily.PCA),
                ModelConfig(model_family=ModelFamily.FAST_ICA)
            ],
            optimization=OptimizationConfig(
                selection_strategy=SelectionStrategy.IMPORTANCE,
                n_iterations=10,
                validation_split=0.25,
                min_features=10
            ),

            verbose=False,
        )
    
    @staticmethod
    def complete(task_type: Optional[TaskType] = TaskType.REGRESSION) -> PipelineConfig:
        """Full-featured configuration with all components."""
        return PipelineConfig(
            task_type=task_type,
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.KNN,
                encoding_strategy=EncodingStrategy.LABEL,
                handle_datetime=True,
                drop_constant=True,
                drop_duplicates=True
            ),
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST),
                ModelConfig(model_family=ModelFamily.LOCAL_OUTLIER_FACTOR),
                ModelConfig(model_family=ModelFamily.ONE_CLASS_SVM),
                ModelConfig(model_family=ModelFamily.ELLIPTIC_ENVELOPE)
            ],
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS),
                ModelConfig(model_family=ModelFamily.BIRCH),
                ModelConfig(model_family=ModelFamily.MINI_BATCH_KMEANS),
                ModelConfig(model_family=ModelFamily.GAUSSIAN_MIXTURE),
                ModelConfig(model_family=ModelFamily.DBSCAN)
            ],
            dimensionality_models=[
                ModelConfig(model_family=ModelFamily.PCA),
                ModelConfig(model_family=ModelFamily.TRUNCATED_SVD),
                ModelConfig(model_family=ModelFamily.FAST_ICA),
            ],
            optimization=OptimizationConfig(
                selection_strategy=SelectionStrategy.IMPORTANCE,
                n_iterations=10,
                validation_split=0.3,
                min_features=10
            ),
            verbose=False,
        )