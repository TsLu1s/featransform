from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from featransform.core.constants import DEFAULT_N_JOBS, DEFAULT_RANDOM_STATE
from featransform.core.enums import (
    EncodingStrategy,
    FeatureType,
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    SelectionStrategy,
    TaskType,
)


@dataclass
class FeatureInfo:
    """Information about an engineered feature."""

    name: str
    feature_type: FeatureType
    model_family: Optional[ModelFamily] = None
    importance_score: float = 0.0
    source_columns: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass
class TransformationResult:
    """Result of a feature transformation."""

    features: pd.DataFrame
    feature_info: List[FeatureInfo]

    @property
    def n_features(self) -> int:
        return len(self.features.columns)

    @lru_cache(maxsize=1)
    def get_summary(self) -> Dict[str, Any]:
        """Get transformation summary with caching."""
        return {
            "n_features": self.n_features,
            "feature_types": {
                ft.value: sum(1 for f in self.feature_info if f.feature_type == ft)
                for ft in FeatureType
            },
        }


@dataclass
class OptimizationHistory:
    """History of optimization iterations."""

    iteration: int
    score: float
    n_features: int
    selected_features: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_family: ModelFamily
    enabled: bool = True
    parameters: Dict[str, Any] = {}

    @field_validator("parameters")
    @classmethod
    def validate_params(cls, v, info):
        if info.data.get("model_family"):
            model = info.data["model_family"]
            if model in [ModelFamily.KMEANS, ModelFamily.MINI_BATCH_KMEANS]:
                if v.get("n_clusters", 2) < 2:
                    raise ValueError("n_clusters must be at least 2")
        return v


class ProcessingConfig(BaseModel):
    """Data processing configuration."""

    imputation_strategy: ImputationStrategy = ImputationStrategy.ITERATIVE
    encoding_strategy: EncodingStrategy = EncodingStrategy.LABEL
    drop_constant: bool = True
    drop_duplicates: bool = True
    handle_datetime: bool = True


@dataclass
class OptimizationConfig:
    """Feature optimization configuration."""

    # Core settings
    selection_strategy: SelectionStrategy = SelectionStrategy.IMPORTANCE
    metric: Optional[OptimizationMetric] = None  # Auto-detected from task type
    n_iterations: int = 10
    validation_split: float = 0.25
    min_features: int = 10
    # Feature importance calculation settings
    use_catboost: bool = True
    use_xgboost: bool = True
    n_estimators: int = 50  # For importance calculation models


class PipelineConfig(BaseModel):
    task_type: Optional[TaskType] = None

    """Complete pipeline configuration."""
    anomaly_models: List[ModelConfig] = Field(default_factory=list)
    clustering_models: List[ModelConfig] = Field(default_factory=list)
    dimensionality_models: List[ModelConfig] = Field(default_factory=list)

    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    optimization: Optional[OptimizationConfig] = None

    random_state: int = DEFAULT_RANDOM_STATE
    n_jobs: int = DEFAULT_N_JOBS
    verbose: bool = False
    enable_caching: bool = True  # New: caching

    class Config:
        arbitrary_types_allowed = True
