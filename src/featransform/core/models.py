from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from featransform.core.constants import (
    DEFAULT_N_JOBS,
    DEFAULT_OPTIMIZATION_ITERATIONS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_VALIDATION_SPLIT,
    MAX_OPTIMIZATION_ITERATIONS,
    MAX_VALIDATION_SPLIT,
    MIN_VALIDATION_SPLIT,
)
from featransform.core.enums import (
    EncodingStrategy,
    FeatureType,
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    ScalingStrategy,
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

    @cached_property
    def summary(self) -> Dict[str, Any]:
        """Cached transformation summary."""
        return {
            "n_features": self.n_features,
            "feature_types": {
                ft.value: sum(1 for f in self.feature_info if f.feature_type == ft)
                for ft in FeatureType
            },
        }

    def get_summary(self) -> Dict[str, Any]:
        """Backwards-compatible accessor for the cached summary."""
        return self.summary


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

    # Unknown fields are rejected (a typo must not be silently ignored) and
    # values assigned after construction are validated like constructor ones.
    model_config = ConfigDict(protected_namespaces=(), extra="forbid", validate_assignment=True)

    model_family: ModelFamily
    enabled: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

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

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    imputation_strategy: ImputationStrategy = ImputationStrategy.ITERATIVE
    encoding_strategy: EncodingStrategy = EncodingStrategy.LABEL
    # Scaling applied ONLY to the copy fed to the feature-engineering ensembles
    # (anomaly / clustering / dimensionality). Passthrough original columns are
    # left in their native scale. Set to None to disable.
    scaling_strategy: Optional[ScalingStrategy] = ScalingStrategy.STANDARD
    drop_constant: bool = True
    drop_duplicates: bool = True
    handle_datetime: bool = True
    # Identifier / index columns that must NOT feed into scaling or the
    # feature-engineering ensembles; they still pass through to the output.
    id_columns: List[str] = Field(default_factory=list)
    # Opt-in heuristic: also treat monotonic, all-unique integer columns as ids.
    auto_detect_id: bool = False


class OptimizationConfig(BaseModel):
    """Feature optimization configuration."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Only ``IMPORTANCE`` is implemented; the other ``SelectionStrategy`` members
    # are reserved names and fall back to it with a warning.
    selection_strategy: SelectionStrategy = SelectionStrategy.IMPORTANCE
    metric: Optional[OptimizationMetric] = None  # Auto-detected from task type
    n_iterations: int = Field(
        default=DEFAULT_OPTIMIZATION_ITERATIONS, ge=1, le=MAX_OPTIMIZATION_ITERATIONS
    )
    validation_split: float = Field(
        default=DEFAULT_VALIDATION_SPLIT, ge=MIN_VALIDATION_SPLIT, le=MAX_VALIDATION_SPLIT
    )
    min_features: int = Field(default=10, ge=1)
    # Feature importance calculation settings
    use_catboost: bool = True
    use_xgboost: bool = True
    n_estimators: int = Field(default=50, ge=1)

    @field_validator("use_xgboost")
    @classmethod
    def at_least_one_calculator(cls, v, info):
        """At least one of CatBoost / XGBoost must be enabled."""
        if not v and not info.data.get("use_catboost", True):
            raise ValueError("At least one of use_catboost or use_xgboost must be True")
        return v


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=(),
        extra="forbid",
        validate_assignment=True,
    )

    # ``None`` resolves the task from the target at fit (see ``detect_task_type``).
    task_type: Optional[TaskType] = None

    anomaly_models: List[ModelConfig] = Field(default_factory=list)
    clustering_models: List[ModelConfig] = Field(default_factory=list)
    dimensionality_models: List[ModelConfig] = Field(default_factory=list)

    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    optimization: Optional[OptimizationConfig] = None

    # Seeds every stochastic component (imputer, ensembles, selector, split).
    random_state: int = DEFAULT_RANDOM_STATE
    # Worker count handed to every component that parallelizes (-1 = all cores).
    n_jobs: int = DEFAULT_N_JOBS
    # Log progress. Visible on the console even without logging configuration.
    verbose: bool = False
    # Reserved: accepted for backwards compatibility, currently has no effect.
    enable_caching: bool = True
