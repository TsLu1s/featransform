from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from featransform.builder import BuilderPipeline
from featransform.core.base import BasePipeline
from featransform.core.enums import PipelineStage, TaskType
from featransform.core.exceptions import NotFittedError
from featransform.core.models import OptimizationHistory, PipelineConfig


@dataclass
class PipelineState:
    """Pipeline execution state."""

    data: pd.DataFrame
    target: Optional[pd.Series] = None
    stage: PipelineStage = PipelineStage.PREPROCESSING
    features_created: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)

    def evolve(self, **kwargs) -> "PipelineState":
        """Create new state with updates."""
        return PipelineState(
            **{
                **self.__dict__,
                **kwargs,
                "features_created": self.features_created.copy(),
                "transformations_applied": self.transformations_applied.copy(),
            }
        )


class Featransform(BasePipeline, BuilderPipeline):
    """
    Feature engineering pipeline with automatic optimization.
    Implements builder pattern for component construction.
    """

    def __init__(self, config: PipelineConfig):
        BuilderPipeline.__init__(self)
        self.config = config
        self._state: Optional[PipelineState] = None
        self._optimization_history: List[OptimizationHistory] = []
        self._feature_importance: Optional[Dict[str, float]] = None

        # Build components using implemented abstract method
        self._components = self.build_components(config)

        BasePipeline.__init__(self, stages=list(self._components.values()), verbose=config.verbose)

    def build_components(self, config: PipelineConfig) -> Dict[str, Any]:
        """Implementation of abstract component building."""
        return self._build_pipeline(config)

    def _select_components(self, config: PipelineConfig) -> List[str]:
        """Select components to build based on configuration."""
        components = []

        if config.anomaly_models:
            components.append("anomaly")
        if config.clustering_models:
            components.append("clustering")
        if config.dimensionality_models:
            components.append("dimensionality")

        return components

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Featransform":
        """Fit pipeline with optional optimization."""
        self._validate_input(X)

        # Store the initial training schema
        # self._store_fit_schema(X)

        # Prepare data splits
        X_train, X_val, y_train, y_val = self.__prepare_splits(X, y)

        # Execute pipeline
        self._state = PipelineState(data=X_train.copy(), target=y_train)
        self._state = self.__process_pipeline(self._state, fit=True)

        # Optimize if validation available
        if X_val is not None:
            self.__optimize_features(X_val, y_val)

        # Update schema after feature engineering
        self._store_fit_schema(self._state.data)

        self._fitted = True
        self._log(f"Pipeline fitted: {self._state.data.shape[1]} features")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through fitted pipeline."""
        self._check_fitted()
        self._validate_input(X)

        state = self.__process_pipeline(PipelineState(data=X.copy()))

        # Apply learned selection if available
        if self.has_component("selector") and self._components["selector"]._fitted:
            state.data = self._components["selector"].transform(state.data)
            # Ensure final output matches training schema
            state.data = self._validate_transform_schema(state.data)

        return state.data

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(X, y).transform(X)

    def __process_pipeline(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Process all pipeline stages."""
        # Preprocessing stage
        state = self.__preprocess(state, fit)

        # Feature engineering stage
        state = self.__engineer_features(state, fit)

        return state

    def __preprocess(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Apply preprocessing transformations."""
        state = state.evolve(stage=PipelineStage.PREPROCESSING)

        # Handle datetime features
        if self.config.processing.handle_datetime:
            state = self.__extract_datetime(state)

        # Apply processors
        for comp in ["imputer", "encoder"]:
            if self.has_component(comp):
                state = self.__apply_component(state, comp, fit)

        # Clean features
        if self.config.processing.drop_constant:
            state.data = state.data.loc[:, state.data.nunique() > 1]

        if self.config.processing.drop_duplicates:
            state.data = state.data.T.drop_duplicates().T

        return state

    def __engineer_features(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Apply feature engineering transformations."""
        state = state.evolve(stage=PipelineStage.FEATURE_ENGINEERING)
        engineered = []

        for comp in ["anomaly", "clustering", "dimensionality"]:
            if self.has_component(comp):
                component = self._components[comp]

                if fit and not component._fitted:
                    component.fit(state.data, state.target)

                result = component.transform(state.data)
                engineered.append(result.features)
                state.features_created.extend(result.features.columns.tolist())
                state.transformations_applied.append(comp)

        if engineered:
            state.data = pd.concat([state.data] + engineered, axis=1)

        return state

    def __apply_component(self, state: PipelineState, name: str, fit: bool) -> PipelineState:
        """Apply single component transformation."""
        component = self._components[name]

        if fit and not component._fitted:
            component.fit(state.data, state.target)

        state.data = component.transform(state.data)
        state.transformations_applied.append(name)
        return state

    def __optimize_features(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Optimize feature selection using integrated selector."""
        val_state = self.__process_pipeline(PipelineState(data=X_val.copy()))

        # Setup selector with metric
        selector = self._components["selector"]
        selector.metric = self.config.optimization.metric

        # FIT THE SELECTOR FIRST - this sets the task_type!
        selector.fit(self._state.data, self._state.target)

        # Generate thresholds
        thresholds = [
            1 - (i / 100) for i in range(0, min(10, self.config.optimization.n_iterations))
        ]

        # Get baseline score
        baseline_score = selector.evaluate_subset(
            self._state.data, self._state.target, val_state.data, y_val
        )
        self._optimization_history.append(
            OptimizationHistory(
                iteration=-1,
                score=baseline_score,
                n_features=self._state.data.shape[1],
                selected_features=list(self._state.data.columns),
            )
        )

        # Optimize selection
        best_features, best_score, history = selector.optimize(
            self._state.data, self._state.target, val_state.data, y_val, thresholds
        )

        # Record history
        for i, result in enumerate(history):
            self._optimization_history.append(
                OptimizationHistory(
                    iteration=i,
                    score=result["score"],
                    n_features=result["n_features"],
                    selected_features=result["features"],
                )
            )
            self._log(
                f"Threshold {result['threshold']:.2f}: {result['n_features']} features, score: {result['score']:.4f}"
            )

        # Apply best selection
        self._state.data = self._state.data[best_features]
        self._state = self._state.evolve(stage=PipelineStage.FEATURE_SELECTION)
        self._feature_importance = selector.get_feature_scores()

    def __extract_datetime(self, state: PipelineState) -> PipelineState:
        """Extract comprehensive datetime features."""
        dt_cols = state.data.select_dtypes(include=["datetime64"]).columns

        for col in dt_cols:
            dt = state.data[col].dt

            # Core temporal features
            state.data[f"{col}_year"] = dt.year
            state.data[f"{col}_month"] = dt.month
            state.data[f"{col}_day"] = dt.day
            state.data[f"{col}_dayofweek"] = dt.dayofweek
            state.data[f"{col}_dayofyear"] = dt.dayofyear
            state.data[f"{col}_quarter"] = dt.quarter
            state.data[f"{col}_weekofyear"] = dt.isocalendar().week

            # Weekend flag
            state.data[f"{col}_is_weekend"] = (dt.dayofweek >= 5).astype(int)

            # Time components if present
            if dt.hour.notna().any():
                state.data[f"{col}_hour"] = dt.hour
                state.data[f"{col}_minute"] = dt.minute
                state.data[f"{col}_second"] = dt.second

            # Cyclic encodings for periodicity
            state.data[f"{col}_month_sin"] = np.sin(2 * np.pi * dt.month / 12)
            state.data[f"{col}_month_cos"] = np.cos(2 * np.pi * dt.month / 12)
            state.data[f"{col}_day_sin"] = np.sin(2 * np.pi * dt.day / 31)
            state.data[f"{col}_day_cos"] = np.cos(2 * np.pi * dt.day / 31)

            # Drop original column
            state.data = state.data.drop(columns=[col])

        return state

    def __prepare_splits(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """Prepare train/validation splits."""
        if not self.config.optimization or self.config.optimization.n_iterations == 0:
            return X, None, y, None

        stratify = y if self.__detect_task_type(y) != TaskType.REGRESSION else None
        return train_test_split(
            X,
            y,
            test_size=self.config.optimization.validation_split,
            random_state=self.config.random_state,
            stratify=stratify,
        )

    def __detect_task_type(self, y: pd.Series) -> TaskType:
        """Detect ML task type."""
        if y.dtype in ["float64", "float32"]:
            return TaskType.REGRESSION
        return (
            TaskType.BINARY_CLASSIFICATION
            if y.nunique() == 2
            else TaskType.MULTICLASS_CLASSIFICATION
        )

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {"config": self.config}

    def set_params(self, **params) -> "Featransform":
        """Set pipeline parameters."""
        if "config" in params:
            self.config = params["config"]
            self._components = self.build_components(self.config)
        return self

    def get_optimization_history(self) -> List[OptimizationHistory]:
        """Get optimization history."""
        return self._optimization_history

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self._feature_importance

    def get_pipeline_schema(self) -> Optional[List[str]]:
        """Get the stored training schema."""
        return self.get_fit_schema()

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        if not self._fitted:
            raise NotFittedError("Pipeline not fitted")

        best = (
            max(self._optimization_history, key=lambda x: x.score)
            if self._optimization_history
            else None
        )

        return {
            "features": {
                "initial": len(self._fit_schema) if self._fit_schema else 0,
                "final": self._state.data.shape[1],
                "engineered": len(self._state.features_created),
            },
            "schema": {
                "columns": self.get_fit_schema(),
                "dtypes": dict(self._state.data.dtypes.astype(str)) if self._state else {},
            },
            "transformations": self._state.transformations_applied,
            "optimization": {
                "best_score": best.score if best else None,
                "best_features": best.n_features if best else None,
                "iterations": len(self._optimization_history),
            },
            "components": {k: type(v).__name__ for k, v in self._components.items()},
        }
