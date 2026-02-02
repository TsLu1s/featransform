from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from featransform.core.base import BaseSelector
from featransform.core.constants import DEFAULT_RANDOM_STATE
from featransform.core.enums import OptimizationMetric, SelectionStrategy, TaskType
from featransform.optimization.evaluator import (
    BaseEvaluator,
    CatBoostImportance,
    ImportanceCalculator,
    XGBoostImportance,
)


class FeatureSelector(BaseSelector, BaseEvaluator):
    """Feature selection with integrated evaluation."""

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.IMPORTANCE,
        task_type: Optional[Union[TaskType, str]] = None,
        min_features: int = 7,
        use_catboost: bool = True,
        use_xgboost: bool = True,
        n_estimators: int = 50,
        metric: Optional[OptimizationMetric] = None,
        **kwargs,
    ):
        BaseSelector.__init__(self, **kwargs)
        BaseEvaluator.__init__(self, metric)
        self.strategy = strategy
        self.min_features = min_features
        self.n_estimators = n_estimators
        self._calculators = self._init_calculators(use_catboost, use_xgboost, n_estimators)
        self._importance_df = None
        self._selected_features = []

        if task_type is not None:
            self._task_type = self._validate_task_type(task_type)

    def _validate_task_type(self, task_type: Union[TaskType, str]) -> TaskType:
        """Validate and convert task_type to TaskType enum."""
        if isinstance(task_type, TaskType):
            return task_type
        if isinstance(task_type, str):
            # Try matching enum value (case-insensitive)
            for task in TaskType:
                if task.value == task_type.lower():
                    return task
            raise ValueError(
                f"Invalid task_type: '{task_type}'. Must be one of: "
                f"{', '.join([t.value for t in TaskType])}"
            )
        raise ValueError(f"task_type must be TaskType enum or string, got {type(task_type)}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        self._validate_input(X)
        # Only detect task_type if not already set
        if self._task_type is None:
            self._task_type = self._detect_task_type(y)
        with self._error_context("importance_calculation"):
            self._importance_df = self._compute_importance(X, y)
        self._fitted = True
        return self

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        thresholds: List[float],
    ) -> Tuple[List[str], float, List[Dict]]:
        if not self._fitted:
            self.fit(X_train, y_train)

        history = []
        best_score = None
        best_features = list(X_train.columns)

        for threshold in thresholds:
            selected = self.select_by_threshold(threshold)
            score = self.evaluate_subset(X_train[selected], y_train, X_val[selected], y_val)

            history.append(
                {
                    "threshold": threshold,
                    "n_features": len(selected),
                    "score": score,
                    "features": selected,
                }
            )

            if best_score is None or self.is_better(score, best_score):
                best_score = score
                best_features = selected

        self.set_selected_features(best_features)
        return best_features, best_score, history

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return X[self._selected_features] if self._selected_features else X

    def select_by_threshold(self, threshold: float) -> List[str]:
        if self._importance_df is None:
            return []
        mask = self._importance_df["Cumulative"] <= threshold
        selected = self._importance_df.loc[mask, "Feature Id"].tolist()
        return (
            selected
            if len(selected) >= self.min_features
            else self._importance_df.head(self.min_features)["Feature Id"].tolist()
        )

    def set_selected_features(self, features: List[str]) -> None:
        self._selected_features = features

    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._importance_df is not None:
            return dict(zip(self._importance_df["Feature Id"], self._importance_df["Importances"]))
        return {}

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "min_features": self.min_features,
            "n_estimators": self.n_estimators,
            "metric": self.metric,
            **self._params,
        }

    def set_params(self, **params) -> "FeatureSelector":
        """Set parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _calculate_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Required abstract method from BaseSelector."""
        if self._importance_df is None:
            self._importance_df = self._compute_importance(X, y)
        return self.get_feature_scores()

    def _init_calculators(
        self, use_catboost: bool, use_xgboost: bool, n_estimators: int
    ) -> List[ImportanceCalculator]:
        calculators = []
        rand_state = self._params.get("random_state", DEFAULT_RANDOM_STATE)
        if use_catboost:
            calculators.append(CatBoostImportance(n_estimators, rand_state))
        if use_xgboost:
            calculators.append(XGBoostImportance(n_estimators, rand_state))
        return calculators

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        importances = [calc.calculate(X, y) for calc in self._calculators]
        df = (
            pd.concat(importances).groupby("Feature Id")["Importances"].mean().reset_index()
            if len(importances) > 1
            else importances[0]
        )
        df = df.sort_values("Importances", ascending=False)
        df["Cumulative"] = df["Importances"].cumsum()
        return df
