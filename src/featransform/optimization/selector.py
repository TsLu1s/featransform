import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from featransform.core.base import BaseSelector
from featransform.core.constants import DEFAULT_N_JOBS, DEFAULT_RANDOM_STATE
from featransform.core.enums import OptimizationMetric, SelectionStrategy, TaskType
from featransform.core.exceptions import ValidationError
from featransform.optimization.evaluator import (
    BaseEvaluator,
    CatBoostImportance,
    ImportanceCalculator,
    XGBoostImportance,
    default_metric,
    validate_metric,
)

# Strategies with a ranking implementation behind them. The remaining
# ``SelectionStrategy`` members are reserved names.
_IMPLEMENTED_STRATEGIES = frozenset({SelectionStrategy.IMPORTANCE})


class FeatureSelector(BaseSelector, BaseEvaluator):
    """Feature selection with integrated evaluation.

    Features are ranked by tree-ensemble importance (CatBoost and/or XGBoost,
    averaged); candidate subsets are the smallest rank prefixes reaching a
    cumulative-importance threshold, and each candidate is scored on a
    validation slice with a freshly trained model.
    """

    # Class-level defaults keep selectors pickled before 2.4.0 loadable.
    _explicit_task_type: bool = True
    best_score_: Optional[float] = None

    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.IMPORTANCE,
        task_type: Optional[Union[TaskType, str]] = None,
        min_features: int = 10,
        use_catboost: bool = True,
        use_xgboost: bool = True,
        n_estimators: int = 50,
        metric: Optional[OptimizationMetric] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS,
        **kwargs,
    ):
        BaseSelector.__init__(self, **kwargs)
        BaseEvaluator.__init__(self, metric, random_state=random_state, n_jobs=n_jobs)
        if not use_catboost and not use_xgboost:
            raise ValueError("At least one of use_catboost or use_xgboost must be True")
        if strategy not in _IMPLEMENTED_STRATEGIES:
            # Say so instead of quietly running a different algorithm than requested.
            warnings.warn(
                f"SelectionStrategy '{getattr(strategy, 'value', strategy)}' is not implemented "
                f"yet; falling back to '{SelectionStrategy.IMPORTANCE.value}'.",
                UserWarning,
                stacklevel=2,
            )
            strategy = SelectionStrategy.IMPORTANCE
        self.strategy = strategy
        self.min_features = min_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._calculators = self._init_calculators(use_catboost, use_xgboost, n_estimators)
        self._importance_df = None
        self._selected_features = []
        self.best_score_ = None
        # Canonical column order seen at fit — used to keep the selected output
        # in schema order rather than importance-rank order.
        self._fit_columns: List[str] = []

        self._explicit_task_type = task_type is not None
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
        self._validate_fit_input(X)
        if X.columns.duplicated().any():
            duplicated = sorted({str(c) for c in X.columns[X.columns.duplicated()]})
            raise ValidationError(f"Feature names must be unique; duplicated: {duplicated}")
        # Remember the input column order so selections can be returned in schema
        # order — anchored on position, never on parsing the column name.
        self._fit_columns = list(X.columns)
        # A task type the caller stated is kept; a detected one is re-derived on
        # every fit so a reused selector never carries the previous target's task.
        if not self._explicit_task_type or self._task_type is None:
            self._task_type = self._detect_task_type(y)
        validate_metric(self.metric, self._task_type)
        # Evaluation state belongs to one fit: drop anything from a previous one.
        self._eval_model = None
        self._target_encoder = None
        self._selected_features = []
        self.best_score_ = None
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
        baseline_score: Optional[float] = None,
    ) -> Tuple[List[str], float, List[Dict]]:
        if not self._fitted:
            self.fit(X_train, y_train)

        history = []
        seen_subsets = set()

        # The full feature set is a real candidate so that the returned
        # best genuinely beats "keep everything". The caller may pass the
        # already-computed baseline score to avoid a redundant evaluation.
        best_features = list(X_train.columns)
        best_score = (
            baseline_score
            if baseline_score is not None
            else self.evaluate_subset(X_train, y_train, X_val, y_val)
        )
        full_key = tuple(sorted(best_features, key=str))
        full_score = best_score

        for threshold in thresholds:
            selected = self.select_by_threshold(threshold)
            subset_key = tuple(sorted(selected, key=str))
            if subset_key in seen_subsets:
                continue
            seen_subsets.add(subset_key)

            # A candidate equal to the full set was already scored as the baseline.
            score = (
                full_score
                if subset_key == full_key
                else self.evaluate_subset(X_train[selected], y_train, X_val[selected], y_val)
            )

            history.append(
                {
                    "threshold": threshold,
                    "n_features": len(selected),
                    "score": score,
                    "features": selected,
                }
            )

            # A strictly better score wins; on an exact tie the smaller subset
            # does (same validation score from fewer columns).
            if self.is_better(score, best_score) or (
                score == best_score and len(selected) < len(best_features)
            ):
                best_score = score
                best_features = selected

        self.set_selected_features(best_features)
        self.best_score_ = best_score
        # Return the schema-ordered selection so callers (and the pipeline)
        # apply columns in input order, not importance rank.
        return self._selected_features, best_score, history

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        return X[self._selected_features] if self._selected_features else X

    def select_by_threshold(self, threshold: float) -> List[str]:
        if self._importance_df is None:
            return []
        # A threshold of 1.0 means "every feature". It is answered directly:
        # importances come from float32 models, so their cumulative sum can land
        # a hair above 1.0 and a comparison would silently drop the tail.
        if threshold >= 1.0:
            return self._importance_df["Feature Id"].tolist()
        mask = self._importance_df["Cumulative"] <= threshold + 1e-9
        selected = self._importance_df.loc[mask, "Feature Id"].tolist()
        return (
            selected
            if len(selected) >= self.min_features
            else self._importance_df.head(self.min_features)["Feature Id"].tolist()
        )

    def set_selected_features(self, features: List[str]) -> None:
        """Validates that features are a subset of those scored at fit.

        The stored selection is kept in fit-time schema order so the
        transformed output column order matches the input/engineered schema
        rather than importance rank.
        """
        self._check_fitted()
        if self._importance_df is not None:
            known = set(self._importance_df["Feature Id"])
            unknown = set(features) - known
            if unknown:
                raise ValidationError(
                    f"Unknown features (not seen at fit): {sorted(unknown, key=str)}"
                )
        self._selected_features = self._order_by_schema(features)

    def _order_by_schema(self, features: List[str]) -> List[str]:
        """Return ``features`` ordered by their position at fit time."""
        selected = set(features)
        if self._fit_columns:
            ordered = [c for c in self._fit_columns if c in selected]
            # Append any features not present in the fit column list (defensive)
            ordered += [c for c in features if c not in self._fit_columns]
            return ordered
        return list(features)

    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._importance_df is not None:
            return dict(zip(self._importance_df["Feature Id"], self._importance_df["Importances"]))
        return {}

    def get_metric(self) -> Optional[OptimizationMetric]:
        """The metric scores are reported in — resolved to the task default when unset."""
        if self.metric is not None:
            return self.metric
        return default_metric(self._task_type) if self._task_type is not None else None

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "min_features": self.min_features,
            "n_estimators": self.n_estimators,
            "metric": self.metric,
            "n_jobs": self.n_jobs,
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
        if use_catboost:
            calculators.append(CatBoostImportance(n_estimators, self.random_state, self.n_jobs))
        if use_xgboost:
            calculators.append(XGBoostImportance(n_estimators, self.random_state, self.n_jobs))
        return calculators

    def _compute_importance(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        # Thread the resolved task type so importance calculators agree with the
        # evaluator instead of independently re-deriving it.
        importances = [calc.calculate(X, y, self._task_type) for calc in self._calculators]
        # Every calculator returns one row per column, in column order, so the
        # scores are averaged by position — independent of the column labels.
        mean_importance = sum(
            imp["Importances"].to_numpy(dtype=float) for imp in importances
        ) / len(importances)
        # Renormalize so the scores sum to exactly 1 and cumulative thresholds
        # mean what they say (the per-model sums are only ~1 in float32).
        total = mean_importance.sum()
        if total > 0:
            mean_importance = mean_importance / total
        df = pd.DataFrame({"Feature Id": list(X.columns), "Importances": mean_importance})
        # Stable sort: equal-importance features keep their schema order.
        df = df.sort_values("Importances", ascending=False, kind="stable").reset_index(drop=True)
        df["Cumulative"] = df["Importances"].cumsum()
        return df
