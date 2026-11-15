from abc import ABC, abstractmethod
from typing import Dict, FrozenSet, Literal, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

from featransform.core.constants import (
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_STATE,
    MAX_CLASSIFICATION_CLASSES,
)
from featransform.core.enums import OptimizationMetric, TaskType
from featransform.core.exceptions import ConfigurationError

METRIC_DIRECTION: Dict[OptimizationMetric, Literal["max", "min"]] = {
    OptimizationMetric.MAE: "min",
    OptimizationMetric.MSE: "min",
    OptimizationMetric.RMSE: "min",
    OptimizationMetric.R2: "max",
    OptimizationMetric.ACCURACY: "max",
    OptimizationMetric.PRECISION: "max",
    OptimizationMetric.RECALL: "max",
    OptimizationMetric.F1: "max",
    OptimizationMetric.AUC_ROC: "max",
}

REGRESSION_METRICS: FrozenSet[OptimizationMetric] = frozenset(
    {
        OptimizationMetric.MAE,
        OptimizationMetric.MSE,
        OptimizationMetric.RMSE,
        OptimizationMetric.R2,
    }
)
CLASSIFICATION_METRICS: FrozenSet[OptimizationMetric] = frozenset(
    {
        OptimizationMetric.ACCURACY,
        OptimizationMetric.PRECISION,
        OptimizationMetric.RECALL,
        OptimizationMetric.F1,
        OptimizationMetric.AUC_ROC,
    }
)


def default_metric(task_type: TaskType) -> OptimizationMetric:
    """Metric used when none is configured."""
    return OptimizationMetric.MSE if task_type == TaskType.REGRESSION else OptimizationMetric.F1


def validate_metric(metric: Optional[OptimizationMetric], task_type: TaskType) -> None:
    """Fail fast when a metric cannot score the resolved task type.

    Without this an F1 score on a regression target (or an MSE over class
    labels) either dies deep inside scikit-learn or silently optimizes
    something meaningless.
    """
    if metric is None:
        return
    allowed = REGRESSION_METRICS if task_type == TaskType.REGRESSION else CLASSIFICATION_METRICS
    if metric not in allowed:
        raise ConfigurationError(
            f"Metric '{metric.value}' cannot be used for a {task_type.value} task. "
            f"Choose one of: {sorted(m.value for m in allowed)}."
        )


def detect_task_type(y: pd.Series, configured: Optional[TaskType] = None) -> TaskType:
    """Single source of truth for task-type resolution.

    Resolution order:
      1. An explicitly configured ``task_type`` always wins.
      2. Float targets: ``{0.0, 1.0}`` is binary classification; any other
         float is regression.
      3. Two distinct values of any dtype is binary classification.
      4. Integer targets with more than ``MAX_CLASSIFICATION_CLASSES`` distinct
         values are treated as regression (counts, ages, prices-as-int);
         otherwise multiclass classification.
      5. Everything else (e.g. string labels) is multiclass classification.
    """
    if configured is not None:
        return configured

    y = pd.Series(y)
    if y.dtype.kind == "f":
        unique = y.dropna().unique()
        if len(unique) == 2 and set(unique).issubset({0.0, 1.0}):
            return TaskType.BINARY_CLASSIFICATION
        return TaskType.REGRESSION

    nunique = y.nunique(dropna=True)
    if nunique == 2:
        return TaskType.BINARY_CLASSIFICATION
    if y.dtype.kind in ("i", "u") and nunique > MAX_CLASSIFICATION_CLASSES:
        return TaskType.REGRESSION
    return TaskType.MULTICLASS_CLASSIFICATION


def _as_model_matrix(X: pd.DataFrame) -> np.ndarray:
    """Feature frame as a plain float matrix.

    Models are fed positional arrays rather than the frame itself so column
    labels never constrain them: XGBoost rejects labels containing ``[``, ``]``
    or ``<`` and CatBoost stringifies non-string labels. Importances are mapped
    back to the original labels by position.
    """
    return X.to_numpy(dtype=np.float64, na_value=np.nan)


class ImportanceCalculator(ABC):
    """Base class for feature importance calculation."""

    @abstractmethod
    def calculate(
        self, X: pd.DataFrame, y: pd.Series, task_type: Optional[TaskType] = None
    ) -> pd.DataFrame:
        """Return DataFrame with 'Feature Id' and 'Importances' columns.

        ``task_type`` is the resolved task type; when ``None`` it is detected
        from ``y`` so the calculator stays consistent with the evaluator.
        """
        pass


class CatBoostImportance(ImportanceCalculator):
    """CatBoost importance calculator."""

    # Class-level default keeps selectors pickled before 2.4.0 loadable.
    n_jobs: int = DEFAULT_N_JOBS

    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def calculate(
        self, X: pd.DataFrame, y: pd.Series, task_type: Optional[TaskType] = None
    ) -> pd.DataFrame:
        task_type = detect_task_type(y, task_type)
        params = {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "thread_count": self.n_jobs,
            "verbose": False,
            # A library must not litter the working directory with catboost_info/.
            "allow_writing_files": False,
        }

        if task_type == TaskType.REGRESSION:
            model = CatBoostRegressor(**params, loss_function="MAE")
        else:
            loss = "Logloss" if task_type == TaskType.BINARY_CLASSIFICATION else "MultiClass"
            model = CatBoostClassifier(**params, loss_function=loss)

        model.fit(_as_model_matrix(X), np.asarray(y))
        importances = np.asarray(model.get_feature_importance(), dtype=float)
        # Normalize defensively against the actual sum, not assumed 100
        total = importances.sum()
        return pd.DataFrame(
            {"Feature Id": list(X.columns), "Importances": importances / max(total, 1e-12)}
        )


class XGBoostImportance(ImportanceCalculator):
    """XGBoost importance calculator."""

    # Class-level default keeps selectors pickled before 2.4.0 loadable.
    n_jobs: int = DEFAULT_N_JOBS

    def __init__(
        self,
        n_estimators: int = 50,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def calculate(
        self, X: pd.DataFrame, y: pd.Series, task_type: Optional[TaskType] = None
    ) -> pd.DataFrame:
        task_type = detect_task_type(y, task_type)
        is_regression = task_type == TaskType.REGRESSION
        model = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
        clf = model(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs
        )
        # Always encode classification targets to contiguous 0..K-1; XGBoost
        # rejects non-contiguous numeric labels like {1,2,3}.
        target = np.asarray(y) if is_regression else LabelEncoder().fit_transform(y)
        clf.fit(_as_model_matrix(X), target)
        return pd.DataFrame(
            {"Feature Id": list(X.columns), "Importances": clf.feature_importances_}
        )


class BaseEvaluator(ABC):
    """Base class with evaluation functionality."""

    # Class-level default keeps selectors pickled before 2.4.0 loadable.
    n_jobs: int = DEFAULT_N_JOBS

    def __init__(
        self,
        metric: Optional[OptimizationMetric] = None,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = DEFAULT_N_JOBS,
    ):
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._task_type = None
        self._eval_model = None
        self._target_encoder: Optional[LabelEncoder] = None

    def evaluate_subset(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """Evaluate a feature subset on a fresh model clone."""
        if self._task_type is None:
            self._task_type = self._detect_task_type(y_train)
        validate_metric(self.metric, self._task_type)
        if self._eval_model is None:
            self._init_eval_model()

        # Encode string targets internally for classification
        y_train_enc, y_val_enc = self._encode_targets(y_train, y_val)

        model = clone(self._eval_model)
        model.fit(_as_model_matrix(X_train), y_train_enc)
        X_val_matrix = _as_model_matrix(X_val)
        y_pred = model.predict(X_val_matrix)
        y_proba = (
            model.predict_proba(X_val_matrix) if self._task_type != TaskType.REGRESSION else None
        )
        return self._calculate_metric(y_val_enc, y_pred, y_proba)

    def _encode_targets(self, y_train: pd.Series, y_val: pd.Series):
        """LabelEncode classification targets to a contiguous 0..K-1 space.

        XGBoost requires class labels in ``[0, n_classes)``; numeric-but-not-
        contiguous targets (e.g. ``{1, 2, 3}`` or ``{10, 20, 30}``) would
        otherwise raise. Encoding every classification target — numeric or not —
        keeps the evaluator robust. Regression targets pass through untouched.
        """
        if self._task_type == TaskType.REGRESSION:
            return np.asarray(y_train), np.asarray(y_val)
        if self._target_encoder is None:
            self._target_encoder = LabelEncoder().fit(y_train)
        return self._target_encoder.transform(y_train), self._target_encoder.transform(y_val)

    def is_better(self, score1: float, score2: float) -> bool:
        """Compare scores using the active metric's direction."""
        direction = METRIC_DIRECTION.get(self.metric, "max")
        return score1 > score2 if direction == "max" else score1 < score2

    def _detect_task_type(self, y: pd.Series) -> TaskType:
        """Delegates to the single source of truth."""
        return detect_task_type(y)

    def _init_eval_model(self):
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbosity": 0,
        }
        if self._task_type == TaskType.REGRESSION:
            self._eval_model = xgb.XGBRegressor(**params)
        else:
            # Targets are label-encoded internally; XGBoost picks the
            # objective-appropriate eval metric (logloss / mlogloss) itself.
            self._eval_model = xgb.XGBClassifier(**params)

    def _calculate_metric(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray = None
    ) -> float:
        if self.metric is None:
            self.metric = default_metric(self._task_type)

        metrics = {
            OptimizationMetric.MAE: lambda: mean_absolute_error(y_true, y_pred),
            OptimizationMetric.MSE: lambda: mean_squared_error(y_true, y_pred),
            OptimizationMetric.RMSE: lambda: float(np.sqrt(mean_squared_error(y_true, y_pred))),
            OptimizationMetric.R2: lambda: r2_score(y_true, y_pred),
            OptimizationMetric.ACCURACY: lambda: accuracy_score(y_true, y_pred),
            OptimizationMetric.PRECISION: lambda: self._classification_metric(
                precision_score, y_true, y_pred
            ),
            OptimizationMetric.RECALL: lambda: self._classification_metric(
                recall_score, y_true, y_pred
            ),
            OptimizationMetric.F1: lambda: self._classification_metric(f1_score, y_true, y_pred),
            OptimizationMetric.AUC_ROC: lambda: self._auc_score(y_true, y_proba),
        }
        return float(metrics[self.metric]())

    def _classification_metric(self, metric_func, y_true: pd.Series, y_pred: np.ndarray) -> float:
        avg = "binary" if self._task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
        return metric_func(y_true, y_pred, average=avg, zero_division=0)

    def _auc_score(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        if self._task_type == TaskType.BINARY_CLASSIFICATION:
            return roc_auc_score(y_true, y_proba[:, 1])

        y_true = np.asarray(y_true)
        present = np.unique(y_true)
        if len(present) == y_proba.shape[1]:
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")

        # A rare class can be absent from the validation slice, which makes the
        # stock multiclass AUC undefined. Score one-vs-rest over the classes that
        # are present, weighted by their support.
        if len(present) < 2:
            raise ConfigurationError(
                "AUC is undefined: the validation slice holds a single class. "
                "Use a larger validation_split or another metric."
            )
        scores = [roc_auc_score(y_true == cls, y_proba[:, cls]) for cls in present]
        weights = [np.sum(y_true == cls) for cls in present]
        return float(np.average(scores, weights=weights))
