from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
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

from featransform.core.constants import DEFAULT_N_JOBS, DEFAULT_RANDOM_STATE
from featransform.core.enums import OptimizationMetric, TaskType


class ImportanceCalculator(ABC):
    """Base class for feature importance calculation."""

    @abstractmethod
    def calculate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Return DataFrame with 'Feature Id' and 'Importances' columns."""
        pass


class CatBoostImportance(ImportanceCalculator):
    """CatBoost importance calculator."""

    def __init__(self, n_estimators: int = 50, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def calculate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        is_regression = y.dtype in ["float64", "float32"]
        params = {
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "verbose": False,
        }

        if is_regression:
            model = CatBoostRegressor(**params, loss_function="MAE")
        else:
            loss = "Logloss" if y.nunique() == 2 else "MultiClass"
            model = CatBoostClassifier(**params, loss_function=loss)

        model.fit(X, y)
        imp = model.get_feature_importance(prettified=True)
        imp["Importances"] = imp["Importances"] / 100
        return imp


class XGBoostImportance(ImportanceCalculator):
    """XGBoost importance calculator."""

    def __init__(self, n_estimators: int = 50, random_state: int = DEFAULT_RANDOM_STATE):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def calculate(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        is_regression = y.dtype in ["float64", "float32"]
        model = xgb.XGBRegressor if is_regression else xgb.XGBClassifier
        clf = model(n_estimators=self.n_estimators, random_state=self.random_state)
        clf.fit(X, y)
        return pd.DataFrame({"Feature Id": X.columns, "Importances": clf.feature_importances_})


class BaseEvaluator(ABC):
    """Base class with evaluation functionality."""

    def __init__(self, metric: OptimizationMetric = None):
        self.metric = metric
        self._task_type = None
        self._eval_model = None

    def evaluate_subset(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """Evaluate a feature subset."""
        if self._eval_model is None:
            self._init_eval_model()
        self._eval_model.fit(X_train, y_train)
        y_pred = self._eval_model.predict(X_val)
        y_proba = (
            self._eval_model.predict_proba(X_val)
            if self._task_type != TaskType.REGRESSION
            else None
        )
        return self._calculate_metric(y_val, y_pred, y_proba)

    def is_better(self, score1: float, score2: float) -> bool:
        return score1 > score2

    def _detect_task_type(self, y: pd.Series) -> TaskType:
        if y.dtype in ["float64", "float32"]:
            return TaskType.REGRESSION
        return (
            TaskType.BINARY_CLASSIFICATION
            if y.nunique() == 2
            else TaskType.MULTICLASS_CLASSIFICATION
        )

    def _init_eval_model(self):
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": DEFAULT_RANDOM_STATE,
            "n_jobs": DEFAULT_N_JOBS,
            "verbosity": 0,
        }
        if self._task_type == TaskType.REGRESSION:
            self._eval_model = xgb.XGBRegressor(**params)
        else:
            self._eval_model = xgb.XGBClassifier(
                **params, use_label_encoder=False, eval_metric="logloss"
            )

    def _calculate_metric(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray = None
    ) -> float:
        if self.metric is None:
            self.metric = (
                OptimizationMetric.MSE
                if self._task_type == TaskType.REGRESSION
                else OptimizationMetric.F1
            )

        metrics = {
            OptimizationMetric.MAE: lambda: -mean_absolute_error(y_true, y_pred),
            OptimizationMetric.MSE: lambda: -mean_squared_error(y_true, y_pred),
            OptimizationMetric.RMSE: lambda: -np.sqrt(mean_squared_error(y_true, y_pred)),
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
        return metrics[self.metric]()

    def _classification_metric(self, metric_func, y_true: pd.Series, y_pred: np.ndarray) -> float:
        avg = "binary" if self._task_type == TaskType.BINARY_CLASSIFICATION else "weighted"
        return metric_func(y_true, y_pred, average=avg, zero_division=0)

    def _auc_score(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        if self._task_type == TaskType.BINARY_CLASSIFICATION:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
