from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from featransform.core.constants import (
    DEFAULT_CONTAMINATION,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_STATE,
)
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.exceptions import NotFittedError, ValidationError
from featransform.core.models import FeatureInfo, TransformationResult
from featransform.features.base import BaseFeatureEngineer, BaseFeatureEnsemble


class AnomalyStrategy(BaseFeatureEngineer):
    """Base strategy for anomaly detection."""

    def __init__(
        self, contamination: float = DEFAULT_CONTAMINATION, include_scores: bool = False, **kwargs
    ):
        super().__init__(feature_type=FeatureType.ANOMALY, **kwargs)
        self.contamination = contamination
        self.include_scores = include_scores
        self._input_features = None

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create anomaly features from fitted models."""
        results = []

        for name, model in self._models.items():
            predictions = model.predict(X)
            predictions = (predictions == -1).astype(int)

            prefix = name.split("_")[0].lower()[:3]
            feature_dict = {f"{prefix}_anomaly": predictions}

            if self.include_scores:
                scores = model.decision_function(X)
                feature_dict[f"{prefix}_score"] = scores

            results.append(pd.DataFrame(feature_dict, index=X.index))

        return pd.concat(results, axis=1)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "AnomalyStrategy":
        self._validate_fit_input(X)

        self._input_features = list(X.columns)

        self._models = self._build_models()
        for _, model in self._models.items():
            model.fit(X)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        self._validate_input(X)
        X = self._align_features(X)

        features = self._create_features(X)
        # Names reflect the columns actually emitted.
        self._feature_names = list(features.columns)

        feature_info = [
            FeatureInfo(
                name=col,
                feature_type=self.feature_type,
                model_family=self.model_family,
                source_columns=list(X.columns),
            )
            for col in features.columns
        ]

        return TransformationResult(
            features=features,
            feature_info=feature_info,
        )

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has same features as during fit."""
        if self._input_features is None:
            raise NotFittedError("No input features stored from fit")

        missing = set(self._input_features) - set(X.columns)
        if missing:
            raise ValidationError(f"Missing features in transform: {sorted(missing, key=str)}")

        extra = set(X.columns) - set(self._input_features)
        if extra:
            self._log(f"Dropping extra features not seen during fit: {extra}")

        return X[self._input_features]

    def _build_models(self) -> Dict[str, Any]:
        """Build anomaly detection models."""
        return {}

    def get_params(self) -> Dict[str, Any]:
        return {"contamination": self.contamination, **self._params}

    def set_params(self, **params) -> "AnomalyStrategy":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class IsolationForestStrategy(AnomalyStrategy):
    """Isolation Forest anomaly detection."""

    def __init__(self, n_estimators: int = DEFAULT_N_ESTIMATORS, **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.model_family = ModelFamily.ISOLATION_FOREST

    def _build_models(self) -> Dict[str, Any]:
        return {
            "isolation_forest": IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
                n_jobs=self._params.get("n_jobs", DEFAULT_N_JOBS),
            )
        }


class LocalOutlierFactorStrategy(AnomalyStrategy):
    """Local Outlier Factor anomaly detection (novelty mode).

    Note: LOF in novelty mode scores a *training* row with that row counted
    among its own neighbours, so slightly fewer training rows are flagged than
    the configured ``contamination`` (and than on unseen rows).
    """

    def __init__(self, n_neighbors: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_neighbors = n_neighbors
        self.model_family = ModelFamily.LOCAL_OUTLIER_FACTOR

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LocalOutlierFactorStrategy":
        self._validate_fit_input(X)

        self._input_features = list(X.columns)

        self._models = self._build_models()
        for _, model in self._models.items():
            # Fit with numpy array to match transform behavior
            model.fit(X.values)

        self._fitted = True
        return self

    def _build_models(self) -> Dict[str, Any]:
        return {
            "lof": LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=self.n_neighbors,
                novelty=True,
                n_jobs=self._params.get("n_jobs", DEFAULT_N_JOBS),
            )
        }

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        results = []

        for name, model in self._models.items():
            # Convert to numpy to match fit behavior & avoid feature name warning
            X_values = X.values

            predictions = model.predict(X_values)
            predictions = (predictions == -1).astype(int)

            feature_dict = {f"{name}_anomaly": predictions}

            if self.include_scores:
                scores = model.decision_function(X_values)
                feature_dict[f"{name}_score"] = scores

            result_df = pd.DataFrame(feature_dict, index=X.index)
            results.append(result_df)

        return pd.concat(results, axis=1) if results else pd.DataFrame()


class OneClassSVMStrategy(AnomalyStrategy):
    """One-Class SVM anomaly detection.

    The share of flagged rows is governed by ``nu``; ``contamination`` is
    accepted for interface parity with the other strategies but not used.
    """

    def __init__(self, nu: float = 0.05, kernel: str = "rbf", **kwargs):
        super().__init__(**kwargs)
        self.nu = nu
        self.kernel = kernel
        self.model_family = ModelFamily.ONE_CLASS_SVM

    def _build_models(self) -> Dict[str, Any]:
        return {"one_class_svm": OneClassSVM(nu=self.nu, kernel=self.kernel, gamma="scale")}


class EllipticEnvelopeStrategy(AnomalyStrategy):
    """Elliptic Envelope anomaly detection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_family = ModelFamily.ELLIPTIC_ENVELOPE

    def _build_models(self) -> Dict[str, Any]:
        return {
            "elliptic_envelope": EllipticEnvelope(
                contamination=self.contamination,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class AnomalyEnsemble(BaseFeatureEnsemble):
    """Ensemble of multiple anomaly detection strategies."""

    def __init__(
        self,
        strategies: Optional[List[AnomalyStrategy]] = None,
        include_scores: bool = False,
        **kwargs,
    ):
        super().__init__(feature_type=FeatureType.ANOMALY, **kwargs)
        self.include_scores = include_scores
        self.strategies = strategies or self._default_strategies()

    def _default_strategies(self) -> List[AnomalyStrategy]:
        """Create default anomaly detection strategies."""
        return [
            IsolationForestStrategy(
                contamination=self._params.get("contamination", DEFAULT_CONTAMINATION),
                include_scores=self.include_scores,
                n_jobs=self._params.get("n_jobs", DEFAULT_N_JOBS),
            )
        ]
