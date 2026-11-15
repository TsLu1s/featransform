import warnings
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD

from featransform.core.constants import DEFAULT_N_COMPONENTS, DEFAULT_RANDOM_STATE
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.exceptions import ConfigurationError
from featransform.core.models import FeatureInfo, TransformationResult
from featransform.features.base import BaseFeatureEngineer, BaseFeatureEnsemble


class DimensionalityStrategy(BaseFeatureEngineer):
    """Base strategy for dimensionality reduction."""

    def __init__(self, n_components: Union[int, float] = DEFAULT_N_COMPONENTS, **kwargs):
        super().__init__(feature_type=FeatureType.DIMENSION, **kwargs)
        self.n_components = n_components

    def _initialize_models(self, X: pd.DataFrame) -> None:
        """Initialize and validate models.

        Float n_components is only valid for PCA (variance threshold);
        TruncatedSVD and FastICA require integer n_components.
        """
        if isinstance(self.n_components, float) and self.model_family != ModelFamily.PCA:
            raise ConfigurationError(
                f"n_components must be an integer for "
                f"{self.model_family.value if self.model_family else self.__class__.__name__}; "
                f"got float {self.n_components}"
            )

        max_components = min(X.shape) - 1
        if isinstance(self.n_components, int) and self.n_components > max_components:
            # Clamp the fallback so it can never itself exceed the data's
            # feature/sample budget (which would crash inside sklearn).
            effective_components = max(1, min(DEFAULT_N_COMPONENTS, max_components))
            # The request is being overridden, so say so even when not verbose.
            warnings.warn(
                f"{self.__class__.__name__}: requested n_components={self.n_components} "
                f"exceeds the data budget ({max_components}); using {effective_components}.",
                UserWarning,
                stacklevel=3,
            )
        else:
            effective_components = self.n_components

        self._models = self._build_models()

        for model in self._models.values():
            if hasattr(model, "n_components"):
                model.n_components = effective_components

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DimensionalityStrategy":
        self._validate_fit_input(X)

        self._initialize_models(X)

        for name, model in self._models.items():
            self._log(f"Fitting {name}")
            model.fit(X)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        self._validate_input(X)

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

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create dimensionality reduced features."""
        results = []

        for name, model in self._models.items():
            transformed = model.transform(X)

            prefix = name[:3]
            cols = [f"{prefix}_{i}" for i in range(transformed.shape[1])]
            results.append(pd.DataFrame(transformed, columns=cols, index=X.index))

        return pd.concat(results, axis=1)

    def _get_fitted_components(self) -> int:
        """Get actual number of components after fitting."""
        for model in self._models.values():
            # After fitting, sklearn stores actual components in n_components_
            if hasattr(model, "n_components_"):
                return model.n_components_
            elif hasattr(model, "components_"):
                return model.components_.shape[0]
        return DEFAULT_N_COMPONENTS

    def _build_models(self) -> Dict[str, Any]:
        """Build dimensionality reduction models."""
        return {}

    def get_params(self) -> Dict[str, Any]:
        return {"n_components": self.n_components, **self._params}

    def set_params(self, **params) -> "DimensionalityStrategy":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class PCAStrategy(DimensionalityStrategy):
    """PCA dimensionality reduction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_family = ModelFamily.PCA

    def _build_models(self) -> Dict[str, Any]:
        return {
            "pca": PCA(
                n_components=self.n_components,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class TruncatedSVDStrategy(DimensionalityStrategy):
    """Truncated SVD dimensionality reduction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_family = ModelFamily.TRUNCATED_SVD

    def _build_models(self) -> Dict[str, Any]:
        return {
            "svd": TruncatedSVD(
                n_components=self.n_components,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class FastICAStrategy(DimensionalityStrategy):
    """FastICA dimensionality reduction."""

    def __init__(self, algorithm: str = "parallel", whiten: str = "unit-variance", **kwargs):
        super().__init__(**kwargs)
        self.algorithm = algorithm
        self.whiten = whiten
        self.model_family = ModelFamily.FAST_ICA

    def _build_models(self) -> Dict[str, Any]:
        return {
            "ica": FastICA(
                n_components=self.n_components,
                algorithm=self.algorithm,
                whiten=self.whiten,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class DimensionalityEnsemble(BaseFeatureEnsemble):
    """Ensemble of dimensionality reduction strategies."""

    def __init__(self, strategies: Optional[List[DimensionalityStrategy]] = None, **kwargs):
        super().__init__(feature_type=FeatureType.DIMENSION, **kwargs)
        self.strategies = strategies or self._default_strategies()

    def _default_strategies(self) -> List[DimensionalityStrategy]:
        return [PCAStrategy(n_components=self._params.get("n_components", DEFAULT_N_COMPONENTS))]
