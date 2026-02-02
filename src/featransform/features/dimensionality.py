import os
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.decomposition import PCA, FastICA, TruncatedSVD

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable TensorFlow optimizations using OneDNN

from featransform.core.constants import COMPONENT_PREFIX, DEFAULT_N_COMPONENTS, DEFAULT_RANDOM_STATE
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.models import FeatureInfo, TransformationResult
from featransform.features.base import BaseFeatureEngineer

warnings.filterwarnings("ignore", category=FutureWarning)


class DimensionalityStrategy(BaseFeatureEngineer):
    """Base strategy for dimensionality dimensionality."""

    def __init__(self, n_components: int = DEFAULT_N_COMPONENTS, **kwargs):
        super().__init__(feature_type=FeatureType.DIMENSION, **kwargs)
        self.n_components = n_components

    def _initialize_models(self, X: pd.DataFrame) -> None:
        """Private method to initialize and validate models."""
        # Validate n_components against data dimensions
        max_components = min(X.shape) - 1

        if isinstance(self.n_components, int) and self.n_components > max_components:
            self._log(
                f"Requested {self.n_components} components exceeds max {max_components}, using default"
            )
            effective_components = DEFAULT_N_COMPONENTS  # Fall back to default
        else:
            effective_components = self.n_components

        # Build models with validated parameters
        self._models = self._build_models()

        # Update n_components in models if needed
        for model in self._models.values():
            if hasattr(model, "n_components"):
                model.n_components = effective_components

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DimensionalityStrategy":
        self._validate_input(X)

        # Initialize models
        self._initialize_models(X)

        # Fit all models
        for name, model in self._models.items():
            self._log(f"Fitting {name}")
            model.fit(X)

        # Get actual components after fitting (for variance-based configs)
        actual_components = self._get_fitted_components()

        self._fitted = True
        self._feature_names = self._generate_feature_names(
            int(actual_components), COMPONENT_PREFIX  # Ensure integer
        )
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        self._validate_input(X)

        features = self._create_features(X)

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
        # Fallback
        return DEFAULT_N_COMPONENTS

    def _build_models(self) -> Dict[str, Any]:
        """Build dimensionality dimensionality models."""
        return {}

    def get_params(self) -> Dict[str, Any]:
        return {"n_components": self.n_components, **self._params}

    def set_params(self, **params) -> "DimensionalityStrategy":
        for key, value in params.items():
            setattr(self, key, value)
        return self


class PCAStrategy(DimensionalityStrategy):
    """PCA dimensionality dimensionality."""

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
    """Truncated SVD dimensionality dimensionality."""

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
    """FastICA dimensionality dimensionality."""

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


class DimensionalityEnsemble(BaseFeatureEngineer):
    """Ensemble of dimensionality dimensionality strategies."""

    def __init__(self, strategies: Optional[List[DimensionalityStrategy]] = None, **kwargs):
        super().__init__(feature_type=FeatureType.DIMENSION, **kwargs)
        self.strategies = strategies or self._default_strategies()

    def _default_strategies(self) -> List[DimensionalityStrategy]:
        return [PCAStrategy(n_components=self._params.get("n_components", DEFAULT_N_COMPONENTS))]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DimensionalityEnsemble":
        self._validate_input(X)

        for strategy in self.strategies:
            self._log(f"Fitting {strategy.__class__.__name__}")
            strategy.fit(X, y)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()

        all_features = []
        all_info = []

        for strategy in self.strategies:
            result = strategy.transform(X)
            all_features.append(result.features)
            all_info.extend(result.feature_info)

        combined_features = pd.concat(all_features, axis=1)

        return TransformationResult(
            features=combined_features,
            feature_info=all_info,
        )

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        pass  # Not used

    def get_params(self) -> Dict[str, Any]:
        return {"strategies": self.strategies, **self._params}

    def set_params(self, **params) -> "DimensionalityEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self
