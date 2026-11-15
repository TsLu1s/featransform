import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional, final

import pandas as pd

from featransform.core.base import BaseComponent
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.models import TransformationResult

# Library-grade logger — application is responsible for handler configuration
logger = logging.getLogger(__name__)


class BaseFeatureEngineer(BaseComponent):
    """Abstract base for feature engineering strategies."""

    def __init__(
        self, feature_type: FeatureType, model_family: Optional[ModelFamily] = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_type = feature_type
        self.model_family = model_family
        self._feature_names: List[str] = []
        self._models: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFeatureEngineer":
        """Fit the feature engineering model. ``y`` is accepted for API symmetry and unused."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> TransformationResult:
        """Generate engineered features."""
        pass

    @abstractmethod
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private abstract method to create features."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> TransformationResult:
        """Fit and generate engineered features in one step."""
        return self.fit(X, y).transform(X)

    def _generate_feature_names(self, n_features: int, prefix: str) -> List[str]:
        """Private method to generate feature names."""
        return [f"{prefix}{i}" for i in range(n_features)]

    @staticmethod
    def _unique_names(columns) -> List[str]:
        """Column labels with duplicates disambiguated by a ``_1``, ``_2`` … suffix."""
        taken = set(columns)
        seen: Dict[str, int] = {}
        new_columns: List[str] = []
        for col in columns:
            if col in seen:
                candidate = col
                while candidate in taken:
                    seen[col] += 1
                    candidate = f"{col}_{seen[col]}"
                taken.add(candidate)
                new_columns.append(candidate)
            else:
                seen[col] = 0
                new_columns.append(col)
        return new_columns

    @classmethod
    def _ensure_unique_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate duplicate column names.

        Strategies derive feature names from short model prefixes, so two
        strategies of the same family (e.g. two KMeans) can emit identical
        column labels. Duplicates get a ``_1``, ``_2`` … suffix so downstream
        ``pd.concat`` / schema alignment stays well-defined.
        """
        if not df.columns.duplicated().any():
            return df
        df = df.copy()
        df.columns = cls._unique_names(list(df.columns))
        return df

    @final
    def get_feature_names(self) -> List[str]:
        """Get generated feature names."""
        return self._feature_names


class BaseFeatureEnsemble(BaseFeatureEngineer):
    """Shared fit / transform logic for an ensemble of feature strategies.

    Subclasses set ``self.strategies`` in ``__init__`` and implement
    ``_default_strategies``.
    """

    strategies: List[BaseFeatureEngineer]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseFeatureEnsemble":
        self._validate_fit_input(X)

        for strategy in self.strategies:
            self._log(f"Fitting {strategy.__class__.__name__}")
            strategy.fit(X, y)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        self._validate_input(X)

        all_features = []
        all_info = []

        for strategy in self.strategies:
            result = strategy.transform(X)
            all_features.append(result.features)
            all_info.extend(result.feature_info)

        combined_features = self._ensure_unique_columns(pd.concat(all_features, axis=1))
        # Names reflect the columns actually emitted — including the
        # feature metadata, which follows any de-duplication rename.
        self._feature_names = list(combined_features.columns)
        for info, name in zip(all_info, self._feature_names):
            info.name = name

        return TransformationResult(
            features=combined_features,
            feature_info=all_info,
        )

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensembles delegate feature creation to their strategies."""
        return self.transform(X).features

    def get_params(self) -> Dict[str, Any]:
        return {"strategies": self.strategies, **self._params}

    def set_params(self, **params) -> "BaseFeatureEnsemble":
        for key, value in params.items():
            setattr(self, key, value)
        return self
