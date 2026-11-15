from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from featransform.core.base import BaseTransformer
from featransform.core.constants import DEFAULT_RANDOM_STATE
from featransform.core.enums import ImputationStrategy
from featransform.core.exceptions import ValidationError

_TEXT_FILL = "missing"


class CoreImputer(BaseTransformer):
    """Missing-value imputation, learned at ``fit`` and replayed at ``transform``.

    * Numeric columns are imputed with the configured strategy. Only the cells
      that are actually missing are written, so complete columns keep their
      dtype (integers stay integers).
    * ``KNN`` imputation runs in standardized space: neighbours are found on
      z-scored columns and the result is mapped back, so a wide-range column
      cannot dominate the distance.
    * ``exclude_columns`` (identifier / index columns) are never used as
      predictors by the multivariate strategies; any gaps in them are filled
      with their own fit-time median.
    * Columns that are entirely missing at fit carry no information and are
      filled with ``0.0``.
    * Non-numeric columns are filled with their fit-time mode.
    """

    # Class-level defaults keep pipelines pickled before 2.4.0 loadable.
    exclude_columns: Sequence[str] = ()
    n_neighbors: int = 5
    _all_nan_columns: Sequence[str] = ()
    _excluded_fill: Optional[Dict[str, float]] = None
    _knn_scaler: Optional[StandardScaler] = None

    def __init__(
        self,
        strategy: ImputationStrategy = ImputationStrategy.ITERATIVE,
        exclude_columns: Optional[Sequence[str]] = None,
        n_neighbors: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.strategy = strategy
        self.exclude_columns = list(exclude_columns or [])
        self.n_neighbors = n_neighbors
        self._imputer = None
        self._knn_scaler = None
        self._numeric_columns: List[str] = []
        self._categorical_columns: List[str] = []
        self._categorical_modes: Dict[str, Any] = {}
        self._all_nan_columns = []
        self._excluded_fill = {}

    def _build_imputer(self):
        if self.strategy == ImputationStrategy.ITERATIVE:
            return IterativeImputer(
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE)
            )
        if self.strategy == ImputationStrategy.KNN:
            return KNNImputer(n_neighbors=self.n_neighbors)
        return SimpleImputer(strategy=self.strategy.value)

    @staticmethod
    def _as_float(block: pd.DataFrame) -> pd.DataFrame:
        """Numeric block as float64 with missing values as NaN (also for nullable dtypes)."""
        return block.astype("float64")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CoreImputer":
        self._validate_fit_input(X)

        excluded = set(self.exclude_columns or ())
        numeric = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Identifier columns: kept out of the multivariate model, own median fill.
        self._excluded_fill = {}
        for col in numeric:
            if col in excluded:
                median = X[col].astype("float64").median()
                self._excluded_fill[col] = 0.0 if pd.isna(median) else float(median)

        candidates = [col for col in numeric if col not in excluded]
        self._all_nan_columns = [col for col in candidates if X[col].isna().all()]
        self._numeric_columns = [col for col in candidates if col not in self._all_nan_columns]

        self._imputer = self._build_imputer()
        self._knn_scaler = None
        if self._numeric_columns:
            block = self._as_float(X[self._numeric_columns])
            if self.strategy == ImputationStrategy.KNN:
                # StandardScaler ignores NaN when fitting and keeps it when transforming.
                self._knn_scaler = StandardScaler().fit(block)
                self._imputer.fit(self._knn_scaler.transform(block))
            else:
                self._imputer.fit(block)

        # Learn categorical modes at fit, replay at transform
        self._categorical_modes = {}
        for col in self._categorical_columns:
            mode = X[col].mode()
            if not mode.empty:
                self._categorical_modes[col] = mode.iloc[0]
            elif self._is_text_like(X[col].dtype):
                self._categorical_modes[col] = _TEXT_FILL
            # else: an entirely-missing non-text column has no sensible fill value

        self._fitted = True
        return self

    @staticmethod
    def _is_text_like(dtype: Any) -> bool:
        return (
            isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )

    def _impute_numeric(self, block: pd.DataFrame) -> np.ndarray:
        """Run the fitted imputer on a float block and return the completed array."""
        if self._knn_scaler is not None:
            scaled = self._knn_scaler.transform(block)
            return self._knn_scaler.inverse_transform(self._imputer.transform(scaled))
        return np.asarray(self._imputer.transform(block))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_input(X)
        X_transformed = X.copy()

        if self._numeric_columns:
            missing_cols = [col for col in self._numeric_columns if col not in X.columns]
            if missing_cols:
                raise ValidationError(
                    f"Columns seen at fit are missing from the input: {missing_cols}"
                )
            block = self._as_float(X[self._numeric_columns])
            values = block.to_numpy()
            gaps = np.isnan(values)
            if gaps.any():
                imputed = self._impute_numeric(block)
                # Write only the missing cells: observed values (and the dtype of
                # complete columns) are left exactly as they came in.
                values = np.where(gaps, imputed, values)
                for idx, col in enumerate(self._numeric_columns):
                    if gaps[:, idx].any():
                        X_transformed[col] = values[:, idx]
            # Nullable extension dtypes (Int64, Float64, …) leave as float64 so
            # the output dtype does not depend on whether a frame had gaps.
            for col in self._numeric_columns:
                if pd.api.types.is_extension_array_dtype(X_transformed[col].dtype):
                    X_transformed[col] = X_transformed[col].astype("float64")

        for col in self._all_nan_columns:
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].astype("float64").fillna(0.0)

        for col, fill in (self._excluded_fill or {}).items():
            if col in X_transformed.columns and X_transformed[col].isna().any():
                X_transformed[col] = X_transformed[col].astype("float64").fillna(fill)

        # Replay learned modes for categorical columns
        for col in self._categorical_columns:
            if col not in X_transformed.columns or col not in self._categorical_modes:
                continue
            series = X_transformed[col]
            if series.isna().any():
                fill = self._categorical_modes[col]
                if isinstance(series.dtype, pd.CategoricalDtype) and (
                    fill not in series.cat.categories
                ):
                    series = series.cat.add_categories([fill])
                X_transformed[col] = series.fillna(fill)

        return X_transformed

    def get_params(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "exclude_columns": list(self.exclude_columns or []),
            "n_neighbors": self.n_neighbors,
            **self._params,
        }

    def set_params(self, **params) -> "CoreImputer":
        for key, value in params.items():
            setattr(self, key, value)
        return self
