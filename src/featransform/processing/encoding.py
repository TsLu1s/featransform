from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from featransform.core.base import BaseTransformer
from featransform.core.enums import EncodingStrategy


def _is_categorical_like(dtype: Any) -> bool:
    """Whether a column dtype holds categorical / text data.

    Checked per dtype rather than through ``select_dtypes`` so the same code
    runs on every supported pandas: ``select_dtypes(include="str")`` raises on
    pandas < 3, while omitting it misses pandas ≥ 3's dedicated string dtype.
    """
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


class CategoricalEncoder(BaseTransformer):
    """Label-encode categorical / text columns.

    Categories are learned at ``fit`` and replayed at ``transform``; a value not
    seen at fit (including a missing value that was not present at fit) maps
    to ``-1``.
    """

    # Class-level default keeps pipelines pickled before 2.4.0 loadable.
    _code_maps: Optional[Dict[str, Dict[str, int]]] = None

    def __init__(self, strategy: EncodingStrategy = EncodingStrategy.LABEL, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._encoders: Dict[str, LabelEncoder] = {}
        self._categorical_columns: List[str] = []
        self._code_maps = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncoder":
        self._validate_fit_input(X)

        self._categorical_columns = [col for col in X.columns if _is_categorical_like(X[col].dtype)]

        # Refit from scratch so a second fit never inherits stale encoders.
        self._encoders = {}
        self._code_maps = {}
        for col in self._categorical_columns:
            encoder = LabelEncoder()
            # Missing values are not a category: they encode to -1 like any
            # other value unseen at fit, on every pandas version.
            encoder.fit(X[col].dropna().astype(str))
            self._encoders[col] = encoder
            self._code_maps[col] = self._build_code_map(encoder)

        self._fitted = True
        return self

    @staticmethod
    def _build_code_map(encoder: LabelEncoder) -> Dict[str, int]:
        """``{class: code}`` in ``classes_`` order — matches ``encoder.transform``."""
        return {cls: idx for idx, cls in enumerate(encoder.classes_)}

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_input(X)
        X_transformed = X.copy()

        code_maps = self._code_maps or {}
        for col in self._categorical_columns:
            if col not in X.columns or col not in self._encoders:
                continue
            # Vectorized encode via the precomputed {class: code} map.
            # Unknown categories map to -1.
            code_map = code_maps.get(col) or self._build_code_map(self._encoders[col])
            series = X[col]
            codes = series.astype(str).map(code_map).astype("float64")
            X_transformed[col] = codes.where(series.notna(), -1.0).fillna(-1.0).astype("int64")

        return X_transformed

    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, **self._params}

    def set_params(self, **params) -> "CategoricalEncoder":
        for key, value in params.items():
            setattr(self, key, value)
        return self
