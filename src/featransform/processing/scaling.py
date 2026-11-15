from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from featransform.core.base import BaseTransformer
from featransform.core.enums import ScalingStrategy

_SCALERS = {
    ScalingStrategy.STANDARD: StandardScaler,
    ScalingStrategy.ROBUST: RobustScaler,
    ScalingStrategy.MINMAX: MinMaxScaler,
}


class FeatureScaler(BaseTransformer):
    """Scales numeric columns for the feature-engineering input.

    Fitted at fit time and replayed at transform (learn-then-replay), matching
    the imputation / drop_constant persistence pattern. Only the numeric columns
    seen at fit are scaled; any other columns pass through untouched, and columns
    absent at transform are skipped.
    """

    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.STANDARD, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._scaler = None
        self._numeric_columns: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureScaler":
        self._validate_fit_input(X)
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        if self._numeric_columns:
            scaler_cls = _SCALERS[self.strategy]
            self._scaler = scaler_cls()
            self._scaler.fit(X[self._numeric_columns])
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_input(X)
        # sklearn scalers need the exact fit-time column set; skip if incomplete.
        if (
            not self._numeric_columns
            or self._scaler is None
            or not all(c in X.columns for c in self._numeric_columns)
        ):
            return X
        X_transformed = X.copy()
        X_transformed[self._numeric_columns] = self._scaler.transform(X[self._numeric_columns])
        return X_transformed

    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, **self._params}

    def set_params(self, **params) -> "FeatureScaler":
        for key, value in params.items():
            setattr(self, key, value)
        return self
