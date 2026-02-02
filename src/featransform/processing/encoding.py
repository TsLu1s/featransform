from typing import Any, Dict, Optional

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from featransform.core.base import BaseTransformer
from featransform.core.enums import EncodingStrategy


class CategoricalEncoder(BaseTransformer):
    """Smart encoding with multiple strategies."""

    def __init__(self, strategy: EncodingStrategy = EncodingStrategy.LABEL, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._encoders = {}
        self._categorical_columns = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CategoricalEncoder":
        self._validate_input(X)

        # Identify categorical columns
        self._categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Create encoders for each categorical column
        for col in self._categorical_columns:
            if self.strategy == EncodingStrategy.LABEL:
                encoder = LabelEncoder()
                encoder.fit(X[col].astype(str))
            else:
                encoder = None  # Possibly add other strategies

            self._encoders[col] = encoder

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X_transformed = X.copy()

        for col in self._categorical_columns:
            if col in self._encoders and self._encoders[col] is not None:
                encoder = self._encoders[col]

                if isinstance(encoder, LabelEncoder):
                    # Handle unknown categories
                    X_transformed[col] = (
                        X[col]
                        .astype(str)
                        .map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
                    )
                else:
                    pass  # Handle other encoders

        return X_transformed

    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, **self._params}

    def set_params(self, **params) -> "CategoricalEncoder":
        for key, value in params.items():
            setattr(self, key, value)
        return self
