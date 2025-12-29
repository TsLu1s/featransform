from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from featransform.core.base import BaseTransformer
from featransform.core.enums import ImputationStrategy
from featransform.core.constants import DEFAULT_RANDOM_STATE


class CoreImputer(BaseTransformer):
    """Smart imputation with multiple strategies."""
    
    def __init__(self, strategy: ImputationStrategy = ImputationStrategy.ITERATIVE, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        self._imputer = None
        self._numeric_columns = []
        self._categorical_columns = []
    
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]= None) -> 'CoreImputer':
        self._validate_input(X)
        
        # Identify column types
        self._numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self._categorical_columns = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Create appropriate imputer
        if self.strategy == ImputationStrategy.ITERATIVE:
            self._imputer = IterativeImputer(
                random_state=self._params.get('random_state', DEFAULT_RANDOM_STATE)
            )
        elif self.strategy == ImputationStrategy.KNN:
            self._imputer = KNNImputer(n_neighbors=5)
        else:
            self._imputer = SimpleImputer(strategy=self.strategy.value)
        
        # Fit only on numeric columns if they exist
        if self._numeric_columns:
            self._imputer.fit(X[self._numeric_columns])
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        X_transformed = X.copy()
        
        # Transform numeric columns
        if self._numeric_columns:
            X_transformed[self._numeric_columns] = self._imputer.transform(X[self._numeric_columns])
        
        # Simple mode imputation for categorical
        for col in self._categorical_columns:
            if X_transformed[col].isnull().any():
                mode_value = X[col].mode()[0] if not X[col].mode().empty else "missing"
                X_transformed[col] = X_transformed[col].fillna(mode_value)
        
        return X_transformed
    
    def get_params(self) -> Dict[str, Any]:
        return {"strategy": self.strategy, **self._params}
    
    def set_params(self, **params) -> 'CoreImputer':
        for key, value in params.items():
            setattr(self, key, value)
        return self
