from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, final
from contextlib import contextmanager
from functools import wraps
import time
import logging
import pandas as pd

from featransform.core.exceptions import (
    NotFittedError, TransformationError,
)
from featransform.core.constants import MIN_SAMPLES_REQUIRED

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class BaseComponent(ABC):
    """Abstract base for all components with common functionality."""
    
    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self._fitted = False
        self._params = kwargs
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get component parameters."""
        pass
    
    @abstractmethod
    def set_params(self, **params) -> 'BaseComponent':
        """Set component parameters."""
        pass
    
    def _check_fitted(self) -> None:
        """Private method to verify fitting status."""
        if not self._fitted:
            raise NotFittedError(f"{self.__class__.__name__} not fitted")
    
    def _log(self, message: str, level: str = "info") -> None:
        """Enhanced logging with levels."""
        if self.verbose:
            getattr(logger, level)(f"[{self.__class__.__name__}] {message}")
    
    @final
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Private final method for input validation."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(X)}")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        if X.shape[0] < MIN_SAMPLES_REQUIRED:
            raise ValueError(f"Need at least {MIN_SAMPLES_REQUIRED} samples")
    
    @contextmanager
    def _error_context(self, operation: str):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            self._log(f"Error in {operation}: {str(e)}", "error")
            raise TransformationError(f"{operation} failed: {str(e)}")


class BaseTransformer(BaseComponent):
    """Abstract base for all transformers."""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]= None) -> 'BaseTransformer':
        """Fit the transformer."""
        self._validate_input(X)
        self._fitted = True
        return self
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._check_fitted()
        self._validate_input(X)
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

class BaseSelector(BaseComponent):
    """Abstract base for feature selection."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_features: List[str] = []
        self._feature_scores: Optional[Dict[str, float]] = None
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseSelector':
        """Fit selector to identify important features."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features based on importance."""
        pass
    
    @abstractmethod
    def _calculate_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Private abstract method to calculate feature importance."""
        pass
    
    @final
    def get_selected_features(self) -> List[str]:
        """Get selected feature names."""
        self._check_fitted()
        return self._selected_features
    
    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self._feature_scores


class BasePipeline(BaseComponent):
    """Abstract base for pipeline implementations."""
    
    def __init__(self, stages: List[BaseComponent], **kwargs):
        super().__init__(**kwargs)
        self.stages = stages
        self._fit_schema: Optional[List[str]] = None  # Store training schema
        self._validate_stages()
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BasePipeline':
        """Fit all pipeline stages."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform through all stages."""
        pass
    
    def _validate_stages(self) -> None:
        """Private method to validate pipeline stages."""
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")
        for stage in self.stages:
            if not isinstance(stage, BaseComponent):
                raise TypeError(f"Invalid stage type: {type(stage)}")
    
    @final
    def _execute_stage(self, stage: BaseComponent, X: pd.DataFrame) -> pd.DataFrame:
        """Private final method to execute a single stage."""
        with self._error_context(f"stage_{stage.__class__.__name__}"):
            self._log(f"Executing {stage.__class__.__name__}")
            return stage.transform(X)
    
    def _store_fit_schema(self, X: pd.DataFrame) -> None:
        """Private method to store training schema."""
        self._fit_schema = list(X.columns)
        self._log(f"Stored training schema with {len(self._fit_schema)} columns")
    
    def _validate_transform_schema(self, X: pd.DataFrame) -> pd.DataFrame:

        if self._fit_schema is None:
            raise NotFittedError("No training schema found. Pipeline must be fitted first.")
        
        # Check for schema differences
        current_columns = set(X.columns)
        expected_columns = set(self._fit_schema)
        
        missing = expected_columns - current_columns
        extra = current_columns - expected_columns
        
        if missing:
            raise ValueError(
                f"Missing columns from training: {missing}\n"
                f"Expected columns: {self._fit_schema}\n"
                f"Received columns: {list(X.columns)}"
            )
        
        if extra:
            self._log(f"Ignoring extra columns not seen during training: {extra}")
        
        # Reorder columns to match training schema
        X_aligned = X[self._fit_schema].copy()
        
        return X_aligned
    
    def get_fit_schema(self) -> Optional[List[str]]:
        """Get the stored training schema."""
        return self._fit_schema.copy() if self._fit_schema else None
