from abc import abstractmethod
from typing import Optional, Dict, List, final

import logging
import pandas as pd

from featransform.core.base import BaseComponent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

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
    
class BaseOptimizer(BaseComponent):
    """Abstract base for hyperparameter optimization strategies."""
    
    def __init__(self, metric: str, n_iterations: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.metric = metric
        self.n_iterations = n_iterations
        self._best_params: Optional[Dict] = None
        self._best_score: Optional[float] = None
        self._optimization_history: List[Dict] = []
    
    @abstractmethod
    def optimize(self, X: pd.DataFrame, y: pd.Series, param_space: Dict) -> Dict:
        """Optimize hyperparameters."""
        pass
    
    @abstractmethod
    def _evaluate(self, X: pd.DataFrame, y: pd.Series, params: Dict) -> float:
        """Evaluate a parameter configuration."""
        pass
    
    def get_optimization_history(self) -> List[Dict]:
        """Get optimization history."""
        return self._optimization_history
    
    def get_best_params(self) -> Optional[Dict]:
        """Get best parameters found."""
        return self._best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved."""
        return self._best_score