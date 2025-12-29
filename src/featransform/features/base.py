from abc import abstractmethod
from typing import Optional, Dict, List, final, Any

import logging
import pandas as pd

from featransform.core.base import BaseComponent
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.models import TransformationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class BaseFeatureEngineer(BaseComponent):
    """Abstract base for feature engineering strategies."""
    
    def __init__(self, feature_type: FeatureType, model_family: Optional[ModelFamily] = None, **kwargs):
        super().__init__(**kwargs)
        self.feature_type = feature_type
        self.model_family = model_family
        self._feature_names: List[str] = []
        self._models: Dict[str, Any] = {}
    
    @abstractmethod
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseFeatureEngineer':
        """Fit the feature engineering model."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> TransformationResult:
        """Generate engineered features."""
        pass
    
    @abstractmethod
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Private abstract method to create features."""
        pass
    
    def _generate_feature_names(self, n_features: int, prefix: str) -> List[str]:
        """Private method to generate feature names."""
        return [f"{prefix}{i}" for i in range(n_features)]
    
    @final
    def get_feature_names(self) -> List[str]:
        """Get generated feature names."""
        return self._feature_names
