from typing import Dict, Any, Optional, List
import pandas as pd
from sklearn.cluster import KMeans, Birch, MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture

from featransform.features.base import BaseFeatureEngineer
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.models import TransformationResult, FeatureInfo
from featransform.core.constants import (
    DEFAULT_N_CLUSTERS, CLUSTER_PREFIX,
    DEFAULT_RANDOM_STATE
)


class ClusteringStrategy(BaseFeatureEngineer):
    """Base strategy for clustering."""
    
    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS, **kwargs):
        super().__init__(feature_type=FeatureType.CLUSTER, **kwargs)
        self.n_clusters = n_clusters
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ClusteringStrategy':
        self._validate_input(X)
        
        with self._error_context("model_building"):
            self._models = self._build_models()
        
        for name, model in self._models.items():
            with self._error_context(f"fitting_{name}"):
                self._log(f"Fitting {name}")
                model.fit(X)
        
        self._fitted = True
        self._feature_names = self._generate_feature_names(
            len(self._models),
            CLUSTER_PREFIX
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
                source_columns=list(X.columns)
            )
            for col in features.columns
        ]
        
        return TransformationResult(
            features=features,
            feature_info=feature_info,
        )
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create cluster features from fitted models."""
        results = []
        
        for name, model in self._models.items():
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                # For models like DBSCAN that don't have predict
                predictions = model.fit_predict(X)
            
            prefix = name.split('_')[0].lower()[:3]
            results.append(pd.DataFrame({
                f"{prefix}_cluster": predictions
            }, index=X.index))
        
        return pd.concat(results, axis=1)
    
    def _build_models(self) -> Dict[str, Any]:
        """Build clustering models."""
        return {}
    
    def get_params(self) -> Dict[str, Any]:
        return {"n_clusters": self.n_clusters, **self._params}
    
    def set_params(self, **params) -> 'ClusteringStrategy':
        for key, value in params.items():
            setattr(self, key, value)
        return self


class KMeansStrategy(ClusteringStrategy):
    """KMeans clustering strategy."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_family = ModelFamily.KMEANS
    
    def _build_models(self) -> Dict[str, Any]:
        return {
            "kmeans": KMeans(
                n_clusters=self.n_clusters,
                random_state=self._params.get('random_state', DEFAULT_RANDOM_STATE),
                n_init='auto'
            )
        }


class BirchStrategy(ClusteringStrategy):
    """Birch clustering strategy."""
    
    def __init__(self, threshold: float = 0.5, branching_factor: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.model_family = ModelFamily.BIRCH
    
    def _build_models(self) -> Dict[str, Any]:
        return {
            "birch": Birch(
                n_clusters=self.n_clusters,
                threshold=self.threshold,
                branching_factor=self.branching_factor
            )
        }


class MiniBatchKMeansStrategy(ClusteringStrategy):
    """MiniBatchKMeans clustering strategy."""
    
    def __init__(self, batch_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.model_family = ModelFamily.MINI_BATCH_KMEANS
    
    def _build_models(self) -> Dict[str, Any]:
        return {
            "mini_batch_kmeans": MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                random_state=self._params.get('random_state', DEFAULT_RANDOM_STATE)
            )
        }


class DBSCANStrategy(ClusteringStrategy):
    """DBSCAN clustering strategy."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        # DBSCAN doesn't use n_clusters
        super().__init__(n_clusters=None, **kwargs)
        self.eps = eps
        self.min_samples = min_samples
        self.model_family = ModelFamily.DBSCAN
    
    def _build_models(self) -> Dict[str, Any]:
        return {
            "dbscan": DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                metric='euclidean',
                n_jobs=self._params.get('n_jobs', -1)
            )
        }


class GaussianMixtureStrategy(ClusteringStrategy):
    """Gaussian Mixture Model clustering strategy."""
    
    def __init__(self, covariance_type: str = 'full', **kwargs):
        super().__init__(**kwargs)
        self.covariance_type = covariance_type
        self.model_family = ModelFamily.GAUSSIAN_MIXTURE
    
    def _build_models(self) -> Dict[str, Any]:
        return {
            "gmm": GaussianMixture(
                n_components=self.n_clusters,
                covariance_type=self.covariance_type,
                random_state=self._params.get('random_state', DEFAULT_RANDOM_STATE)
            )
        }


class ClusteringEnsemble(BaseFeatureEngineer):
    """Ensemble of clustering strategies."""
    
    def __init__(self, strategies: Optional[List[ClusteringStrategy]] = None, **kwargs):
        super().__init__(feature_type=FeatureType.CLUSTER, **kwargs)
        self.strategies = strategies or self._default_strategies()
    
    def _default_strategies(self) -> List[ClusteringStrategy]:
        return [KMeansStrategy(n_clusters=self._params.get('n_clusters', DEFAULT_N_CLUSTERS))]
    
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series]=None) -> 'ClusteringEnsemble':
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
    
    def set_params(self, **params) -> 'ClusteringEnsemble':
        for key, value in params.items():
            setattr(self, key, value)
        return self

