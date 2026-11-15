from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, Birch, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from featransform.core.constants import (
    DBSCAN_AUTO_EPS_MAX_SAMPLES,
    DBSCAN_AUTO_EPS_QUANTILE,
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_JOBS,
    DEFAULT_RANDOM_STATE,
)
from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.exceptions import ConfigurationError
from featransform.core.models import FeatureInfo, TransformationResult
from featransform.features.base import BaseFeatureEngineer, BaseFeatureEnsemble

_NOISE = -1


class _DBSCANPredictor:
    """DBSCAN with a deterministic, fit-once ``predict``.

    DBSCAN has no native ``predict``. A new point is assigned the way DBSCAN
    itself treats a border point: it joins the cluster of its nearest *core*
    sample when that sample lies within ``eps``, and is noise (``-1``)
    otherwise — so a point far from every cluster is not forced into one.

    ``eps="auto"`` resolves the radius from the data with the k-distance
    heuristic: the ``DBSCAN_AUTO_EPS_QUANTILE`` quantile of each point's
    distance to its ``min_samples``-th nearest neighbour. A fixed radius is
    meaningless across feature spaces of different dimensionality, where the
    typical neighbour distance grows with the number of columns.
    """

    # Class-level defaults keep pipelines pickled before 2.4.0 loadable: those
    # carry ``_nn`` / ``_labels`` over every training row instead of a core index.
    _core_nn: Optional[NearestNeighbors] = None
    _core_labels: Optional[np.ndarray] = None
    _nn: Optional[NearestNeighbors] = None
    _labels: Optional[np.ndarray] = None
    eps_: Optional[float] = None

    def __init__(
        self,
        eps: Union[float, str],
        min_samples: int,
        n_jobs: int = DEFAULT_N_JOBS,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.eps_ = None
        self._dbscan: Optional[DBSCAN] = None
        self._core_nn = None
        self._core_labels = None

    def _resolve_eps(self, values: np.ndarray) -> float:
        if not isinstance(self.eps, str):
            return float(self.eps)
        if len(values) > DBSCAN_AUTO_EPS_MAX_SAMPLES:
            rng = np.random.RandomState(self.random_state)
            values = values[rng.choice(len(values), DBSCAN_AUTO_EPS_MAX_SAMPLES, replace=False)]
        # +1 neighbour: the query set is the fitted set, so each point is its own
        # nearest neighbour at distance 0.
        k = int(min(self.min_samples + 1, len(values)))
        distances, _ = (
            NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs).fit(values).kneighbors(values)
        )
        eps = float(np.quantile(distances[:, -1], DBSCAN_AUTO_EPS_QUANTILE))
        # Identical rows give a zero k-distance; DBSCAN requires eps > 0.
        return eps if eps > 0 else float(np.finfo(float).eps)

    def fit(self, X: pd.DataFrame) -> "_DBSCANPredictor":
        values = np.asarray(X, dtype=float)
        self.eps_ = self._resolve_eps(values)
        self._dbscan = DBSCAN(
            eps=self.eps_, min_samples=self.min_samples, metric="euclidean", n_jobs=self.n_jobs
        )
        labels = self._dbscan.fit_predict(values)
        core_idx = self._dbscan.core_sample_indices_
        if len(core_idx):
            self._core_nn = NearestNeighbors(n_neighbors=1, n_jobs=self.n_jobs).fit(
                values[core_idx]
            )
            self._core_labels = labels[core_idx]
        else:  # every training row is noise
            self._core_nn = None
            self._core_labels = np.empty(0, dtype=labels.dtype)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        if self._core_labels is None:
            if self._nn is not None and self._labels is not None:
                # Pipeline pickled before 2.4.0: nearest training row's label.
                idx = self._nn.kneighbors(values, return_distance=False)[:, 0]
                return self._labels[idx]
            raise RuntimeError("DBSCAN predictor must be fit before predict")
        if self._core_nn is None:
            return np.full(len(values), _NOISE, dtype=int)
        distances, idx = self._core_nn.kneighbors(values)
        labels = self._core_labels[idx[:, 0]]
        return np.where(distances[:, 0] <= self.eps_, labels, _NOISE)


class ClusteringStrategy(BaseFeatureEngineer):
    """Base strategy for clustering."""

    def __init__(self, n_clusters: int = DEFAULT_N_CLUSTERS, **kwargs):
        super().__init__(feature_type=FeatureType.CLUSTER, **kwargs)
        self.n_clusters = n_clusters

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClusteringStrategy":
        self._validate_fit_input(X)

        with self._error_context("model_building"):
            self._models = self._build_models()

        for name, model in self._models.items():
            with self._error_context(f"fitting_{name}"):
                self._log(f"Fitting {name}")
                model.fit(X)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> TransformationResult:
        self._check_fitted()
        self._validate_input(X)

        features = self._create_features(X)
        # Names reflect the columns actually emitted.
        self._feature_names = list(features.columns)

        feature_info = [
            FeatureInfo(
                name=col,
                feature_type=self.feature_type,
                model_family=self.model_family,
                source_columns=list(X.columns),
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
            # Every registered model exposes ``predict`` (DBSCAN via
            # ``_DBSCANPredictor``): transform never refits.
            predictions = model.predict(X)

            prefix = name.split("_")[0].lower()[:3]
            results.append(pd.DataFrame({f"{prefix}_cluster": predictions}, index=X.index))

        return pd.concat(results, axis=1)

    def _build_models(self) -> Dict[str, Any]:
        """Build clustering models."""
        return {}

    def get_params(self) -> Dict[str, Any]:
        return {"n_clusters": self.n_clusters, **self._params}

    def set_params(self, **params) -> "ClusteringStrategy":
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
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
                n_init="auto",
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
                branching_factor=self.branching_factor,
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
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class DBSCANStrategy(ClusteringStrategy):
    """DBSCAN clustering strategy with a deterministic, fit-once predict.

    ``eps="auto"`` (the default) derives the neighbourhood radius from the data;
    pass a number to pin it. See :class:`_DBSCANPredictor`.
    """

    def __init__(self, eps: Union[float, str] = "auto", min_samples: int = 5, **kwargs):
        # DBSCAN doesn't use n_clusters
        super().__init__(n_clusters=None, **kwargs)
        if isinstance(eps, str):
            if eps != "auto":
                raise ConfigurationError(f"eps must be a positive number or 'auto'; got {eps!r}")
        elif isinstance(eps, bool) or not eps > 0:
            raise ConfigurationError(f"eps must be a positive number or 'auto'; got {eps!r}")
        self.eps = eps
        self.min_samples = min_samples
        self.model_family = ModelFamily.DBSCAN

    def _build_models(self) -> Dict[str, Any]:
        return {
            "dbscan": _DBSCANPredictor(
                eps=self.eps,
                min_samples=self.min_samples,
                n_jobs=self._params.get("n_jobs", DEFAULT_N_JOBS),
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class GaussianMixtureStrategy(ClusteringStrategy):
    """Gaussian Mixture Model clustering strategy."""

    def __init__(self, n_components: Optional[int] = None, covariance_type: str = "full", **kwargs):
        # ``n_components`` is sklearn's native name for the cluster count; accept
        # it as an alias for the strategy-level ``n_clusters`` so the passed value
        # is honored instead of silently falling back to DEFAULT_N_CLUSTERS.
        if n_components is not None:
            kwargs["n_clusters"] = n_components
        super().__init__(**kwargs)
        self.covariance_type = covariance_type
        self.model_family = ModelFamily.GAUSSIAN_MIXTURE

    def _build_models(self) -> Dict[str, Any]:
        return {
            "gmm": GaussianMixture(
                n_components=self.n_clusters,
                covariance_type=self.covariance_type,
                random_state=self._params.get("random_state", DEFAULT_RANDOM_STATE),
            )
        }


class ClusteringEnsemble(BaseFeatureEnsemble):
    """Ensemble of clustering strategies."""

    def __init__(self, strategies: Optional[List[ClusteringStrategy]] = None, **kwargs):
        super().__init__(feature_type=FeatureType.CLUSTER, **kwargs)
        self.strategies = strategies or self._default_strategies()

    def _default_strategies(self) -> List[ClusteringStrategy]:
        return [KMeansStrategy(n_clusters=self._params.get("n_clusters", DEFAULT_N_CLUSTERS))]
