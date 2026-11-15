"""Tests for feature engineering: anomaly, clustering, dimensionality."""

import numpy as np
import pandas as pd
import pytest

from featransform.core.enums import FeatureType, ModelFamily
from featransform.core.exceptions import NotFittedError
from featransform.core.models import TransformationResult
from featransform.features.anomaly import (
    AnomalyEnsemble,
    AnomalyStrategy,
    EllipticEnvelopeStrategy,
    IsolationForestStrategy,
    LocalOutlierFactorStrategy,
    OneClassSVMStrategy,
)
from featransform.features.clustering import (
    BirchStrategy,
    ClusteringEnsemble,
    DBSCANStrategy,
    GaussianMixtureStrategy,
    KMeansStrategy,
    MiniBatchKMeansStrategy,
    _DBSCANPredictor,
)
from featransform.features.dimensionality import (
    DimensionalityEnsemble,
    FastICAStrategy,
    PCAStrategy,
    TruncatedSVDStrategy,
)


# ── Anomaly Detection ──────────────────────────────────────────────────


class TestAnomalyStrategies:
    @pytest.mark.parametrize(
        "strategy_cls",
        [
            IsolationForestStrategy,
            LocalOutlierFactorStrategy,
            OneClassSVMStrategy,
            EllipticEnvelopeStrategy,
        ],
    )
    def test_fit_transform(self, numeric_df, strategy_cls):
        strategy = strategy_cls(random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        assert len(result.features) == len(numeric_df)
        assert result.n_features >= 1
        for info in result.feature_info:
            assert info.feature_type == FeatureType.ANOMALY

    def test_isolation_forest_with_scores(self, numeric_df):
        strategy = IsolationForestStrategy(include_scores=True, random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        # Should have both anomaly flag and score columns
        assert result.n_features == 2

    def test_transform_before_fit_raises(self, numeric_df):
        strategy = IsolationForestStrategy()
        with pytest.raises(NotFittedError):
            strategy.transform(numeric_df)

    def test_align_features_drops_extra_columns(self, numeric_df):
        strategy = IsolationForestStrategy(random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)

        df_extra = numeric_df.copy()
        df_extra["extra_col"] = 999
        result = strategy.transform(df_extra)
        assert isinstance(result, TransformationResult)

    def test_align_features_raises_on_missing(self, numeric_df):
        strategy = IsolationForestStrategy(random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)

        df_missing = numeric_df.drop(columns=["feat_0"])
        with pytest.raises(ValueError, match="Missing features"):
            strategy.transform(df_missing)

    def test_get_params(self):
        strategy = IsolationForestStrategy(contamination=0.01, random_state=42)
        params = strategy.get_params()
        assert params["contamination"] == 0.01


class TestAnomalyEnsemble:
    def test_ensemble_fit_transform(self, numeric_df):
        strategies = [
            IsolationForestStrategy(random_state=42),
            EllipticEnvelopeStrategy(random_state=42),
        ]
        ensemble = AnomalyEnsemble(strategies=strategies)
        y = pd.Series(np.zeros(len(numeric_df)))
        ensemble.fit(numeric_df, y)
        result = ensemble.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        # 2 strategies, each producing at least 1 feature
        assert result.n_features >= 2

    def test_ensemble_transform_before_fit_raises(self, numeric_df):
        strategies = [IsolationForestStrategy(random_state=42)]
        ensemble = AnomalyEnsemble(strategies=strategies)
        with pytest.raises(NotFittedError):
            ensemble.transform(numeric_df)

    def test_ensemble_default_construction(self, numeric_df):
        """AnomalyEnsemble() with no args must not raise AttributeError."""
        ensemble = AnomalyEnsemble()
        assert ensemble.include_scores is False
        assert len(ensemble.strategies) >= 1
        y = pd.Series(np.zeros(len(numeric_df)))
        ensemble.fit(numeric_df, y)
        result = ensemble.transform(numeric_df)
        assert isinstance(result, TransformationResult)

    def test_ensemble_default_construction_with_scores(self, numeric_df):
        """include_scores forwards into default strategies."""
        ensemble = AnomalyEnsemble(include_scores=True)
        assert ensemble.include_scores is True


# ── Clustering ─────────────────────────────────────────────────────────


class TestClusteringStrategies:
    @pytest.mark.parametrize(
        "strategy_cls",
        [
            KMeansStrategy,
            BirchStrategy,
            MiniBatchKMeansStrategy,
            GaussianMixtureStrategy,
        ],
    )
    def test_fit_transform(self, numeric_df, strategy_cls):
        strategy = strategy_cls(n_clusters=3, random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        assert len(result.features) == len(numeric_df)
        for info in result.feature_info:
            assert info.feature_type == FeatureType.CLUSTER

    def test_dbscan_fit_transform(self, numeric_df):
        strategy = DBSCANStrategy(eps=1.0, min_samples=5)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        assert isinstance(result, TransformationResult)

    def test_dbscan_transform_is_deterministic(self, numeric_df):
        """DBSCAN.transform must NOT refit; two calls produce identical labels."""
        strategy = DBSCANStrategy(eps=1.0, min_samples=5)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        a = strategy.transform(numeric_df).features
        b = strategy.transform(numeric_df).features
        pd.testing.assert_frame_equal(a, b)

    def test_dbscan_transform_on_unseen_data(self, numeric_df):
        """DBSCAN can label unseen rows via the nearest-neighbor predictor."""
        strategy = DBSCANStrategy(eps=1.5, min_samples=5)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        # Take rows that were not in fit data
        new_rows = numeric_df.iloc[:10] + 0.01  # slightly perturbed
        result = strategy.transform(new_rows)
        assert isinstance(result, TransformationResult)
        assert len(result.features) == 10

    def test_transform_before_fit_raises(self, numeric_df):
        strategy = KMeansStrategy()
        with pytest.raises(NotFittedError):
            strategy.transform(numeric_df)

    def test_cluster_values_are_integers(self, numeric_df):
        strategy = KMeansStrategy(n_clusters=3, random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        for col in result.features.columns:
            assert result.features[col].dtype in [np.int32, np.int64, int]


class TestClusteringEnsemble:
    def test_ensemble_fit_transform(self, numeric_df):
        strategies = [
            KMeansStrategy(n_clusters=3, random_state=42),
            BirchStrategy(n_clusters=3),
        ]
        ensemble = ClusteringEnsemble(strategies=strategies)
        y = pd.Series(np.zeros(len(numeric_df)))
        ensemble.fit(numeric_df, y)
        result = ensemble.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        assert result.n_features >= 2


# ── Dimensionality Reduction ──────────────────────────────────────────


class TestDimensionalityStrategies:
    @pytest.mark.parametrize(
        "strategy_cls",
        [PCAStrategy, TruncatedSVDStrategy, FastICAStrategy],
    )
    def test_fit_transform(self, numeric_df, strategy_cls):
        strategy = strategy_cls(n_components=2, random_state=42)
        y = pd.Series(np.zeros(len(numeric_df)))
        strategy.fit(numeric_df, y)
        result = strategy.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        assert len(result.features) == len(numeric_df)
        for info in result.feature_info:
            assert info.feature_type == FeatureType.DIMENSION

    def test_n_components_capped(self):
        """n_components > min(n_samples, n_features)-1 should fall back."""
        df = pd.DataFrame(np.random.randn(20, 8), columns=[f"c{i}" for i in range(8)])
        strategy = PCAStrategy(n_components=100, random_state=42)
        y = pd.Series(np.zeros(20))
        # Should not raise, falls back to DEFAULT_N_COMPONENTS=4
        strategy.fit(df, y)
        result = strategy.transform(df)
        assert isinstance(result, TransformationResult)

    def test_n_components_fallback_clamped_to_feature_budget(self):
        """When n_components exceeds the data budget the fallback must clamp
        to it; DEFAULT_N_COMPONENTS=4 would otherwise crash on a 3-column frame."""
        df = pd.DataFrame(np.random.randn(50, 3), columns=list("abc"))
        strategy = PCAStrategy(n_components=10, random_state=42)
        strategy.fit(df, pd.Series(np.zeros(50)))  # must not raise
        result = strategy.transform(df)
        # max_components = min(50, 3) - 1 = 2
        assert result.n_features <= 2
        assert result.n_features >= 1

    def test_transform_before_fit_raises(self, numeric_df):
        strategy = PCAStrategy(n_components=2)
        with pytest.raises(NotFittedError):
            strategy.transform(numeric_df)


class TestDimensionalityEnsemble:
    def test_ensemble_fit_transform(self, numeric_df):
        strategies = [
            PCAStrategy(n_components=2, random_state=42),
            TruncatedSVDStrategy(n_components=2, random_state=42),
        ]
        ensemble = DimensionalityEnsemble(strategies=strategies)
        y = pd.Series(np.zeros(len(numeric_df)))
        ensemble.fit(numeric_df, y)
        result = ensemble.transform(numeric_df)
        assert isinstance(result, TransformationResult)
        assert result.n_features >= 4  # 2 strategies x 2 components each

    def test_same_family_strategies_get_unique_columns(self, numeric_df):
        """Two strategies of the same family must not emit colliding column
        names; duplicates are suffixed so the schema stays well-defined."""
        strategies = [
            PCAStrategy(n_components=2, random_state=42),
            PCAStrategy(n_components=2, random_state=0),
        ]
        ensemble = DimensionalityEnsemble(strategies=strategies)
        ensemble.fit(numeric_df, pd.Series(np.zeros(len(numeric_df))))
        result = ensemble.transform(numeric_df)
        assert result.features.columns.is_unique
        assert result.n_features == 4
