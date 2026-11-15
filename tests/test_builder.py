"""Tests for the BuilderPipeline and component registration."""

import pytest

from featransform.builder import BuilderPipeline
from featransform.core.enums import ModelFamily
from featransform.core.models import ModelConfig, PipelineConfig
from featransform.features.anomaly import IsolationForestStrategy
from featransform.features.clustering import KMeansStrategy
from featransform.features.dimensionality import PCAStrategy
from featransform.pipeline import Featransform


class TestBuilderPipeline:
    def test_components_property_returns_copy(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        components = pipe.components
        assert isinstance(components, dict)
        # Modifying the copy should not affect the pipeline
        components["fake"] = "test"
        assert not pipe.has_component("fake")

    def test_get_component(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        anomaly = pipe.get_component("anomaly")
        assert anomaly is not None
        assert pipe.get_component("nonexistent") is None

    def test_has_component(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        assert pipe.has_component("imputer")
        assert pipe.has_component("encoder")
        assert pipe.has_component("anomaly")
        assert not pipe.has_component("clustering")

    def test_all_anomaly_strategies(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST),
                ModelConfig(model_family=ModelFamily.LOCAL_OUTLIER_FACTOR),
                ModelConfig(model_family=ModelFamily.ONE_CLASS_SVM),
                ModelConfig(model_family=ModelFamily.ELLIPTIC_ENVELOPE),
            ],
        )
        pipe = Featransform(config)
        anomaly = pipe.get_component("anomaly")
        assert len(anomaly.strategies) == 4

    def test_all_clustering_strategies(self):
        config = PipelineConfig(
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS),
                ModelConfig(model_family=ModelFamily.BIRCH),
                ModelConfig(model_family=ModelFamily.MINI_BATCH_KMEANS),
                ModelConfig(model_family=ModelFamily.GAUSSIAN_MIXTURE),
                ModelConfig(model_family=ModelFamily.DBSCAN),
            ],
        )
        pipe = Featransform(config)
        clustering = pipe.get_component("clustering")
        assert len(clustering.strategies) == 5

    def test_all_dimensionality_strategies(self):
        config = PipelineConfig(
            dimensionality_models=[
                ModelConfig(model_family=ModelFamily.PCA),
                ModelConfig(model_family=ModelFamily.TRUNCATED_SVD),
                ModelConfig(model_family=ModelFamily.FAST_ICA),
            ],
        )
        pipe = Featransform(config)
        dim = pipe.get_component("dimensionality")
        assert len(dim.strategies) == 3

    def test_disabled_models_excluded(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST, enabled=True),
                ModelConfig(model_family=ModelFamily.ONE_CLASS_SVM, enabled=False),
            ],
        )
        pipe = Featransform(config)
        anomaly = pipe.get_component("anomaly")
        assert len(anomaly.strategies) == 1
