"""Tests for core module: enums, models, exceptions, constants, base classes."""

import importlib

import numpy as np
import pandas as pd
import pytest

from featransform.core.base import BaseComponent, BasePipeline, BaseSelector, BaseTransformer
from featransform.core.constants import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_RANDOM_STATE,
    MIN_SAMPLES_REQUIRED,
)
from featransform.core.enums import (
    EncodingStrategy,
    FeatureType,
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    PipelineStage,
    SelectionStrategy,
    TaskType,
)
from featransform.core.exceptions import (
    ConfigurationError,
    FeatureTransformError,
    NotFittedError,
    PipelineError,
    TransformationError,
    ValidationError,
)
from featransform.core.models import (
    FeatureInfo,
    ModelConfig,
    OptimizationConfig,
    OptimizationHistory,
    PipelineConfig,
    ProcessingConfig,
    TransformationResult,
)


# ── Enums ──────────────────────────────────────────────────────────────


class TestEnums:
    def test_model_family_values(self):
        assert ModelFamily.ISOLATION_FOREST == "isolation_forest"
        assert ModelFamily.PCA == "pca"
        assert ModelFamily.KMEANS == "kmeans"
        assert ModelFamily.CATBOOST == "catboost"

    def test_feature_type_values(self):
        assert set(FeatureType) == {
            FeatureType.ANOMALY,
            FeatureType.CLUSTER,
            FeatureType.DIMENSION,
            FeatureType.ORIGINAL,
        }

    def test_task_type_values(self):
        assert TaskType.REGRESSION.value == "regression"
        assert TaskType.BINARY_CLASSIFICATION.value == "binary_classification"
        assert TaskType.MULTICLASS_CLASSIFICATION.value == "multiclass_classification"

    def test_imputation_strategy_values(self):
        expected = {"mean", "median", "most_frequent", "constant", "iterative", "knn"}
        assert {s.value for s in ImputationStrategy} == expected

    def test_encoding_strategy_has_no_target(self):
        """TARGET was removed because it had no implementation."""
        assert not hasattr(EncodingStrategy, "TARGET")
        assert EncodingStrategy.LABEL.value == "label"

    def test_selection_strategy_values(self):
        assert SelectionStrategy.IMPORTANCE.value == "importance"

    def test_optimization_metric_values(self):
        assert OptimizationMetric.MAE.value == "mae"
        assert OptimizationMetric.F1.value == "f1"

    def test_pipeline_stage_values(self):
        assert PipelineStage.PREPROCESSING.value == "preprocessing"
        assert PipelineStage.FEATURE_SELECTION.value == "feature_selection"


# ── Exceptions ─────────────────────────────────────────────────────────


class TestExceptions:
    def test_base_exception(self):
        err = FeatureTransformError("test error", details={"key": "val"})
        assert str(err) == "test error"
        assert err.details == {"key": "val"}

    def test_not_fitted_error(self):
        err = NotFittedError("MyModel")
        assert "MyModel" in str(err)

    def test_not_fitted_error_accepts_freeform_message(self):
        """NotFittedError must accept arbitrary sentences from call sites."""
        err = NotFittedError("Pipeline not fitted")
        assert str(err) == "Pipeline not fitted"
        # Default message when called with no arg
        default_err = NotFittedError()
        assert "must be fitted" in str(default_err)

    def test_configuration_error(self):
        err = ConfigurationError("bad config")
        assert isinstance(err, FeatureTransformError)

    def test_transformation_error(self):
        err = TransformationError("transform failed")
        assert isinstance(err, FeatureTransformError)

    def test_validation_error(self):
        assert issubclass(ValidationError, FeatureTransformError)

    def test_pipeline_error(self):
        assert issubclass(PipelineError, FeatureTransformError)


# ── Models ─────────────────────────────────────────────────────────────


class TestModels:
    def test_feature_info_creation(self):
        fi = FeatureInfo(name="feat_0", feature_type=FeatureType.ANOMALY)
        assert fi.name == "feat_0"
        assert fi.importance_score == 0.0
        assert fi.source_columns == []

    def test_feature_info_hash(self):
        a = FeatureInfo(name="f1", feature_type=FeatureType.ANOMALY)
        b = FeatureInfo(name="f1", feature_type=FeatureType.CLUSTER)
        assert hash(a) == hash(b)  # hash is based on name only

    def test_transformation_result(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        info = [
            FeatureInfo(name="a", feature_type=FeatureType.ANOMALY),
            FeatureInfo(name="b", feature_type=FeatureType.CLUSTER),
        ]
        result = TransformationResult(features=df, feature_info=info)
        assert result.n_features == 2

    def test_transformation_result_summary_cached(self):
        """Summary uses cached_property (computes once per instance)."""
        df = pd.DataFrame({"a": [1, 2]})
        info = [FeatureInfo(name="a", feature_type=FeatureType.ANOMALY)]
        result = TransformationResult(features=df, feature_info=info)
        first = result.summary
        second = result.summary
        assert first is second  # cached_property returns the same object
        # Backwards-compat method also works
        assert result.get_summary() == first

    def test_model_config_defaults(self):
        mc = ModelConfig(model_family=ModelFamily.KMEANS)
        assert mc.enabled is True
        assert mc.parameters == {}

    def test_model_config_validates_n_clusters(self):
        with pytest.raises(Exception):
            ModelConfig(
                model_family=ModelFamily.KMEANS,
                parameters={"n_clusters": 1},
            )

    def test_processing_config_defaults(self):
        pc = ProcessingConfig()
        assert pc.imputation_strategy == ImputationStrategy.ITERATIVE
        assert pc.encoding_strategy == EncodingStrategy.LABEL
        assert pc.drop_constant is True

    def test_optimization_config_defaults(self):
        oc = OptimizationConfig()
        assert oc.n_iterations == 10
        assert oc.validation_split == 0.25
        assert oc.use_catboost is True
        assert oc.use_xgboost is True
        assert oc.min_features == 10

    def test_optimization_config_rejects_invalid_iterations(self):
        """Pydantic now enforces n_iterations bounds."""
        with pytest.raises(Exception):
            OptimizationConfig(n_iterations=0)
        with pytest.raises(Exception):
            OptimizationConfig(n_iterations=999)

    def test_optimization_config_rejects_invalid_split(self):
        """Pydantic now enforces validation_split bounds."""
        with pytest.raises(Exception):
            OptimizationConfig(validation_split=0.001)
        with pytest.raises(Exception):
            OptimizationConfig(validation_split=0.9)

    def test_optimization_config_rejects_no_calculator(self):
        """At least one of catboost/xgboost must be enabled."""
        with pytest.raises(Exception):
            OptimizationConfig(use_catboost=False, use_xgboost=False)

    def test_optimization_config_serializes(self):
        """Pydantic conversion makes JSON round-trip safe."""
        oc = OptimizationConfig(n_iterations=5, min_features=8)
        dumped = oc.model_dump()
        assert dumped["n_iterations"] == 5
        assert dumped["min_features"] == 8
        rebuilt = OptimizationConfig(**dumped)
        assert rebuilt.n_iterations == 5

    def test_pipeline_config_defaults(self):
        pc = PipelineConfig()
        assert pc.random_state == DEFAULT_RANDOM_STATE
        assert pc.verbose is False
        assert pc.anomaly_models == []
        assert pc.optimization is None

    def test_pipeline_config_with_models(self):
        pc = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        assert len(pc.anomaly_models) == 1
        assert len(pc.clustering_models) == 1

    def test_optimization_history(self):
        oh = OptimizationHistory(
            iteration=0, score=0.85, n_features=10, selected_features=["a", "b"]
        )
        assert oh.iteration == 0
        assert oh.score == 0.85
        assert oh.timestamp is not None


# ── Constants ──────────────────────────────────────────────────────────


class TestConstants:
    def test_defaults_exist(self):
        assert DEFAULT_RANDOM_STATE == 42
        assert DEFAULT_N_CLUSTERS == 5
        assert DEFAULT_N_COMPONENTS == 4
        assert MIN_SAMPLES_REQUIRED == 5

    def test_default_contamination_is_meaningful(self):
        """Contamination must be high enough to flag actual anomalies."""
        from featransform.core.constants import DEFAULT_CONTAMINATION

        assert DEFAULT_CONTAMINATION >= 0.01


# ── Public API surface ─────────────────────────────────────────────────


class TestPublicAPI:
    def test_all_advertised_names_are_importable(self):
        """Every name in featransform.__all__ must be importable from the package."""
        pkg = importlib.import_module("featransform")
        for name in pkg.__all__:
            assert hasattr(pkg, name), f"featransform.{name} listed in __all__ but not importable"


# ── Base Classes ───────────────────────────────────────────────────────


class TestBaseComponent:
    def test_validate_input_rejects_non_dataframe(self):
        class Dummy(BaseComponent):
            def get_params(self):
                return {}

            def set_params(self, **p):
                return self

        d = Dummy()
        with pytest.raises(TypeError, match="Expected DataFrame"):
            d._validate_input([1, 2, 3])

    def test_validate_input_rejects_empty(self):
        class Dummy(BaseComponent):
            def get_params(self):
                return {}

            def set_params(self, **p):
                return self

        d = Dummy()
        with pytest.raises(ValueError, match="empty"):
            d._validate_input(pd.DataFrame())

    def test_validate_fit_input_rejects_too_few_samples(self):
        """Only fit-side validation enforces MIN_SAMPLES_REQUIRED."""

        class Dummy(BaseComponent):
            def get_params(self):
                return {}

            def set_params(self, **p):
                return self

        d = Dummy()
        df = pd.DataFrame({"a": [1, 2, 3]})  # < MIN_SAMPLES_REQUIRED
        with pytest.raises(ValueError, match="at least"):
            d._validate_fit_input(df)
        # Lightweight _validate_input must NOT enforce the floor (transform path)
        d._validate_input(df)  # no exception

    def test_check_fitted_raises(self):
        class Dummy(BaseComponent):
            def get_params(self):
                return {}

            def set_params(self, **p):
                return self

        d = Dummy()
        with pytest.raises(NotFittedError):
            d._check_fitted()
