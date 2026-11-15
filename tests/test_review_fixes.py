"""Regression tests for the 2.x review fixes (A1-A4, B1, B2, C1, C2, C4, D1).

Each test documents the concrete defect it guards against so the intent survives
future refactors.
"""

import json

import numpy as np
import pandas as pd
import pytest

from featransform.configs.baseline import FTconfig
from featransform.core.constants import DEFAULT_N_CLUSTERS
from featransform.core.enums import ModelFamily, ScalingStrategy, TaskType
from featransform.core.exceptions import ConfigurationError
from featransform.core.models import ModelConfig, PipelineConfig, ProcessingConfig
from featransform.features.anomaly import AnomalyEnsemble
from featransform.features.clustering import ClusteringEnsemble, GaussianMixtureStrategy
from featransform.features.dimensionality import DimensionalityEnsemble
from featransform.optimization.evaluator import XGBoostImportance
from featransform.optimization.selector import FeatureSelector
from featransform.pipeline import Featransform
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.scaling import FeatureScaler
from featransform.utils.data_generator import DatasetGenerator, make_dataset
from featransform.utils.serializer import PipelineSerializer


def _dataset_with_id(n=200, seed=0, monotonic_id=True):
    """Feature frame with an id column and off-standard-scale features."""
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    if not monotonic_id:
        rng.shuffle(ids)
    X = pd.DataFrame(
        {
            "id": ids,
            "f0": rng.normal(100.0, 5.0, n),
            "f1": rng.normal(-50.0, 2.0, n),
            "f2": rng.normal(0.0, 1000.0, n),
        }
    )
    y = pd.Series(rng.integers(0, 2, n), name="target")
    return X, y


class TestTaskAliases:
    @pytest.mark.parametrize(
        "alias,expected_classes",
        [("binary", 2), ("multiclass", 3), ("regression", None)],
    )
    def test_make_dataset_accepts_short_aliases(self, alias, expected_classes):
        kwargs = {"n_classes": 3} if alias == "multiclass" else {}
        X, y = make_dataset(task=alias, n_samples=80, complexity="simple", **kwargs)
        assert len(X) == 80
        if expected_classes is not None:
            assert y.nunique() == expected_classes

    def test_convenience_helpers_do_not_raise(self):
        # Previously quick_binary/quick_multiclass/complex_dataset passed short
        # aliases straight into generate() and raised ValueError.
        DatasetGenerator.quick_binary(n_samples=60)
        DatasetGenerator.quick_multiclass(n_samples=60, n_classes=3)
        DatasetGenerator.complex_dataset(n_samples=60)

    def test_make_dataset_default_task(self):
        X, y = make_dataset(n_samples=60, complexity="simple")  # default task="binary"
        assert y.nunique() == 2

    def test_genuinely_unknown_task_still_raises(self):
        with pytest.raises(ValueError):
            DatasetGenerator.generate(task="not_a_task", n_samples=30)


class TestDatetimeGeneration:
    def test_medium_complexity_emits_datetime(self):
        X, _ = make_dataset(task="binary", n_samples=60, complexity="medium")
        assert X.select_dtypes(include=["datetime64"]).shape[1] >= 1

    def test_complex_dataset_emits_datetime(self):
        X, _ = DatasetGenerator.complex_dataset(n_samples=60, task="binary_classification")
        assert X.select_dtypes(include=["datetime64"]).shape[1] >= 1


class TestGaussianMixtureComponents:
    def test_n_components_alias_is_honored(self):
        strat = GaussianMixtureStrategy(n_components=3, random_state=0)
        strat.fit(pd.DataFrame(np.random.rand(50, 4), columns=list("abcd")), None)
        assert strat.n_clusters == 3
        assert strat._models["gmm"].n_components == 3

    def test_default_falls_back_to_default_n_clusters(self):
        strat = GaussianMixtureStrategy(random_state=0)
        assert strat.n_clusters == DEFAULT_N_CLUSTERS

    def test_n_components_through_builder(self):
        config = PipelineConfig(
            clustering_models=[
                ModelConfig(
                    model_family=ModelFamily.GAUSSIAN_MIXTURE,
                    parameters={"n_components": 3},
                )
            ]
        )
        pipe = Featransform(config)
        assert pipe._components["clustering"].strategies[0].n_clusters == 3


class TestNonContiguousMulticlass:
    @pytest.mark.parametrize("labels", [[1, 2, 3], [10, 20, 30]])
    def test_selector_evaluates_numeric_non_contiguous(self, labels):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.random((90, 4)), columns=list("abcd"))
        y = pd.Series(rng.choice(labels, 90))
        sel = FeatureSelector(task_type=TaskType.MULTICLASS_CLASSIFICATION, min_features=2)
        sel.fit(X, y)
        score = sel.evaluate_subset(X.iloc[:70], y.iloc[:70], X.iloc[70:], y.iloc[70:])
        assert np.isfinite(score)

    def test_xgboost_importance_numeric_non_contiguous(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.random((80, 4)), columns=list("abcd"))
        y = pd.Series(rng.choice([10, 20, 30], 80))
        out = XGBoostImportance(n_estimators=10).calculate(
            X, y, TaskType.MULTICLASS_CLASSIFICATION
        )
        assert set(out["Feature Id"]) == set("abcd")


class TestFeatureScaling:
    def test_default_processing_enables_standard_scaling(self):
        assert ProcessingConfig().scaling_strategy == ScalingStrategy.STANDARD

    def test_feature_scaler_standardizes(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {"a": rng.normal(100, 5, 200), "b": rng.normal(-3, 2, 200)}
        )
        scaler = FeatureScaler(strategy=ScalingStrategy.STANDARD).fit(X)
        out = scaler.transform(X)
        assert np.allclose(out.mean().values, 0, atol=1e-6)
        assert np.allclose(out.std(ddof=0).values, 1, atol=1e-6)

    def test_passthrough_columns_remain_unscaled(self):
        X, y = _dataset_with_id(n=200)
        config = PipelineConfig(
            task_type="binary_classification",
            processing=ProcessingConfig(
                scaling_strategy=ScalingStrategy.STANDARD,
                imputation_strategy="mean",
                id_columns=["id"],
                handle_datetime=False,
                drop_constant=False,
                drop_duplicates=False,
                auto_detect_id=False,
            ),
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})
            ],
        )
        pipe = Featransform(config).fit(X, y)
        out = pipe.transform(X)
        # Original feature columns pass through in their native scale.
        assert np.allclose(out["f0"].values, X["f0"].values)
        assert np.allclose(out["f2"].values, X["f2"].values)
        assert pipe._scaler is not None and pipe._scaler._fitted

    def test_scaling_can_be_disabled(self):
        config = PipelineConfig(
            processing=ProcessingConfig(scaling_strategy=None),
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        pipe = Featransform(config)
        assert pipe._scaler is None


class TestIdColumnExclusion:
    def test_explicit_id_excluded_from_fe_input(self):
        X, y = _dataset_with_id(n=200)
        config = PipelineConfig(
            task_type="binary_classification",
            processing=ProcessingConfig(
                id_columns=["id"], handle_datetime=False, drop_constant=False
            ),
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        pipe = Featransform(config).fit(X, y)
        assert pipe._id_columns == ["id"]
        # id feeds neither the scaler nor the feature-engineering ensembles.
        assert "id" not in pipe._scaler._numeric_columns
        # ...but it still passes through to the output.
        assert "id" in pipe.transform(X).columns

    def test_auto_detect_finds_shuffled_id(self):
        X, y = _dataset_with_id(n=200, monotonic_id=False)
        config = PipelineConfig(
            task_type="binary_classification",
            processing=ProcessingConfig(
                auto_detect_id=True, handle_datetime=False, drop_constant=False
            ),
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        pipe = Featransform(config).fit(X, y)
        assert "id" in pipe._id_columns

    def test_auto_detect_off_by_default(self):
        X, _ = _dataset_with_id(n=50)
        config = PipelineConfig(
            processing=ProcessingConfig(handle_datetime=False),
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        pipe = Featransform(config)
        assert pipe._resolve_id_columns(X) == []


class TestUnknownParamValidation:
    def test_unknown_param_raises(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(
                    model_family=ModelFamily.ISOLATION_FOREST,
                    parameters={"totally_bogus": 1},
                )
            ]
        )
        with pytest.raises(ConfigurationError) as exc:
            Featransform(config)
        assert "totally_bogus" in str(exc.value)

    def test_valid_params_accepted(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(
                    model_family=ModelFamily.ISOLATION_FOREST,
                    parameters={"n_estimators": 50, "contamination": 0.1},
                )
            ],
            clustering_models=[
                ModelConfig(
                    model_family=ModelFamily.GAUSSIAN_MIXTURE,
                    parameters={"n_components": 3, "covariance_type": "diag"},
                )
            ],
        )
        Featransform(config)  # must not raise

    def test_injected_params_allowed(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(
                    model_family=ModelFamily.ISOLATION_FOREST,
                    parameters={"n_jobs": 1, "random_state": 7},
                )
            ]
        )
        Featransform(config)  # must not raise


class TestMetadataSerialization:
    def test_export_metadata_config_is_json_dict(self, binary_data):
        X, y = binary_data
        pipe = Featransform(FTconfig.minimal(task_type="binary_classification")).fit(X, y)
        meta = PipelineSerializer.export_metadata(pipe)
        assert isinstance(meta["config"], dict)
        # Round-trips through json without a stringified pydantic repr.
        json.dumps(meta, default=str)
        assert meta["config"]["task_type"] == "binary_classification"


class TestFeatureNamesMatchColumns:
    @pytest.mark.parametrize("ensemble_cls", [AnomalyEnsemble, ClusteringEnsemble, DimensionalityEnsemble])
    def test_get_feature_names_matches_emitted_columns(self, ensemble_cls, small_numeric_df):
        ens = ensemble_cls()
        ens.fit(small_numeric_df, None)
        result = ens.transform(small_numeric_df)
        assert ens.get_feature_names() == list(result.features.columns)


class TestEncoderVectorized:
    def test_known_categories_match_classes_order(self):
        enc = CategoricalEncoder().fit(pd.DataFrame({"c": ["a", "b", "c", "a", "b"]}))
        out = enc.transform(pd.DataFrame({"c": ["a", "b", "c"]}))
        # classes_ is sorted -> a=0, b=1, c=2
        assert out["c"].tolist() == [0, 1, 2]

    def test_unknown_category_maps_to_minus_one(self):
        enc = CategoricalEncoder().fit(pd.DataFrame({"c": ["a", "b", "a", "b", "a"]}))
        out = enc.transform(pd.DataFrame({"c": ["a", "zzz"]}))
        assert out["c"].tolist() == [0, -1]
        assert out["c"].dtype.kind in ("i", "u")
