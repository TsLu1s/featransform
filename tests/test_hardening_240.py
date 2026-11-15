"""Regression tests for the 2.4.0 hardening pass.

Each class pins one defect found in the fourth review: the behaviour that used
to crash, silently do the wrong thing, or ignore configuration.
"""

import logging
import pickle
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs, make_classification

import featransform
from featransform import (
    CategoricalEncoder,
    ConfigurationError,
    CoreImputer,
    DatetimeExpander,
    Featransform,
    FeatureSelector,
    FTconfig,
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    PipelineSerializer,
    ProcessingConfig,
    TransformationError,
    ValidationError,
    make_dataset,
)
from featransform.core.enums import (
    ImputationStrategy,
    ModelFamily,
    OptimizationMetric,
    SelectionStrategy,
    TaskType,
)
from featransform.core.exceptions import FeatureTransformError, NotFittedError
from featransform.features.anomaly import IsolationForestStrategy
from featransform.features.clustering import DBSCANStrategy, KMeansStrategy
from featransform.features.dimensionality import PCAStrategy

# ── helpers ───────────────────────────────────────────────────────────


def _fast_optimization(**overrides):
    params = dict(
        n_iterations=3,
        min_features=3,
        validation_split=0.25,
        use_catboost=False,
        use_xgboost=True,
        n_estimators=10,
    )
    params.update(overrides)
    return OptimizationConfig(**params)


def _config(optimize=False, task_type=None, **kwargs):
    kwargs.setdefault("processing", ProcessingConfig(imputation_strategy=ImputationStrategy.MEAN))
    kwargs.setdefault("anomaly_models", [ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)])
    kwargs.setdefault(
        "clustering_models",
        [ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})],
    )
    kwargs.setdefault(
        "dimensionality_models",
        [ModelConfig(model_family=ModelFamily.PCA, parameters={"n_components": 2})],
    )
    return PipelineConfig(
        task_type=task_type,
        optimization=_fast_optimization() if optimize else None,
        n_jobs=1,
        **kwargs,
    )


@pytest.fixture
def clf_data():
    X, y = make_classification(
        n_samples=240, n_features=8, n_informative=5, n_redundant=1, random_state=0
    )
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(8)]), pd.Series(y, name="target")


@pytest.fixture
def mixed_data():
    return make_dataset(task="binary_classification", n_samples=300, complexity="medium")


# ── Refit starts from scratch ─────────────────────────────────────────


class TestRefit:
    def test_refit_relearns_every_component(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        first_mean = pipe._scaler._scaler.mean_.copy()
        first_history = len(pipe.get_optimization_history())

        pipe.fit(X * 100 + 5, y)

        assert not np.allclose(first_mean, pipe._scaler._scaler.mean_)
        # History belongs to the latest fit only — it must not accumulate.
        assert len(pipe.get_optimization_history()) == first_history

    def test_refit_matches_a_fresh_pipeline(self, clf_data):
        X, y = clf_data
        other = X * 3 - 1
        refitted = Featransform(_config(optimize=True)).fit(X, y).fit(other, y)
        fresh = Featransform(_config(optimize=True)).fit(other, y)
        pd.testing.assert_frame_equal(refitted.transform(other), fresh.transform(other))

    def test_refit_with_a_different_task(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        assert pipe.get_summary()["task_type"] == "binary_classification"

        continuous = pd.Series(np.random.RandomState(0).randn(len(X)))
        pipe.fit(X, continuous)
        assert pipe.get_summary()["task_type"] == "regression"
        assert pipe.get_summary()["optimization"]["metric"] == "mse"


# ── transform aligns the raw input to the fit-time columns ────────────


class TestTransformInputAlignment:
    def test_extra_columns_are_ignored(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        expected = pipe.transform(X)
        pd.testing.assert_frame_equal(pipe.transform(X.assign(extra=1.0)), expected)

    def test_column_order_does_not_matter(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config()).fit(X, y)
        pd.testing.assert_frame_equal(pipe.transform(X[X.columns[::-1]]), pipe.transform(X))

    def test_missing_column_is_named(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config()).fit(X, y)
        with pytest.raises(ValidationError, match="f3"):
            pipe.transform(X.drop(columns=["f3"]))

    def test_input_columns_accessor(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config())
        assert pipe.get_input_columns() is None
        assert pipe.fit(X, y).get_input_columns() == list(X.columns)


# ── A fitted pipeline does not carry its training data ────────────────


class TestNoTrainingDataRetention:
    def test_state_holds_schema_only(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        assert len(pipe._state.data) == 0
        assert pipe._state.target is None
        assert list(pipe._state.data.columns) == pipe.get_fit_schema()

    def test_pickle_size_independent_of_row_count(self):
        sizes = []
        for n in (200, 2000):
            X, y = make_classification(n_samples=n, n_features=6, random_state=0)
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
            config = _config(
                clustering_models=[],
                dimensionality_models=[],
                anomaly_models=[
                    ModelConfig(
                        model_family=ModelFamily.ISOLATION_FOREST,
                        parameters={"n_estimators": 5},
                    )
                ],
            )
            sizes.append(len(pickle.dumps(Featransform(config).fit(X, pd.Series(y)))))
        assert sizes[1] < sizes[0] * 1.5

    def test_fit_transform_equals_fit_then_transform(self, mixed_data):
        X, y = mixed_data
        config = _config(optimize=True, task_type="binary_classification")
        direct = Featransform(config).fit_transform(X, y)
        two_step = Featransform(config).fit(X, y).transform(X)
        pd.testing.assert_frame_equal(direct, two_step)


# ── set_params rebuilds from an empty registry ────────────────────────


class TestSetParamsRebuild:
    def test_components_absent_from_new_config_are_dropped(self):
        pipe = Featransform(FTconfig.complete())
        assert pipe.has_component("selector") and pipe.has_component("clustering")
        pipe.set_params(config=FTconfig.minimal())
        assert sorted(pipe.components) == ["anomaly", "encoder", "imputer"]


# ── Column labels ─────────────────────────────────────────────────────


class TestColumnLabels:
    def test_integer_labels(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(200, 6))
        y = pd.Series((X[0] + X[1] > 0).astype(int))
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        out = pipe.transform(X)
        assert all(isinstance(col, str) for col in out.columns)
        assert len(out) == len(X)

    def test_duplicate_labels_are_rejected(self):
        X = pd.DataFrame(np.random.RandomState(0).randn(50, 3), columns=["a", "b", "a"])
        with pytest.raises(ValidationError, match="unique"):
            Featransform(_config()).fit(X, pd.Series(np.arange(50) % 2))

    def test_labels_with_characters_xgboost_rejects(self, clf_data):
        X, y = clf_data
        X = X.rename(columns={"f0": "price[usd]", "f1": "a<b", "f2": "weird]name"})
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        assert pipe.transform(X).shape[0] == len(X)

    def test_engineered_name_collision_is_renamed(self, clf_data):
        X, y = clf_data
        X = X.assign(pca_0=1.5 * X["f0"], iso_anomaly=X["f1"] ** 2)
        pipe = Featransform(_config()).fit(X, y)
        out = pipe.transform(X)
        assert not out.columns.duplicated().any()
        assert {"pca_0", "iso_anomaly", "pca_0_ft", "iso_anomaly_ft"} <= set(out.columns)
        # The user's own columns are untouched.
        pd.testing.assert_series_equal(out["pca_0"], X["pca_0"])


# ── Awkward input columns ─────────────────────────────────────────────


class TestAwkwardColumns:
    def test_all_missing_numeric_column(self, clf_data):
        X, y = clf_data
        out = (
            Featransform(_config()).fit(X.assign(empty=np.nan), y).transform(X.assign(empty=np.nan))
        )
        assert not out.isna().any().any()

    @pytest.mark.parametrize("strategy", list(ImputationStrategy))
    def test_imputer_handles_all_missing_column(self, strategy):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0], "empty": np.nan})
        out = CoreImputer(strategy=strategy).fit(df).transform(df)
        assert not out.isna().any().any()
        assert (out["empty"] == 0.0).all()

    def test_timezone_aware_datetime(self, clf_data):
        X, y = clf_data
        X = X.assign(when=pd.date_range("2021-03-01", periods=len(X), freq="h", tz="UTC"))
        out = Featransform(_config()).fit(X, y).transform(X)
        assert "when" not in out.columns and "when_hour" in out.columns

    def test_timedelta_becomes_seconds(self, clf_data):
        X, y = clf_data
        X = X.assign(elapsed=pd.to_timedelta(np.arange(len(X)), unit="h"))
        out = Featransform(_config()).fit(X, y).transform(X)
        assert out["elapsed"].iloc[2] == 7200.0

    def test_missing_timestamps_become_nan_features(self):
        df = pd.DataFrame(
            {"ts": pd.date_range("2022-01-01", periods=10, freq="D"), "v": np.arange(10.0)}
        )
        df.loc[[2, 5], "ts"] = pd.NaT
        out = DatetimeExpander().fit(df).transform(df)
        date_parts = [c for c in out.columns if c.startswith("ts_")]
        assert out.loc[[2, 5], date_parts].isna().all().all()
        assert not out.drop(index=[2, 5])[date_parts].isna().any().any()

    def test_datetime_feature_dtypes_are_stable(self):
        """The output dtypes must not depend on whether a frame holds a NaT."""
        df = pd.DataFrame(
            {"ts": pd.date_range("2022-01-01", periods=10, freq="D"), "v": np.arange(10.0)}
        )
        expander = DatetimeExpander().fit(df)
        clean = expander.transform(df)
        gappy = expander.transform(df.assign(ts=df["ts"].where(df.index != 3)))
        assert (clean.dtypes == gappy.dtypes).all()

    def test_no_pandas_deprecation_warnings(self):
        df = pd.DataFrame(
            {"ts": pd.date_range("2022-01-01", periods=10, freq="D"), "v": np.arange(10.0)}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DatetimeExpander().fit(df).transform(df)


# ── Target validation ─────────────────────────────────────────────────


class TestTargetValidation:
    @pytest.mark.parametrize("convert", [np.asarray, list, lambda y: y.to_frame()])
    def test_array_like_targets(self, clf_data, convert):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, convert(y))
        assert pipe.get_summary()["task_type"] == "binary_classification"

    def test_target_is_paired_by_position(self, clf_data):
        X, y = clf_data
        shifted = y.copy()
        shifted.index = shifted.index + 10_000
        a = Featransform(_config(optimize=True)).fit(X, y).transform(X)
        b = Featransform(_config(optimize=True)).fit(X, shifted).transform(X)
        pd.testing.assert_frame_equal(a, b)

    def test_length_mismatch(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValidationError, match="different lengths"):
            Featransform(_config()).fit(X, y.iloc[:50])

    def test_missing_target_values(self, clf_data):
        X, y = clf_data
        y = y.astype(float)
        y.iloc[:3] = np.nan
        with pytest.raises(ValidationError, match="missing"):
            Featransform(_config(optimize=True)).fit(X, y)

    def test_constant_target(self, clf_data):
        X, _ = clf_data
        with pytest.raises(ValidationError, match="two distinct"):
            Featransform(_config(optimize=True)).fit(X, pd.Series(np.zeros(len(X), dtype=int)))

    def test_target_optional_without_optimization(self, clf_data):
        X, _ = clf_data
        assert Featransform(_config()).fit(X).transform(X).shape[0] == len(X)

    def test_target_required_with_optimization(self, clf_data):
        X, _ = clf_data
        with pytest.raises(ValidationError, match="y is required"):
            Featransform(_config(optimize=True)).fit(X)

    def test_two_dimensional_target(self, clf_data):
        X, y = clf_data
        with pytest.raises(ValidationError, match="one-dimensional"):
            Featransform(_config()).fit(X, np.column_stack([y, y]))

    def test_two_valued_target_under_regression_warns(self, clf_data):
        X, y = clf_data
        with pytest.warns(UserWarning, match="two distinct values"):
            Featransform(_config(optimize=True, task_type="regression")).fit(X, y)

    def test_regression_task_with_text_target(self, clf_data):
        X, y = clf_data
        labels = y.map({0: "no", 1: "yes"})
        with pytest.raises(ConfigurationError, match="regression"):
            Featransform(_config(optimize=True, task_type="regression")).fit(X, labels)


# ── The selection split copes with rare classes ───────────────────────


class TestSelectionSplit:
    def test_singleton_class(self, clf_data):
        X, y = clf_data
        y = y.copy()
        y.iloc[0] = 2
        with pytest.warns(UserWarning, match="single row"):
            pipe = Featransform(_config(optimize=True, task_type="multiclass_classification"))
            pipe.fit(X, y)
        assert pipe.get_summary()["optimization"]["best_score"] is not None

    def test_auc_with_class_absent_from_validation(self, clf_data):
        X, y = clf_data
        y = y.copy()
        y.iloc[:2] = 2  # 2 rows: one lands in validation at most, maybe none
        y.iloc[2] = 3  # singleton: stays in the training slice
        config = _config(optimize=True, task_type="multiclass_classification")
        config.optimization.metric = OptimizationMetric.AUC_ROC
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = Featransform(config).fit(X, y)
        assert 0.0 <= pipe.get_summary()["optimization"]["best_score"] <= 1.0


# ── Metric / task consistency ─────────────────────────────────────────


class TestMetricValidation:
    def test_classification_metric_on_regression(self, clf_data):
        X, _ = clf_data
        y = pd.Series(np.random.RandomState(0).randn(len(X)))
        config = _config(optimize=True)
        config.optimization.metric = OptimizationMetric.F1
        with pytest.raises(ConfigurationError, match="regression"):
            Featransform(config).fit(X, y)

    def test_regression_metric_on_classification(self, clf_data):
        X, y = clf_data
        config = _config(optimize=True)
        config.optimization.metric = OptimizationMetric.MSE
        with pytest.raises(ConfigurationError, match="binary_classification"):
            Featransform(config).fit(X, y)

    def test_selector_validates_metric_standalone(self, clf_data):
        X, y = clf_data
        selector = FeatureSelector(metric=OptimizationMetric.R2, use_catboost=False)
        with pytest.raises(ConfigurationError):
            selector.fit(X, y)

    def test_summary_reports_resolved_metric(self, clf_data):
        X, y = clf_data
        summary = Featransform(_config(optimize=True)).fit(X, y).get_summary()
        assert summary["optimization"]["metric"] == "f1"


# ── Pipeline-level settings reach the components ──────────────────────


class TestSettingsThreading:
    def test_n_jobs_random_state_verbose(self, clf_data):
        X, y = clf_data
        config = _config(
            optimize=True,
            processing=ProcessingConfig(imputation_strategy=ImputationStrategy.ITERATIVE),
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST),
                ModelConfig(model_family=ModelFamily.LOCAL_OUTLIER_FACTOR),
            ],
        )
        config.n_jobs = 2
        config.random_state = 7
        pipe = Featransform(config).fit(X, y)

        iso, lof = pipe.get_component("anomaly").strategies
        assert iso._models["isolation_forest"].n_jobs == 2
        assert iso._models["isolation_forest"].random_state == 7
        assert lof._models["lof"].n_jobs == 2
        assert pipe.get_component("imputer")._imputer.random_state == 7
        selector = pipe.get_component("selector")
        assert selector.n_jobs == 2
        assert all(calc.n_jobs == 2 for calc in selector._calculators)

    def test_model_parameters_override_pipeline_settings(self):
        config = _config(
            anomaly_models=[
                ModelConfig(
                    model_family=ModelFamily.ISOLATION_FOREST,
                    parameters={"n_jobs": 3, "random_state": 11},
                )
            ]
        )
        strategy = Featransform(config).get_component("anomaly").strategies[0]
        assert strategy._params["n_jobs"] == 3
        assert strategy._params["random_state"] == 11


# ── verbose=True is visible ───────────────────────────────────────────


class TestVerboseLogging:
    def test_records_reach_application_logging(self, clf_data, caplog):
        X, y = clf_data
        config = _config()
        config.verbose = True
        with caplog.at_level(logging.INFO, logger="featransform"):
            Featransform(config).fit(X, y)
        assert any("Pipeline fitted" in record.getMessage() for record in caplog.records)

    def test_console_output_without_logging_config(self):
        script = (
            "import numpy as np, pandas as pd\n"
            "from featransform import Featransform, FTconfig\n"
            "X = pd.DataFrame(np.random.RandomState(0).randn(40, 4), columns=list('abcd'))\n"
            "c = FTconfig.minimal(task_type=None); c.verbose = True\n"
            "Featransform(c).fit(X)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr
        assert "[Featransform] Pipeline fitted" in result.stderr

    def test_silent_by_default(self, clf_data, caplog):
        X, y = clf_data
        with caplog.at_level(logging.INFO, logger="featransform"):
            Featransform(_config()).fit(X, y)
        assert not caplog.records


# ── Selection strategies ──────────────────────────────────────────────


class TestSelectionStrategies:
    @pytest.mark.parametrize(
        "strategy",
        [SelectionStrategy.CORRELATION, SelectionStrategy.VARIANCE, SelectionStrategy.RECURSIVE],
    )
    def test_unimplemented_strategy_warns_and_falls_back(self, strategy):
        with pytest.warns(UserWarning, match="not implemented"):
            selector = FeatureSelector(strategy=strategy)
        assert selector.strategy == SelectionStrategy.IMPORTANCE

    def test_importance_strategy_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            FeatureSelector(strategy=SelectionStrategy.IMPORTANCE)

    def test_tie_goes_to_the_smaller_subset(self, clf_data, monkeypatch):
        X, y = clf_data
        selector = FeatureSelector(min_features=2, use_catboost=False, n_estimators=10)
        selector.fit(X, y)
        monkeypatch.setattr(selector, "evaluate_subset", lambda *args, **kwargs: 0.5)
        features, score, _ = selector.optimize(X, y, X, y, thresholds=[0.5, 0.8, 1.0])
        assert score == 0.5
        assert len(features) == len(selector.select_by_threshold(0.5)) < X.shape[1]

    def test_full_set_candidate_reuses_baseline_score(self, clf_data, monkeypatch):
        X, y = clf_data
        selector = FeatureSelector(min_features=X.shape[1], use_catboost=False, n_estimators=10)
        selector.fit(X, y)
        calls = []
        monkeypatch.setattr(
            selector, "evaluate_subset", lambda *args, **kwargs: calls.append(1) or 0.1
        )
        _, score, history = selector.optimize(X, y, X, y, thresholds=[1.0], baseline_score=0.9)
        assert not calls  # the full set was not trained a second time
        assert score == 0.9 and history[0]["score"] == 0.9

    def test_reused_selector_redetects_task(self, clf_data):
        X, y = clf_data
        selector = FeatureSelector(use_catboost=False, n_estimators=10)
        selector.fit(X, y)
        assert selector._task_type == TaskType.BINARY_CLASSIFICATION
        selector.fit(X, pd.Series(np.random.RandomState(1).randn(len(X))))
        assert selector._task_type == TaskType.REGRESSION

    def test_explicit_task_survives_refit(self, clf_data):
        X, y = clf_data
        selector = FeatureSelector(task_type="regression", use_catboost=False, n_estimators=10)
        selector.fit(X, y.astype(float))
        selector.fit(X, y.astype(float))
        assert selector._task_type == TaskType.REGRESSION

    def test_importance_keyed_by_original_labels(self):
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(120, 4), columns=[0, 1, "two", ("t", 3)])
        y = pd.Series((X[0] > 0).astype(int))
        selector = FeatureSelector(n_estimators=10).fit(X, y)
        assert set(selector.get_feature_scores()) == set(X.columns)


# ── No files written to the working directory ─────────────────────────


class TestNoWorkingDirectoryLitter:
    def test_catboost_writes_nothing(self, clf_data, tmp_path, monkeypatch):
        X, y = clf_data
        monkeypatch.chdir(tmp_path)
        FeatureSelector(n_estimators=5).fit(X, y)
        assert list(tmp_path.iterdir()) == []


# ── Strategies ────────────────────────────────────────────────────────


class TestStrategies:
    @pytest.mark.parametrize(
        "strategy",
        [KMeansStrategy(n_clusters=2), PCAStrategy(n_components=2), IsolationForestStrategy()],
    )
    def test_fit_without_target(self, strategy, clf_data):
        X, _ = clf_data
        result = strategy.fit(X).transform(X)
        assert len(result.features) == len(X)

    def test_fit_transform_shortcut(self, clf_data):
        X, _ = clf_data
        assert KMeansStrategy(n_clusters=2).fit_transform(X).n_features == 1

    def test_deduplicated_names_reach_feature_info(self, clf_data):
        from featransform import ClusteringEnsemble

        X, _ = clf_data
        ensemble = ClusteringEnsemble(
            strategies=[KMeansStrategy(n_clusters=2), KMeansStrategy(n_clusters=3)]
        )
        result = ensemble.fit(X).transform(X)
        assert [info.name for info in result.feature_info] == list(result.features.columns)
        assert len(set(result.features.columns)) == 2


class TestDBSCAN:
    @pytest.fixture
    def blobs(self):
        X, _ = make_blobs(n_samples=400, centers=3, n_features=12, cluster_std=1.0, random_state=0)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        return pd.DataFrame(X, columns=[f"b{i}" for i in range(12)])

    def test_auto_eps_finds_structure_in_scaled_data(self, blobs):
        strategy = DBSCANStrategy().fit(blobs)
        labels = strategy.transform(blobs).features["dbs_cluster"]
        assert (labels != -1).mean() > 0.5
        assert labels[labels != -1].nunique() >= 2
        assert strategy._models["dbscan"].eps_ > 0

    def test_fixed_eps_is_honored(self, blobs):
        strategy = DBSCANStrategy(eps=0.123).fit(blobs)
        assert strategy._models["dbscan"].eps_ == 0.123

    def test_far_points_are_noise(self, blobs):
        strategy = DBSCANStrategy().fit(blobs)
        far = pd.DataFrame(np.full((3, blobs.shape[1]), 1e3), columns=blobs.columns)
        assert (strategy.transform(far).features["dbs_cluster"] == -1).all()

    def test_training_core_points_keep_their_cluster(self, blobs):
        strategy = DBSCANStrategy().fit(blobs)
        model = strategy._models["dbscan"]
        core = model._dbscan.core_sample_indices_
        predicted = strategy.transform(blobs).features["dbs_cluster"].to_numpy()
        assert (predicted[core] == model._dbscan.labels_[core]).all()

    def test_transform_is_deterministic(self, blobs):
        strategy = DBSCANStrategy().fit(blobs)
        first = strategy.transform(blobs).features
        pd.testing.assert_frame_equal(first, strategy.transform(blobs).features)

    @pytest.mark.parametrize("eps", ["median", 0, -1.0])
    def test_invalid_eps(self, eps):
        with pytest.raises(ConfigurationError):
            DBSCANStrategy(eps=eps)

    def test_all_noise_training_set(self):
        X = pd.DataFrame(np.random.RandomState(0).randn(30, 3) * 100, columns=list("abc"))
        strategy = DBSCANStrategy(eps=1e-6).fit(X)
        assert (strategy.transform(X).features["dbs_cluster"] == -1).all()


# ── Imputation ────────────────────────────────────────────────────────


class TestImputation:
    def test_knn_is_not_dominated_by_wide_columns(self):
        rng = np.random.RandomState(0)
        X, _ = make_classification(
            n_samples=600, n_features=6, n_informative=4, n_redundant=2, random_state=0
        )
        full = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
        full.insert(0, "wide", rng.permutation(len(full)) * 1000.0)
        mask = rng.rand(*full.shape) < 0.1
        mask[:, 0] = False
        gappy = full.mask(mask)

        def rmse(frame):
            return float(np.sqrt(np.mean((frame.to_numpy()[mask] - full.to_numpy()[mask]) ** 2)))

        knn = CoreImputer(strategy=ImputationStrategy.KNN).fit(gappy).transform(gappy)
        mean = CoreImputer(strategy=ImputationStrategy.MEAN).fit(gappy).transform(gappy)
        # In raw space the 'wide' column decides every neighbour and KNN is no
        # better than the mean; in standardized space it is clearly better.
        assert rmse(knn) < 0.8 * rmse(mean)

    def test_observed_values_and_dtypes_are_untouched(self):
        df = pd.DataFrame(
            {
                "ints": np.arange(8, dtype="int64"),
                "floats": [1.5, np.nan, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                "f32": np.linspace(0, 1, 8).astype("float32"),
            }
        )
        for strategy in ImputationStrategy:
            out = CoreImputer(strategy=strategy).fit(df).transform(df)
            assert out["ints"].dtype == "int64"
            assert out["f32"].dtype == "float32"
            observed = df["floats"].notna()
            assert (out.loc[observed, "floats"] == df.loc[observed, "floats"]).all()
            assert not out.isna().any().any()

    def test_nullable_integers_become_float(self):
        df = pd.DataFrame(
            {"n": pd.array([1, 2, None, 4, 5, 6], dtype="Int64"), "x": np.arange(6.0)}
        )
        imputer = CoreImputer(strategy=ImputationStrategy.MEAN).fit(df)
        assert imputer.transform(df)["n"].dtype == "float64"
        assert imputer.transform(df.dropna())["n"].dtype == "float64"

    def test_excluded_columns_are_not_predictors(self):
        df = pd.DataFrame({"id": np.arange(20), "a": np.arange(20.0), "b": np.arange(20.0)})
        df.loc[3, "a"] = np.nan
        imputer = CoreImputer(strategy=ImputationStrategy.ITERATIVE, exclude_columns=["id"]).fit(df)
        assert "id" not in imputer._numeric_columns
        out = imputer.transform(df)
        assert out["id"].equals(df["id"])
        assert abs(out.loc[3, "a"] - 3.0) < 0.5

    def test_pipeline_keeps_identifiers_out_of_the_imputer(self, mixed_data):
        X, y = mixed_data
        pipe = Featransform(FTconfig.minimal(task_type="binary_classification")).fit(X, y)
        assert pipe.get_summary()["id_columns"] == ["id"]
        assert "id" not in pipe.get_component("imputer")._numeric_columns
        assert pipe.transform(X)["id"].dtype == X["id"].dtype

    def test_missing_fit_column_is_reported(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3, 4, 5], "b": np.arange(5.0)})
        imputer = CoreImputer(strategy=ImputationStrategy.MEAN).fit(df)
        with pytest.raises(ValidationError, match="'b'"):
            imputer.transform(df[["a"]])

    def test_all_missing_categorical(self):
        df = pd.DataFrame({"c": pd.Categorical([None] * 6, categories=["x"]), "v": np.arange(6.0)})
        out = CoreImputer(strategy=ImputationStrategy.MEAN).fit(df).transform(df)
        assert (out["c"] == "missing").all()


# ── Encoding ──────────────────────────────────────────────────────────


class TestEncoding:
    @pytest.mark.parametrize("dtype", ["object", "category", "string"])
    def test_text_like_dtypes_are_encoded(self, dtype):
        df = pd.DataFrame({"c": pd.Series(["b", "a", "c", "a", "b", "c"], dtype=dtype)})
        out = CategoricalEncoder().fit(df).transform(df)
        assert out["c"].tolist() == [1, 0, 2, 0, 1, 2]
        assert out["c"].dtype == "int64"

    def test_missing_and_unknown_map_to_minus_one(self):
        train = pd.DataFrame({"c": ["a", "b", None, "a", "b", "a"]})
        encoder = CategoricalEncoder().fit(train)
        assert list(encoder._encoders["c"].classes_) == ["a", "b"]
        out = encoder.transform(pd.DataFrame({"c": ["a", None, "zzz", "b"]}))
        assert out["c"].tolist() == [0, -1, -1, 1]

    def test_refit_discards_previous_columns(self):
        encoder = CategoricalEncoder().fit(pd.DataFrame({"old": list("abcde")}))
        encoder.fit(pd.DataFrame({"new": list("vwxyz")}))
        assert list(encoder._encoders) == ["new"]


# ── Engineered-feature pruning ────────────────────────────────────────


class TestConstantEngineeredFeatures:
    def _config(self, drop_constant):
        return PipelineConfig(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, drop_constant=drop_constant
            ),
            clustering_models=[
                # eps this small leaves every training row as noise: one constant column.
                ModelConfig(model_family=ModelFamily.DBSCAN, parameters={"eps": 1e-9}),
                ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 2}),
            ],
        )

    def test_dropped_at_fit_and_transform(self, clf_data):
        X, y = clf_data
        pipe = Featransform(self._config(drop_constant=True)).fit(X, y)
        assert "dbs_cluster" not in pipe.get_fit_schema()
        assert "kme_cluster" in pipe.get_fit_schema()
        assert "dbs_cluster" not in pipe.transform(X.iloc[:3]).columns
        assert pipe.get_summary()["features"]["engineered"] == 1

    def test_kept_when_drop_constant_is_off(self, clf_data):
        X, y = clf_data
        pipe = Featransform(self._config(drop_constant=False)).fit(X, y)
        assert "dbs_cluster" in pipe.get_fit_schema()


class TestFeatureEngineeringInput:
    def test_skipped_when_only_identifiers_remain(self):
        X = pd.DataFrame({"row": np.arange(30), "key": np.arange(30)[::-1]})
        config = _config(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, id_columns=["row", "key"]
            )
        )
        with pytest.warns(UserWarning, match="skipped"):
            pipe = Featransform(config).fit(X)
        assert list(pipe.transform(X).columns) == ["row", "key"]

    def test_unknown_id_column_warns(self, clf_data):
        X, y = clf_data
        config = _config(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, id_columns=["nope"]
            )
        )
        with pytest.warns(UserWarning, match="nope"):
            Featransform(config).fit(X, y)


# ── Summary / report ──────────────────────────────────────────────────


class TestSummary:
    def test_initial_counts_input_columns(self, mixed_data):
        X, y = mixed_data
        config = _config(optimize=True, task_type="binary_classification")
        summary = Featransform(config).fit(X, y).get_summary()
        features = summary["features"]
        assert features["initial"] == X.shape[1]
        assert features["final"] == len(summary["schema"]["columns"])
        assert 0 <= features["engineered_selected"] <= features["engineered"]
        assert summary["version"] == featransform.__version__

    def test_best_matches_applied_selection(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        optimization = pipe.get_summary()["optimization"]
        assert optimization["best_features"] == pipe.transform(X).shape[1]
        scores = [h.score for h in pipe.get_optimization_history()]
        assert optimization["best_score"] == max(scores)

    def test_zero_best_score_is_still_reported(self, clf_data, capsys):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        pipe._best_result = {"score": 0.0, "n_features": 4}
        pipe.report_optimization()
        assert "Best optimization score: 0.0000" in capsys.readouterr().out


# ── Configuration models ──────────────────────────────────────────────


class TestConfigValidation:
    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError):
            ProcessingConfig(scaling="robust")
        with pytest.raises(ValueError):
            PipelineConfig(n_job=2)

    def test_assignment_is_validated(self):
        config = PipelineConfig()
        config.task_type = "binary_classification"
        assert config.task_type is TaskType.BINARY_CLASSIFICATION
        with pytest.raises(ValueError):
            config.task_type = "ranking"

        optimization = OptimizationConfig()
        with pytest.raises(ValueError):
            optimization.validation_split = 0.9

    def test_parameters_default_is_not_shared(self):
        first, second = (ModelConfig(model_family=ModelFamily.PCA) for _ in range(2))
        first.parameters["n_components"] = 3
        assert second.parameters == {}


# ── Custom strategies ─────────────────────────────────────────────────


class TestRegisterStrategy:
    def test_registration_takes_effect(self, clf_data):
        class WideForest(IsolationForestStrategy):
            def __init__(self, **kwargs):
                super().__init__(n_estimators=7, **kwargs)

        X, y = clf_data
        pipe = Featransform(_config())
        pipe.register_strategy(ModelFamily.ISOLATION_FOREST, WideForest, "anomaly")
        strategy = pipe.get_component("anomaly").strategies[0]
        assert isinstance(strategy, WideForest)
        pipe.fit(X, y)
        assert strategy._models["isolation_forest"].n_estimators == 7

    def test_registration_on_fitted_pipeline_warns(self, clf_data):
        X, y = clf_data
        pipe = Featransform(_config()).fit(X, y)
        with pytest.warns(UserWarning, match="next fit"):
            pipe.register_strategy(ModelFamily.ISOLATION_FOREST, IsolationForestStrategy)

    def test_invalid_registration(self):
        pipe = Featransform(_config())
        with pytest.raises(ConfigurationError):
            pipe.register_strategy(ModelFamily.PCA, PCAStrategy, "reduction")
        with pytest.raises(ConfigurationError):
            pipe.register_strategy(ModelFamily.PCA, PCAStrategy(), "dimensionality")


# ── Exceptions ────────────────────────────────────────────────────────


class TestExceptions:
    def test_validation_error_is_a_value_error(self):
        assert issubclass(ValidationError, ValueError)
        assert issubclass(ValidationError, FeatureTransformError)

    def test_error_context_keeps_own_exceptions_and_chains_others(self):
        component = KMeansStrategy()
        with pytest.raises(ConfigurationError):
            with component._error_context("step"):
                raise ConfigurationError("bad config")
        with pytest.raises(TransformationError) as caught:
            with component._error_context("step"):
                raise KeyError("boom")
        assert isinstance(caught.value.__cause__, KeyError)

    def test_transform_before_fit(self, clf_data):
        X, _ = clf_data
        with pytest.raises(NotFittedError):
            Featransform(_config()).transform(X)


# ── Dataset generator ─────────────────────────────────────────────────


class TestDatasetGenerator:
    def test_global_rng_is_untouched(self):
        np.random.seed(123)
        expected = np.random.rand()
        np.random.seed(123)
        make_dataset(task="binary", n_samples=40)
        assert np.random.rand() == expected

    def test_reproducible(self):
        a, _ = make_dataset(task="regression", n_samples=60, random_state=5)
        b, _ = make_dataset(task="regression", n_samples=60, random_state=5)
        pd.testing.assert_frame_equal(a, b)

    def test_multi_target_regression_is_rejected(self):
        from featransform import DatasetGenerator

        with pytest.raises(ValueError, match="single-target"):
            DatasetGenerator.generate(task="regression", n_samples=20, n_targets=2)


# ── Serialization ─────────────────────────────────────────────────────


class TestSerialization:
    def test_round_trip_is_silent_and_exact(self, clf_data, tmp_path):
        X, y = clf_data
        pipe = Featransform(_config(optimize=True)).fit(X, y)
        path = tmp_path / "nested" / "pipe.pkl"
        PipelineSerializer.save(pipe, path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = PipelineSerializer.load(path)
        pd.testing.assert_frame_equal(loaded.transform(X), pipe.transform(X))

    def test_version_mismatch_warns(self, clf_data, tmp_path):
        X, y = clf_data
        pipe = Featransform(_config()).fit(X, y)
        pipe._fitted_version = "2.3.0"
        PipelineSerializer.save(pipe, tmp_path / "old.pkl")
        with pytest.warns(UserWarning, match="2.3.0"):
            PipelineSerializer.load(tmp_path / "old.pkl")

    def test_load_rejects_other_objects(self, tmp_path):
        path = tmp_path / "other.pkl"
        path.write_bytes(pickle.dumps({"not": "a pipeline"}))
        with pytest.raises(TypeError, match="Featransform"):
            PipelineSerializer.load(path)

    def test_save_rejects_other_objects(self, tmp_path):
        with pytest.raises(TypeError):
            PipelineSerializer.save({"not": "a pipeline"}, tmp_path / "x.pkl")

    def test_config_round_trip(self, tmp_path):
        config = FTconfig.complete(task_type="binary_classification")
        PipelineSerializer.save_config(config, tmp_path / "config.json")
        assert PipelineSerializer.load_config(tmp_path / "config.json") == config

    def test_pipeline_pickled_before_240_still_transforms(self, mixed_data):
        """Strip every attribute introduced in 2.4.0 — what unpickling an older
        pipeline yields — and check the class-level defaults carry it."""
        X, y = mixed_data
        config = FTconfig.complete(task_type="binary_classification")
        config.optimization = None
        pipe = Featransform(config).fit(X, y)
        expected = pipe.transform(X)

        for name in (
            "_input_columns",
            "_engineered_name_maps",
            "_engineered_dropped",
            "_fe_skipped",
            "_best_result",
            "_task_type",
            "_fitted_version",
        ):
            pipe.__dict__.pop(name, None)
        imputer = pipe.get_component("imputer")
        for name in ("exclude_columns", "n_neighbors", "_all_nan_columns", "_excluded_fill"):
            imputer.__dict__.pop(name, None)
        pipe.get_component("encoder").__dict__.pop("_code_maps", None)
        pipe._datetime_expander.__dict__.pop("_timedelta_columns", None)

        pd.testing.assert_frame_equal(pipe.transform(X), expected)
        assert pipe.get_summary()["features"]["final"] == expected.shape[1]


# ── Public API ────────────────────────────────────────────────────────


class TestPublicApi:
    def test_all_names_resolve(self):
        for name in featransform.__all__:
            assert hasattr(featransform, name), name

    def test_new_exports(self):
        from featransform import FeatureScaler, ScalingStrategy  # noqa: F401

        assert featransform.__version__ == "2.4.0"


class TestThresholdSemantics:
    def test_threshold_one_keeps_every_feature(self):
        """Many zero-importance columns: float32 importances used to push the
        cumulative sum past 1.0 and drop part of the tail at threshold 1.0."""
        rng = np.random.RandomState(0)
        X = pd.DataFrame(rng.randn(300, 60), columns=[f"c{i}" for i in range(60)])
        y = pd.Series((X["c0"] + X["c1"] > 0).astype(int))
        selector = FeatureSelector(min_features=1, n_estimators=10).fit(X, y)
        assert set(selector.select_by_threshold(1.0)) == set(X.columns)
        assert abs(sum(selector.get_feature_scores().values()) - 1.0) < 1e-9
