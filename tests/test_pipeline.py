"""Tests for the main Featransform pipeline."""

import numpy as np
import pandas as pd
import pytest

from featransform.configs.baseline import FTconfig
from featransform.core.enums import (
    EncodingStrategy,
    ImputationStrategy,
    ModelFamily,
    PipelineStage,
    SelectionStrategy,
    TaskType,
)
from featransform.core.exceptions import NotFittedError
from featransform.core.models import (
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    ProcessingConfig,
)
from featransform.pipeline import Featransform, PipelineState


# ── PipelineState ──────────────────────────────────────────────────────


class TestPipelineState:
    def test_creation(self, numeric_df):
        state = PipelineState(data=numeric_df)
        assert state.target is None
        assert state.stage == PipelineStage.PREPROCESSING
        assert state.features_created == []
        assert state.transformations_applied == []

    def test_evolve(self, numeric_df):
        state = PipelineState(data=numeric_df)
        state.features_created.append("test")
        new_state = state.evolve(stage=PipelineStage.FEATURE_ENGINEERING)
        assert new_state.stage == PipelineStage.FEATURE_ENGINEERING
        # evolve copies lists
        assert new_state.features_created == ["test"]
        new_state.features_created.append("other")
        assert "other" not in state.features_created


# ── Pipeline Construction ──────────────────────────────────────────────


class TestPipelineConstruction:
    def test_minimal_config(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        assert pipe.has_component("imputer")
        assert pipe.has_component("encoder")
        assert pipe.has_component("anomaly")
        assert not pipe.has_component("clustering")
        assert not pipe.has_component("dimensionality")

    def test_full_components(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
            dimensionality_models=[ModelConfig(model_family=ModelFamily.PCA)],
        )
        pipe = Featransform(config)
        assert pipe.has_component("anomaly")
        assert pipe.has_component("clustering")
        assert pipe.has_component("dimensionality")

    def test_optimization_component(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            optimization=OptimizationConfig(
                selection_strategy=SelectionStrategy.IMPORTANCE,
                n_iterations=3,
            ),
        )
        pipe = Featransform(config)
        assert pipe.has_component("selector")

    def test_no_optimization(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            optimization=None,
        )
        pipe = Featransform(config)
        assert not pipe.has_component("selector")

    def test_disabled_model(self):
        config = PipelineConfig(
            anomaly_models=[
                ModelConfig(model_family=ModelFamily.ISOLATION_FOREST, enabled=False)
            ],
        )
        pipe = Featransform(config)
        assert not pipe.has_component("anomaly")


# ── Pipeline fit/transform ─────────────────────────────────────────────


class TestPipelineFitTransform:
    def _simple_config(self):
        return PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
            verbose=False,
        )

    def test_fit_regression(self, regression_data):
        X, y = regression_data
        pipe = Featransform(self._simple_config())
        pipe.fit(X, y)
        assert pipe._fitted

    def test_fit_binary(self, binary_data):
        X, y = binary_data
        pipe = Featransform(self._simple_config())
        pipe.fit(X, y)
        assert pipe._fitted

    def test_transform_after_fit(self, regression_data):
        X, y = regression_data
        pipe = Featransform(self._simple_config())
        pipe.fit(X, y)
        result = pipe.transform(X)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)
        # Should have original features + engineered features
        assert result.shape[1] >= X.shape[1]

    def test_fit_transform(self, regression_data):
        X, y = regression_data
        pipe = Featransform(self._simple_config())
        result = pipe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)

    def test_transform_before_fit_raises(self, regression_data):
        X, _ = regression_data
        pipe = Featransform(self._simple_config())
        with pytest.raises(NotFittedError):
            pipe.transform(X)

    def test_output_has_no_nans(self, regression_data):
        X, y = regression_data
        X_nan = X.copy()
        X_nan.iloc[0, 0] = np.nan
        X_nan.iloc[5, 3] = np.nan
        pipe = Featransform(self._simple_config())
        result = pipe.fit_transform(X_nan, y)
        assert result.isnull().sum().sum() == 0

    def test_with_categorical_data(self, binary_data):
        X, y = binary_data
        X["cat_col"] = np.random.choice(["a", "b", "c"], len(X))
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                encoding_strategy=EncodingStrategy.LABEL,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        assert "cat_col" in result.columns
        assert result["cat_col"].dtype != object

    def test_datetime_extraction(self):
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(np.random.randn(n, 5), columns=[f"feat_{i}" for i in range(5)])
        X["date_col"] = pd.date_range("2020-01-01", periods=n, freq="D")
        y = pd.Series(np.random.randn(n), name="target")
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=True,
                drop_constant=False,
                drop_duplicates=False,
            ),
        )
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        # Original datetime column should be dropped, new temporal features added
        assert "date_col" not in result.columns
        assert any("date_col_year" in c for c in result.columns)
        assert any("date_col_month" in c for c in result.columns)


# ── Pipeline with Optimization ─────────────────────────────────────────


class TestPipelineOptimization:
    def test_fit_with_optimization(self, regression_data):
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})
            ],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
            optimization=OptimizationConfig(
                selection_strategy=SelectionStrategy.IMPORTANCE,
                n_iterations=3,
                validation_split=0.25,
                min_features=3,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        assert pipe._fitted
        history = pipe.get_optimization_history()
        assert len(history) > 0

    def test_optimization_history(self, regression_data):
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
            optimization=OptimizationConfig(
                n_iterations=3,
                min_features=3,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        importance = pipe.get_feature_importance()
        assert importance is not None
        assert len(importance) > 0

    def test_transformers_fit_on_full_data(self, regression_data):
        """With optimization configured, transformers are fit on the full
        dataset (the holdout is selection-only) rather than the ~75% train
        split used previously. Since 2.4.0 the fitted pipeline no longer keeps
        the training rows, so the evidence is the fitted scaler's sample count."""
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
            optimization=OptimizationConfig(
                n_iterations=3, min_features=3, validation_split=0.25,
                use_catboost=False, use_xgboost=True, n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        out = pipe.fit_transform(X, y)
        assert out.shape[0] == X.shape[0]
        assert pipe._scaler._scaler.n_samples_seen_ == X.shape[0]

    def test_transform_output_follows_schema_order(self, regression_data):
        """The filtered output columns must be a subsequence of the
        full engineered schema order, not importance rank."""
        X, y = regression_data

        def _config(with_opt):
            return PipelineConfig(
                anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
                clustering_models=[
                    ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})
                ],
                processing=ProcessingConfig(
                    imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
                ),
                optimization=(
                    OptimizationConfig(
                        n_iterations=5, min_features=4,
                        use_catboost=False, use_xgboost=True, n_estimators=10,
                    )
                    if with_opt
                    else None
                ),
            )

        # Full engineered order (no selection)
        full_order = list(Featransform(_config(False)).fit_transform(X, y).columns)

        # With optimization the output is a filtered subset — must keep relative order
        pipe = Featransform(_config(True))
        pipe.fit(X, y)
        out_cols = list(pipe.transform(X).columns)

        assert set(out_cols).issubset(set(full_order))
        positions = [full_order.index(c) for c in out_cols]
        assert positions == sorted(positions), "selected columns are out of schema order"

    def test_summary_best_matches_applied_selection(self, regression_data):
        """get_summary's reported best feature count must equal the number
        of features the pipeline actually applied."""
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
            optimization=OptimizationConfig(
                n_iterations=4, min_features=3,
                use_catboost=False, use_xgboost=True, n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        summary = pipe.get_summary()
        applied = pipe.transform(X).shape[1]
        assert summary["optimization"]["best_features"] == applied

    def test_summary_picks_best_under_min_direction(self, regression_data):
        """get_summary must pick the lowest MSE as 'best', not highest."""
        from featransform.core.enums import OptimizationMetric

        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
            optimization=OptimizationConfig(
                n_iterations=3,
                min_features=3,
                metric=OptimizationMetric.MSE,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        summary = pipe.get_summary()
        history = pipe.get_optimization_history()
        if len(history) > 1:
            min_score = min(h.score for h in history)
            assert summary["optimization"]["best_score"] == min_score


# ── Pipeline Schema ────────────────────────────────────────────────────


class TestPipelineSchema:
    def _config(self):
        return PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )

    def test_schema_stored_after_fit(self, regression_data):
        X, y = regression_data
        pipe = Featransform(self._config())
        pipe.fit(X, y)
        schema = pipe.get_pipeline_schema()
        assert schema is not None
        assert len(schema) > 0

    def test_transform_preserves_schema(self, regression_data):
        X, y = regression_data
        pipe = Featransform(self._config())
        pipe.fit(X, y)
        result = pipe.transform(X)
        schema = pipe.get_pipeline_schema()
        # All schema columns should be in the result
        for col in schema:
            assert col in result.columns


# ── Pipeline Summary ───────────────────────────────────────────────────


class TestPipelineSummary:
    def test_summary_after_fit(self, regression_data):
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        summary = pipe.get_summary()
        assert "features" in summary
        assert "transformations" in summary
        assert "optimization" in summary
        assert "components" in summary
        assert summary["features"]["final"] > 0

    def test_summary_before_fit_raises(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        with pytest.raises(NotFittedError):
            pipe.get_summary()


# ── get/set params ─────────────────────────────────────────────────────


class TestPipelineParams:
    def test_get_params(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        params = pipe.get_params()
        assert "config" in params
        assert params["config"] is config

    def test_set_params_rebuilds(self):
        config1 = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        config2 = PipelineConfig(
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
        )
        pipe = Featransform(config1)
        assert pipe.has_component("anomaly")
        pipe.set_params(config=config2)
        assert pipe.has_component("clustering")

    def test_set_params_resets_fit_state_and_warns(self, regression_data):
        """set_params on a fitted pipeline resets state and emits UserWarning."""
        import warnings as warnings_module

        X, y = regression_data
        config1 = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
        )
        config2 = PipelineConfig(
            clustering_models=[ModelConfig(model_family=ModelFamily.KMEANS)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
        )
        pipe = Featransform(config1)
        pipe.fit(X, y)
        assert pipe._fitted is True

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            pipe.set_params(config=config2)
            assert any(
                "fitted state was discarded" in str(w.message) for w in caught
            ), "Expected a UserWarning about discarded fitted state"

        # Pipeline must require refit
        assert pipe._fitted is False
        assert pipe._state is None
        assert pipe._fit_schema is None
        with pytest.raises(NotFittedError):
            pipe.transform(X)


# ── Task type honoring ─────────────────────────────────────────────────


class TestTaskTypeHonoring:
    def test_explicit_task_type_overrides_dtype_heuristic(self, binary_data):
        """config.task_type must override the dtype-based detection."""
        X, y = binary_data
        # Convert binary y to float (would otherwise be detected as regression by old code)
        y_float = y.astype("float64")
        config = PipelineConfig(
            task_type=TaskType.BINARY_CLASSIFICATION,
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
            optimization=OptimizationConfig(
                n_iterations=2,
                min_features=3,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y_float)
        # Selector must have detected (or been told) BINARY_CLASSIFICATION
        assert pipe._components["selector"]._task_type == TaskType.BINARY_CLASSIFICATION

    def test_float_binary_detected_as_classification(self, binary_data):
        """Without explicit task_type, 0.0/1.0 floats are still classification."""
        X, y = binary_data
        y_float = y.astype("float64")
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
            optimization=OptimizationConfig(
                n_iterations=2,
                min_features=3,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y_float)
        assert pipe._components["selector"]._task_type == TaskType.BINARY_CLASSIFICATION


# ── Datetime extraction ───────────────────────────────────────────────


class TestDatetimeExtraction:
    def test_date_only_omits_time_features(self):
        """Date-only inputs must NOT produce hour/minute/second columns."""
        n = 30
        df = pd.DataFrame(
            {
                "value": np.arange(n, dtype=float),
                "ts": pd.date_range("2020-01-01", periods=n, freq="D"),  # midnight
            }
        )
        y = pd.Series(np.arange(n, dtype=float))
        config = PipelineConfig(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=True,
                drop_constant=False,
                drop_duplicates=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(df, y)
        out = pipe.transform(df)
        cols = list(out.columns)
        assert "ts_year" in cols
        assert "ts_hour" not in cols
        assert "ts_minute" not in cols
        assert "ts_second" not in cols

    def test_with_time_keeps_time_features(self):
        """Inputs with non-zero time must produce hour/minute/second columns."""
        n = 30
        df = pd.DataFrame(
            {
                "value": np.arange(n, dtype=float),
                "ts": pd.date_range("2020-01-01 09:30:00", periods=n, freq="h"),
            }
        )
        y = pd.Series(np.arange(n, dtype=float))
        config = PipelineConfig(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=True,
                drop_constant=False,
                drop_duplicates=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(df, y)
        out = pipe.transform(df)
        assert "ts_hour" in out.columns

    def test_extraction_does_not_corrupt_dtypes(self, datetime_df):
        """Datetime extraction should not coerce other columns to object."""
        df = datetime_df.copy()
        # Pad to MIN_SAMPLES_REQUIRED
        df = pd.concat([df] * 2, ignore_index=True)
        y = pd.Series(np.arange(len(df), dtype=float))
        config = PipelineConfig(
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=True,
                drop_constant=False,
                drop_duplicates=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(df, y)
        out = pipe.transform(df)
        # 'value' column was numeric, must stay numeric
        assert pd.api.types.is_numeric_dtype(out["value"])


# ── Single-row transform inference ────────────────────────────────────


class TestSingleRowTransform:
    def test_transform_accepts_single_row(self, regression_data):
        """Fitted pipelines must accept 1-row inference frames."""
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
                drop_constant=False,
                drop_duplicates=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        out = pipe.transform(X.head(1))
        assert len(out) == 1

    def test_fit_still_requires_min_samples(self):
        """Fit-side validation still enforces MIN_SAMPLES_REQUIRED."""
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # < 5
        y = pd.Series([0.0, 1.0, 0.0])
        with pytest.raises(ValueError, match="at least"):
            pipe.fit(df, y)


# ── Float n_components rejection ──────────────────────────────────────


class TestFloatNComponentsRejection:
    def test_truncated_svd_rejects_float(self, regression_data):
        """Float n_components must be rejected for non-PCA strategies."""
        from featransform.core.exceptions import ConfigurationError

        X, y = regression_data
        config = PipelineConfig(
            dimensionality_models=[
                ModelConfig(
                    model_family=ModelFamily.TRUNCATED_SVD,
                    parameters={"n_components": 0.95},
                )
            ],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
        )
        pipe = Featransform(config)
        with pytest.raises(ConfigurationError, match="n_components must be an integer"):
            pipe.fit(X, y)


# ── No module-level basicConfig ───────────────────────────────────────


class TestLoggingHygiene:
    def test_no_basicconfig_at_import(self):
        """featransform must not call logging.basicConfig at import time."""
        import importlib
        import logging

        # Reset any pre-existing handlers before re-importing
        root = logging.getLogger()
        prior_handlers = list(root.handlers)
        for h in prior_handlers:
            root.removeHandler(h)

        try:
            importlib.import_module("featransform")
            # If basicConfig had been called, root would now have handlers
            assert (
                len(root.handlers) == 0
            ), f"featransform installed root handlers: {root.handlers}"
        finally:
            for h in prior_handlers:
                root.addHandler(h)


# ── Preprocessing persistence ─────────────────────────────────────────


class TestPreprocessingPersistence:
    def _basic_config(self):
        return PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
                drop_constant=True,
                drop_duplicates=True,
            ),
        )

    def test_drop_constant_preserves_dtypes(self, regression_data):
        """drop_duplicates must not coerce numeric dtypes to object."""
        X, y = regression_data
        pipe = Featransform(self._basic_config())
        pipe.fit(X, y)
        result = pipe.transform(X)
        # All numeric columns must remain numeric (object dtype indicates corruption)
        assert not any(result.dtypes == object), f"object cols leaked: {result.dtypes}"

    def test_drop_constant_persisted(self, regression_data):
        """Constant cols learned at fit must be applied at transform."""
        X, y = regression_data
        X_with_constant = X.copy()
        X_with_constant["always_zero"] = 0.0
        pipe = Featransform(self._basic_config())
        pipe.fit(X_with_constant, y)
        # The constant col must be in the dropped list
        assert "always_zero" in pipe._constant_cols_dropped
        # When transforming a frame where 'always_zero' has variance, the col is still dropped
        X_test = X.copy()
        X_test["always_zero"] = np.arange(len(X_test), dtype=float)
        result = pipe.transform(X_test)
        assert "always_zero" not in result.columns

    def test_drop_duplicates_persisted(self, regression_data):
        """Duplicate cols learned at fit must be applied at transform."""
        X, y = regression_data
        X_with_dup = X.copy()
        X_with_dup["dup_of_0"] = X_with_dup["feat_0"].copy()
        pipe = Featransform(self._basic_config())
        pipe.fit(X_with_dup, y)
        # One of the two duplicates is dropped
        dropped = pipe._duplicate_cols_dropped
        assert "dup_of_0" in dropped or "feat_0" in dropped

    def test_schema_validated_unconditionally(self, regression_data):
        """Schema alignment runs even without a selector."""
        X, y = regression_data
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN, handle_datetime=False
            ),
            # No optimization → no selector
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        # Schema is stored even without selector
        assert pipe.get_pipeline_schema() is not None
        # Transform output must match training schema exactly
        out = pipe.transform(X)
        assert list(out.columns) == pipe.get_pipeline_schema()
