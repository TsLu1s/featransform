"""Tests for optimization module: evaluator and selector."""

import numpy as np
import pandas as pd
import pytest

from featransform.core.enums import OptimizationMetric, SelectionStrategy, TaskType
from featransform.core.exceptions import NotFittedError
from featransform.optimization.evaluator import (
    METRIC_DIRECTION,
    BaseEvaluator,
    CatBoostImportance,
    XGBoostImportance,
    detect_task_type,
)
from featransform.optimization.selector import FeatureSelector


# ── Importance Calculators ─────────────────────────────────────────────


class TestCatBoostImportance:
    def test_regression(self, regression_data):
        X, y = regression_data
        calc = CatBoostImportance(n_estimators=10, random_state=42)
        result = calc.calculate(X, y)
        assert "Feature Id" in result.columns
        assert "Importances" in result.columns
        assert len(result) == X.shape[1]

    def test_binary_classification(self, binary_data):
        X, y = binary_data
        calc = CatBoostImportance(n_estimators=10, random_state=42)
        result = calc.calculate(X, y)
        assert len(result) == X.shape[1]

    def test_multiclass_classification(self, multiclass_data):
        X, y = multiclass_data
        calc = CatBoostImportance(n_estimators=10, random_state=42)
        result = calc.calculate(X, y)
        assert len(result) == X.shape[1]


class TestXGBoostImportance:
    def test_regression(self, regression_data):
        X, y = regression_data
        calc = XGBoostImportance(n_estimators=10, random_state=42)
        result = calc.calculate(X, y)
        assert "Feature Id" in result.columns
        assert "Importances" in result.columns
        assert len(result) == X.shape[1]

    def test_binary_classification(self, binary_data):
        X, y = binary_data
        calc = XGBoostImportance(n_estimators=10, random_state=42)
        result = calc.calculate(X, y)
        assert len(result) == X.shape[1]


# ── BaseEvaluator ──────────────────────────────────────────────────────


class TestBaseEvaluator:
    def test_detect_task_type_regression(self):
        class ConcreteEvaluator(BaseEvaluator):
            pass

        ev = ConcreteEvaluator()
        y = pd.Series([1.0, 2.5, 3.3])
        assert ev._detect_task_type(y) == TaskType.REGRESSION

    def test_detect_task_type_binary(self):
        class ConcreteEvaluator(BaseEvaluator):
            pass

        ev = ConcreteEvaluator()
        y = pd.Series([0, 1, 0, 1])
        assert ev._detect_task_type(y) == TaskType.BINARY_CLASSIFICATION

    def test_detect_task_type_multiclass(self):
        class ConcreteEvaluator(BaseEvaluator):
            pass

        ev = ConcreteEvaluator()
        y = pd.Series([0, 1, 2, 3])
        assert ev._detect_task_type(y) == TaskType.MULTICLASS_CLASSIFICATION

    def test_evaluate_subset_regression(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector._task_type = TaskType.REGRESSION
        selector.metric = OptimizationMetric.MSE
        selector._fitted = True
        score = selector.evaluate_subset(X[:150], y[:150], X[150:], y[150:])
        assert isinstance(score, float)
        # MSE is now reported in its natural orientation (non-negative)
        assert score >= 0.0

    def test_evaluate_subset_classification(self, binary_data):
        X, y = binary_data
        selector = FeatureSelector(
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector._task_type = TaskType.BINARY_CLASSIFICATION
        selector.metric = OptimizationMetric.F1
        selector._fitted = True
        score = selector.evaluate_subset(X[:150], y[:150], X[150:], y[150:])
        assert 0.0 <= score <= 1.0


# ── FeatureSelector ────────────────────────────────────────────────────


class TestFeatureSelector:
    def test_fit_regression(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=True,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X, y)
        assert selector._fitted
        scores = selector.get_feature_scores()
        assert len(scores) == X.shape[1]

    def test_fit_classification(self, binary_data):
        X, y = binary_data
        selector = FeatureSelector(
            use_catboost=True,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X, y)
        assert selector._fitted

    def test_select_by_threshold(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(
            min_features=3,
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X, y)
        selected = selector.select_by_threshold(0.8)
        assert len(selected) >= 3  # respects min_features

    def test_transform_returns_selected(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X, y)
        selector.set_selected_features(["feat_0", "feat_1"])
        result = selector.transform(X)
        assert list(result.columns) == ["feat_0", "feat_1"]

    def test_transform_without_selection_returns_all(self, regression_data):
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X, y)
        result = selector.transform(X)
        assert result.shape == X.shape  # no selection applied yet

    def test_transform_before_fit_raises(self, regression_data):
        X, _ = regression_data
        selector = FeatureSelector()
        with pytest.raises(NotFittedError):
            selector.transform(X)

    def test_validate_task_type_string(self):
        selector = FeatureSelector(task_type="regression", random_state=42)
        assert selector._task_type == TaskType.REGRESSION

    def test_validate_task_type_enum(self):
        selector = FeatureSelector(
            task_type=TaskType.BINARY_CLASSIFICATION, random_state=42
        )
        assert selector._task_type == TaskType.BINARY_CLASSIFICATION

    def test_validate_task_type_invalid(self):
        with pytest.raises(ValueError, match="Invalid task_type"):
            FeatureSelector(task_type="invalid_task")

    def test_optimize(self, regression_data):
        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        selector = FeatureSelector(
            min_features=3,
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X_train, y_train)

        best_features, best_score, history = selector.optimize(
            X_train, y_train, X_val, y_val, thresholds=[0.9, 0.8, 0.7]
        )
        # History may have <= len(thresholds) entries since duplicate subsets are deduped
        assert 1 <= len(history) <= 3
        assert len(best_features) >= 3
        assert isinstance(best_score, float)
        # No two history entries should evaluate the same feature subset
        subsets = [tuple(sorted(h["features"])) for h in history]
        assert len(set(subsets)) == len(subsets)

    def test_get_params(self):
        selector = FeatureSelector(
            strategy=SelectionStrategy.IMPORTANCE,
            min_features=5,
            n_estimators=20,
            random_state=42,
        )
        params = selector.get_params()
        assert params["strategy"] == SelectionStrategy.IMPORTANCE
        assert params["min_features"] == 5
        assert params["n_estimators"] == 20

    def test_default_min_features_matches_config(self):
        """FeatureSelector default must match OptimizationConfig default (10)."""
        selector = FeatureSelector()
        assert selector.min_features == 10


# ── Metric direction ───────────────────────────────────────────────────


class TestMetricDirection:
    def test_direction_map_covers_all_metrics(self):
        """Every OptimizationMetric must declare a direction."""
        for metric in OptimizationMetric:
            assert metric in METRIC_DIRECTION

    def test_direction_for_regression_losses_is_min(self):
        """Losses (MAE/MSE/RMSE) must be 'min'."""
        assert METRIC_DIRECTION[OptimizationMetric.MAE] == "min"
        assert METRIC_DIRECTION[OptimizationMetric.MSE] == "min"
        assert METRIC_DIRECTION[OptimizationMetric.RMSE] == "min"

    def test_direction_for_scores_is_max(self):
        """Scores (R2/F1/Accuracy/etc.) must be 'max'."""
        for m in (
            OptimizationMetric.R2,
            OptimizationMetric.ACCURACY,
            OptimizationMetric.PRECISION,
            OptimizationMetric.RECALL,
            OptimizationMetric.F1,
            OptimizationMetric.AUC_ROC,
        ):
            assert METRIC_DIRECTION[m] == "max"

    def test_is_better_respects_direction_for_min(self, regression_data):
        """For MSE (min), lower scores must beat higher ones."""
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector.metric = OptimizationMetric.MSE
        assert selector.is_better(0.5, 0.8) is True
        assert selector.is_better(1.0, 0.5) is False

    def test_is_better_respects_direction_for_max(self):
        """For F1 (max), higher scores must beat lower ones."""
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector.metric = OptimizationMetric.F1
        assert selector.is_better(0.9, 0.7) is True
        assert selector.is_better(0.6, 0.9) is False

    def test_mse_is_non_negative(self, regression_data):
        """MSE/MAE/RMSE are now reported in natural orientation."""
        X, y = regression_data
        for metric in (OptimizationMetric.MSE, OptimizationMetric.MAE, OptimizationMetric.RMSE):
            selector = FeatureSelector(
                use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
            )
            selector._task_type = TaskType.REGRESSION
            selector.metric = metric
            selector._fitted = True
            score = selector.evaluate_subset(X[:150], y[:150], X[150:], y[150:])
            assert score >= 0.0, f"{metric} should be non-negative, got {score}"


# ── Selector hardening ────────────────────────────────────────────────


class TestSelectorHardening:
    def test_set_selected_features_validates_unknown(self, regression_data):
        """set_selected_features must reject features not seen at fit."""
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector.fit(X, y)
        with pytest.raises(ValueError, match="Unknown features"):
            selector.set_selected_features(["feat_0", "feat_does_not_exist"])

    def test_set_selected_features_requires_fit(self, regression_data):
        """set_selected_features must require fit first."""
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        with pytest.raises(NotFittedError):
            selector.set_selected_features(["feat_0"])

    def test_selected_features_kept_in_schema_order(self, regression_data):
        """set_selected_features stores selection in fit-time column order,
        not in the (arbitrary) order it was passed."""
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector.fit(X, y)
        # Pass features deliberately out of order
        selector.set_selected_features(["feat_7", "feat_1", "feat_3", "feat_0"])
        # Stored / transformed order must follow X.columns order
        assert selector.get_selected_features() == ["feat_0", "feat_1", "feat_3", "feat_7"]
        assert list(selector.transform(X).columns) == ["feat_0", "feat_1", "feat_3", "feat_7"]

    def test_selection_order_independent_of_column_names(self):
        """Ordering anchors on fit-time position, never on parsing the
        column name (works for arbitrary user-supplied names with no index)."""
        np.random.seed(0)
        cols = ["zeta", "alpha", "mike", "bravo", "yankee"]  # no parseable index
        X = pd.DataFrame(np.random.randn(120, 5), columns=cols)
        y = pd.Series(np.random.randn(120))
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector._task_type = TaskType.REGRESSION
        selector.fit(X, y)
        selector.set_selected_features(["yankee", "alpha", "mike"])
        # Order follows fit position (zeta=0, alpha=1, mike=2, bravo=3, yankee=4)
        assert selector.get_selected_features() == ["alpha", "mike", "yankee"]

    def test_random_state_promoted_to_named_param(self):
        """random_state is a real attribute, not pulled from _params."""
        selector = FeatureSelector(random_state=123)
        assert selector.random_state == 123

    def test_eval_model_clones_per_call(self, regression_data):
        """evaluate_subset must use a fresh clone each time."""
        X, y = regression_data
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector._task_type = TaskType.REGRESSION
        selector.metric = OptimizationMetric.MSE
        selector._fitted = True
        # Two calls in sequence on different subsets must not crash; the
        # second call cannot inherit booster state from the first.
        score1 = selector.evaluate_subset(X[:150], y[:150], X[150:], y[150:])
        score2 = selector.evaluate_subset(
            X[:150][["feat_0", "feat_1", "feat_2"]],
            y[:150],
            X[150:][["feat_0", "feat_1", "feat_2"]],
            y[150:],
        )
        assert isinstance(score1, float)
        assert isinstance(score2, float)

    def test_optimize_dedupes_subsets(self, regression_data):
        """Duplicate selected subsets across thresholds are skipped."""
        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]
        selector = FeatureSelector(
            min_features=10,  # forces all 10 features for any threshold
            use_catboost=False,
            use_xgboost=True,
            n_estimators=10,
            random_state=42,
        )
        selector.fit(X_train, y_train)
        # All thresholds will collapse to the same all-10-features set
        _, _, history = selector.optimize(
            X_train, y_train, X_val, y_val, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        assert len(history) == 1, f"expected 1 deduped entry, got {len(history)}"


class TestEvaluatorStringTargets:
    def test_string_classification_targets(self, binary_data):
        """Evaluator handles string targets via internal LabelEncoder."""
        X, y = binary_data
        y_str = y.map({0: "neg", 1: "pos"})
        selector = FeatureSelector(
            use_catboost=False, use_xgboost=True, n_estimators=10, random_state=42
        )
        selector._task_type = TaskType.BINARY_CLASSIFICATION
        selector.metric = OptimizationMetric.F1
        selector._fitted = True
        score = selector.evaluate_subset(X[:150], y_str[:150], X[150:], y_str[150:])
        assert 0.0 <= score <= 1.0


class TestTaskTypeDetection:
    """Single source of truth for task-type resolution."""

    def test_configured_task_type_always_wins(self):
        y = pd.Series(np.arange(200))  # would be regression by heuristic
        assert detect_task_type(y, TaskType.MULTICLASS_CLASSIFICATION) == (
            TaskType.MULTICLASS_CLASSIFICATION
        )

    def test_integer_high_cardinality_is_regression(self):
        """An integer target with many distinct values is regression."""
        y = pd.Series(np.arange(200))
        assert detect_task_type(y) == TaskType.REGRESSION

    def test_few_integer_classes_stay_multiclass(self):
        assert detect_task_type(pd.Series([0, 1, 2, 3])) == TaskType.MULTICLASS_CLASSIFICATION

    def test_integer_cardinality_boundary(self):
        """The integer/regression cutoff is MAX_CLASSIFICATION_CLASSES."""
        from featransform.core.constants import MAX_CLASSIFICATION_CLASSES

        at_limit = pd.Series(np.arange(MAX_CLASSIFICATION_CLASSES))
        over_limit = pd.Series(np.arange(MAX_CLASSIFICATION_CLASSES + 1))
        assert detect_task_type(at_limit) == TaskType.MULTICLASS_CLASSIFICATION
        assert detect_task_type(over_limit) == TaskType.REGRESSION

    def test_string_target_is_classification(self):
        assert detect_task_type(pd.Series(["a", "b", "c", "a"])) == (
            TaskType.MULTICLASS_CLASSIFICATION
        )

    def test_float_binary_is_classification(self):
        assert detect_task_type(pd.Series([0.0, 1.0, 0.0, 1.0])) == (
            TaskType.BINARY_CLASSIFICATION
        )

    def test_float_continuous_is_regression(self):
        assert detect_task_type(pd.Series([1.0, 2.5, 3.3])) == TaskType.REGRESSION

    def test_calculators_honor_resolved_task_type(self):
        """Importance calculators must use the resolved task type, not
        independently re-derive it (an integer regression target must not be
        treated as a high-cardinality classification problem)."""
        np.random.seed(0)
        X = pd.DataFrame(np.random.randn(120, 4), columns=list("abcd"))
        y = pd.Series(np.arange(120))  # integer regression
        for calc in (
            CatBoostImportance(n_estimators=10, random_state=42),
            XGBoostImportance(n_estimators=10, random_state=42),
        ):
            result = calc.calculate(X, y, TaskType.REGRESSION)
            assert len(result) == X.shape[1]


class TestOptimizationIntegrity:
    """The applied selection matches the reported best."""

    def test_full_feature_set_competes(self):
        """When the full set is genuinely best, optimize must keep it
        rather than returning a worse subset."""
        np.random.seed(0)
        cols = [f"f{i}" for i in range(8)]
        Xtr = pd.DataFrame(np.random.randn(200, 8), columns=cols)
        ytr = pd.Series(Xtr.values.sum(axis=1) + 0.01 * np.random.randn(200))
        Xv = pd.DataFrame(np.random.randn(80, 8), columns=cols)
        yv = pd.Series(Xv.values.sum(axis=1) + 0.01 * np.random.randn(80))
        selector = FeatureSelector(
            min_features=2, use_catboost=False, use_xgboost=True,
            n_estimators=20, random_state=42, metric=OptimizationMetric.MSE,
        )
        selector._task_type = TaskType.REGRESSION
        selector.fit(Xtr, ytr)
        # thresholds never reach 1.0 -> the full set is only reachable via the seed
        best_features, best_score, _ = selector.optimize(
            Xtr, ytr, Xv, yv, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]
        )
        assert len(best_features) == 8

    def test_threshold_one_selects_all_features(self):
        """Float cumsum rounding must not drop the last feature at 1.0."""
        np.random.seed(1)
        X = pd.DataFrame(np.random.randn(120, 6), columns=[f"f{i}" for i in range(6)])
        y = pd.Series(X.values @ np.arange(1, 7) + np.random.randn(120))
        selector = FeatureSelector(
            min_features=1, use_catboost=False, use_xgboost=True,
            n_estimators=20, random_state=42,
        )
        selector._task_type = TaskType.REGRESSION
        selector.fit(X, y)
        assert len(selector.select_by_threshold(1.0)) == 6


class TestThresholdSweep:
    def test_sweep_covers_meaningful_range(self, regression_data):
        """Pipeline-level sweep covers 0.5 to 1.0, not 0.91 to 1.0."""
        from featransform.configs.baseline import FTconfig
        from featransform.pipeline import Featransform

        X, y = regression_data
        config = FTconfig.standard()
        config.optimization.n_iterations = 5
        config.optimization.use_catboost = False
        pipe = Featransform(config)
        pipe.fit(X, y)
        history = pipe.get_optimization_history()
        # The non-baseline (iteration >= 0) entries reflect the sweep
        sweep_entries = [h for h in history if h.iteration >= 0]
        # Different thresholds must produce different feature counts, on average
        n_feats = [h.n_features for h in sweep_entries]
        assert len(set(n_feats)) >= 2, f"sweep produced no diversity: {n_feats}"
