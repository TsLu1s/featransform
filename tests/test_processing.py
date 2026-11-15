"""Tests for processing module: imputation and encoding."""

import numpy as np
import pandas as pd
import pytest

from featransform.core.enums import EncodingStrategy, ImputationStrategy
from featransform.core.exceptions import NotFittedError
from featransform.processing.datetime_features import DatetimeExpander
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.imputation import CoreImputer


# ── CoreImputer ────────────────────────────────────────────────────────


class TestCoreImputer:
    def test_fit_transform_no_missing(self, numeric_df):
        imp = CoreImputer(strategy=ImputationStrategy.MEAN)
        result = imp.fit(numeric_df).transform(numeric_df)
        assert result.shape == numeric_df.shape
        assert result.isnull().sum().sum() == 0

    def test_fit_transform_with_missing(self, numeric_df_with_missing):
        imp = CoreImputer(strategy=ImputationStrategy.MEAN)
        result = imp.fit(numeric_df_with_missing).transform(numeric_df_with_missing)
        assert result.isnull().sum().sum() == 0

    @pytest.mark.parametrize(
        "strategy",
        [
            ImputationStrategy.MEAN,
            ImputationStrategy.MEDIAN,
            ImputationStrategy.MODE,
            ImputationStrategy.ITERATIVE,
            ImputationStrategy.KNN,
        ],
    )
    def test_all_numeric_strategies(self, numeric_df_with_missing, strategy):
        imp = CoreImputer(strategy=strategy)
        result = imp.fit(numeric_df_with_missing).transform(numeric_df_with_missing)
        assert result.isnull().sum().sum() == 0

    def test_categorical_imputation(self):
        df = pd.DataFrame(
            {
                "num": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
                "cat": ["a", "b", None, "a", "b", "a"],
            }
        )
        imp = CoreImputer(strategy=ImputationStrategy.MEAN)
        result = imp.fit(df).transform(df)
        assert result["cat"].isnull().sum() == 0
        assert result["num"].isnull().sum() == 0

    def test_transform_before_fit_raises(self, numeric_df):
        imp = CoreImputer()
        with pytest.raises(NotFittedError):
            imp.transform(numeric_df)

    def test_get_set_params(self):
        imp = CoreImputer(strategy=ImputationStrategy.KNN)
        params = imp.get_params()
        assert params["strategy"] == ImputationStrategy.KNN
        imp.set_params(strategy=ImputationStrategy.MEAN)
        assert imp.strategy == ImputationStrategy.MEAN

    def test_preserves_index(self, numeric_df_with_missing):
        df = numeric_df_with_missing.copy()
        df.index = range(100, 200)
        imp = CoreImputer(strategy=ImputationStrategy.MEAN)
        result = imp.fit(df).transform(df)
        assert list(result.index) == list(df.index)

    def test_categorical_mode_uses_fit_value(self):
        """Categorical mode must come from fit data, not transform data."""
        train = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat": ["A", "A", "A", "B", "B", "C"],  # mode is 'A'
            }
        )
        # Test data has DIFFERENT mode ('Z')
        test = pd.DataFrame(
            {
                "num": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "cat": ["Z", "Z", "Z", "Z", None, "Z"],  # transform-mode would be 'Z'
            }
        )
        imp = CoreImputer(strategy=ImputationStrategy.MEAN).fit(train)
        result = imp.transform(test)
        # Missing value must be filled with FIT mode ('A'), not transform mode ('Z')
        assert result.loc[4, "cat"] == "A"


# ── CategoricalEncoder ────────────────────────────────────────────────


class TestCategoricalEncoder:
    def test_fit_transform_label(self, categorical_df):
        enc = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        result = enc.fit(categorical_df).transform(categorical_df)
        assert result["cat_a"].dtype != object
        assert result["cat_b"].dtype != object
        assert result["num_a"].equals(categorical_df["num_a"])

    def test_unknown_categories_get_minus_one(self):
        train = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat": ["a", "b", "c", "a", "b", "c"],
            }
        )
        test = pd.DataFrame(
            {
                "num": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "cat": ["a", "b", "UNKNOWN", "a", "NEW", "c"],
            }
        )
        enc = CategoricalEncoder()
        enc.fit(train)
        result = enc.transform(test)
        assert (result["cat"] == -1).sum() == 2

    def test_transform_before_fit_raises(self, categorical_df):
        enc = CategoricalEncoder()
        with pytest.raises(NotFittedError):
            enc.transform(categorical_df)

    def test_no_categorical_columns(self, numeric_df):
        enc = CategoricalEncoder()
        result = enc.fit(numeric_df).transform(numeric_df)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_get_params(self):
        enc = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
        params = enc.get_params()
        assert params["strategy"] == EncodingStrategy.LABEL

    def test_transform_skips_missing_columns(self):
        """Encoder must not KeyError when an expected col is absent at transform."""
        train = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "cat": ["a", "b", "c", "a", "b", "c"],
            }
        )
        # Test frame is missing "cat" entirely
        test = pd.DataFrame({"num": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]})
        enc = CategoricalEncoder()
        enc.fit(train)
        result = enc.transform(test)
        assert "num" in result.columns
        assert "cat" not in result.columns

    def test_transform_validates_input_type(self):
        """Non-DataFrame inputs should raise TypeError, not pass through."""
        train = pd.DataFrame(
            {"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "cat": ["a", "b", "c", "a", "b", "c"]}
        )
        enc = CategoricalEncoder().fit(train)
        with pytest.raises(TypeError, match="Expected DataFrame"):
            enc.transform([1, 2, 3])

    def test_fit_does_not_emit_select_dtypes_deprecation(self):
        """L2 — categorical detection must not raise the object-includes-str
        Pandas4Warning on object-dtype columns."""
        import warnings

        df = pd.DataFrame(
            {"num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "cat": ["a", "b", "c", "a", "b", "c"]}
        )
        with warnings.catch_warnings():
            # Pandas4Warning derives from DeprecationWarning
            warnings.simplefilter("error", category=DeprecationWarning)
            CategoricalEncoder().fit(df)


# ── Validation: imputation also validates input type ───────────────────


class TestImputationValidation:
    def test_transform_validates_input_type(self, numeric_df):
        """Non-DataFrame inputs should raise TypeError."""
        imp = CoreImputer(strategy=ImputationStrategy.MEAN).fit(numeric_df)
        with pytest.raises(TypeError, match="Expected DataFrame"):
            imp.transform(np.array([[1, 2, 3]]))


# ── DatetimeExpander ───────────────────────────────────────────────────


class TestDatetimeExpander:
    def test_replaces_datetime_with_numeric_features(self, datetime_df):
        result = DatetimeExpander().fit(datetime_df).transform(datetime_df)

        assert "ts" not in result.columns
        assert result.select_dtypes(include=["datetime64"]).empty
        assert "value" in result.columns
        for part in ("year", "month", "day", "dayofweek", "quarter", "is_weekend"):
            assert f"ts_{part}" in result.columns
        assert {"ts_month_sin", "ts_month_cos", "ts_day_sin", "ts_day_cos"} <= set(result.columns)

    def test_output_is_sklearn_consumable(self, datetime_df):
        """Regression: datetime dtypes used to reach sklearn and raise DTypePromotionError."""
        from sklearn.ensemble import IsolationForest

        result = DatetimeExpander().fit(datetime_df).transform(datetime_df)
        IsolationForest(n_estimators=10, random_state=42).fit(result)

    def test_date_only_column_omits_time_parts(self, datetime_df):
        result = DatetimeExpander().fit(datetime_df).transform(datetime_df)
        assert "ts_hour" not in result.columns

    def test_time_component_emitted_when_present(self):
        df = pd.DataFrame(
            {
                "value": np.arange(24, dtype=float),
                "ts": pd.date_range("2020-01-01", periods=24, freq="h"),
            }
        )
        result = DatetimeExpander().fit(df).transform(df)
        for part in ("hour", "minute", "second"):
            assert f"ts_{part}" in result.columns

    def test_schema_is_stable_between_fit_and_transform(self):
        """A midnight-only test split must not lose the fitted hour/minute/second columns."""
        train = pd.DataFrame(
            {
                "value": np.arange(24, dtype=float),
                "ts": pd.date_range("2020-01-01", periods=24, freq="h"),
            }
        )
        test = pd.DataFrame(
            {
                "value": np.arange(10, dtype=float),
                "ts": pd.date_range("2021-01-01", periods=10, freq="D"),
            }
        )
        expander = DatetimeExpander().fit(train)
        assert list(expander.transform(train).columns) == list(expander.transform(test).columns)

    def test_no_datetime_columns_is_passthrough(self, numeric_df):
        result = DatetimeExpander().fit(numeric_df).transform(numeric_df)
        pd.testing.assert_frame_equal(result, numeric_df)

    def test_does_not_mutate_input(self, datetime_df):
        original = datetime_df.copy()
        DatetimeExpander().fit(datetime_df).transform(datetime_df)
        pd.testing.assert_frame_equal(datetime_df, original)

    def test_transform_before_fit_raises(self, datetime_df):
        with pytest.raises(NotFittedError):
            DatetimeExpander().transform(datetime_df)

    def test_get_feature_names_reports_source_columns(self, datetime_df):
        assert DatetimeExpander().fit(datetime_df).get_feature_names() == ["ts"]
