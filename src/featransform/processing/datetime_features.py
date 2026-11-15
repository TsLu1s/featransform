from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from featransform.core.base import BaseTransformer


class DatetimeExpander(BaseTransformer):
    """Expand datetime columns into numeric temporal features.

    Datetime dtypes cannot be consumed by the scikit-learn estimators behind the
    anomaly / clustering / dimensionality components, so they are replaced by
    calendar parts plus cyclic encodings and the originals are dropped.
    Timezone-aware columns are expanded in their own local time. Timedelta
    columns are converted in place to their duration in seconds.

    Which columns are expanded — and whether each carries a time-of-day
    component — is learned at ``fit`` and replayed at ``transform``, so a train
    and a test frame always yield the same schema even when the test split
    happens to hold midnight-only timestamps. Every emitted feature is
    ``float64`` and a missing timestamp (``NaT``) yields ``NaN`` in each of its
    features, so the output dtypes never depend on whether a frame had gaps.
    """

    # Class-level default keeps pipelines pickled before 2.4.0 loadable.
    _timedelta_columns: Sequence[str] = ()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._datetime_columns: List[str] = []
        self._timedelta_columns = []
        self._has_time: Dict[str, bool] = {}

    @staticmethod
    def _column_has_time(dt) -> bool:
        """Whether a datetime accessor carries a non-zero time component."""
        return bool(
            (dt.hour.fillna(0) != 0).any()
            or (dt.minute.fillna(0) != 0).any()
            or (dt.second.fillna(0) != 0).any()
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DatetimeExpander":
        self._validate_fit_input(X)

        # Per-dtype checks cover naive and timezone-aware datetimes on every
        # supported pandas version.
        self._datetime_columns = [
            col for col in X.columns if pd.api.types.is_datetime64_any_dtype(X[col].dtype)
        ]
        self._timedelta_columns = [
            col for col in X.columns if pd.api.types.is_timedelta64_dtype(X[col].dtype)
        ]
        # Learn the hour/minute/second decision once so transform stays schema-stable.
        self._has_time = {col: self._column_has_time(X[col].dt) for col in self._datetime_columns}

        self._fitted = True
        return self

    @staticmethod
    def _part(values) -> pd.Series:
        """A calendar part as float64 (NaT → NaN)."""
        return pd.Series(values).astype("float64")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        self._validate_input(X)

        result = X.copy()

        for col in self._timedelta_columns:
            if col in result.columns and pd.api.types.is_timedelta64_dtype(result[col].dtype):
                result[col] = result[col].dt.total_seconds()

        cols = [c for c in self._datetime_columns if c in X.columns]
        if not cols:
            return result

        new_features: Dict[str, Any] = {}
        for col in cols:
            dt = X[col].dt
            day_of_week = self._part(dt.day_of_week)
            month = self._part(dt.month)
            day = self._part(dt.day)

            # Core temporal features
            new_features[f"{col}_year"] = self._part(dt.year)
            new_features[f"{col}_month"] = month
            new_features[f"{col}_day"] = day
            new_features[f"{col}_dayofweek"] = day_of_week
            new_features[f"{col}_dayofyear"] = self._part(dt.day_of_year)
            new_features[f"{col}_quarter"] = self._part(dt.quarter)
            new_features[f"{col}_weekofyear"] = self._part(dt.isocalendar().week)
            new_features[f"{col}_is_weekend"] = (
                (day_of_week >= 5).astype("float64").where(day_of_week.notna())
            )

            # Emit hour/minute/second only when the fitted column had them
            if self._has_time.get(col, False):
                new_features[f"{col}_hour"] = self._part(dt.hour)
                new_features[f"{col}_minute"] = self._part(dt.minute)
                new_features[f"{col}_second"] = self._part(dt.second)

            # Cyclic encodings for periodicity. Day-of-month is normalized by the
            # column's actual month length so 28/29/30/31-day months each
            # close the cycle correctly instead of distorting against a fixed 31.
            new_features[f"{col}_month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
            new_features[f"{col}_month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
            days_in_month = self._part(dt.days_in_month)
            new_features[f"{col}_day_sin"] = np.sin(2 * np.pi * (day - 1) / days_in_month)
            new_features[f"{col}_day_cos"] = np.cos(2 * np.pi * (day - 1) / days_in_month)

        return pd.concat(
            [
                result.drop(columns=cols),
                pd.DataFrame(new_features, index=X.index),
            ],
            axis=1,
        )

    def get_feature_names(self) -> List[str]:
        """Datetime columns consumed by this expander."""
        return list(self._datetime_columns)

    def get_converted_columns(self) -> List[str]:
        """Every column this expander rewrites: datetimes plus timedeltas."""
        return list(self._datetime_columns) + list(self._timedelta_columns)

    def get_params(self) -> Dict[str, Any]:
        return {**self._params}

    def set_params(self, **params) -> "DatetimeExpander":
        for key, value in params.items():
            setattr(self, key, value)
        return self
