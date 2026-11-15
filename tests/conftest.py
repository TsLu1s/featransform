import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture
def numeric_df():
    """Simple numeric DataFrame for basic tests."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(100, 5),
        columns=[f"feat_{i}" for i in range(5)],
    )


@pytest.fixture
def numeric_df_with_missing(numeric_df):
    """Numeric DataFrame with NaN values."""
    df = numeric_df.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(df.shape) < 0.1
    df[mask] = np.nan
    return df


@pytest.fixture
def categorical_df():
    """DataFrame with categorical columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "num_a": np.random.randn(n),
            "num_b": np.random.randn(n),
            "cat_a": np.random.choice(["x", "y", "z"], n),
            "cat_b": np.random.choice(["alpha", "beta"], n),
        }
    )


@pytest.fixture
def datetime_df():
    """DataFrame with datetime columns."""
    n = 50
    return pd.DataFrame(
        {
            "value": np.random.randn(n),
            "ts": pd.date_range("2020-01-01", periods=n, freq="D"),
        }
    )


@pytest.fixture
def regression_data():
    """Regression dataset (X, y)."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=6, noise=5, random_state=42
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def binary_data():
    """Binary classification dataset (X, y)."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def multiclass_data():
    """Multiclass classification dataset (X, y)."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y, name="target")
    return X, y


@pytest.fixture
def small_numeric_df():
    """Small numeric DataFrame (just above MIN_SAMPLES_REQUIRED=5)."""
    np.random.seed(42)
    return pd.DataFrame(
        np.random.randn(10, 4),
        columns=[f"f_{i}" for i in range(4)],
    )
