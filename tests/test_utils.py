"""Tests for utility modules: data_generator and serializer."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from featransform.core.enums import ImputationStrategy, ModelFamily
from featransform.core.models import ModelConfig, PipelineConfig, ProcessingConfig
from featransform.pipeline import Featransform
from featransform.utils.data_generator import DatasetGenerator, make_dataset
from featransform.utils.serializer import PipelineSerializer


# ── DatasetGenerator ───────────────────────────────────────────────────


class TestDatasetGenerator:
    @pytest.mark.parametrize(
        "task",
        ["binary_classification", "multiclass_classification", "regression"],
    )
    def test_generate_all_tasks(self, task):
        X, y = DatasetGenerator.generate(
            task=task, n_samples=100, n_features=10, n_informative=6, n_redundant=2,
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == 100

    def test_generate_with_datetime(self):
        X, y = DatasetGenerator.generate(
            task="binary_classification",
            n_samples=50,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            add_datetime=True,
            n_datetime_cols=2,
        )
        dt_cols = X.select_dtypes(include=["datetime64"]).columns
        assert len(dt_cols) == 2

    def test_generate_with_categorical(self):
        X, y = DatasetGenerator.generate(
            task="binary_classification",
            n_samples=50,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            add_categorical=True,
            n_categorical=3,
            cat_cardinality=4,
        )
        cat_cols = X.select_dtypes(include=["category"]).columns
        assert len(cat_cols) == 3
        for col in cat_cols:
            assert X[col].nunique() <= 4

    def test_generate_with_missing(self):
        X, y = DatasetGenerator.generate(
            task="regression",
            n_samples=200,
            n_features=10,
            add_missing=True,
            missing_rate=0.2,
            add_id_column=False,
        )
        assert X.isnull().sum().sum() > 0

    def test_generate_without_extras(self):
        X, y = DatasetGenerator.generate(
            task="regression",
            n_samples=50,
            n_features=5,
            add_datetime=False,
            add_categorical=False,
            add_missing=False,
            add_id_column=False,
        )
        assert X.shape[1] == 5
        assert X.isnull().sum().sum() == 0

    def test_generate_with_id_column(self):
        X, y = DatasetGenerator.generate(
            task="regression", n_samples=50, n_features=5, add_id_column=True
        )
        assert "id" in X.columns

    def test_generate_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            DatasetGenerator.generate(task="unknown_task")

    def test_quick_regression(self):
        X, y = DatasetGenerator.quick_regression(n_samples=100)
        assert len(X) == 100
        assert y.dtype == np.float64

    def test_quick_binary(self):
        X, y = DatasetGenerator.generate(
            task="binary_classification", n_samples=100, n_features=20,
            n_informative=15, n_redundant=5,
        )
        assert len(X) == 100
        assert set(y.unique()) == {0, 1}

    def test_quick_multiclass(self):
        X, y = DatasetGenerator.generate(
            task="multiclass_classification", n_samples=100, n_features=20,
            n_informative=15, n_classes=4, n_redundant=2,
        )
        assert len(X) == 100
        assert y.nunique() == 4

    def test_as_frame_false(self):
        X, y = DatasetGenerator.generate(
            task="regression",
            n_samples=50,
            n_features=5,
            add_datetime=False,
            add_categorical=False,
            add_missing=False,
            add_id_column=False,
            as_frame=False,
        )
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_reproducibility(self):
        X1, y1 = DatasetGenerator.generate(
            task="regression", n_samples=50, n_features=5, random_state=42
        )
        X2, y2 = DatasetGenerator.generate(
            task="regression", n_samples=50, n_features=5, random_state=42
        )
        # Numeric columns should match
        num_cols = X1.select_dtypes(include=[np.number]).columns
        pd.testing.assert_frame_equal(X1[num_cols], X2[num_cols])


class TestMakeDataset:
    @pytest.mark.parametrize("complexity", ["simple", "medium", "complex"])
    def test_all_complexities(self, complexity):
        X, y = make_dataset(
            task="binary_classification", n_samples=100, complexity=complexity
        )
        assert len(X) == 100

    def test_simple_has_no_extras(self):
        X, y = make_dataset(task="regression", n_samples=50, complexity="simple")
        assert X.select_dtypes(include=["datetime64"]).shape[1] == 0
        assert X.select_dtypes(include=["category"]).shape[1] == 0
        assert X.isnull().sum().sum() == 0


# ── PipelineSerializer ─────────────────────────────────────────────────


class TestPipelineSerializer:
    def _fitted_pipeline(self):
        X, y = DatasetGenerator.generate(
            task="regression",
            n_samples=100,
            n_features=8,
            add_datetime=False,
            add_categorical=False,
            add_missing=False,
            add_id_column=False,
        )
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        return pipe, X

    def test_save_load_pipeline(self):
        pipe, X = self._fitted_pipeline()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            PipelineSerializer.save(pipe, path)
            loaded = PipelineSerializer.load(path)
            assert loaded._fitted
            result = loaded.transform(X)
            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(path)

    def test_save_load_config(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            PipelineSerializer.save_config(config, path)
            loaded = PipelineSerializer.load_config(path)
            assert isinstance(loaded, PipelineConfig)
            assert len(loaded.anomaly_models) == 1
        finally:
            os.unlink(path)

    def test_export_metadata(self):
        pipe, _ = self._fitted_pipeline()
        metadata = PipelineSerializer.export_metadata(pipe)
        assert "config" in metadata
        assert "summary" in metadata

    def test_export_metadata_before_fit_raises(self):
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
        )
        pipe = Featransform(config)
        with pytest.raises(ValueError, match="fitted"):
            PipelineSerializer.export_metadata(pipe)
