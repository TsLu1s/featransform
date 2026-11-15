"""Integration tests: end-to-end pipeline execution with different configurations."""

import numpy as np
import pandas as pd
import pytest

from featransform.configs.baseline import FTconfig
from featransform.core.enums import ImputationStrategy, ModelFamily, SelectionStrategy, TaskType
from featransform.core.models import (
    ModelConfig,
    OptimizationConfig,
    PipelineConfig,
    ProcessingConfig,
)
from featransform.pipeline import Featransform
from featransform.utils.data_generator import DatasetGenerator


class TestEndToEndRegression:
    def test_minimal_preset(self):
        X, y = DatasetGenerator.generate(
            task="regression", n_samples=200, n_features=10,
            add_datetime=False, add_categorical=False, add_missing=False,
            add_id_column=False,
        )
        config = FTconfig.minimal(task_type=TaskType.REGRESSION)
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)
        assert result.isnull().sum().sum() == 0

    def test_with_missing_and_categorical(self):
        X, y = DatasetGenerator.generate(
            task="regression", n_samples=200, n_features=10,
            add_datetime=False, add_categorical=True, n_categorical=2,
            add_missing=True, missing_rate=0.1, add_id_column=False,
        )
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        assert result.isnull().sum().sum() == 0
        assert result.select_dtypes(include=["object", "category"]).shape[1] == 0


class TestEndToEndClassification:
    def test_binary_with_multiple_components(self):
        X, y = DatasetGenerator.generate(
            task="binary_classification", n_samples=200, n_features=10,
            n_informative=6, n_redundant=2,
            add_datetime=False, add_categorical=False, add_missing=False,
            add_id_column=False,
        )
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})
            ],
            dimensionality_models=[
                ModelConfig(model_family=ModelFamily.PCA, parameters={"n_components": 3})
            ],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        assert result.shape[1] > X.shape[1]  # engineered features added

    def test_multiclass(self):
        X, y = DatasetGenerator.generate(
            task="multiclass_classification", n_samples=300, n_features=10,
            n_informative=6, n_redundant=2, n_classes=4,
            add_datetime=False, add_categorical=False,
            add_missing=False, add_id_column=False,
        )
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        result = pipe.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)


class TestEndToEndWithOptimization:
    def test_optimization_selects_features(self):
        X, y = DatasetGenerator.generate(
            task="binary_classification", n_samples=300, n_features=15,
            n_informative=8, n_redundant=4, add_datetime=False, add_categorical=False,
            add_missing=False, add_id_column=False,
        )
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
                min_features=5,
                use_catboost=False,
                use_xgboost=True,
                n_estimators=10,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        result = pipe.transform(X)
        assert pipe._fitted
        history = pipe.get_optimization_history()
        assert len(history) > 0
        importance = pipe.get_feature_importance()
        assert importance is not None


class TestEndToEndTransformConsistency:
    def test_fit_and_transform_produce_same_schema(self):
        X, y = DatasetGenerator.generate(
            task="regression", n_samples=200, n_features=10,
            add_datetime=False, add_categorical=False, add_missing=False,
            add_id_column=False,
        )
        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            clustering_models=[
                ModelConfig(model_family=ModelFamily.KMEANS, parameters={"n_clusters": 3})
            ],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X, y)
        result1 = pipe.transform(X)
        result2 = pipe.transform(X)
        assert list(result1.columns) == list(result2.columns)
        assert result1.shape == result2.shape

    def test_transform_on_new_data(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(200, 8), columns=[f"f{i}" for i in range(8)])
        y_train = pd.Series(np.random.randn(200))
        X_test = pd.DataFrame(np.random.randn(50, 8), columns=[f"f{i}" for i in range(8)])

        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(
                imputation_strategy=ImputationStrategy.MEAN,
                handle_datetime=False,
            ),
        )
        pipe = Featransform(config)
        pipe.fit(X_train, y_train)
        result = pipe.transform(X_test)
        assert len(result) == 50
        schema = pipe.get_pipeline_schema()
        assert set(schema) == set(result.columns)


class TestEndToEndDatetime:
    """Datetime-bearing data through the full pipeline.

    Regression guard: the rest of the suite disables datetime generation, so a
    pipeline that choked on datetime dtypes could pass everything else.
    """

    @staticmethod
    def _datetime_data(n_samples=200):
        return DatasetGenerator.generate(
            task="binary_classification", n_samples=n_samples, n_features=10,
            n_informative=5, n_redundant=2,
            add_datetime=True, n_datetime_cols=2, add_categorical=False,
            add_missing=False, add_id_column=False,
        )

    def test_pipeline_expands_datetime_columns(self):
        X, y = self._datetime_data()
        assert X.select_dtypes(include=["datetime64"]).shape[1] == 2

        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(handle_datetime=True),
        )
        result = Featransform(config).fit_transform(X, y)

        assert result.select_dtypes(include=["datetime64"]).empty
        assert len(result) == len(X)
        assert result.isnull().sum().sum() == 0

    def test_transform_matches_fit_schema_on_unseen_data(self):
        X, y = self._datetime_data(n_samples=300)
        X_train, X_test = X.iloc[:200], X.iloc[200:]
        y_train = y.iloc[:200]

        config = PipelineConfig(
            anomaly_models=[ModelConfig(model_family=ModelFamily.ISOLATION_FOREST)],
            processing=ProcessingConfig(handle_datetime=True),
        )
        pipe = Featransform(config).fit(X_train, y_train)

        train_out = pipe.transform(X_train)
        test_out = pipe.transform(X_test)

        assert list(train_out.columns) == list(test_out.columns)
        assert test_out.select_dtypes(include=["datetime64"]).empty
        assert len(test_out) == len(X_test)

