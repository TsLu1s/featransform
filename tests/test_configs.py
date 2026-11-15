"""Tests for preset configurations and FTconfig."""

import pytest

from featransform.configs.baseline import FTconfig
from featransform.core.enums import TaskType
from featransform.core.models import PipelineConfig
from featransform.pipeline import Featransform


class TestFTconfig:
    @pytest.mark.parametrize("preset", ["minimal", "standard", "optimized", "complete"])
    def test_preset_returns_pipeline_config(self, preset):
        config = getattr(FTconfig, preset)()
        assert isinstance(config, PipelineConfig)

    @pytest.mark.parametrize("preset", ["minimal", "standard", "optimized", "complete"])
    def test_preset_with_task_type(self, preset):
        config = getattr(FTconfig, preset)(task_type=TaskType.BINARY_CLASSIFICATION)
        assert config.task_type == TaskType.BINARY_CLASSIFICATION

    def test_minimal_has_anomaly_only(self):
        config = FTconfig.minimal()
        assert len(config.anomaly_models) >= 1
        assert len(config.clustering_models) == 0
        assert len(config.dimensionality_models) == 0
        assert config.optimization is None

    def test_standard_has_optimization(self):
        config = FTconfig.standard()
        assert config.optimization is not None
        assert len(config.anomaly_models) >= 1
        assert len(config.clustering_models) >= 1
        assert len(config.dimensionality_models) >= 1

    def test_complete_has_all_models(self):
        config = FTconfig.complete()
        assert len(config.anomaly_models) == 4
        assert len(config.clustering_models) == 5
        assert len(config.dimensionality_models) == 3

    def test_get_by_name(self):
        config = FTconfig._get("minimal")
        assert isinstance(config, PipelineConfig)

    def test_get_unknown_falls_back_to_standard(self, capsys):
        config = FTconfig._get("nonexistent")
        captured = capsys.readouterr()
        assert "Unknown config" in captured.out
        assert isinstance(config, PipelineConfig)

    @pytest.mark.parametrize("preset", ["minimal", "standard", "optimized", "complete"])
    def test_preset_creates_valid_pipeline(self, preset):
        config = getattr(FTconfig, preset)()
        pipe = Featransform(config)
        assert pipe is not None
