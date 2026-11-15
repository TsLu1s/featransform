import json
import pickle
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Union

from featransform._version import __version__
from featransform.core.models import PipelineConfig
from featransform.pipeline import Featransform

PathLike = Union[str, Path]


class PipelineSerializer:
    """Utility for saving and loading pipelines.

    Pipelines are stored with :mod:`pickle`. Unpickling executes code, so only
    load files you created yourself or received from a source you trust.
    """

    @staticmethod
    def save(pipeline: Featransform, filepath: PathLike) -> None:
        """Save fitted pipeline to disk."""
        if not isinstance(pipeline, Featransform):
            raise TypeError(f"Expected a Featransform pipeline, got {type(pipeline).__name__}")
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath: PathLike) -> Featransform:
        """Load fitted pipeline from disk.

        Only load files from a trusted source — see the class docstring. A
        pipeline fitted by another Featransform version loads with a warning:
        transforms keep working across compatible releases, but refitting with
        the installed version is the safe path.
        """
        with open(filepath, "rb") as f:
            pipeline = pickle.load(f)

        if not isinstance(pipeline, Featransform):
            raise TypeError(
                f"{filepath} does not hold a Featransform pipeline "
                f"(found {type(pipeline).__name__})"
            )

        fitted_version = getattr(pipeline, "_fitted_version", None)
        if pipeline._fitted and fitted_version != __version__:
            warnings.warn(
                f"This pipeline was fitted with featransform "
                f"{fitted_version or '<= 2.3.0'}; the installed version is {__version__}. "
                "Refit the pipeline if its output looks inconsistent.",
                UserWarning,
                stacklevel=2,
            )
        return pipeline

    @staticmethod
    def save_config(config: PipelineConfig, filepath: PathLike) -> None:
        """Save pipeline configuration as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(config.model_dump(mode="json"), f, indent=2)

    @staticmethod
    def load_config(filepath: PathLike) -> PipelineConfig:
        """Load pipeline configuration from JSON."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)

        return PipelineConfig(**config_dict)

    @staticmethod
    def export_metadata(pipeline: Featransform) -> Dict[str, Any]:
        """Export pipeline metadata without fitted models."""
        if not pipeline._fitted:
            raise ValueError("Pipeline must be fitted before exporting metadata")

        summary = pipeline.get_summary()

        return {
            "featransform_version": __version__,
            "config": pipeline.config.model_dump(mode="json"),
            "summary": summary,
            "optimization_history": [asdict(h) for h in pipeline.get_optimization_history()],
            "feature_importance": pipeline.get_feature_importance(),
        }

    @staticmethod
    def save_metadata(pipeline: Featransform, filepath: PathLike) -> None:
        """Save pipeline metadata as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        metadata = PipelineSerializer.export_metadata(pipeline)
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
