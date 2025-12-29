import pickle
import json
from pathlib import Path
from typing import Any, Dict
from featransform.core.models import PipelineConfig
from featransform.pipeline import Featransform


class PipelineSerializer:
    """Utility for saving and loading pipelines."""
    
    @staticmethod
    def save(pipeline: Featransform, filepath: str) -> None:
        """Save fitted pipeline to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(pipeline, f)
    
    @staticmethod
    def load(filepath: str) -> Featransform:
        """Load fitted pipeline from disk."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_config(config: PipelineConfig, filepath: str) -> None:
        """Save pipeline configuration as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = config.model_dump()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    @staticmethod
    def load_config(filepath: str) -> PipelineConfig:
        """Load pipeline configuration from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return PipelineConfig(**config_dict)
    
    @staticmethod
    def export_metadata(pipeline: Featransform) -> Dict[str, Any]:
        """Export pipeline metadata without fitted models."""
        if not pipeline._fitted:
            raise ValueError("Pipeline must be fitted before exporting metadata")
        
        summary = pipeline.get_summary()
        
        return {
            'config': pipeline.config,
            'summary': summary,
            'optimization_history': [
                h.dict() for h in pipeline.get_optimization_history()
            ],
            'feature_importance': pipeline.get_feature_importance()
        }
    
    @staticmethod
    def save_metadata(pipeline: Featransform, filepath: str) -> None:
        """Save pipeline metadata as JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        metadata = PipelineSerializer.export_metadata(pipeline)
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)