from typing import Any, Dict, Optional


class FeatureTransformError(Exception):
    """Base exception for all Featransform errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class NotFittedError(FeatureTransformError):
    """Raised when transform is called before fit."""

    def __init__(self, model_name: str):
        super().__init__(f"{model_name} must be fitted before transform")


class ConfigurationError(FeatureTransformError):
    """Raised for configuration issues."""

    pass


class ValidationError(FeatureTransformError):
    """Raised for data validation failures."""

    pass


class TransformationError(FeatureTransformError):
    """Raised when transformation fails."""

    pass


class PipelineError(FeatureTransformError):
    """Raised for pipeline execution errors."""

    pass
