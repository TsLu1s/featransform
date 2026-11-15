from typing import Any, Dict, Optional


class FeatureTransformError(Exception):
    """Base exception for all Featransform errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class NotFittedError(FeatureTransformError):
    """Raised when transform is called before fit."""

    def __init__(self, message: str = "Component must be fitted before transform"):
        super().__init__(message)


class ConfigurationError(FeatureTransformError):
    """Raised for configuration issues."""

    pass


class ValidationError(FeatureTransformError, ValueError):
    """Raised for data validation failures.

    Also a ``ValueError`` so callers that guard input problems with
    ``except ValueError`` keep working.
    """

    pass


class TransformationError(FeatureTransformError):
    """Raised when transformation fails."""

    pass


class PipelineError(FeatureTransformError):
    """Raised for pipeline execution errors."""

    pass
