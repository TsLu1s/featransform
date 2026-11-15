import logging
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, final

import pandas as pd

from featransform.core.constants import MIN_SAMPLES_REQUIRED
from featransform.core.exceptions import (
    FeatureTransformError,
    NotFittedError,
    TransformationError,
    ValidationError,
)

# Library-grade logger — application is responsible for handler configuration
logger = logging.getLogger(__name__)

_PACKAGE_LOGGER = logging.getLogger("featransform")


class _VerboseHandler(logging.StreamHandler):
    """Fallback console handler behind ``verbose=True``.

    The library never configures logging on import. But ``verbose=True`` is an
    explicit opt-in, and in an application that has not configured logging the
    INFO records would otherwise be dropped silently. This handler prints them
    — and stands down as soon as the application installs its own root
    handlers, so records are never duplicated.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)

    @property
    def stream(self):
        # Resolved on every record so redirected / captured stderr is honored.
        return sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        if logging.getLogger().handlers:
            return
        super().emit(record)


def _ensure_verbose_logging() -> None:
    """Make ``verbose=True`` output visible without touching the root logger."""
    if not any(isinstance(h, _VerboseHandler) for h in _PACKAGE_LOGGER.handlers):
        handler = _VerboseHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        _PACKAGE_LOGGER.addHandler(handler)
    if _PACKAGE_LOGGER.getEffectiveLevel() > logging.INFO:
        _PACKAGE_LOGGER.setLevel(logging.INFO)


class BaseComponent(ABC):
    """Abstract base for all components with common functionality."""

    def __init__(self, verbose: bool = False, **kwargs):
        self.verbose = verbose
        self._fitted = False
        self._params = kwargs

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get component parameters."""
        pass

    @abstractmethod
    def set_params(self, **params) -> "BaseComponent":
        """Set component parameters."""
        pass

    def _check_fitted(self) -> None:
        """Private method to verify fitting status."""
        if not self._fitted:
            raise NotFittedError(f"{self.__class__.__name__} not fitted")

    def _log(self, message: str, level: str = "info") -> None:
        """Log ``message`` when the component is verbose."""
        if self.verbose:
            _ensure_verbose_logging()
            getattr(logger, level)(f"[{self.__class__.__name__}] {message}")

    @final
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Lightweight input validation — used in transform paths."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(X)}")
        if X.empty:
            raise ValidationError("Input DataFrame is empty")

    @final
    def _validate_fit_input(self, X: pd.DataFrame) -> None:
        """Stricter validation — used in fit paths. Enforces MIN_SAMPLES_REQUIRED."""
        self._validate_input(X)
        if X.shape[0] < MIN_SAMPLES_REQUIRED:
            raise ValidationError(f"Need at least {MIN_SAMPLES_REQUIRED} samples")

    @contextmanager
    def _error_context(self, operation: str):
        """Wrap unexpected failures in ``TransformationError``.

        Featransform's own exceptions (configuration / validation / not-fitted)
        already carry a precise meaning and pass through untouched; anything
        else is re-raised as ``TransformationError`` chained to its cause.
        """
        try:
            yield
        except FeatureTransformError:
            raise
        except Exception as e:
            self._log(f"Error in {operation}: {str(e)}", "error")
            raise TransformationError(f"{operation} failed: {str(e)}") from e


class BaseTransformer(BaseComponent):
    """Abstract base for all transformers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BaseTransformer":
        """Fit the transformer."""
        self._validate_fit_input(X)
        self._fitted = True
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        self._check_fitted()
        self._validate_input(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class BaseSelector(BaseComponent):
    """Abstract base for feature selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._selected_features: List[str] = []
        self._feature_scores: Optional[Dict[str, float]] = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseSelector":
        """Fit selector to identify important features."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select features based on importance."""
        pass

    @abstractmethod
    def _calculate_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Private abstract method to calculate feature importance."""
        pass

    @final
    def get_selected_features(self) -> List[str]:
        """Get selected feature names."""
        self._check_fitted()
        return self._selected_features

    def get_feature_scores(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self._feature_scores


class BasePipeline(BaseComponent):
    """Abstract base for pipeline implementations."""

    def __init__(self, stages: List[BaseComponent], **kwargs):
        super().__init__(**kwargs)
        self.stages = stages
        self._fit_schema: Optional[List[str]] = None
        self._validate_stages()

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePipeline":
        """Fit all pipeline stages."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform through all stages."""
        pass

    def _validate_stages(self) -> None:
        """Private method to validate pipeline stages."""
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage")
        for stage in self.stages:
            if not isinstance(stage, BaseComponent):
                raise TypeError(f"Invalid stage type: {type(stage)}")

    @final
    def _execute_stage(self, stage: BaseComponent, X: pd.DataFrame) -> pd.DataFrame:
        """Private final method to execute a single stage."""
        with self._error_context(f"stage_{stage.__class__.__name__}"):
            self._log(f"Executing {stage.__class__.__name__}")
            return stage.transform(X)

    def _store_fit_schema(self, X: pd.DataFrame) -> None:
        """Private method to store training schema."""
        self._fit_schema = list(X.columns)
        self._log(f"Stored training schema with {len(self._fit_schema)} columns")

    def _validate_transform_schema(self, X: pd.DataFrame) -> pd.DataFrame:

        if self._fit_schema is None:
            raise NotFittedError("No training schema found. Pipeline must be fitted first.")

        current_columns = set(X.columns)
        expected_columns = set(self._fit_schema)

        missing = expected_columns - current_columns
        extra = current_columns - expected_columns

        if missing:
            raise ValidationError(
                f"Missing columns from training: {sorted(missing, key=str)}\n"
                f"Expected columns: {self._fit_schema}\n"
                f"Received columns: {list(X.columns)}"
            )

        if extra:
            self._log(f"Ignoring extra columns not seen during training: {extra}")

        X_aligned = X[self._fit_schema].copy()

        return X_aligned

    def get_fit_schema(self) -> Optional[List[str]]:
        """Get the stored training schema."""
        return self._fit_schema.copy() if self._fit_schema else None
