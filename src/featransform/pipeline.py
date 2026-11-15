import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from featransform._version import __version__
from featransform.builder import BuilderPipeline
from featransform.core.base import BasePipeline
from featransform.core.constants import ENGINEERED_COLLISION_SUFFIX
from featransform.core.enums import ModelFamily, PipelineStage, TaskType
from featransform.core.exceptions import ConfigurationError, NotFittedError, ValidationError
from featransform.core.models import OptimizationHistory, PipelineConfig
from featransform.optimization.evaluator import (
    METRIC_DIRECTION,
    detect_task_type,
    validate_metric,
)
from featransform.processing.datetime_features import DatetimeExpander
from featransform.processing.scaling import FeatureScaler

_FE_COMPONENTS = ("anomaly", "clustering", "dimensionality")


@dataclass
class PipelineState:
    """Pipeline execution state."""

    data: pd.DataFrame
    target: Optional[pd.Series] = None
    stage: PipelineStage = PipelineStage.PREPROCESSING
    features_created: List[str] = field(default_factory=list)
    transformations_applied: List[str] = field(default_factory=list)

    def evolve(self, **kwargs) -> "PipelineState":
        """Create new state with updates."""
        return PipelineState(
            **{
                **self.__dict__,
                **kwargs,
                "features_created": self.features_created.copy(),
                "transformations_applied": self.transformations_applied.copy(),
            }
        )


class Featransform(BasePipeline, BuilderPipeline):
    """
    Feature engineering pipeline with automatic optimization.
    Implements builder pattern for component construction.

    ``fit`` learns every step on the training frame; ``transform`` replays them
    on any frame holding the same input columns (in any order — extra columns
    are ignored). A fitted pipeline keeps only what it needs to transform new
    data: the training rows and target are released once fitting is done.
    """

    # Class-level defaults: a pipeline pickled by an older release has none of
    # these on the instance and falls back to them, so it still transforms.
    _input_columns: Optional[List[str]] = None
    _engineered_name_maps: Optional[Dict[str, Dict[str, str]]] = None
    _engineered_dropped: Sequence[str] = ()
    _fe_skipped: bool = False
    _best_result: Optional[Dict[str, Any]] = None
    _task_type: Optional[TaskType] = None
    _fitted_version: Optional[str] = None

    def __init__(self, config: PipelineConfig):
        BuilderPipeline.__init__(self)
        self.config = config
        self._state: Optional[PipelineState] = None
        self._optimization_history: List[OptimizationHistory] = []
        self._feature_importance: Optional[Dict[str, float]] = None
        # Persistent preprocessing state (learned at fit, replayed at transform)
        self._input_columns = None
        self._constant_cols_dropped: List[str] = []
        self._duplicate_cols_dropped: List[str] = []
        # Feature-engineering input state: id columns to exclude and the
        # scaler applied to the copy fed to anomaly/clustering/dimensionality.
        self._id_columns: List[str] = []
        self._scaler: Optional[FeatureScaler] = None
        # Datetime expansion is learned at fit and replayed at transform
        self._datetime_expander: Optional[DatetimeExpander] = None
        # Engineered-feature bookkeeping: collision renames and constant drops.
        self._engineered_name_maps = {}
        self._engineered_dropped = []
        self._fe_skipped = False
        self._best_result = None
        self._task_type = None
        self._fitted_version = None

        self._components = self.build_components(config)
        self._build_scaler()

        BasePipeline.__init__(self, stages=list(self._components.values()), verbose=config.verbose)

    # ------------------------------------------------------------------ build

    def _build_scaler(self) -> None:
        """Build the feature-engineering-input scaler from config."""
        strategy = self.config.processing.scaling_strategy
        self._scaler = (
            FeatureScaler(strategy=strategy, verbose=self.config.verbose)
            if strategy is not None
            else None
        )

    def build_components(self, config: PipelineConfig) -> Dict[str, Any]:
        """Implementation of abstract component building."""
        return self._build_pipeline(config)

    def _select_components(self, config: PipelineConfig) -> List[str]:
        """Select components to build based on configuration."""
        components = []

        if config.anomaly_models:
            components.append("anomaly")
        if config.clustering_models:
            components.append("clustering")
        if config.dimensionality_models:
            components.append("dimensionality")

        return components

    def _rebuild_components(self) -> None:
        """Replace every component with a fresh, unfitted one built from config."""
        self._components = self.build_components(self.config)
        self._build_scaler()
        self.stages = list(self._components.values())

    def _has_fitted_parts(self) -> bool:
        """Whether any component still carries state from an earlier fit."""
        return (
            self._fitted
            or any(getattr(component, "_fitted", False) for component in self._components.values())
            or (self._scaler is not None and self._scaler._fitted)
        )

    def _reset_learned_state(self) -> None:
        """Forget everything learned by a previous ``fit``."""
        self._fitted = False
        self._state = None
        self._fit_schema = None
        self._input_columns = None
        self._optimization_history = []
        self._feature_importance = None
        self._constant_cols_dropped = []
        self._duplicate_cols_dropped = []
        self._id_columns = []
        self._datetime_expander = None
        self._engineered_name_maps = {}
        self._engineered_dropped = []
        self._fe_skipped = False
        self._best_result = None
        self._task_type = None
        self._fitted_version = None

    def register_strategy(
        self, model_family: ModelFamily, strategy_class: Any, comp_type: str = "anomaly"
    ) -> None:
        """Register a custom strategy class and rebuild the components with it.

        On a fitted pipeline the fitted components are left untouched and the
        new strategy takes effect at the next ``fit``.
        """
        super().register_strategy(model_family, strategy_class, comp_type)
        if self._has_fitted_parts():
            warnings.warn(
                "Strategy registered on a fitted pipeline; it takes effect at the next fit().",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._rebuild_components()

    # -------------------------------------------------------------- fit / transform

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "Featransform":
        """Fit pipeline with optional optimization.

        All processing and feature-engineering components are fit on the full
        dataset so no rows are discarded from the fitted transformers.
        When optimization is configured, feature *selection* carves an internal
        validation slice out of the engineered features — the holdout governs
        which columns to keep, not which rows the transformers see.

        ``y`` is required when optimization is configured and optional
        otherwise. Calling ``fit`` again refits everything from scratch.
        """
        self._fit(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and return the transformed training frame.

        Equivalent to ``fit(X, y).transform(X)`` but reuses the frame built
        while fitting instead of running the whole pipeline a second time.
        """
        return self._fit(X, y)

    def _fit(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        data = self._prepare_frame(X, fit=True)
        target = self._prepare_target(data, y)

        # A refit starts from fresh components: nothing learned on a previous
        # dataset (encoders, scaler, models, selection, history) may survive.
        if self._has_fitted_parts():
            self._rebuild_components()
        self._reset_learned_state()

        self._task_type = self._resolve_task_type(target)
        self._input_columns = list(data.columns)

        # Resolve id/index columns on the raw input — names persist through
        # preprocessing, so we exclude them from the feature-engineering input.
        self._id_columns = self._resolve_id_columns(data)
        if self._id_columns:
            self._log(f"Identifier columns kept out of feature engineering: {self._id_columns}")

        # Fit every component on the full dataset
        self._state = PipelineState(data=data, target=target)
        self._state = self.__process_pipeline(self._state, fit=True)

        # Optimize feature selection on a holdout of the engineered features
        if self._needs_target():
            self.__optimize_features()

        self._store_fit_schema(self._state.data)
        result = self._state.data

        # Release the training rows and target: a fitted pipeline (and its
        # pickle) keeps the output schema, not the data it was fitted on.
        self._state = self._state.evolve(data=result.iloc[0:0], target=None)

        self._fitted_version = __version__
        self._fitted = True
        self._log(f"Pipeline fitted: {result.shape[1]} features")
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data through fitted pipeline."""
        self._check_fitted()
        data = self._prepare_frame(X, fit=False)

        state = self.__process_pipeline(PipelineState(data=data))

        if self.has_component("selector") and self._components["selector"]._fitted:
            state.data = self._components["selector"].transform(state.data)

        state.data = self._validate_transform_schema(state.data)

        return state.data

    # ------------------------------------------------------------ input validation

    def _needs_target(self) -> bool:
        """Whether fitting uses the target (only feature selection does)."""
        return self.config.optimization is not None and self.has_component("selector")

    def _prepare_frame(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Validate the input frame and return the working copy.

        Column labels are normalized to strings (the engineered features are
        named with strings, and a frame mixing label types cannot be consumed
        downstream). At transform the frame is aligned to the columns seen at
        fit: order is restored, extra columns are ignored and missing ones are
        reported by name.
        """
        if fit:
            self._validate_fit_input(X)
        else:
            self._validate_input(X)

        data = X.copy()
        if not all(isinstance(col, str) for col in data.columns):
            data.columns = [str(col) for col in data.columns]

        duplicated = data.columns[data.columns.duplicated()]
        if len(duplicated):
            raise ValidationError(
                f"Column names must be unique; duplicated: {sorted(set(duplicated))}"
            )

        if not fit and self._input_columns is not None:
            present = set(data.columns)
            missing = [col for col in self._input_columns if col not in present]
            if missing:
                raise ValidationError(
                    f"Input is missing columns seen at fit: {missing}\n"
                    f"Expected columns: {self._input_columns}\n"
                    f"Received columns: {list(data.columns)}"
                )
            if len(data.columns) != len(self._input_columns):
                expected = set(self._input_columns)
                extra = [col for col in data.columns if col not in expected]
                self._log(f"Ignoring extra columns not seen during fit: {extra}")
            data = data[self._input_columns]

        return data

    def _prepare_target(self, data: pd.DataFrame, y: Any) -> Optional[pd.Series]:
        """Validate the target and align it to ``data`` by position."""
        if y is None:
            if self._needs_target():
                raise ValidationError(
                    "y is required: feature-selection optimization is configured. "
                    "Pass the target, or set optimization=None."
                )
            return None

        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValidationError(
                    f"y must be one-dimensional; got a DataFrame with {y.shape[1]} columns"
                )
            y = y.iloc[:, 0]
        if not isinstance(y, pd.Series):
            values = np.asarray(y)
            if values.ndim != 1:
                raise ValidationError(f"y must be one-dimensional; got shape {values.shape}")
            y = pd.Series(values, name="target")

        if len(y) != len(data):
            raise ValidationError(
                f"X and y have different lengths: {len(data)} rows vs {len(y)} target values"
            )

        # Rows pair up by position (the scikit-learn convention), so the target
        # takes over the frame's index.
        target = y.copy()
        target.index = data.index

        if self._needs_target():
            n_missing = int(target.isna().sum())
            if n_missing:
                raise ValidationError(
                    f"y contains {n_missing} missing value(s); drop or impute them before fit"
                )
            if target.nunique() < 2:
                raise ValidationError("y must hold at least two distinct values")

        return target

    def _resolve_task_type(self, target: Optional[pd.Series]) -> Optional[TaskType]:
        """Resolve the task once per fit and check it against target and metric."""
        if target is None:
            return self.config.task_type

        task_type = self.__detect_task_type(target)
        if not self._needs_target():
            return task_type

        if task_type == TaskType.REGRESSION and (
            not pd.api.types.is_numeric_dtype(target) or pd.api.types.is_bool_dtype(target)
        ):
            raise ConfigurationError(
                f"task_type is regression but the target dtype is {target.dtype}. "
                "Set task_type to a classification task or pass a numeric target."
            )
        if task_type == TaskType.REGRESSION and target.nunique() == 2:
            # The presets default to regression, so this is the usual symptom of
            # a classification target fitted without stating the task.
            warnings.warn(
                "task_type is regression but the target holds only two distinct values. "
                "If this is a classification problem, set "
                "task_type='binary_classification' (or task_type=None to auto-detect).",
                UserWarning,
                stacklevel=4,
            )
        validate_metric(self.config.optimization.metric, task_type)
        return task_type

    # ------------------------------------------------------------------ stages

    def __process_pipeline(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Process all pipeline stages."""
        state = self.__preprocess(state, fit)
        return self.__engineer_features(state, fit)

    def __preprocess(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Apply preprocessing transformations."""
        state = state.evolve(stage=PipelineStage.PREPROCESSING)

        if self.config.processing.handle_datetime:
            state = self.__extract_datetime(state, fit)

        for comp in ["imputer", "encoder"]:
            if self.has_component(comp):
                state = self.__apply_component(state, comp, fit)

        # Clean features — learn at fit, replay at transform
        if self.config.processing.drop_constant:
            if fit:
                self._constant_cols_dropped = state.data.columns[state.data.nunique() <= 1].tolist()
                if self._constant_cols_dropped:
                    self._log(f"Dropping constant columns: {self._constant_cols_dropped}")
            cols_to_drop = [c for c in self._constant_cols_dropped if c in state.data.columns]
            if cols_to_drop:
                state.data = state.data.drop(columns=cols_to_drop)

        # Drop duplicate columns — the mask-only transpose preserves dtypes
        if self.config.processing.drop_duplicates:
            if fit:
                duplicated_mask = state.data.T.duplicated().values
                self._duplicate_cols_dropped = state.data.columns[duplicated_mask].tolist()
                if self._duplicate_cols_dropped:
                    self._log(f"Dropping duplicate columns: {self._duplicate_cols_dropped}")
            cols_to_drop = [c for c in self._duplicate_cols_dropped if c in state.data.columns]
            if cols_to_drop:
                state.data = state.data.drop(columns=cols_to_drop)

        return state

    def __engineer_features(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Apply feature engineering transformations.

        Feature-engineering components consume a prepared input (id columns
        excluded, then scaled — B1/B2); the original processed frame passes
        through unscaled and keeps its id columns, so the engineered features are
        concatenated onto the untouched originals.
        """
        state = state.evolve(stage=PipelineStage.FEATURE_ENGINEERING)

        fe_components = [comp for comp in _FE_COMPONENTS if self.has_component(comp)]
        if not fe_components:
            return state

        if fit:
            n_fe_columns = state.data.shape[1] - sum(
                1 for col in self._id_columns if col in state.data.columns
            )
            self._fe_skipped = n_fe_columns == 0
            if self._fe_skipped:
                warnings.warn(
                    "No columns are left for feature engineering once identifier columns "
                    "are excluded; the anomaly / clustering / dimensionality components "
                    "are skipped.",
                    UserWarning,
                    stacklevel=4,
                )
        if self._fe_skipped:
            return state

        fe_input = self._prepare_fe_input(state.data, fit)
        taken = set(state.data.columns)
        engineered = []

        for comp in fe_components:
            component = self._components[comp]

            if fit:
                self._log(f"Fitting {comp} features on {fe_input.shape[1]} columns")
                component.fit(fe_input, state.target)

            features = component.transform(fe_input).features
            features = self.__finalize_engineered(comp, features, taken, fit)
            taken.update(features.columns)

            engineered.append(features)
            state.features_created.extend(features.columns.tolist())
            state.transformations_applied.append(comp)

        if engineered:
            state.data = pd.concat([state.data] + engineered, axis=1)

        return state

    def __finalize_engineered(
        self, comp: str, features: pd.DataFrame, taken: set, fit: bool
    ) -> pd.DataFrame:
        """Name and prune one component's engineered features.

        Two decisions are learned at fit and replayed at transform:

        * a feature whose name collides with an existing column is renamed
          (``pca_0`` → ``pca_0_ft``) so the output never holds duplicate labels;
        * when ``drop_constant`` is on, a feature that is constant on the
          training data (an anomaly flag that never fires, a DBSCAN run that
          found a single group) carries no information and is dropped.
        """
        if fit:
            rename: Dict[str, str] = {}
            reserved = set(taken) | set(features.columns)
            for col in features.columns:
                if col not in taken:
                    continue
                candidate = f"{col}{ENGINEERED_COLLISION_SUFFIX}"
                counter = 1
                while candidate in reserved:
                    counter += 1
                    candidate = f"{col}{ENGINEERED_COLLISION_SUFFIX}{counter}"
                rename[col] = candidate
                reserved.add(candidate)
            self._engineered_name_maps[comp] = rename
            if rename:
                self._log(f"Renamed engineered features colliding with input columns: {rename}")
        else:
            rename = (self._engineered_name_maps or {}).get(comp, {})

        if rename:
            features = features.rename(columns=rename)

        if fit and self.config.processing.drop_constant:
            constant = features.columns[features.nunique(dropna=False) <= 1].tolist()
            if constant:
                self._engineered_dropped = list(self._engineered_dropped) + constant
                self._log(f"Dropping constant engineered features: {constant}")

        to_drop = [col for col in self._engineered_dropped if col in features.columns]
        if to_drop:
            features = features.drop(columns=to_drop)

        return features

    def _resolve_id_columns(self, data: pd.DataFrame) -> List[str]:
        """Resolve id/index columns to exclude from feature engineering.

        Explicit ``processing.id_columns`` are always honored. When
        ``processing.auto_detect_id`` is on, integer-valued, fully-unique columns
        (the classic identifier signature) are also excluded. Auto-detection is a
        heuristic — prefer the explicit list when a column could be a real feature.
        """
        proc = self.config.processing
        requested = [str(col) for col in proc.id_columns]
        resolved = [col for col in requested if col in data.columns]
        absent = [col for col in requested if col not in data.columns]
        if absent:
            warnings.warn(
                f"id_columns not found in the input and ignored: {absent}",
                UserWarning,
                stacklevel=4,
            )

        if proc.auto_detect_id:
            n = len(data)
            for col in data.columns:
                if col in resolved:
                    continue
                s = data[col]
                if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                    continue
                values = s.dropna()
                if len(values) != n:  # any missing → not an identifier
                    continue
                is_integer_valued = pd.api.types.is_integer_dtype(s) or bool(
                    np.all(np.mod(values.to_numpy(dtype="float64"), 1) == 0)
                )
                if is_integer_valued and values.nunique() == n:
                    resolved.append(col)

        return resolved

    def _prepare_fe_input(self, data: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Build the (id-excluded, scaled) input for feature-engineering."""
        working = data.drop(columns=[c for c in self._id_columns if c in data.columns])

        if self._scaler is not None and working.shape[1] > 0:
            if fit:
                self._scaler.fit(working)
            if self._scaler._fitted:
                working = self._scaler.transform(working)

        return working

    def __apply_component(self, state: PipelineState, name: str, fit: bool) -> PipelineState:
        """Apply single component transformation."""
        component = self._components[name]

        if fit:
            if name == "imputer" and hasattr(component, "exclude_columns"):
                # Identifiers must not act as predictors for the imputation model.
                component.exclude_columns = list(self._id_columns)
            component.fit(state.data, state.target)

        state.data = component.transform(state.data)
        state.transformations_applied.append(name)
        return state

    # ------------------------------------------------------------ optimization

    def __split_for_selection(
        self, target: pd.Series, task_type: TaskType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Row positions of the train / validation slices used for selection.

        Classification splits are stratified. A class with a single row cannot
        be stratified, so such rows stay in the training slice; and whatever the
        split, every class that reaches the validation slice is also present in
        the training slice (the evaluator cannot score a class it never saw).
        """
        positions = np.arange(len(target))
        test_size = self.config.optimization.validation_split
        seed = self.config.random_state

        if task_type == TaskType.REGRESSION:
            train_idx, val_idx = train_test_split(positions, test_size=test_size, random_state=seed)
            return train_idx, val_idx

        labels = target.to_numpy()
        counts = target.value_counts()
        singleton_mask = target.isin(counts.index[counts < 2]).to_numpy()
        if singleton_mask.any():
            warnings.warn(
                f"{int(singleton_mask.sum())} target class(es) have a single row; those rows "
                "are kept in the training slice of the feature-selection split.",
                UserWarning,
                stacklevel=5,
            )

        splittable = positions[~singleton_mask]
        try:
            train_idx, val_idx = train_test_split(
                splittable,
                test_size=test_size,
                random_state=seed,
                stratify=labels[~singleton_mask],
            )
        except ValueError:
            # e.g. more classes than validation rows — fall back to a plain split.
            warnings.warn(
                "The feature-selection split could not be stratified by the target; "
                "using a random split instead.",
                UserWarning,
                stacklevel=5,
            )
            train_idx, val_idx = train_test_split(
                splittable, test_size=test_size, random_state=seed
            )

        train_idx = np.concatenate([train_idx, positions[singleton_mask]])

        unseen = ~pd.Series(labels[val_idx]).isin(pd.unique(labels[train_idx])).to_numpy()
        if unseen.any():
            train_idx = np.concatenate([train_idx, val_idx[unseen]])
            val_idx = val_idx[~unseen]

        if len(val_idx) == 0:
            raise ValidationError(
                "The feature-selection validation slice is empty; "
                "provide more rows or a larger validation_split."
            )
        return train_idx, val_idx

    def __optimize_features(self) -> None:
        """Optimize feature selection on a holdout of the engineered features.

        The transformers have already been fit on the full dataset; here we
        split the *engineered* feature matrix only to decide which columns to
        keep, then apply that selection to every row.

        Caveat: because the anomaly / clustering / dimensionality / scaling
        components were fit on all rows (including this holdout), the validation
        rows are not strictly unseen by those unsupervised transforms. The
        selection *evaluator* (a fresh model per subset) never trains on the
        holdout, but the engineered-feature values it reads were shaped with
        holdout rows present. The scores reported by ``get_summary`` /
        ``report_optimization`` are therefore a mildly optimistic estimate of
        the selected subset's generalization, and should be read as a relative
        ranking signal across subsets rather than an unbiased holdout score.
        """
        data = self._state.data
        target = self._state.target
        task_type = self._task_type

        train_idx, val_idx = self.__split_for_selection(target, task_type)
        X_tr, X_val = data.iloc[train_idx], data.iloc[val_idx]
        y_tr, y_val = target.iloc[train_idx], target.iloc[val_idx]

        # The task was resolved once on the full target; the selector reuses it
        # rather than re-deriving it from the training slice.
        selector = self._components["selector"]
        selector.metric = self.config.optimization.metric
        selector._task_type = task_type
        selector._explicit_task_type = True
        selector.fit(X_tr, y_tr)

        n = max(2, min(self.config.optimization.n_iterations, 10))
        thresholds = list(np.linspace(0.5, 1.0, n))

        # Get baseline score (full engineered feature set on the holdout)
        baseline_score = selector.evaluate_subset(X_tr, y_tr, X_val, y_val)
        self._optimization_history.append(
            OptimizationHistory(
                iteration=-1,
                score=baseline_score,
                n_features=data.shape[1],
                selected_features=list(data.columns),
            )
        )
        self._log(f"Baseline ({data.shape[1]} features): score {baseline_score:.4f}")

        # Optimize selection. Reuse the baseline score so the full feature set
        # competes inside optimize without a second evaluation.
        best_features, best_score, history = selector.optimize(
            X_tr,
            y_tr,
            X_val,
            y_val,
            thresholds,
            baseline_score=baseline_score,
        )

        for i, result in enumerate(history):
            self._optimization_history.append(
                OptimizationHistory(
                    iteration=i,
                    score=result["score"],
                    n_features=result["n_features"],
                    selected_features=result["features"],
                )
            )
            self._log(
                f"Threshold {result['threshold']:.2f}: {result['n_features']} features, score: {result['score']:.4f}"
            )

        # Apply best selection — and remember exactly what was applied, so the
        # summary never has to re-derive it from the history.
        self._best_result = {"score": best_score, "n_features": len(best_features)}
        self._state.data = self._state.data[best_features]
        self._state = self._state.evolve(stage=PipelineStage.FEATURE_SELECTION)
        self._feature_importance = selector.get_feature_scores()
        self._log(f"Selected {len(best_features)} features, score {best_score:.4f}")

    def __extract_datetime(self, state: PipelineState, fit: bool = False) -> PipelineState:
        """Expand datetime columns without in-place mutation of state.data.

        Delegates to :class:`DatetimeExpander` so the pipeline and the
        component-level API share one implementation. The expander is fitted on
        the training frame and replayed afterwards, keeping the emitted columns
        identical between fit and transform.
        """
        if fit or self._datetime_expander is None:
            self._datetime_expander = DatetimeExpander(verbose=self.verbose)
            self._datetime_expander.fit(state.data)
            if self._datetime_expander.get_converted_columns():
                self._log(
                    "Expanding datetime / timedelta columns: "
                    f"{self._datetime_expander.get_converted_columns()}"
                )

        if not self._datetime_expander.get_converted_columns():
            return state

        state.data = self._datetime_expander.transform(state.data)
        return state

    def __detect_task_type(self, y: pd.Series) -> TaskType:
        """Detect ML task type, honoring config.task_type if explicitly set."""
        return detect_task_type(y, self.config.task_type)

    # ------------------------------------------------------------------- params

    def get_params(self) -> Dict[str, Any]:
        """Get pipeline parameters."""
        return {"config": self.config}

    def set_params(self, **params) -> "Featransform":
        """Set pipeline parameters. Replacing the config resets fitted state."""
        if "config" in params:
            was_fitted = self._fitted
            self.config = params["config"]
            self.verbose = self.config.verbose
            self._rebuild_components()
            # Clear all fit-time state — pipeline must be refit
            self._reset_learned_state()
            if was_fitted:
                warnings.warn(
                    "Pipeline config was replaced; fitted state was discarded. "
                    "Call .fit() again before .transform().",
                    UserWarning,
                    stacklevel=2,
                )
        return self

    # ------------------------------------------------------------------ reports

    def get_optimization_history(self) -> List[OptimizationHistory]:
        """Get optimization history."""
        return self._optimization_history

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance scores."""
        return self._feature_importance

    def get_pipeline_schema(self) -> Optional[List[str]]:
        """Get the stored training schema."""
        return self.get_fit_schema()

    def get_input_columns(self) -> Optional[List[str]]:
        """Columns the pipeline expects at ``transform`` (as seen at ``fit``)."""
        return list(self._input_columns) if self._input_columns is not None else None

    def report_optimization(self, top_n: int = 10) -> None:
        """Print a human-readable optimization report."""
        self._check_fitted()
        self.optimization_report(top_n=top_n)

    def _best_history_entry(self) -> Optional[Dict[str, Any]]:
        """The applied selection's score and size."""
        if self._best_result is not None:
            return self._best_result
        if not self._optimization_history:
            return None
        # Pipeline fitted by a release that did not record the applied result:
        # recover it from the history.
        metric = self._components["selector"].metric if self.has_component("selector") else None
        pick = max if METRIC_DIRECTION.get(metric, "max") == "max" else min
        best = pick(self._optimization_history, key=lambda x: x.score)
        return {"score": best.score, "n_features": best.n_features}

    def get_summary(self) -> Dict[str, Any]:
        """Get pipeline summary."""
        if not self._fitted:
            raise NotFittedError("Pipeline not fitted")

        best = self._best_history_entry()
        selector = self._components.get("selector")
        metric = selector.get_metric() if hasattr(selector, "get_metric") else None

        final_columns = list(self._state.data.columns)
        created = self._state.features_created
        kept = set(final_columns)
        n_input = len(self._input_columns) if self._input_columns is not None else None

        return {
            "features": {
                # Columns of the raw frame passed to fit.
                "initial": n_input if n_input is not None else len(final_columns),
                # Columns of the transformed output.
                "final": len(final_columns),
                # Engineered features created / still present after selection.
                "engineered": len(created),
                "engineered_selected": sum(1 for col in created if col in kept),
            },
            "schema": {
                "columns": self.get_fit_schema(),
                "dtypes": dict(self._state.data.dtypes.astype(str)),
            },
            "task_type": getattr(self._task_type, "value", self._task_type),
            "id_columns": list(self._id_columns),
            "transformations": self._state.transformations_applied,
            "optimization": {
                "best_score": best["score"] if best else None,
                "best_features": best["n_features"] if best else None,
                "iterations": len(self._optimization_history),
                "metric": metric.value if metric is not None and best else None,
            },
            "components": {k: type(v).__name__ for k, v in self._components.items()},
            "version": self._fitted_version,
        }
