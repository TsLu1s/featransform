import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type

from featransform.core.enums import ModelFamily
from featransform.core.exceptions import ConfigurationError
from featransform.core.models import ModelConfig, PipelineConfig
from featransform.features.anomaly import (
    AnomalyEnsemble,
    EllipticEnvelopeStrategy,
    IsolationForestStrategy,
    LocalOutlierFactorStrategy,
    OneClassSVMStrategy,
)
from featransform.features.clustering import (
    BirchStrategy,
    ClusteringEnsemble,
    DBSCANStrategy,
    GaussianMixtureStrategy,
    KMeansStrategy,
    MiniBatchKMeansStrategy,
)
from featransform.features.dimensionality import (
    DimensionalityEnsemble,
    FastICAStrategy,
    PCAStrategy,
    TruncatedSVDStrategy,
)
from featransform.optimization.evaluator import METRIC_DIRECTION
from featransform.optimization.selector import FeatureSelector
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.imputation import CoreImputer

# Parameters the builder injects itself (via config), always accepted regardless
# of a strategy's explicit signature since they flow through **kwargs.
_INJECTED_PARAMS: Set[str] = {"random_state", "n_jobs", "verbose"}


class BuilderPipeline(ABC):
    """
    Abstract builder for pipeline components using strategy pattern.
    Subclasses must implement component selection logic.
    """

    __ANOMALY_STRATEGIES = {
        ModelFamily.ISOLATION_FOREST: IsolationForestStrategy,
        ModelFamily.LOCAL_OUTLIER_FACTOR: LocalOutlierFactorStrategy,
        ModelFamily.ONE_CLASS_SVM: OneClassSVMStrategy,
        ModelFamily.ELLIPTIC_ENVELOPE: EllipticEnvelopeStrategy,
    }

    __CLUSTERING_STRATEGIES = {
        ModelFamily.KMEANS: KMeansStrategy,
        ModelFamily.BIRCH: BirchStrategy,
        ModelFamily.MINI_BATCH_KMEANS: MiniBatchKMeansStrategy,
        ModelFamily.DBSCAN: DBSCANStrategy,
        ModelFamily.GAUSSIAN_MIXTURE: GaussianMixtureStrategy,
    }

    __DIMENSIONALITY_STRATEGIES = {
        ModelFamily.PCA: PCAStrategy,
        ModelFamily.TRUNCATED_SVD: TruncatedSVDStrategy,
        ModelFamily.FAST_ICA: FastICAStrategy,
    }

    __ENSEMBLE_CLASSES = {
        "anomaly": AnomalyEnsemble,
        "clustering": ClusteringEnsemble,
        "dimensionality": DimensionalityEnsemble,
    }

    def __init__(self):
        self._components: Dict[str, Any] = {}
        self._custom_strategies: Dict[str, Dict[ModelFamily, Any]] = {
            "anomaly": {},
            "clustering": {},
            "dimensionality": {},
        }

    @abstractmethod
    def build_components(self, config: PipelineConfig) -> Dict[str, Any]:
        """Build pipeline components from configuration."""
        pass

    @abstractmethod
    def _select_components(self, config: PipelineConfig) -> List[str]:
        """Select which components to build based on config."""
        pass

    def _build_pipeline(self, config: PipelineConfig) -> Dict[str, Any]:
        """Internal pipeline building logic.

        Always starts from an empty registry so a rebuild (new config, refit)
        can never inherit a component the new configuration does not ask for.
        """
        self._components = {}

        self.__build_processors(config)

        for comp_type in self._select_components(config):
            self.__build_feature_component(comp_type, config)

        self.__build_optimizer(config)

        return self._components

    def __build_processors(self, config: PipelineConfig) -> None:
        """Build preprocessing components."""
        if config.processing.imputation_strategy:
            self._components["imputer"] = CoreImputer(
                strategy=config.processing.imputation_strategy,
                random_state=config.random_state,
                verbose=config.verbose,
            )

        if config.processing.encoding_strategy:
            self._components["encoder"] = CategoricalEncoder(
                strategy=config.processing.encoding_strategy, verbose=config.verbose
            )

    def __build_feature_component(self, comp_type: str, config: PipelineConfig) -> None:
        """Build single feature engineering component."""
        models = getattr(config, f"{comp_type}_models", None)
        if not models:
            return

        strategies = self.__create_strategies(comp_type, models, config)
        if strategies:
            self._components[comp_type] = self.__create_ensemble(comp_type, strategies, config)

    def __build_optimizer(self, config: PipelineConfig) -> None:
        """Build optimization component with integrated evaluation."""
        if config.optimization is not None and config.optimization.selection_strategy is not None:
            opt = config.optimization
            self._components["selector"] = FeatureSelector(
                strategy=opt.selection_strategy,
                task_type=config.task_type,
                min_features=opt.min_features,
                metric=opt.metric,
                use_catboost=opt.use_catboost,
                use_xgboost=opt.use_xgboost,
                n_estimators=opt.n_estimators,
                random_state=config.random_state,
                n_jobs=config.n_jobs,
                verbose=config.verbose,
            )

    def __create_strategies(
        self, comp_type: str, models: List[ModelConfig], config: PipelineConfig
    ) -> List[Any]:
        """Create strategy instances for component type."""
        return [
            strategy
            for strategy in [
                self.__create_single_strategy(m, comp_type, config) for m in models if m.enabled
            ]
            if strategy is not None
        ]

    def __create_single_strategy(
        self, model: ModelConfig, comp_type: str, config: PipelineConfig
    ) -> Optional[Any]:
        """Create single strategy instance."""
        strategy_class = self.__get_strategy_class(model.model_family, comp_type)
        if strategy_class is None:
            return None

        # Fail fast on unknown parameters instead of silently dropping them.
        self.__validate_model_params(strategy_class, model)

        params = {**model.parameters}
        # Pipeline-level settings reach every strategy unless the model's own
        # parameters override them. `is not None` keeps an explicit seed of 0.
        injected = {
            "random_state": config.random_state,
            "n_jobs": config.n_jobs,
            "verbose": config.verbose,
        }
        for name, value in injected.items():
            if (
                value is not None
                and name not in params
                and self.__accepts_param(strategy_class, name)
            ):
                params[name] = value

        return strategy_class(**params)

    @classmethod
    def __accepts_param(cls, strategy_class: Type[Any], name: str) -> bool:
        """Whether ``strategy_class(**{name: ...})`` is a valid call."""
        if name in cls.__accepted_params(strategy_class):
            return True
        try:
            signature = inspect.signature(strategy_class.__init__)
        except (TypeError, ValueError):
            return False
        return any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())

    def __validate_model_params(self, strategy_class: Type[Any], model: ModelConfig) -> None:
        """Reject ModelConfig.parameters keys the target strategy doesn't accept."""
        accepted = self.__accepted_params(strategy_class)
        unknown = sorted(
            k for k in model.parameters if k not in accepted and k not in _INJECTED_PARAMS
        )
        if unknown:
            allowed = sorted(accepted | _INJECTED_PARAMS)
            raise ConfigurationError(
                f"{strategy_class.__name__} ({model.model_family.value}) received unknown "
                f"parameter(s) {unknown}. Accepted parameters: {allowed}."
            )

    @staticmethod
    def __accepted_params(strategy_class: Type[Any]) -> Set[str]:
        """Collect explicit __init__ parameter names across the strategy's MRO."""
        accepted: Set[str] = set()
        for klass in strategy_class.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            try:
                signature = inspect.signature(init)
            except (TypeError, ValueError):
                continue
            for name, param in signature.parameters.items():
                if name == "self" or param.kind in (
                    param.VAR_POSITIONAL,
                    param.VAR_KEYWORD,
                ):
                    continue
                accepted.add(name)
        return accepted

    def __get_strategy_class(self, model_family: ModelFamily, comp_type: str) -> Optional[Any]:
        """Get strategy class from mappings."""
        # Check custom strategies first
        if model_family in self._custom_strategies[comp_type]:
            return self._custom_strategies[comp_type][model_family]

        strategy_map = {
            "anomaly": self.__ANOMALY_STRATEGIES,
            "clustering": self.__CLUSTERING_STRATEGIES,
            "dimensionality": self.__DIMENSIONALITY_STRATEGIES,
        }.get(comp_type, {})

        return strategy_map.get(model_family)

    def __create_ensemble(
        self, comp_type: str, strategies: List[Any], config: PipelineConfig
    ) -> Any:
        """Create ensemble from strategies."""
        ensemble_class = self.__ENSEMBLE_CLASSES[comp_type]
        kwargs = {"strategies": strategies, "verbose": config.verbose}

        if comp_type == "anomaly":
            kwargs["n_jobs"] = config.n_jobs

        return ensemble_class(**kwargs)

    def optimization_report(self, top_n: int = 10) -> None:
        """
        Report pipeline summary, optimization results, and feature importance.
        """
        summary = self.get_summary()

        print("\nPipeline Summary:")
        print(f"  Final features: {summary['features']['final']}")
        print(f"  Engineered features: {summary['features']['engineered']}")
        print(f"  Transformations: {summary['transformations']}")

        # ``is None``: a legitimate best score of 0.0 must still be reported.
        if summary["optimization"]["best_score"] is None:
            return

        selector = self._components["selector"]
        metric_used = selector.get_metric() if hasattr(selector, "get_metric") else selector.metric
        metric_name = metric_used.value if metric_used else "auto-detected"
        direction = METRIC_DIRECTION.get(metric_used, "max")
        score = summary["optimization"]["best_score"]

        print(f"  Best optimization score: {score:.4f} ({metric_name}, {direction})")
        print(f"  Best feature count: {summary['optimization']['best_features']}")

        if self.has_component("selector"):
            importance = self._components["selector"].get_feature_scores()
            if importance:
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
                print(f"\nTop {len(top_features)} Important Features:")
                for feat, score in top_features:
                    print(f"  {feat}: {score:.4f}")

    def register_strategy(
        self, model_family: ModelFamily, strategy_class: Any, comp_type: str = "anomaly"
    ) -> None:
        """Register a custom strategy class for ``model_family``.

        ``comp_type`` is one of ``"anomaly"``, ``"clustering"`` or
        ``"dimensionality"``. The class replaces the built-in strategy for that
        family wherever the configuration requests it.
        """
        if comp_type not in self._custom_strategies:
            raise ConfigurationError(
                f"Unknown component type '{comp_type}'. "
                f"Expected one of: {sorted(self._custom_strategies)}."
            )
        if not inspect.isclass(strategy_class):
            raise ConfigurationError(
                f"strategy_class must be a class, got {type(strategy_class).__name__}"
            )
        self._custom_strategies[comp_type][model_family] = strategy_class

    def get_component(self, name: str) -> Optional[Any]:
        """Get component by name."""
        return self._components.get(name)

    def has_component(self, name: str) -> bool:
        """Check if component exists."""
        return name in self._components

    @property
    def components(self) -> Dict[str, Any]:
        """Get all components (read-only)."""
        return self._components.copy()
