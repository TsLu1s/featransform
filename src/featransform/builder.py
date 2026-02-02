from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from featransform.core.enums import ModelFamily, OptimizationMetric
from featransform.core.models import ModelConfig, PipelineConfig

# Component imports
from featransform.features.anomaly import *
from featransform.features.clustering import *
from featransform.features.dimensionality import *
from featransform.optimization.selector import FeatureSelector
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.imputation import CoreImputer


class BuilderPipeline(ABC):
    """
    Abstract builder for pipeline components using strategy pattern.
    Subclasses must implement component selection logic.
    """

    # Private strategy mappings
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
        """Internal pipeline building logic."""
        # Build processing components
        self.__build_processors(config)

        # Build feature engineering components
        for comp_type in self._select_components(config):
            self.__build_feature_component(comp_type, config)

        # Build optimization component
        self.__build_optimizer(config)

        return self._components

    def __build_processors(self, config: PipelineConfig) -> None:
        """Build preprocessing components."""
        if config.processing.imputation_strategy:
            self._components["imputer"] = CoreImputer(
                strategy=config.processing.imputation_strategy, verbose=config.verbose
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
        if config.optimization and config.optimization.selection_strategy:
            opt = config.optimization
            self._components["selector"] = FeatureSelector(
                strategy=opt.selection_strategy,
                min_features=opt.min_features,
                metric=opt.metric,
                use_catboost=opt.use_catboost,
                use_xgboost=opt.use_xgboost,
                n_estimators=opt.n_estimators,
                random_state=config.random_state,
                verbose=config.verbose,
            )

    def __create_strategies(
        self, comp_type: str, models: List[ModelConfig], config: PipelineConfig
    ) -> List[Any]:
        """Create strategy instances for component type."""
        return [
            strategy
            for strategy in [
                self.__create_single_strategy(m, comp_type, config.random_state)
                for m in models
                if m.enabled
            ]
            if strategy is not None
        ]

    def __create_single_strategy(
        self, model: ModelConfig, comp_type: str, random_state: Optional[int] = None
    ) -> Optional[Any]:
        """Create single strategy instance."""
        params = {**model.parameters}
        if random_state and "random_state" not in params:
            params["random_state"] = random_state

        # Get strategy class from mappings
        strategy_class = self.__get_strategy_class(model.model_family, comp_type)
        return strategy_class(**params) if strategy_class else None

    def __get_strategy_class(self, model_family: ModelFamily, comp_type: str) -> Optional[Any]:
        """Get strategy class from mappings."""
        # Check custom strategies first
        if model_family in self._custom_strategies[comp_type]:
            return self._custom_strategies[comp_type][model_family]

        # Check built-in strategies
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

        if not summary["optimization"]["best_score"]:
            return

        # Report optimization score
        metric_used = self._components["selector"].metric
        metric_name = metric_used.value if metric_used else "auto-detected"

        # Negated metrics should be displayed as positive
        negated_metrics = [OptimizationMetric.MSE, OptimizationMetric.MAE, OptimizationMetric.RMSE]

        score = summary["optimization"]["best_score"]
        display_score = abs(score) if metric_used in negated_metrics else score

        print(f"  Best optimization score: {display_score:.4f} ({metric_name})")
        print(f"  Best feature count: {summary['optimization']['best_features']}")

        # Report feature importance
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
        """Register custom strategy."""
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
