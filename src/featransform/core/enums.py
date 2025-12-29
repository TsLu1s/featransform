from enum import Enum

class ModelFamily(str, Enum):
    """Available model families."""
    # Anomaly Detection
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    
    # Clustering
    KMEANS = "kmeans"
    BIRCH = "birch"
    MINI_BATCH_KMEANS = "mini_batch_kmeans"
    DBSCAN = "dbscan"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    
    # Dimensionality dimensionality
    PCA = "pca"
    TRUNCATED_SVD = "truncated_svd"
    FAST_ICA = "fast_ica"
    
    # Feature Selection
    CATBOOST = "catboost"
    XGBOOST = "xgboost"


class FeatureType(str, Enum):
    """Types of engineered features."""
    ANOMALY = "anomaly"
    CLUSTER = "cluster"
    DIMENSION = "dimension"
    ORIGINAL = "original"


class TaskType(str, Enum):
    """Machine learning task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class ImputationStrategy(str, Enum):
    """Missing value imputation strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "most_frequent"
    CONSTANT = "constant"
    ITERATIVE = "iterative"
    KNN = "knn"


class EncodingStrategy(str, Enum):
    """Categorical encoding strategies."""
    LABEL = "label"
    TARGET = "target"

class SelectionStrategy(str, Enum):
    """Feature selection strategies."""
    IMPORTANCE = "importance"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    RECURSIVE = "recursive"


class OptimizationMetric(str, Enum):
    """Optimization metrics."""
    # Regression
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC_ROC = "auc_roc"


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    PREPROCESSING = "preprocessing"
    IMPUTATION = "imputation"
    ENCODING = "encoding"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    EVALUATION = "evaluation"