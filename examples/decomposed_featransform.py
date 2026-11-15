import pandas as pd
from sklearn.model_selection import train_test_split
from featransform.utils.data_generator import DatasetGenerator
from featransform.processing.imputation import CoreImputer
from featransform.processing.encoding import CategoricalEncoder
from featransform.processing.datetime_features import DatetimeExpander
from featransform.processing.scaling import FeatureScaler
from featransform.features.anomaly import AnomalyEnsemble, IsolationForestStrategy #, EllipticEnvelopeStrategy, OneClassSVMStrategy, LocalOutlierFactorStrategy
from featransform.features.clustering import ClusteringEnsemble, KMeansStrategy # DBSCANStrategy, BirchStrategy, MiniBatchKMeansStrategy, GaussianMixtureStrategy
from featransform.features.dimensionality import DimensionalityEnsemble, PCAStrategy
from featransform.optimization.selector import FeatureSelector
from featransform.core.enums import (
    ImputationStrategy, EncodingStrategy, ScalingStrategy, SelectionStrategy
)

# Identifier / index columns: they pass through to the output but must not act as
# imputation predictors or feed the distance / variance based ensembles.
ID_COLUMNS = ["id"]

print("\n" + "=" * 70)
print(" COMPLETE PIPELINE - TRAIN/TEST FLOW")
print("=" * 70)

# Generate a sample dataset — for demonstration only.
# When using Featransform for real, swap this for your own data (X) and target (y).
X_full, y_full = DatasetGenerator.complex_dataset(
    n_samples=10000,
    task='binary_classification'  # Options: 'regression', 'binary_classification', 'multiclass_classification'
)

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.3, random_state=42
)
print("\nComplex dataset shapes:")
print(f"  Train: {X_train_full.shape}")
print(f"  Test: {X_test_full.shape}")
print(f"  Dtypes: {X_full.dtypes.value_counts().to_dict()}")
print(f"  Train missing: {X_train_full.isnull().sum().sum()}")
print(f"  Test missing: {X_test_full.isnull().sum().sum()}")

# Step 2: Datetime expansion
print("\n2. Datetime expansion...")
# Datetime columns cannot be consumed by the scikit-learn estimators behind the
# anomaly / clustering / dimensionality components, so expand them into calendar
# parts and cyclic encodings first. The full Featransform pipeline does this
# automatically (ProcessingConfig.handle_datetime); component-level usage calls it directly.
dt_expander = DatetimeExpander()
dt_expander.fit(X_train_full)
X_train_full = dt_expander.transform(X_train_full)
X_test_full = dt_expander.transform(X_test_full)
print(f"  Expanded datetime columns: {dt_expander.get_feature_names()}")
print(f"  Shape after expansion: Train {X_train_full.shape}, Test {X_test_full.shape}")

# Step 3: Encoding
print("\n3. Categorical encoding...")
# Available encoding strategies:
# - EncodingStrategy.LABEL: Label encoding for ordinal categories
encoder = CategoricalEncoder(strategy=EncodingStrategy.LABEL)
encoder.fit(X_train_full, y_train_full)
X_train_proc = encoder.transform(X_train_full)
X_test_proc = encoder.transform(X_test_full)
print(f"  Shape after encoding: Train {X_train_proc.shape}, Test {X_test_proc.shape}")

# Step 4: Imputation
print("\n4. Imputation...")
# Available imputation strategies:
# - ImputationStrategy.MEAN: Replace with mean (numeric only)
# - ImputationStrategy.MEDIAN: Replace with median (numeric only)
# - ImputationStrategy.MODE: Replace with most frequent value
# - ImputationStrategy.ITERATIVE: Iterative imputation using other features
# - ImputationStrategy.KNN: K-Nearest Neighbors imputation (run in standardized space)
imputer = CoreImputer(strategy=ImputationStrategy.ITERATIVE, exclude_columns=ID_COLUMNS)
imputer.fit(X_train_proc)
X_train_proc = imputer.transform(X_train_proc)
X_test_proc = imputer.transform(X_test_proc)
print(f"  Train missing after: {X_train_proc.isnull().sum().sum()}")
print(f"  Test missing after: {X_test_proc.isnull().sum().sum()}")

# Step 5: Feature engineering
print("\n5. Feature engineering...")

# The ensembles below are distance / variance based, so they get a dedicated input:
# identifier columns removed and every column standardized. The processed frame
# itself (X_train_proc / X_test_proc) stays in its native scale and is what the
# engineered features are concatenated onto. The full Featransform pipeline does
# both automatically (ProcessingConfig.id_columns / auto_detect_id, scaling_strategy).
# Available scaling strategies: ScalingStrategy.STANDARD, ROBUST, MINMAX
scaler = FeatureScaler(strategy=ScalingStrategy.STANDARD)
X_train_fe = scaler.fit(X_train_proc.drop(columns=ID_COLUMNS)).transform(
    X_train_proc.drop(columns=ID_COLUMNS)
)
X_test_fe = scaler.transform(X_test_proc.drop(columns=ID_COLUMNS))

# ANOMALY DETECTION OPTIONS:
# Available strategies (can use multiple):
# - IsolationForestStrategy(n_estimators=100, contamination=0.1, include_scores=False)
# - LocalOutlierFactorStrategy(n_neighbors=20, contamination=0.1, include_scores=False)
# - OneClassSVMStrategy(nu=0.05, kernel='rbf', contamination=0.1, include_scores=False)
# - EllipticEnvelopeStrategy(contamination=0.1, include_scores=False)
anomaly_strategies = [
    IsolationForestStrategy(n_estimators=100, contamination=0.1),
    # LocalOutlierFactorStrategy(n_neighbors=20),
    # OneClassSVMStrategy(nu=0.05, kernel='rbf'),
    # EllipticEnvelopeStrategy(contamination=0.1)  # May cause warnings with high-dim data
]
anomaly_ens = AnomalyEnsemble(strategies=anomaly_strategies, verbose=False)
anomaly_ens.fit(X_train_fe)
train_anom = anomaly_ens.transform(X_train_fe).features
test_anom = anomaly_ens.transform(X_test_fe).features

# CLUSTERING OPTIONS:
# Available strategies (can use multiple):
# - KMeansStrategy(n_clusters=3, random_state=42)
# - BirchStrategy(n_clusters=3, threshold=0.5, branching_factor=50)
# - MiniBatchKMeansStrategy(n_clusters=3, batch_size=100, random_state=42)
# - DBSCANStrategy(eps="auto", min_samples=5)  # Note: doesn't use n_clusters;
#   eps="auto" derives the radius from the data, or pass a number to pin it
# - GaussianMixtureStrategy(n_clusters=3, covariance_type='full', random_state=42)
#   covariance_type options: 'full', 'tied', 'diag', 'spherical'
clustering_strategies = [
    KMeansStrategy(n_clusters=3, random_state=42),
    # BirchStrategy(n_clusters=3, threshold=0.5),
    # MiniBatchKMeansStrategy(n_clusters=3, batch_size=100),
    # DBSCANStrategy(eps="auto", min_samples=5),
    # GaussianMixtureStrategy(n_clusters=3, covariance_type='full')
]
cluster_ens = ClusteringEnsemble(strategies=clustering_strategies, verbose=False)
cluster_ens.fit(X_train_fe)
train_clust = cluster_ens.transform(X_train_fe).features
test_clust = cluster_ens.transform(X_test_fe).features

# DIMENSIONALITY REDUCTION
# Available strategies:
# - PCAStrategy(n_components=0.95)  # Keep 95% variance or specify number
# - TruncatedSVDStrategy(n_components=10)  # For sparse data
# - FastICAStrategy(n_components=10, algorithm='parallel', whiten='unit-variance')
print("  - Dimensionality reduction...")
dim_strategies = [
    PCAStrategy(n_components=5),
    # TruncatedSVDStrategy(n_components=10)
]
dim_ens = DimensionalityEnsemble(strategies=dim_strategies, verbose=False)
dim_ens.fit(X_train_fe)  # unsupervised: the target is not needed
train_dim = dim_ens.transform(X_train_fe).features
test_dim = dim_ens.transform(X_test_fe).features
print(f"    Added {train_dim.shape[1]} dimensionality features")

X_train_all = pd.concat([
    X_train_proc,
    train_anom,
    train_clust,
    train_dim
], axis=1)

X_test_all = pd.concat([
    X_test_proc,
    test_anom,
    test_clust,
    test_dim
], axis=1)
print("\n  Combined feature matrix:")
print(f"    Train: {X_train_proc.shape[1]} original + {train_anom.shape[1]} anomaly + {train_clust.shape[1]} cluster + {train_dim.shape[1]} dim = {X_train_all.shape[1]} total")
print(f"    Test:  {X_test_proc.shape[1]} original + {test_anom.shape[1]} anomaly + {test_clust.shape[1]} cluster + {test_dim.shape[1]} dim = {X_test_all.shape[1]} total")


# Step 6: Feature selection
print("\n6. Feature selection...")
selector_final = FeatureSelector(
    strategy=SelectionStrategy.IMPORTANCE,
    min_features=10,
    verbose=False
)

selector_final.fit(X_train_all, y_train_full)

threshold = 0.7  # Keep features covering 70% of cumulative importance (at least min_features)
selected_features = selector_final.select_by_threshold(threshold)
selector_final.set_selected_features(selected_features)

X_train_final = selector_final.transform(X_train_all)
X_test_final = selector_final.transform(X_test_all)
print(f"  Final shapes after selection: Train {X_train_final.shape}, Test {X_test_final.shape}")

print(f"  Selected features: {len(selected_features)} features retained")

scores = selector_final.get_feature_scores()
if scores:
    top_10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 features by importance:")
    for feat, score in top_10:
        print(f"    {feat}: {score:.4f}")

print("\n" + "=" * 70)
print("TRAIN/TEST COMPONENT TESTING COMPLETED SUCCESSFULLY!")
print("=" * 70)
