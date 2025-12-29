
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional, Literal
from sklearn.datasets import make_classification, make_regression


class DatasetGenerator:
    """Generate synthetic datasets for testing Featransform pipeline."""
    
    @staticmethod
    def generate(
        task: Literal['binary_classification', 'multiclass_classification', 'regression'] = 'binary_classification',
        n_samples: int = 5000,
        n_features: int = 25,
        n_informative: int = 10,
        n_redundant: int = 2,
        n_classes: int = 2,
        n_targets: int = 1,
        noise: float = 0.1,
        class_sep: float = 1.0,
        flip_y: float = 0.01,
        weights: Optional[list] = None,
        # Additional features
        add_datetime: bool = True,
        n_datetime_cols: int = 2,
        add_categorical: bool = True,
        n_categorical: int = 3,
        cat_cardinality: int = 5,
        add_missing: bool = True,
        missing_rate: float = 0.1,
        add_id_column: bool = True,
        # Output options
        feature_names: Optional[list] = None,
        target_name: str = 'target',
        as_frame: bool = True,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic dataset with various feature types.
        
        Args:
            task: Type of ML task ('binary', 'multiclass', 'regression')
            n_samples: Number of samples to generate
            n_features: Total number of numeric features
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_classes: Number of classes (for classification)
            n_targets: Number of targets (for multi-output)
            noise: Standard deviation of Gaussian noise
            class_sep: Factor multiplying hypercube size (classification)
            flip_y: Fraction of samples with randomly flipped class
            weights: Weights for each class
            add_datetime: Whether to add datetime features
            n_datetime_cols: Number of datetime columns
            add_categorical: Whether to add categorical features
            n_categorical: Number of categorical features
            cat_cardinality: Number of unique values per categorical
            add_missing: Whether to add missing values
            missing_rate: Proportion of missing values
            add_id_column: Whether to add ID column
            feature_names: Custom feature names
            target_name: Name for target column
            as_frame: Return as DataFrame/Series (vs numpy arrays)
            random_state: Random seed for reproducibility
            
        Returns:
            X: Feature DataFrame
            y: Target Series
        """
        np.random.seed(random_state)
        
        # Generate base dataset
        if task == 'regression':
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_targets=n_targets,
                noise=noise,
                random_state=random_state
            )
        
        elif task == 'binary_classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=2,
                n_clusters_per_class=2,
                weights=weights,
                flip_y=flip_y,
                class_sep=class_sep,
                random_state=random_state
            )
        
        elif task == 'multiclass_classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=n_classes,
                n_clusters_per_class=2,
                weights=weights,
                flip_y=flip_y,
                class_sep=class_sep,
                random_state=random_state
            )
        
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # Convert to DataFrame
        if not feature_names:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        X = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
        
        # Add ID column
        if add_id_column:
            X.insert(0, 'id', range(1, len(X) + 1))
        
        # Add datetime features
        if add_datetime:
            X = DatasetGenerator._add_datetime_features(X, n_datetime_cols, random_state)
        
        # Add categorical features
        if add_categorical:
            X = DatasetGenerator._add_categorical_features(
                X, n_categorical, cat_cardinality, random_state
            )
        
        # Add missing values
        if add_missing:
            X = DatasetGenerator._add_missing_values(X, missing_rate, random_state)
        
        # Convert target to Series
        y = pd.Series(y, name=target_name)
        
        if not as_frame:
            return X.values, y.values
        
        return X, y
    
    @staticmethod
    def _add_datetime_features(
        X: pd.DataFrame,
        n_cols: int,
        random_state: int
    ) -> pd.DataFrame:
        """Add datetime columns with various patterns."""
        np.random.seed(random_state)
        
        base_date = datetime(2020, 1, 1)
        
        for i in range(n_cols):
            if i == 0:
                # Sequential dates
                dates = [base_date + timedelta(days=j) for j in range(len(X))]
            elif i == 1:
                # Random dates within 2 years
                days = np.random.randint(0, 730, len(X))
                dates = [base_date + timedelta(days=int(d)) for d in days]
            else:
                # Dates with hourly precision
                hours = np.random.randint(0, 24*365, len(X))
                dates = [base_date + timedelta(hours=int(h)) for h in hours]
            
            X[f'datetime_{i}'] = pd.to_datetime(dates)
        
        return X
    
    @staticmethod
    def _add_categorical_features(
        X: pd.DataFrame,
        n_cats: int,
        cardinality: int,
        random_state: int
    ) -> pd.DataFrame:
        """Add categorical columns."""
        np.random.seed(random_state)
        
        for i in range(n_cats):
            if i == 0:
                # Ordinal categories
                categories = [f'level_{j}' for j in range(cardinality)]
            elif i == 1:
                # Nominal categories
                categories = [f'cat_{chr(65+j)}' for j in range(cardinality)]
            else:
                # Mixed alphanumeric
                categories = [f'type_{j}{chr(65+j%26)}' for j in range(cardinality)]
            
            X[f'categorical_{i}'] = np.random.choice(categories, len(X))
            X[f'categorical_{i}'] = X[f'categorical_{i}'].astype('category')
        
        return X
    
    @staticmethod
    def _add_missing_values(
        X: pd.DataFrame,
        missing_rate: float,
        random_state: int
    ) -> pd.DataFrame:
        """Add missing values to numeric columns."""
        np.random.seed(random_state)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        # Don't add missing to ID column
        numeric_cols = [col for col in numeric_cols if col != 'id']
        
        for col in numeric_cols:
            mask = np.random.random(len(X)) < missing_rate
            X.loc[mask, col] = np.nan
        
        return X
    
    @staticmethod
    def quick_regression(n_samples: int = 1000, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Quick regression dataset generation."""
        return DatasetGenerator.generate(
            task='regression',
            n_samples=n_samples,
            n_features=20,
            n_informative=18,
            noise=10.0,
            **kwargs
        )
    
    @staticmethod
    def quick_binary(n_samples: int = 1000, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Quick binary classification dataset."""
        return DatasetGenerator.generate(
            task='binary',
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            **kwargs
        )
    
    @staticmethod
    def quick_multiclass(n_samples: int = 1000, n_classes: int = 3, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Quick multiclass dataset."""
        return DatasetGenerator.generate(
            task='multiclass',
            n_samples=n_samples,
            n_features=20,
            n_informative=15,
            n_classes=n_classes,
            **kwargs
        )
    
    @staticmethod
    def complex_dataset(n_samples: int = 2000, task: str = 'binary', **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate complex dataset with all feature types."""
        return DatasetGenerator.generate(
            task=task,
            n_samples=n_samples,
            n_features=30,
            n_informative=20,
            add_datetime=True,
            n_datetime_cols=0,
            add_categorical=True,
            n_categorical=5,
            cat_cardinality=7,
            add_missing=True,
            missing_rate=0.15,
            add_id_column=True,
            **kwargs
        )
    
    @staticmethod
    def benchmark_datasets() -> dict:
        """Generate multiple datasets for benchmarking."""
        datasets = {}
        
        # Small datasets
        datasets['small_binary'] = DatasetGenerator.quick_binary(n_samples=1000)
        datasets['small_multiclass'] = DatasetGenerator.quick_multiclass(n_samples=1000)
        datasets['small_regression'] = DatasetGenerator.quick_regression(n_samples=1000)
        
        # Medium datasets
        datasets['medium_binary'] = DatasetGenerator.complex_dataset(n_samples=5000, task='binary_classification')
        datasets['medium_multiclass'] = DatasetGenerator.complex_dataset(n_samples=5000, task='multiclass_classification')
        datasets['medium_regression'] = DatasetGenerator.complex_dataset(n_samples=5000, task='regression')
        
        # Large datasets
        datasets['large_binary'] = DatasetGenerator.complex_dataset(n_samples=20000, task='binary_classification')
        datasets['large_multiclass'] = DatasetGenerator.complex_dataset(n_samples=20000, task='multiclass_classification')
        datasets['large_regression'] = DatasetGenerator.complex_dataset(n_samples=20000, task='regression')
        
        return datasets


# Convenience functions
def make_dataset(
    task: str = 'binary',
    n_samples: int = 10000,
    complexity: Literal['simple', 'medium', 'complex'] = 'medium',
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function for quick dataset generation.

    Returns:
        X: Features DataFrame
        y: Target Series
    """
    if complexity == 'simple':
        return DatasetGenerator.generate(
            task=task,
            n_samples=n_samples,
            add_datetime=False,
            add_categorical=False,
            add_missing=False,
            add_id_column=False,
            **kwargs
        )
    
    elif complexity == 'medium':
        return DatasetGenerator.generate(
            task=task,
            n_samples=n_samples,
            add_datetime=True,
            n_datetime_cols=0,
            add_categorical=True,
            n_categorical=3,
            add_missing=True,
            missing_rate=0.1,
            **kwargs
        )
    
    else:  # complex
        return DatasetGenerator.complex_dataset(
            n_samples=n_samples,
            task=task,
            **kwargs
        )