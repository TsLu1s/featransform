[![LinkedIn][linkedin-shield]][linkedin-url]
[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![MIT License][license-shield]][license-url]
[![Downloads][downloads-shield]][downloads-url]
[![Month Downloads][downloads-month-shield]][downloads-month-url]

[contributors-shield]: https://img.shields.io/github/contributors/TsLu1s/Featransform.svg?style=for-the-badge&logo=github&logoColor=white
[contributors-url]: https://github.com/TsLu1s/Featransform/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/TsLu1s/Featransform.svg?style=for-the-badge&logo=github&logoColor=white
[stars-url]: https://github.com/TsLu1s/Featransform/stargazers
[license-shield]: https://img.shields.io/github/license/TsLu1s/Featransform.svg?style=for-the-badge&logo=opensource&logoColor=white
[license-url]: https://github.com/TsLu1s/Featransform/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/luisfssantos98/
[downloads-shield]: https://static.pepy.tech/personalized-badge/featransform?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Total%20Downloads
[downloads-url]: https://pepy.tech/project/featransform
[downloads-month-shield]: https://static.pepy.tech/personalized-badge/featransform?period=month&units=international_system&left_color=grey&right_color=blue&left_text=Month%20Downloads
[downloads-month-url]: https://pepy.tech/project/featransform

<br>
<p align="center">
  <h2 align="center"> Featransform: Automated Feature Engineering Framework for Supervised Machine Learning
  <br>

## Framework Contextualization <a name = "ta"></a>

The `Featransform` project constitutes an objective and modern proposition to automate feature engineering framework through the integration of various approachs of input pattern recognition known in Machine Learning such as dimensionality reduction, anomaly detection, clustering approaches and datetime feature constrution. Built with advanced design patterns and a modular architecture, it seamlessly orchestrates multiple feature engineering techniques including anomaly detection, clustering, dimensionality reduction, and temporal feature extraction—all optimized through intelligent validation-driven feature selection.

In order to avoid generation of noisy data for predictive consumption, after the engineered features ensemble are concatenated with the original features, a backwards wrapper feature selection also known as backward elimination is implemented to iteratively remove features based on evaluation of relevance, maintaining only valuable columns available for future models performance improvement purposes.

The architecture design includes three main sections, these being: data preprocessing, diverse feature engineering ensembles and optimized feature selection validation.

This project aims at providing the following application capabilities:

* General applicability on tabular datasets: The developed feature engineering procedures are applicable on any data table associated with any Supervised ML scopes, based on input data columns to be built up on.
    
* Improvement of predictive results: The application of the `Featransform` aims at improve the predictive performance of future applied Machine Learning models through added feature construction, increased pattern recognition and optimization of existing input features.

* Continuous integration: After the train data is fitted, the created object can be saved and implemented in future data with the same structure. 
   
#### Main Development Tools <a name = "pre1"></a>

Major frameworks used to built this project: 

* [Pandas](https://pandas.pydata.org/)
* [Sklearn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [Optuna](https://optuna.org/)
    
## Where to get it <a name = "ta"></a>
    
Binary installer for the latest released version is available at the Python Package Index [(PyPI)](https://pypi.org/project/featransform/).   

GitHub Project Link: [https://github.com/TsLu1s/Featransform](https://github.com/TsLu1s/Featransform)

## Installation  

To install this package from Pypi repository run the following command:

```
pip install featransform
```

# Usage Example
    
## Featransform - Automated Feature Engineering Pipeline

# Usage Example
    
## Featransform - Automated Feature Engineering Pipeline

In order to be able to apply the automated feature engineering `featransform` pipeline you need first to import the package. 
The following needed step is to load a dataset and define your to be predicted target column name into the variable `target`.
You can customize the pipeline by selecting from preset configurations (`minimal`, `standard`, `optimized`, `complete`) and specifying the task type (`regression`, `binary_classification`, `multiclass_classification`).

```py

import pandas as pd
from sklearn.model_selection import train_test_split
from featransform.pipeline import Featransform
from featransform.configs.baseline import FTconfig
from featransform.utils.serializer import PipelineSerializer
from featransform.utils.data_generator import make_dataset
import warnings
warnings.filterwarnings("ignore", category=Warning)

# Generate sample dataset
X, y = make_dataset(task='multiclass_classification', n_samples=5000, complexity='medium', n_classes=3)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline with preset configuration
config = FTconfig.complete(task_type="multiclass_classification")
# Available presets: minimal(), standard(), optimized(), complete()
# Task types: "regression", "binary_classification", "multiclass_classification"

pipeline = Featransform(config)

# Fit and transform
pipeline.fit(X_train, y_train)
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# View optimization results
pipeline.report_optimization()

# Save & Load Pipeline
serializer = PipelineSerializer()
serializer.save(pipeline, 'fitted_pipeline.pkl')

loaded_pipeline = serializer.load('fitted_pipeline.pkl')
X_loaded_transform = loaded_pipeline.transform(X_test)

```

## Usage Interactive Examples

Further automated and customizable feature engineering applications:

* [Baseline Example](https://github.com/TsLu1s/featransform/blob/main/examples/baseline_featransform.py) - Get started right way with intelligent preset configurations, synthetic dataset generation, and seamless pipeline serialization for production deployment
* [Advanced Configuration](https://github.com/TsLu1s/featransform/blob/main/examples/automated_featransform.py) - Build fully customized pipelines from scratch with complete control over preprocessing strategies, feature engineering components, and optimization parameters
* [Component Testing](https://github.com/TsLu1s/featransform/blob/main/examples/decomposed_featransform.py) - Deep-dive into individual components with comprehensive train-test evaluation across encoding, imputation, anomaly detection, clustering, and dimensionality reduction methods


**Prefer interactive notebooks?** Check out the [notebook examples](https://github.com/TsLu1s/featransform/blob/main/examples/notebooks) with step-by-step execution.

## Core Capabilities

**Feature Engineering Methods:**
- Anomaly Detection (Isolation Forest, LOF, One-Class SVM, Elliptic Envelope)
- Clustering (KMeans, Birch, DBSCAN, Gaussian Mixture)
- Dimensionality Reduction (PCA, SVD, FastICA)
- Temporal Features (Cyclic encoding, datetime decomposition)

**Intelligent Processing:**
- Advanced Imputation (Mean, Median, Iterative, KNN)
- Categorical Encoding (Label)
- Automated Feature Selection (Importance-based)

## Built With

Major frameworks used to build this project:

* [Pandas](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)
* [CatBoost](https://catboost.ai/)
* [Pydantic](https://docs.pydantic.dev/)

## Citation

If you use Featransform in your research, please cite:

```bibtex
@software{featransform2023,
  author = {Luis Fernando Santos},
  title = {Featransform: Automated Feature Engineering Framework for Supervised Machine Learning},
  year = {2023},
  publisher = {PyPI},
  url = {https://pypi.org/project/featransform/}
}
```

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TsLu1s/featransform/blob/main/LICENSE) for more information.

## Contact 
 
[Luis Santos - LinkedIn](https://www.linkedin.com/in/luisfssantos98/)







