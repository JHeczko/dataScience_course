# Data Science - University Course

This repository collects a set of Jupyter notebooks that document the work carried out during a university courses on classical data
science and an associated deep learning course. The goal of each notebook is to present a concrete problem, describe the dataset used,
outline the methods applied (from preprocessing through modelling), and to visualise the results.  The folders are organised by topic, and
within each topic you will find notebooks


---

## Folder structure overview

- **Anomaly_Detection** – examples of detecting unusual observations in time‑series data.
- **Classification** – binary and multiclass classification tasks; model building and evaluation.
- **Clusterization** – unsupervised grouping of data points with K‑means and other algorithms.
- **Data_Exploration** – exploratory data analysis (EDA), statistical summaries and visualisation.
- **Regression** – predicting continuous targets using linear models, trees, and more.
- **Recommender_System** – building recommenders with similarity measures and co‑occurrence.
- **Statistical_Inference** – inferential statistics and hypothesis testing (work in progress).
- **Monte_Carlo (not done)** – Monte Carlo simulation techniques (unfinished).
- **deep_learning** – deep learning experiments using PyTorch and Keras.

Each topic often has subfolders `Basic`, `Advanced`, and `Other` depending on the level of material.

---

## Data Exploration

### Primer
**`classic_data_science/Data_Exploration/Basic/explo_primer.ipynb`**  
A walk‑through of simple exploratory techniques applied to the `kc_house_data.csv` dataset,
which contains real estate transactions in King County, WA.  The notebook loads the CSV, 
computes descriptive statistics, inspects missing values, and visualises distributions
and pairwise relationships using histograms, scatter plots and correlation heatmaps.  The
purpose is to familiarise the reader with pandas, seaborn/plotly, and the iterative EDA process.

### Advanced
**`classic_data_science/Data_Exploration/Advanced/explo_advanced.ipynb`**  
This notebook performs a deeper investigation of two NASA datasets `impacts.csv` and
`orbits.csv` that record potentially hazardous near‑Earth objects.  Methods include data
cleaning, merging tables, temporal analysis, and geospatial plotting.  It demonstrates
techniques such as feature engineering (e.g. calculating orbital parameters), time‑series
analysis, and interactive visualisations.


## Regression

### Primer
**`classic_data_science/Regression/Basic/regression_primer.ipynb`**  
Using the `kc_house_data.csv` dataset again, this notebook introduces simple linear
regression to predict house prices from features like square footage and number of
en‑suites.  It covers train/test splitting, model fitting with scikit‑learn,
coefficients interpretation, residual analysis, and basic performance metrics (MSE, R²).


### Advanced
**`classic_data_science/Regression/Advanced/regression_advanced.ipynb`**  
An expanded regression study on the wine quality datasets (`winequality-red.csv` and
`winequality-white.csv`).  After preprocessing (handling categorical data, scaling,
and combining vintages) numerous models are trained: ridge, lasso, decision trees,
and gradient boosting.  Cross‑validation, hyperparameter tuning, and feature importance
are demonstrated.  The notebook illustrates how to compare model performance across
pipelines and interpret results for continuous target prediction.


### Other
**`classic_data_science/Regression/Other/model_building.ipynb`**  
A separate regression example using `Xtrain.csv`/`ytrain.csv`, from a Kaggle-style
competition. It focuses on feature selection, model stacking, and submission file
creation. The dataset isn't included due to size but the notebook describes the workflow.


## Classification

### Primer
**`classic_data_science/Classification/Basic/classification_primer.ipynb`**  
An introductory classification notebook that could use a standard dataset (e.g.
Iris or similar).  It demonstrates loading data, visualising class separability, training
logistic regression and k‑nearest neighbours, and evaluating with confusion matrices,
precision/recall, and ROC curves.


### Advanced
**`classic_data_science/Classification/Advanced/classification_advanced.ipynb`**  
A more sophisticated classification project using the wine quality CSV files.  The goal
is to predict quality scores converted to binary or multi‑class labels.  Techniques
include feature engineering, scaling, handling class imbalance, training SVMs, random
forests, and ensemble methods.  Model evaluation uses cross‑validation and detailed
classification reports.  The notebook also explores the impact of hyperparameters
and visualises decision boundaries.

### Other
**`classic_data_science/Classification/Other/model_build.ipynb`**  
Additional classification examples from outside the core course. The notebook walks
through training models on an external dataset, likely focusing on rapid prototyping
and baseline comparisons. Data may not be included in the repository.


## Clusterization

### Primer
**`classic_data_science/Clusterization/Basic/clusterization_primer.ipynb`**  
Explores unsupervised learning with K‑means clustering on a simple dataset. Covers
standardisation, choosing `k` with the elbow method and silhouette score, and
visualising clusters.  The notebook highlights how to interpret cluster centroids
and assign labels to new observations.


### Advanced
**`classic_data_science/Clusterization/Advanced/clusterization_advanced.ipynb`**  
Demonstrates more advanced clustering algorithms (e.g. hierarchical clustering,
DBSCAN) and techniques such as dimensionality reduction (PCA, t‑SNE) to visualise
high‑dimensional data.  It may also include a case study on a real-world dataset
with preprocessing steps like outlier removal and feature selection.

### Other
**`classic_data_science/Clusterization/Other/model_building.ipynb`**  
Supplementary clustering work from an external project. Contains experiments
with alternative algorithms and evaluation metrics.  he notebook may be more
experimental in nature and serves as additional practice.


## Anomaly Detection

**`classic_data_science/Anomaly_Detection/model_testing.ipynb`**  
A notebook dedicated to identifying anomalies in a time‑series dataset (`Xtrain.csv`
and `y_train.csv`).  The problem is framed as detecting outliers that deviate
from expected patterns.  Methods could include statistical thresholds, rolling means,
Isolation Forest, or autoencoder‑based approaches.  The notebook compares detection
performance and visualises anomalous points.


## Recommender System

Two notebooks illustrate collaborative filtering and content‑based recommendations
using Steam games and user behaviour data.

* **`classic_data_science/Recommender_System/preprocessing.ipynb`** – data cleaning,
  transformation, and creation of user‑item matrices.
* **`classic_data_science/Recommender_System/recommending-system.ipynb`** – builds
  recommendation models:
  - **Product similarity** using game descriptions and categories (content‑based).
  - **Co‑occurrence matrix** constructing a user‑item co‑occurrence for
    collaborative filtering.  It includes similarity computations, ranking, and
    evaluation of recommendations.

The notebooks explain the underlying mathematics (cosine similarity, Pearson
correlation) and discuss cold‑start issues.


## Statistical Inference (work in progress)

**`classic_data_science/Statistical_Inference/statistical_inference.ipynb`**  
Contains exercises on hypothesis testing, confidence intervals, and parameter
estimation. Datasets may include samples drawn from known distributions.  This
folder is marked "not done"; the notebook may be incomplete but outlines classical
statistical procedures relevant to data science.


## Monte Carlo (work in progress)

**`classic_data_science/Monte_Carlo/montecarlo.ipynb`**  
An unfinished notebook intending to demonstrate Monte Carlo simulation techniques
for numerical integration, option pricing, or risk assessment.  It likely contains
initial code scaffolding and explanatory comments but lacks full experiments.


## Deep Learning

The `deep_learning` directory houses various experiments carried out with **PyTorch and Keras**. The notebooks cover feed‑forward neural networks (FFNN), convolutional networks, recurrent architectures, and hyperparameter tuning with Ray Tune. Specific files include:

* **`Highway_FFNN.ipynb`** – implements highway networks for regression or
  classification tasks, showing how skip connections can ease training.
* **`Keras_FFNN_regression.ipynb`** – builds a dense neural network in Keras to
  perform regression on a dataset (likely housing or wine).  Includes data scaling,
  training/validation splits, and callbacks.
* **`LSTM_ATT_token_generator.ipynb`** – constructs a character‑level or word‑level
  language model using LSTM with attention for sequence generation.  Explores
  text preprocessing, embedding layers, and sampling strategies.
* **`Raytune_on_highwayFFNN.ipynb`** – uses Ray Tune to perform automated
  hyperparameter search on the Highway FFNN architecture.  Demonstrates parallel
  tuning and result analysis.
* **`SimpleTorchConv.ipynb`** – builds a basic convolutional network in PyTorch
  for tasks such as MNIST digit classification. Includes forward/backward passes
  and loss tracking.
* **`Torch_FFNN_regression.ipynb`** – implements a feed‑forward neural network using
  PyTorch for a regression problem, illustrating manual training loops and
  tensor operations.

### `ResNet.ipynb`
It is the most important notebook here which was the one of the final proejct for this course. It recreates a residual convolutional network (ResNet) for image data. Contains data augmentation, training loops, and evaluation on a small image dataset. This notebook containt full paper based implementantion of ResNet model

The notebook contains:
1. Full paper-based implementation of ResNet34 and ResNet56.
2. Implemeneted standard and bottleneck blocks for ResNet interface
3. Nice and adjustable interface of the ResNet architecture
4. Training cycle with optimal arguments
5. Trained on **Cifar100**

**The outcome:**
- **ResNet34** scored 78% accuracy
- **ResNet56** scored 71% accuracy


---

## How to use this repository

1. **Install dependencies** – most notebooks rely on standard packages:
   pandas, numpy, scikit‑learn, matplotlib/seaborn, plotly, PyTorch, and Keras.
   Create a virtual environment and install via `pip install -r requirements.txt`
   if available, or manually add packages.
2. **Open a notebook** – launch Jupyter or VS Code's notebook support and navigate
   to the desired `.ipynb` file.
3. **Execute cells in order** – some notebooks assume earlier cells define functions
   or load data.  Running all cells sequentially reproduces the analysis.
4. **Inspect data paths** – many notebooks read CSV files from nearby `data/`
   subdirectories. Ensure these files exist or adjust paths accordingly.

Feel free to adapt the notebooks for your own datasets or extend them with new
models and visualisations.  The structure is intended to guide learners from
fundamental techniques to advanced modelling practices.