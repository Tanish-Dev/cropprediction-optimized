al# Agricultural Crop Yield Prediction (CYP)

## Overview

Agricultural Crop Yield Prediction (CYP) is an end-to-end machine learning pipeline designed to forecast crop yields using agricultural, environmental, and temporal data collected across 101 countries and 10 crop types from 1990–2013.

The system combines traditional machine learning techniques with a hybrid deep learning architecture consisting of a 1D Convolutional Neural Network (CNN) and a Recursive Bidirectional Long Short-Term Memory (BiLSTM) network. The pipeline performs data analysis, preprocessing, feature selection, model training, evaluation, and interactive visualization through a Dash dashboard.

---

## Dataset

| Metric          | Value         |
| --------------- | ------------- |
| Total Samples   | 25,932        |
| Countries       | 101           |
| Crop Types      | 10            |
| Time Period     | 1990–2013     |
| Source Datasets | 9             |
| Target Variable | Yield (hg/ha) |

---

## System Workflow

```text
Raw Datasets
     │
     ▼
Data Analysis
     │
     ▼
Preprocessing
     │
     ▼
Feature Selection
     │
     ▼
Model Training
     │
     ▼
Evaluation
     │
     ▼
Interactive Dashboard
```

### 1. Data Analysis

The initial stage performs exploratory analysis on all source datasets, including:

* Statistical profiling
* Missing value analysis
* Correlation analysis
* Outlier detection
* Yield trend visualization

---

### 2. Data Preprocessing

The consolidated dataset is cleaned and transformed using:

* Smart missing-value imputation
* Label encoding of categorical features
* StandardScaler normalization
* IQR-based outlier clipping
* Train / Validation / Test splitting

---

### 3. Feature Selection

A hybrid feature selection strategy combines:

#### LASSO Regression

LassoCV is used to rank features and remove less informative variables through L1 regularization.

#### Selfish Herd Optimization (SHO)

A custom metaheuristic optimization algorithm is applied to identify the most effective feature subset.

Selected Features:

* Year
* Pesticides (tonnes)
* Average Temperature
* Crop Item

---

### 4. Model Training

Five models are trained and compared:

* Random Forest Regressor
* Gradient Boosting Regressor
* 1D CNN
* Recursive BiLSTM
* Proposed CNN-BiLSTM Hybrid

Training includes:

* Early Stopping
* ReduceLROnPlateau
* Validation-based monitoring

---

### 5. Evaluation

Models are evaluated using:

* RMSE
* MSE
* MAE
* MBE
* R² Score
* Accuracy
* Precision
* Recall
* F1 Score

The evaluation stage also generates prediction analysis, residual analysis, and model comparison visualizations.

---

## Proposed CNN–BiLSTM Architecture

```text
Input Features
      │
      ▼
1D CNN
(64 → 128 → 64 Filters)
      │
      ▼
Global Average Pooling
      │
      ▼
Recursive BiLSTM
(64 → 64 → 32 Units)
      │
      ▼
Dense Layers + Dropout
      │
      ▼
Yield Prediction
```

The CNN component extracts local feature interactions while the Recursive BiLSTM captures temporal dependencies from agricultural time-series data. Combining both enables the model to learn spatial and sequential patterns simultaneously.

---

## Dashboard

The project includes an interactive Dash dashboard that provides:

* Dataset statistics
* Model comparison tables
* Training history visualizations
* Evaluation plots
* Actual vs Predicted comparisons
* Yield prediction interface

The dashboard runs locally on:

```text
http://localhost:8050
```

---

## Technology Stack

* Python
* TensorFlow / Keras
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly Dash
