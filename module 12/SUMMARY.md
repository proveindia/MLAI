# Module 12: K-Nearest Neighbors (KNN) and Model Selection

## Overview
This module focused on the K-Nearest Neighbors (KNN) algorithm and the process of selecting the optimal hyperparameters using Grid Search and Cross-Validation.

## Key Concepts
*   **K-Nearest Neighbors (KNN):** A distance-based algorithm used for classification (and regression). It classifies a data point based on the majority class of its 'K' nearest neighbors.
*   **Pipelines:** Using `sklearn.pipeline.Pipeline` to chain preprocessing steps (like scaling and encoding) with the estimator. This ensures that validation data is processed exactly like training data and prevents data leakage.
*   **ColumnTransformer:** Applying different preprocessing steps to different subsets of features (e.g., scaling numeric features, one-hot encoding categorical features).
*   **Grid Search (GridSearchCV):** systematically working through multiple combinations of parameter tunes, cross-validating as it goes
    *   **Grid Search:** Exhaustive search over specified parameter values for an estimator.
    *   **Cross Validation:** Evaluating estimator performance by splitting data into train/test sets multiple times (e.g., K-Fold).

## Key Formulas

### Euclidean Distance
The distance between two points $p$ and $q$ in n-dimensional space, used to find nearest neighbors:

$$ d(p, q) = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2} $$

## Assignment Highlights
*   **Dataset:** Credit card default dataset (or similar).
*   **Goal:** Identify the best 'K' (n_neighbors) for the KNN model.
*   **Process:** Built a pipeline with scaling and KNN, then used GridSearchCV to find the optimal number of neighbors that maximizes model accuracy.

## Implementation Details

### 1. K-Nearest Neighbors (KNN)
We initialized and trained a KNN classifier.

```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize the model with K=1
model = KNeighborsClassifier(n_neighbors=1)

# Fit the model (using 'Income' and 'Debt' to predict 'Status')
model.fit(df[['Income', 'Debt']], df['Status'])
```

### 2. Hyperparameter Tuning with GridSearchCV
We used `GridSearchCV` to find the optimal number of neighbors.

```python
from sklearn.model_selection import GridSearchCV

# Define limits for K
model = KNeighborsClassifier()
parameters_to_try = {'n_neighbors': range(1, len(df))}

# Setup GridSearchCV
model_finder = GridSearchCV(
    estimator=model,
    param_grid=parameters_to_try,
    scoring="accuracy",
    cv=5
)

# Fit the grid search
model_finder.fit(df[['Income', 'Debt']], df["Status"])

# Best K and score will be stored in model_finder.best_params_ and model_finder.best_score_
```
