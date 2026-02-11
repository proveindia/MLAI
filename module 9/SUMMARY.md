# Module 9: Feature Selection Summary

## Overview
Feature Selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

## Key Concepts

### 1. The Curse of Dimensionality
As the number of features (dimensions) increases, the amount of data needed to generalize accurately grows exponentially.
*   **Consequence:** Models become complex, slower to train, and more prone to **overfitting**.
*   **Solution:** Feature Selection / Dimensionality Reduction.

### 2. Feature Selection Methods
*   **Filter Methods:** Select features based on statistical scores (e.g., Correlation, Chi-Square) independent of the model. Fast but ignores feature interactions.
*   **Wrapper Methods:** Evaluate subsets of features by training a model (e.g., SFS, RFE). Computationally expensive but usually accurate.
*   **Embedded Methods:** Perform feature selection during the model training process (e.g., Lasso, Tree-based importance).

## Key Formulas

### 1. Lasso Regression (L1 Regularization)
Lasso adds a penalty equal to the absolute value of the magnitude of coefficients. This can shrink some coefficients to exactly **zero**, effectively performing feature selection.
$$ J(\beta) = \text{MSE} + \lambda \sum_{j=1}^p |\beta_j| $$
*   **$\lambda$** (Pronounced: *Lambda*): Penalty term. Higher $\lambda$ $\rightarrow$ More coefficients become zero.
*   **$\beta$** (Pronounced: *Beta*): The model coefficients (weights).

### 2. Ridge Regression (L2 Regularization)
Adds a penalty equal to the square of the magnitude of coefficients. Shrinks coefficients but rarely to zero.
$$ J(\beta) = \text{MSE} + \lambda \sum_{j=1}^p \beta_j^2 $$
*   **$\beta^2$** (Pronounced: *Beta squared*): The squared magnitude of the coefficients.

## Code for Learning

### Setup and Import
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector, RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
```

### 1. Sequential Feature Selection (SFS) - Wrapper
Iteratively adds (Forward) or removes (Backward) features.

```python
# Load Dataset
df = sns.load_dataset('mpg').dropna()
X = df.drop(['mpg', 'name', 'origin'], axis=1)
y = df['mpg']

# Base Model
model = LinearRegression()

# Forward Selection: Select best 3 features
sfs = SequentialFeatureSelector(model, n_features_to_select=3, direction='forward', cv=5)
sfs.fit(X, y)

print("Selected Features (SFS):", sfs.get_feature_names_out())
```

### 2. Recursive Feature Elimination (RFE) - Wrapper
Recursively removes the least important feature based on model weights.

```python
# RFE to select top 3 features
rfe = RFE(estimator=model, n_features_to_select=3)
rfe.fit(X, y)

print("Selected Features (RFE):", rfe.get_feature_names_out())

# Ranking of all features (1 = selected)
feature_ranking = pd.DataFrame({'Feature': X.columns, 'Rank': rfe.ranking_})
print(feature_ranking.sort_values('Rank'))
```

### 3. Lasso Regression - Embedded
Using L1 regularization to zero out unimportant coefficients.

```python
# Scale data (Crucial for Regularization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Lasso with specific alpha (lambda)
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

# Check Coefficients
coefs = pd.DataFrame({'Feature': X.columns, 'Coefficient': lasso.coef_})
print(coefs)

# Visualize Coefficients
plt.figure(figsize=(8, 5))
plt.barh(X.columns, lasso.coef_)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("Lasso Coefficients (Zero = Functionally Removed)")
plt.show()
```

### 4. Polynomial Features & selection
Handling non-linear relationships and interactions.

```python
# Generate Polynomial Features (Degree 2 results in interactions)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
feature_names = poly.get_feature_names_out(X.columns)

print(f"Original features: {X.shape[1]}")
print(f"Poly features: {X_poly.shape[1]}") # Explodes to many features

# Use Lasso to select only useful polynomial features
lasso_poly = Lasso(alpha=0.5, max_iter=10000)
lasso_poly.fit(scaler.fit_transform(X_poly), y)

# Identify selected features (non-zero coeff)
selected_mask = lasso_poly.coef_ != 0
selected_features = feature_names[selected_mask]
print(f"Selected Poly Features ({len(selected_features)}):")
print(selected_features)
```
