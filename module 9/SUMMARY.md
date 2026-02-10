# Module 9: Feature Selection Summary

This module covers advanced techniques for selecting the most relevant features to build efficient and effective models, avoiding the "Curse of Dimensionality".

## ⏱️ Quick Review (20 Mins)

### 1. Sequential Feature Selection (SFS)
An iterative method that adds (Forward) or removes (Backward) features to find the best subset.

- **Forward Selection**: Starts with 0 features, adds the one that improves the model most, repeats.
- **Backward Selection**: Starts with all features, removes the one that hurts the model least, repeats.

```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# Base Model
model = LinearRegression()

# Selector: Select best 4 features using Forward Selection
sfs = SequentialFeatureSelector(estimator=model, 
                                n_features_to_select=4,
                                direction='forward',
                                scoring='neg_mean_squared_error',
                                cv=5)

# Fit and Transform
X_selected = sfs.fit_transform(X, y)

# Which features were picked?
print(sfs.get_support()) 
```

### 2. Cross-Validation (CV)
Used inside selection to ensure features generalize well, not just memorize the training set.

`cv=[[train_idx, test_idx]]` allows custom split indices, vital for time-series or specific data structures where random shuffling isn't appropriate.

### 3. Verification
Always verify the selected features by training a model and checking the Error Metric (e.g., MSE).

```python
from sklearn.metrics import mean_squared_error

# Train on selected features
lr = LinearRegression().fit(X_selected, y)
preds = lr.predict(X_selected)

print(f"MSE: {mean_squared_error(y, preds)}")
```

---
*Reference: Assignment 9.1*
