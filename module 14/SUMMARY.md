# Module 14: Decision Trees

## Overview
This module covered Decision Trees, a versatile non-parametric supervised learning method used for classification and regression.

## Key Concepts
*   **Decision Tree:** A flowchart-like structure in which each internal node represents a "test" on an attribute (e.g. whether a coin flip comes up heads or tails), each branch represents the outcome of the test, and each leaf node represents a class label (decision taken after computing all attributes).
*   **Splitting Criteria:**
    *   **Gini Impurity:** A measurement of the likelihood of an incorrect classification of a new instance of a random variable, if that new instance were randomly classified according to the distribution of class labels from the data set.
    *   **Entropy:** A measure of randomness or "disorder" in the information being processed.
*   **Hyperparameters:**
    *   `max_depth`: The maximum depth of the tree. Limiting this helps prevent overfitting.
    *   `min_samples_split`: The minimum number of samples required to split an internal node.
*   **Overfitting:** Creating a tree that is too complex and matches the training data too closely, leading to poor generalization on new data.
*   **Decision Boundaries:** Visualizing how the tree partitions the feature space into classes.

## Assignment Highlights
*   **Dataset:** Penguins dataset.
*   **Goal:** Classify penguin species.
*   **Process:**
    *   Built Decision Tree models with Scikit-Learn.
    *   Experimented with hyperparameters like `max_depth` to observe the effect on model complexity and performance.
    *   Visualized the resulting trees and decision boundaries.

## Implementation Details

### 1. Decision Tree Classification and Visualization
We trained Decision Trees with different depths and visualized them using `plot_tree`.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Train a shallow tree (max_depth=1)
tree1 = DecisionTreeClassifier(max_depth=1).fit(X, y)
plot_tree(tree1, feature_names=list(X.columns), filled=True)

# Train a deeper tree (max_depth=5)
tree2 = DecisionTreeClassifier(max_depth=5).fit(X, y)
plot_tree(tree2, feature_names=list(X.columns), filled=True)
```

### 2. Hyperparameter Tuning
We used `GridSearchCV` (and other search methods) to find optimal hyperparameters like `max_depth`, `min_samples_split`, and `criterion`.

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Grid Search
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```
