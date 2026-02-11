# Module 16: Support Vector Machines (SVM)

## Overview
This module focused on Support Vector Machines (SVM), specifically the Maximum Margin Classifier.

## Key Concepts
*   **Support Vector Classifier (SVC):** A supervised learning algorithm used for classification/regression. It finds a hyperplane in an N-dimensional space that distinctly classifies the data points.
*   **Maximum Margin:** The objective of SVM is to find the hyperplane that maximizes the margin (distance) between the classes. The vectors that define the hyperplane are the support vectors.
*   **Support Vectors:** Data points that are closer to the hyperplane and influence the position and orientation of the hyperplane.
*   **Kernels:** Functions that transform data into a higher-dimensional space to make it separable.
    *   `kernel='linear'`: For linearly separable data.
    *   `kernel='poly'`: Polynomial kernel for non-linear boundaries.
    *   `kernel='rbf'`: Radial Basis Function (infinite dimensions).
*   **Decision Function:** A function that returns the distance of samples to the separating hyperplane.

## Key Formulas

### 3. Cost Function (Hinge Loss)
SVM minimizes the Hinge Loss function to find the maximum margin. It penalizes misclassifications:

$$ L(y, f(x)) = \max(0, 1 - y \cdot f(x)) $$

*   **$y$**: True label (-1 or 1).
*   **$f(x)$**: Predicted score ($w^T x + b$).
*   **$L$**: Loss is 0 if correctly classified with sufficient margin ($y \cdot f(x) \geq 1$), otherwise increases linearly.

### 4. Kernel Functions & The Kernel Trick
**The Problem:** Many datasets are not linearly separable in their original dimensions (e.g., concentric circles).

**The Solution (The Kernel Trick):**
Instead of explicitly calculating the coordinates of data points in a high-dimensional space (which is computationally expensive), SVM uses a **Kernel Function** to compute the dot product between two vectors as if they were in that higher-dimensional space.

*   **Concept:** Like lifting the data into a 3D space where a flat sheet (hyperplane) can separate points that were mixed together on a 2D table.
*   **Efficiency:** It avoids the "Curse of Dimensionality" by operating in the original input space while benefiting from high-dimensional separability.

**Common Kernels:**
*   **Linear Kernel:** For linearly separable data.
    $$ K(x, x') = x^T x' $$
*   **Polynomial Kernel:** Maps data into a polynomial feature space.
    $$ K(x, x') = (\gamma x^T x' + r)^d $$
    *   $d$: Degree of the polynomial.
*   **Radial Basis Function (RBF) Kernel:** (Default) Maps data into an infinite-dimensional space. Effective for complex, non-linear boundaries.
    $$ K(x, x') = \exp(-\gamma ||x - x'||^2) $$
    *   $\gamma$ (gamma): Controls the influence of a single training example.

## Hyperparameters

### 1. Regularization (`C`)
*   **Definition:** Controls the trade-off between achieving a low error on the training data and minimizing the norm of the weights (maximizing margin).
*   **High C:** Strict penalty for misclassification. Results in a smaller margin and complex boundary (Risk: Overfitting).
*   **Low C:** More tolerance for misclassification. Results in a wider margin and simpler boundary (Risk: Underfitting).

### 2. Gamma (`gamma`)
*   **Definition:** Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Defines how far the influence of a single training example reaches.
*   **High Gamma:** Only close points influence the boundary. Creates islands of decision boundaries around points (Risk: Overfitting).
*   **Low Gamma:** Far points influence the boundary. Result is a smoother, more linear-like boundary.

## Assignment Highlights
*   **Dataset:** Synthetic blobs.
*   **Goal:** Implement specific SVC estimators and visualize decision boundaries.
*   **Process:**
    *   Instantiated `SVC` with a linear kernel.
    *   Identified support vectors using `support_vectors_`.
    *   Calculated decision boundary characteristics.
    *   Visualized margins and support vectors.
    *   Implemented Polynomial SVM for non-linear data.

## Implementation Details

### Linear Support Vector Classifier
The `SVC` class with `kernel='linear'` fits a linear decision boundary.

```python
svc_linear = SVC(kernel='linear', C=1.0)
svc_linear.fit(X_train, y_train)

# Get support vectors
support_vectors = svc_linear.support_vectors_
print(f"Number of support vectors: {len(support_vectors)}")
```

### Decision Function and Visualization
Using `decision_function` to plot decision boundaries and margins.

```python
# Create grid
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict scores
Z = svc_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Linear SVM Decision Boundary")
plt.show()
```

### Polynomial Kernel
For non-linear data, use `kernel='poly'`.

```python
svc_poly = SVC(kernel='poly', degree=3, C=10)
svc_poly.fit(X_train, y_train)
```

### Grid Search for Hyperparameters
Tuning `C`, `kernel`, and `gamma` to find the best model.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)

print(f"Best Parameters: {grid.best_params_}")
print(f"Best Estimator: {grid.best_estimator_}")
```
