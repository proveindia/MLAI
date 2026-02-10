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

## Assignment Highlights
*   **Dataset:** Synthetic blobs.
*   **Goal:** Implement specific SVC estimators and visualize decision boundaries.
*   **Process:**
    *   Instantiated `SVC` with a linear kernel.
    *   Identified support vectors.
    *   Calculated the slope and margins of the decision boundary.
    *   Visualized the decision boundary and support vectors.
    *   Compared Linear SVC with Logistic Regression and Polynomial SVC.

## Implementation Details

### Linear Support Vector Classifier
The `SVC` class with `kernel='linear'` is used to fit a linear decision boundary. The `support_vectors_` attribute provides the indices of the support vectors.

```python
svc_1 = SVC(kernel = 'linear').fit(X_train, y_train)
support_vectors = svc_1.support_vectors_
```

### Decision Function and Visualization
The `decision_function` method returns the distance of samples to the separating hyperplane, which is useful for visualizing the decision boundary and margins.

```python
# Grid of points to plot decision boundaries 
XX, YY = np.meshgrid(X_train[:, 0], X_train[:, 1])
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Output from grid of points based on decision function
Z = svc_1.decision_function(xy).reshape(XX.shape)

# Plots of points and support vectors
fig, ax = plt.subplots()
ax.contour(XX, YY,  Z, levels = [0], colors = ['black'])
ax.scatter(support_vectors[:, 0], support_vectors[:, 1], color = 'red', s = 80, marker = 'x')
ax.set_title('Support Vectors and Maximum Margin\nNote Black Line as Decision Boundary');
```

### Polynomial Kernel
For non-linear decision boundaries, a polynomial kernel can be used by setting `kernel='poly'`.

```python
svc2 = SVC(kernel='poly').fit(X_train, y_train)
```

### Grid Search with SVM
`GridSearchCV` can be used to tune hyperparameters like `C` and `kernel`.

```python
params = {'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}

svc = SVC()
grid = GridSearchCV(svc, param_grid=params, cv=5)
grid.fit(X_train, y_train)

print(f'Best score: {grid.best_score_}')
print(f'Best params: {grid.best_params_}')
```
