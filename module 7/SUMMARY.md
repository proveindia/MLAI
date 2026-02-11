# Module 7: Optimization Summary

## Overview
Optimization is the engine of Machine Learning. It involves finding the best parameters ($\theta$) that minimize a **Loss Function**. The most common algorithm for this is **Gradient Descent**.

## Key Concepts

### 1. The Loss Function ($J(\theta)$)
A mathematical function that quantifies the error between the model's predictions and the actual data.
*   **Goal:** Minimize $J(\theta)$.
*   **Convexity:** A convex function is bowl-shaped, guaranteeing that any local minimum is also the global minimum. This is ideal for optimization.

### 2. Gradient Descent
An iterative algorithm that moves parameters in the opposite direction of the gradient (slope) to find the minimum.
*   **Learning Rate ($\alpha$):** Controlled step size.
    *   **Too Small:** Converges very slowly.
    *   **Too Large:** Can overshoot the minimum and diverge.

## Key Formulas

### 1. Mean Squared Error (MSE) - The Loss Function
$$ J(\theta) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

### 2. The Gradient ($\nabla J$)
The derivative of the loss function with respect to the weights. It points "uphill".
$$ \nabla J(\theta) = -\frac{2}{n} \sum_{i=1}^n (y_i - \hat{y}_i) \cdot x_i $$

### 3. Update Rule (The "Learning" Step)
We move "downhill" by subtracting the gradient.
$$ \theta_{new} = \theta_{old} - \alpha \cdot \nabla J(\theta) $$

## Code for Learning

### Setup
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise
```

### 1. Manual Gradient Descent
Implementation from scratch to understand the math.

```python
# Hyperparameters
learning_rate = 0.1
n_iterations = 1000
m = len(X) # Number of samples

# Initialization
theta = np.random.randn(2, 1) # Random weights
X_b = np.c_[np.ones((m, 1)), X] # Add bias term (x0 = 1)

# Optimization Loop
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Intercept:", theta[0][0])
print("Slope:", theta[1][0])
```

### 2. Visualizing Convergence
How does the path to the minimum look?

```python
theta_path = []
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients
    theta_path.append(theta)

path = np.array(theta_path)
plt.plot(path[:, 0], path[:, 1], "b-o")
plt.xlabel("Theta 0 (Intercept)")
plt.ylabel("Theta 1 (Slope)")
plt.title("Gradient Descent Path")
plt.show()
```

### 3. Using Scipy Minimize (Black Box)
For complex functions where gradients are hard to derive manually.

```python
from scipy.optimize import minimize

def loss_func(theta):
    theta = theta.reshape(-1, 1)
    return np.mean((y - X_b.dot(theta))**2)

# Initial guess
x0 = np.random.randn(2, 1)

# Minimize
result = minimize(loss_func, x0)
print("Scipy Result:", result.x)
```
