# Module 7: Optimization Summary

This module dives into the mathematical foundation of Machine Learning: Optimization. Specifically, minimizing Loss Functions.

## ⏱️ Quick Review (20 Mins)

### 1. Loss Functions
A metric that quantifies how "wrong" a model is. The goal of ML is to minimize this value.

**L2 Loss (Mean Squared Error):**
$$ L(\theta) = \frac{1}{n} \sum (y_{true} - y_{pred})^2 $$

Where $y_{pred} = \theta x$ (for a simple linear model).

```python
import numpy as np

def l2_loss(theta, x, y):
    y_pred = theta * x
    return np.mean((y - y_pred)**2)
```

### 2. Numerical Optimization
Using algorithms to find the parameters ($\theta$) that minimize the loss, rather than solving algebraically.

**Using `scipy.optimize`:**
```python
from scipy.optimize import minimize

# Define the function to minimize (wrapper to handle extra args)
def objective_function(theta):
    return l2_loss(theta, x, y)

# Initial guess
x0 = [0.0]

# Run Optimizer
result = minimize(objective_function, x0)

print(f"Optimal Theta: {result.x}")
print(f"Minimum Loss: {result.fun}")
```

### 3. Key Takeaway
Even complex Neural Networks use this same principle: define a loss function (error) and use an optimizer (like Gradient Descent) to tweak specific values (weights) until the error is minimized.

---
*Reference: Assignment 7.1*
