# Module 15: Gradient Descent for Linear Regression

## Overview
This module explored the optimization algorithm Gradient Descent and its application to Linear Regression.

## Key Concepts
*   **Gradient Descent:** An optimization algorithm used to minimize some function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient.
*   **Linear Regression (Optimization perspective):** Finding the coefficients (weights) that minimize the error term.
*   **Cost Function (MSE):** Mean Squared Error. The function we want to minimize. It measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value.
*   **Learning Rate (Alpha):** A hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated.
    *   Too small: Convergence is slow.
    *   Too large: Can overshoot the minimum and fail to converge.
*   **Epochs:** The number of times the algorithm iterates through the entire training dataset.

## Assignment Highlights
*   **Dataset:** Credit dataset.
*   **Goal:** Implement Gradient Descent to predict 'Balance' based on another feature (e.g., 'Rating' or 'Limit').
*   **Process:**
    *   Defined the MSE cost function.
    *   Implemented the gradient descent update rule manually.
    *   Visualized the cost reduction over iterations.
    *   Compared the Manual implementation results with Scikit-Learn's `LinearRegression` or `SGDRegressor`.

## Implementation Details

### 1. Manual Gradient Descent
We implemented the gradient descent algorithm from scratch to understand how it iteratively updates weights to minimize the cost function.

```python
def gradient_descent(df, initial_guess, alpha, n):
    """
    Performs n steps of gradient descent on df using learning rate alpha.
    """
    guesses = [initial_guess]
    current_guess = initial_guess
    while len(guesses) < n:
        # Update rule: theta = theta - alpha * gradient
        current_guess = current_guess - alpha * df(current_guess)
        guesses.append(current_guess)
        
    return np.array(guesses)
```

### 2. MSE Gradient Calculation
We calculated the gradient of the Mean Squared Error (MSE) with respect to the parameters ($\theta_0$, $\theta_1$).

```python
def mse_gradient(theta, X, y_obs):
    """Returns the gradient of the MSE on our data for the given theta"""    
    x0 = X.iloc[:, 0]
    x1 = X.iloc[:, 1]
    # Partial derivatives
    dth0 = np.mean(-2 * (y_obs - theta[0]*x0 - theta[1]*x1) * x0)
    dth1 = np.mean(-2 * (y_obs - theta[0]*x0 - theta[1]*x1) * x1)
    return np.array([dth0, dth1])

# Running Gradient Descent
guesses = gradient_descent(mse_gradient_single_arg, np.array([0, 0]), 0.001, 10000)
```

### 3. Stochastic Gradient Descent (SGD)
We also implemented SGD, which updates weights using a random subset (batch) of data at each step, making it more efficient for large datasets.

```python
def stochastic_gradient_descent(df, initial_guess, alpha, n, num_dps, number_of_batches):
    guesses = [initial_guess]
    guess = initial_guess
    while len(guesses) < n:
        dp_indices = np.random.permutation(np.arange(num_dps))
        for batch_indices in np.split(dp_indices, number_of_batches):            
            guess = guess - alpha * df(guess, batch_indices)
            guesses.append(guess)
    return np.array(guesses)
```
