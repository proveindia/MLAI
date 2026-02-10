# Module 2: Probability & Distributions Summary

This module focuses on continuous probability distributions, specifically Uniform and Gaussian (Normal) distributions, and how to analyze them using Python.

## ⏱️ Quick Review (20 Mins)

### 1. Continuous Uniform Distribution
A distribution where all outcomes are equally likely within a range.

**Key Concepts:**
- **PDF (Probability Density Function)**: Likelihood of a specific value (height of curve).
- **CDF (Cumulative Distribution Function)**: Probability that a variable takes a value less than or equal to $x$.

```python
from scipy.stats import uniform

# Create a uniform distribution starting at 10 with width 3 (range 10-13)
dist = uniform(loc=10, scale=3)

# Statistics
mean = dist.mean()
std = dist.std()

# Probability P(x < 12)
prob = dist.cdf(12)
```

### 2. Gaussian (Normal) Distribution
The bell curve distribution, defined by its mean ($\mu$) and standard deviation ($\sigma$).

**Key Concepts:**
- **Central Tendency**: Mean, Median, Mode.
- **Dispersion**: Variance, Standard Deviation.
- **68-95-99.7 Rule**: 99.7% of data lies within 3 standard deviations of the mean.

```python
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Create Normal Distribution (mean=5, std=2)
gauss = norm(loc=5, scale=2)

# Generate Random Samples
samples = gauss.rvs(size=100, random_state=12)

# Calculate Sample Statistics
sample_mean = np.mean(samples)
sample_std = np.std(samples)
```

### 3. Visualizing Distributions
Comparing theoretical distributions with sample data.

```python
# Plotting
x = np.linspace(-1, 11, 1000)
plt.plot(x, gauss.pdf(x), label='Theoretical PDF')
plt.hist(samples, density=True, alpha=0.5, label='Sample Histogram')
plt.legend()
plt.show()
```

---
*Reference: Assignment 2.1 (Uniform), Self Study 2.1 (Gaussian)*
