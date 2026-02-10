# Module 3: Data Visualization Summary

This module covers techniques for visualizing data using `matplotlib` and `seaborn` to uncover patterns and insights.

## ⏱️ Quick Review (20 Mins)

### 1. Histograms
Used to visualize the distribution of a single numerical variable.

```python
import matplotlib.pyplot as plt
import pandas as pd

# Using Pandas built-in plotting
df['lifeExp'].hist(bins=20)
plt.title("Histogram of Life Expectancy")
plt.xlabel("Life Expectancy")
plt.ylabel("Frequency")
plt.show()
```

### 2. Boxplots
Displays the five-number summary (min, Q1, median, Q3, max) and detects outliers.

```python
import seaborn as sns

# Boxplot of Life Expectancy by Continent
sns.boxplot(data=df, x='lifeExp', y='continent')
plt.title("Life Expectancy Distribution per Continent")
plt.show()
```

### 3. Bar Plots
Compares categorical data using the height of bars (often representing mean or count).

```python
# Bar plot of GDP per Capita by Continent
sns.barplot(data=df, x='gdpPercap', y='continent')
plt.show()
```

### 4. Scatterplots
Visualizes the relationship between two numerical variables.

```python
# Relationship between GDP and Life Expectancy
sns.scatterplot(data=df, x='gdpPercap', y='lifeExp', hue='continent')
plt.title("GDP vs Life Expectancy")
plt.show()
```

### 5. Best Practices
- **Titles & Labels**: Always label axes and give a title.
- **Legends**: Use legends to distinguish groups (e.g., `hue` in seaborn).
- **Saving**: `plt.savefig('plot.png')` to save high-quality images.

---
*Reference: Assignment 3.1 (Replicating Plots)*
