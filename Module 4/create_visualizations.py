import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

os.makedirs('images', exist_ok=True)
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

print('Creating Module 4 visualizations...')

# 1. Pandas Merge Types Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

merge_types = [
    ('Inner Join', 'Only matching rows'),
    ('Left Join', 'All from left + matching from right'),
    ('Right Join', 'All from right + matching from left'),
    ('Outer Join', 'All rows from both tables')
]

colors_left = ['#3498db', '#3498db', 'lightgray', '#3498db']
colors_right = ['lightgray', '#e74c3c', '#e74c3c', '#e74c3c']
colors_result = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

for idx, (title, desc) in enumerate(merge_types):
    ax = axes[idx//2, idx%2]
    
    # Draw tables
    ax.add_patch(plt.Rectangle((0.1, 0.6), 0.25, 0.3, facecolor=colors_left[idx], alpha=0.6, edgecolor='black', linewidth=2))
    ax.add_patch(plt.Rectangle((0.65, 0.6), 0.25, 0.3, facecolor=colors_right[idx], alpha=0.6, edgecolor='black', linewidth=2))
    ax.add_patch(plt.Rectangle((0.35, 0.15), 0.3, 0.3, facecolor=colors_result[idx], alpha=0.7, edgecolor='black', linewidth=3))
    
    # Labels
    ax.text(0.225, 0.95, 'Table A', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.775, 0.95, 'Table B', ha='center', fontsize=11, fontweight='bold')
    ax.text(0.5, 0.5, title, ha='center', fontsize=14, fontweight='bold')
    ax.text(0.5, 0.08, desc, ha='center', fontsize=9, style='italic')
    ax.text(0.5, 0.32, 'Result', ha='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(0.4, 0.4), xytext=(0.25, 0.65), 
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.6, 0.4), xytext=(0.75, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.suptitle('Pandas Merge/Join Types', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('images/pandas_merge_types.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Created: pandas_merge_types.png')

# 2. Normalization Comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

np.random.seed(42)
data1 = np.random.randn(100) * 20 + 50  # Mean=50, std=20
data2 = np.random.randn(100) * 5 + 10   # Mean=10, std=5

# Original data
axes[0, 0].scatter(data1, data2, alpha=0.6, s=30)
axes[0, 0].set_title('Original Data', fontweight='bold')
axes[0, 0].set_xlabel('Feature 1 (range: 10-90)')
axes[0, 0].set_ylabel('Feature 2 (range: 0-20)')
axes[0, 0].grid(True, alpha=0.3)

# Min-Max Scaling
data1_minmax = (data1 - data1.min()) / (data1.max() - data1.min())
data2_minmax = (data2 - data2.min()) / (data2.max() - data2.min())
axes[0, 1].scatter(data1_minmax, data2_minmax, alpha=0.6, s=30, color='green')
axes[0, 1].set_title('Min-Max Scaling [0,1]', fontweight='bold')
axes[0, 1].set_xlabel('Scaled Feature 1')
axes[0, 1].set_ylabel('Scaled Feature 2')
axes[0, 1].grid(True, alpha=0.3)

# Z-Score Standardization
data1_std = (data1 - data1.mean()) / data1.std()
data2_std = (data2 - data2.mean()) / data2.std()
axes[0, 2].scatter(data1_std, data2_std, alpha=0.6, s=30, color='purple')
axes[0, 2].set_title('Z-Score Standardization', fontweight='bold')
axes[0, 2].set_xlabel('Standardized Feature 1')
axes[0, 2].set_ylabel('Standardized Feature 2')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.5)

# Distributions before normalization
axes[1, 0].hist(data1, bins=20, alpha=0.7, label='Feature 1', color='blue')
axes[1, 0].hist(data2, bins=20, alpha=0.7, label='Feature 2', color='orange')
axes[1, 0].set_title('Original Distributions', fontweight='bold')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# After Min-Max
axes[1, 1].hist(data1_minmax, bins=20, alpha=0.7, label='Feature 1', color='blue')
axes[1, 1].hist(data2_minmax, bins=20, alpha=0.7, label='Feature 2', color='orange')
axes[1, 1].set_title('After Min-Max', fontweight='bold')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# After Z-Score
axes[1, 2].hist(data1_std, bins=20, alpha=0.7, label='Feature 1', color='blue')
axes[1, 2].hist(data2_std, bins=20, alpha=0.7, label='Feature 2', color='orange')
axes[1, 2].set_title('After Z-Score', fontweight='bold')
axes[1, 2].set_xlabel('Value (std units)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Feature Normalization Methods Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/normalization_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Created: normalization_comparison.png')

# 3. Statistical Distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Normal Distribution
x = np.linspace(-4, 4, 1000)
y_normal = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2)
axes[0, 0].plot(x, y_normal, 'b-', linewidth=2.5)
axes[0, 0].fill_between(x, y_normal, alpha=0.3)
axes[0, 0].set_title('Normal Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Mean', alpha=0.7)
axes[0, 0].legend()

# Uniform Distribution
x_uniform = np.linspace(0, 10, 1000)
y_uniform = np.ones_like(x_uniform) * 0.1
axes[0, 1].plot(x_uniform, y_uniform, 'g-', linewidth=2.5)
axes[0, 1].fill_between(x_uniform, y_uniform, alpha=0.3, color='green')
axes[0, 1].set_title('Uniform Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 0.15)

# Exponential Distribution
x_exp = np.linspace(0, 5, 1000)
y_exp = np.exp(-x_exp)
axes[0, 2].plot(x_exp, y_exp, 'r-', linewidth=2.5)
axes[0, 2].fill_between(x_exp, y_exp, alpha=0.3, color='red')
axes[0, 2].set_title('Exponential Distribution', fontweight='bold')
axes[0, 2].set_xlabel('Value')
axes[0, 2].set_ylabel('Probability Density')
axes[0, 2].grid(True, alpha=0.3)

# Sample data visualizations
np.random.seed(42)
data_normal = np.random.normal(50, 10, 500)
data_skewed = np.random.exponential(2, 500)
data_bimodal = np.concatenate([np.random.normal(30, 5, 250), np.random.normal(70, 5, 250)])

# Histogram
axes[1, 0].hist(data_normal, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
axes[1, 0].set_title('Histogram (Normal Data)', fontweight='bold')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Box Plot
axes[1, 1].boxplot([data_normal, data_skewed, data_bimodal], labels=['Normal', 'Skewed', 'Bimodal'])
axes[1, 1].set_title('Box Plots (Comparing Distributions)', fontweight='bold')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# KDE (Kernel Density Estimate)
from scipy import stats
kde_normal = stats.gaussian_kde(data_normal)
kde_x = np.linspace(data_normal.min(), data_normal.max(), 200)
axes[1, 2].plot(kde_x, kde_normal(kde_x), linewidth=2.5, color='purple')
axes[1, 2].fill_between(kde_x, kde_normal(kde_x), alpha=0.3, color='purple')
axes[1, 2].set_title('KDE (Smooth Density)', fontweight='bold')
axes[1, 2].set_xlabel('Value')
axes[1, 2].set_ylabel('Density')
axes[1, 2].grid(True, alpha=0.3)

plt.suptitle('Statistical Distributions and Visualizations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/statistical_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Created: statistical_distributions.png')

# 4. Correlation Heatmap Example
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Create sample correlated data
np.random.seed(42)
n = 100
data_dict = {
    'Age': np.random.randint(20, 70, n),
    'Income': np.random.randint(30000, 150000, n),
    'Years_Edu': np.random.randint(10, 20, n),
    'Credit_Score': np.random.randint(300, 850, n),
}
# Add correlations
data_dict['Spending'] = data_dict['Income'] * 0.3 + np.random.randn(n) * 5000
data_dict['Savings'] = data_dict['Income'] * 0.15 + data_dict['Years_Edu'] * 1000 + np.random.randn(n) * 3000

df = pd.DataFrame(data_dict)

# Correlation matrix
corr = df.corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Correlation Heatmap', fontweight='bold', fontsize=14)

# Scatter matrix example (subset)
scatter_data = df[['Income', 'Spending', 'Savings']].sample(50)
axes[1].scatter(scatter_data['Income'], scatter_data['Spending'], 
                s=scatter_data['Savings']/100, alpha=0.6, c=scatter_data.index, cmap='viridis')
axes[1].set_xlabel('Income', fontweight='bold')
axes[1].set_ylabel('Spending', fontweight='bold')
axes[1].set_title('Income vs Spending (size=Savings)', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(scatter_data['Income'], scatter_data['Spending'], 1)
p = np.poly1d(z)
axes[1].plot(scatter_data['Income'].sort_values(), p(scatter_data['Income'].sort_values()), 
             "r--", linewidth=2, label='Trend')
axes[1].legend()

plt.tight_layout()
plt.savefig('images/correlation_heatmap_example.png', dpi=300, bbox_inches='tight')
plt.close()
print('✓ Created: correlation_heatmap_example.png')

print('\n All Module 4 images created successfully!')
