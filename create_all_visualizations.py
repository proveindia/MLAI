"""
Master script to create all visualizations for Modules 1, 2, and 3
This will generate professional images for all three foundation modules
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import os

# Global Imports
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import l1_min_c

sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

print("="*70)
print("CREATING ALL VISUALIZATIONS FOR MODULES 1, 2, AND 3")
print("="*70)

# ============================================================================
# MODULE 1 VISUALIZATIONS
# ============================================================================
print("\n### MODULE 1: ML Fundamentals ###\n")
os.makedirs('module 1/images', exist_ok=True)

# 1. ML Types Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Supervised Learning
ax = axes[0]
X_sup = np.random.rand(50, 2) * 10
y_sup = (X_sup[:, 0] + X_sup[:, 1] > 10).astype(int)
colors = ['blue' if y == 0 else 'red' for y in y_sup]
ax.scatter(X_sup[:, 0], X_sup[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')
ax.set_title('Supervised Learning\n(Labeled Data)', fontweight='bold', fontsize=13)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend(['Class 0', 'Class 1'], loc='upper left')
ax.grid(True, alpha=0.3)
ax.text(0.5, -0.15, 'Input X → Output y (known)', ha='center', transform=ax.transAxes, 
        fontsize=10, style='italic')

# Unsupervised Learning  
ax = axes[1]
from sklearn.cluster import KMeans
X_unsup = np.vstack([
    np.random.randn(25, 2) + [3, 3],
    np.random.randn(25, 2) + [8, 3],
    np.random.randn(25, 2) + [5.5, 8]
])
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_unsup)
ax.scatter(X_unsup[:, 0], X_unsup[:, 1], c=labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
          marker='*', s=500, c='red', edgecolors='black', linewidth=2)
ax.set_title('Unsupervised Learning\n(Unlabeled Data)', fontweight='bold', fontsize=13)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)
ax.text(0.5, -0.15, 'Find patterns without labels', ha='center', transform=ax.transAxes,
        fontsize=10, style='italic')

# Reinforcement Learning
ax = axes[2]
states = np.arange(10)
rewards = np.cumsum(np.random.randn(10) * 0.5 + 0.3)
ax.plot(states, rewards, 'go-', linewidth=2.5, markersize=10, label='Cumulative Reward')
ax.fill_between(states, rewards, alpha=0.3, color='green')
ax.set_title('Reinforcement Learning\n(Trial & Error)', fontweight='bold', fontsize=13)
ax.set_xlabel('Time Step')
ax.set_ylabel('Cumulative Reward')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(0.5, -0.15, 'Agent learns from rewards/penalties', ha='center', transform=ax.transAxes,
        fontsize=10, style='italic')

plt.suptitle('Three Types of Machine Learning', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 1/images/ml_types_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ml_types_comparison.png")

# 2. NumPy Array Operations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Element-wise operations
ax = axes[0, 0]
ax.text(0.5, 0.85, 'Element-wise Operations', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.25, 0.5, '[1, 2, 3]\n+\n[4, 5, 6]\n=\n[5, 7, 9]', ha='center', va='center',
        fontsize=14, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
ax.text(0.75, 0.5, '[1, 2, 3]\n*\n[4, 5, 6]\n=\n[4, 10, 18]', ha='center', va='center',
        fontsize=14, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax.axis('off')

# Broadcasting
ax = axes[0, 1]
ax.text(0.5, 0.85, 'Broadcasting', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.5, '[[1, 2, 3],\n [4, 5, 6]]\n+\n[10, 20, 30]\n=\n[[11, 22, 33],\n [14, 25, 36]]',
        ha='center', va='center', fontsize=12, transform=ax.transAxes, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
ax.text(0.5, 0.15, 'Smaller array stretched to match shape', ha='center', fontsize=10,
        transform=ax.transAxes, style='italic')
ax.axis('off')

# Reshaping
ax = axes[1, 0]
ax.text(0.5, 0.85, 'Array Reshaping', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
original = np.arange(1, 7)
ax.text(0.5, 0.65, f'Original: {original}', ha='center', fontsize=11, transform=ax.transAxes)
reshaped = original.reshape(2, 3)
ax.text(0.5, 0.45, f'Reshaped (2x3):\n{reshaped}', ha='center', fontsize=11, transform=ax.transAxes,
        family='monospace')
ax.axis('off')

# Aggregations
ax = axes[1, 1]
data_agg = np.array([[1, 2, 3], [4, 5, 6]])
ax.text(0.5, 0.85, 'Aggregations', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.65, f'Array:\n{data_agg}', ha='center', fontsize=11, transform=ax.transAxes, family='monospace')
ax.text(0.2, 0.35, f'sum():\n{data_agg.sum()}', ha='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.6))
ax.text(0.5, 0.35, f'mean():\n{data_agg.mean():.1f}', ha='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
ax.text(0.8, 0.35, f'max():\n{data_agg.max()}', ha='center', fontsize=10, transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))
ax.axis('off')

plt.suptitle('NumPy Array Operations', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('module 1/images/numpy_array_operations.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: numpy_array_operations.png")

# 3. Pandas DataFrame Structure
fig = plt.figure(figsize=(12, 8))

# Create sample DataFrame visualization
ax = plt.subplot(111)
ax.text(0.5, 0.9, 'Pandas DataFrame Anatomy', ha='center', fontsize=16, fontweight='bold', transform=ax.transAxes)

# Draw table structure
table_data = [
    ['', 'Name', 'Age', 'City', 'Salary'],
    ['0', 'Alice', '25', 'NYC', '70000'],
    ['1', 'Bob', '30', 'LA', '80000'],
    ['2', 'Charlie', '35', 'SF', '90000']
]

# Draw grid
cell_w, cell_h = 0.15, 0.08
start_x, start_y = 0.15, 0.6

for i, row in enumerate(table_data):
    for j, cell in enumerate(row):
        color = 'lightblue' if i == 0 else ('lightyellow' if j == 0 else 'white')
        ax.add_patch(plt.Rectangle((start_x + j*cell_w, start_y - i*cell_h), 
                                   cell_w, cell_h, facecolor=color, edgecolor='black', linewidth=1.5))
        ax.text(start_x + j*cell_w + cell_w/2, start_y - i*cell_h + cell_h/2, 
               cell, ha='center', va='center', fontsize=10, fontweight='bold' if i==0 or j==0 else 'normal')

# Annotations
ax.annotate('Index', xy=(start_x + cell_w/2, start_y + cell_h/2), xytext=(0.05, 0.75),
           fontsize=11, color='blue', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.annotate('Columns', xy=(start_x + 2.5*cell_w, start_y + cell_h/2), xytext=(0.55, 0.8),
           fontsize=11, color='darkgreen', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
ax.annotate('Data Values', xy=(start_x + 2*cell_w, start_y - 1.5*cell_h), xytext=(0.7, 0.45),
           fontsize=11, color='darkred', fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='darkred', lw=2))

# Key operations
ops_text = """
Common Operations:
• df.head() - First 5 rows
• df.describe() - Statistics
• df['column'] - Select column
• df.loc[row] - Select by label
• df.iloc[index] - Select by position
• df.groupby() - Group aggregation
"""
ax.text(0.15, 0.25, ops_text, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('module 1/images/pandas_dataframe_structure.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: pandas_dataframe_structure.png")

# 4. ML Workflow Detailed
fig, ax = plt.subplots(figsize=(14, 8))

stages = [
    ('1. Data\nCollection', 0.1, 0.7, 'lightblue'),
    ('2. Data\nCleaning', 0.25, 0.7, 'lightgreen'),
    ('3. Feature\nEngineering', 0.4, 0.7, 'lightyellow'),
    ('4. Train/Test\nSplit', 0.55, 0.7, 'lightcoral'),
    ('5. Model\nTraining', 0.7, 0.7, 'plum'),
    ('6. Evaluation', 0.85, 0.7, 'lightgray'),
]

for i, (label, x, y, color) in enumerate(stages):
    ax.add_patch(plt.Rectangle((x, y), 0.12, 0.15, facecolor=color, edgecolor='black', linewidth=2))
    ax.text(x + 0.06, y + 0.075, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw arrows
    if i < len(stages) - 1:
        ax.annotate('', xy=(stages[i+1][1], y + 0.075), xytext=(x + 0.12, y + 0.075),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Details for each stage
details = {
    0.1: 'CSV, APIs,\nDatabases',
    0.25: 'Handle nulls,\noutliers',
    0.4: 'Create new\nfeatures',
    0.55: '80% train\n20% test',
    0.7: 'Fit model\non train',
    0.85: 'Test on\ntest set'
}

for x, detail in details.items():
    ax.text(x + 0.06, 0.5, detail, ha='center', fontsize=9, style='italic')

# Feedback loop for deployment
ax.annotate('', xy=(0.1, 0.65), xytext=(0.9, 0.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
ax.text(0.5, 0.4, '7. Deploy & Monitor (iterate if needed)', ha='center', fontsize=11,
        color='red', fontweight='bold')

ax.set_xlim(0, 1)
ax.set_ylim(0.3, 1)
ax.set_title('End-to-End Machine Learning Workflow', fontsize=16, fontweight='bold')
ax.axis('off')

plt.tight_layout()
plt.savefig('module 1/images/ml_workflow_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ml_workflow_detailed.png")

print("\nModule 1 complete: 4 visualizations created\n")

# ============================================================================
# MODULE 2 VISUALIZATIONS
# ============================================================================
print("### MODULE 2: Probability & Distributions ###\n")
os.makedirs('module 2/images', exist_ok=True)

# 1. Distribution Types
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Normal Distribution
x = np.linspace(-4, 4, 1000)
y_normal = stats.norm.pdf(x, 0, 1)
axes[0, 0].plot(x, y_normal, 'b-', linewidth=2.5)
axes[0, 0].fill_between(x, y_normal, alpha=0.3)
axes[0, 0].set_title('Normal (Gaussian) Distribution', fontweight='bold', fontsize=12)
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=0, color='red', linestyle='--', label='Mean=0', alpha=0.7)
axes[0, 0].legend()

# Uniform Distribution
x_uniform = np.linspace(0, 10, 1000)
y_uniform = stats.uniform.pdf(x_uniform, 0, 10)
axes[0, 1].plot(x_uniform, y_uniform, 'g-', linewidth=2.5)
axes[0, 1].fill_between(x_uniform, y_uniform, alpha=0.3, color='green')
axes[0, 1].set_title('Uniform Distribution', fontweight='bold', fontsize=12)
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(0, 0.15)

# Exponential Distribution
x_exp = np.linspace(0, 5, 1000)
y_exp = stats.expon.pdf(x_exp, scale=1)
axes[1, 0].plot(x_exp, y_exp, 'r-', linewidth=2.5)
axes[1, 0].fill_between(x_exp, y_exp, alpha=0.3, color='red')
axes[1, 0].set_title('Exponential Distribution', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].grid(True, alpha=0.3)

# Binomial Distribution
n, p = 20, 0.5
x_binom = np.arange(0, n+1)
y_binom = stats.binom.pmf(x_binom, n, p)
axes[1, 1].bar(x_binom, y_binom, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].set_title('Binomial Distribution (n=20, p=0.5)', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Number of Successes')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.suptitle('Common Probability Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 2/images/distribution_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: distribution_types.png")

# 2. Normal Distribution 68-95-99.7 Rule
fig, ax = plt.subplots(figsize=(12, 7))

x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)

ax.plot(x, y, 'b-', linewidth=3)
ax.fill_between(x, y, alpha=0.2, color='blue')

# 68% (1 sigma)
x1 = x[(x >= -1) & (x <= 1)]
y1 = stats.norm.pdf(x1, 0, 1)
ax.fill_between(x1, y1, alpha=0.4, color='green', label='68% (±1σ)')

# 95% (2 sigma)
x2 = x[(x >= -2) & (x <= -1)]
y2_left = stats.norm.pdf(x2, 0, 1)
x2_right = x[(x >= 1) & (x <= 2)]
y2_right = stats.norm.pdf(x2_right, 0, 1)
ax.fill_between(x2, y2_left, alpha=0.4, color='yellow', label='95% (±2σ)')
ax.fill_between(x2_right, y2_right, alpha=0.4, color='yellow')

# 99.7% (3 sigma)
x3_left = x[(x >= -3) & (x <= -2)]
y3_left = stats.norm.pdf(x3_left, 0, 1)
x3_right = x[(x >= 2) & (x <= 3)]
y3_right = stats.norm.pdf(x3_right, 0, 1)
ax.fill_between(x3_left, y3_left, alpha=0.4, color='orange', label='99.7% (±3σ)')
ax.fill_between(x3_right, y3_right, alpha=0.4, color='orange')

# Vertical lines for standard deviations
for i in range(-3, 4):
    ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5)
    ax.text(i, -0.02, f'{i}σ', ha='center', fontsize=10)

ax.set_title('68-95-99.7 Rule (Empirical Rule)', fontsize=16, fontweight='bold')
ax.set_xlabel('Standard Deviations from Mean', fontsize=12, fontweight='bold')
ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 0.45)

plt.tight_layout()
plt.savefig('module 2/images/normal_distribution_rules.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: normal_distribution_rules.png")

# 3. Skewness Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Normal
x = np.linspace(-4, 4, 1000)
axes[1].plot(x, stats.norm.pdf(x), 'b-', linewidth=2)
axes[1].fill_between(x, stats.norm.pdf(x), alpha=0.3, color='blue')
axes[1].set_title('Symmetrical (Normal)\nMean = Median = Mode', fontweight='bold')
axes[1].axvline(0, color='red', linestyle='--', label='Mean/Median/Mode')
axes[1].legend()

# Positive Skew (Right)
from scipy.stats import skewnorm
x_pos = np.linspace(-1, 5, 1000)
p_pos = skewnorm.pdf(x_pos, 4)
axes[2].plot(x_pos, p_pos, 'g-', linewidth=2)
axes[2].fill_between(x_pos, p_pos, alpha=0.3, color='green')
axes[2].set_title('Positive (Right) Skew\nMode < Median < Mean', fontweight='bold')
axes[2].axvline(stats.skewnorm.mean(4), color='red', linestyle='--', label='Mean')
axes[2].axvline(stats.skewnorm.median(4), color='orange', linestyle='-.', label='Median')
axes[2].legend()

# Negative Skew (Left)
x_neg = np.linspace(-5, 1, 1000)
p_neg = skewnorm.pdf(x_neg, -4)
axes[0].plot(x_neg, p_neg, 'purple', linewidth=2)
axes[0].fill_between(x_neg, p_neg, alpha=0.3, color='purple')
axes[0].set_title('Negative (Left) Skew\nMean < Median < Mode', fontweight='bold')
axes[0].axvline(stats.skewnorm.mean(-4), color='red', linestyle='--', label='Mean')
axes[0].axvline(stats.skewnorm.median(-4), color='orange', linestyle='-.', label='Median')
axes[0].legend()

plt.suptitle('Skewness and Measures of Central Tendency', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 2/images/skewness_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: skewness_comparison.png")

# 4. Kurtosis Comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(-5, 5, 1000)

# Mesokurtic (Normal)
y_meso = stats.norm.pdf(x, 0, 1)
ax.plot(x, y_meso, 'b-', linewidth=3, label='Mesokurtic (Normal)\nKurtosis ≈ 3')
ax.fill_between(x, y_meso, alpha=0.1, color='blue')

# Leptokurtic (Laplace - Peaked, Heavy Tails)
# Scale so variance is similar for visual comparison of shape
y_lepto = stats.laplace.pdf(x, 0, 1/np.sqrt(2)) 
ax.plot(x, y_lepto, 'r--', linewidth=2, label='Leptokurtic (Peaked, Heavy Tails)\nKurtosis > 3')

# Platykurtic (Wigner Semicircle / Uniform-like - Flat, Light Tails)
# Using Cosine distribution as a smooth platykurtic example
y_platy = stats.cosine.pdf(x, 0, np.pi) # Variance is different but shape is clear
ax.plot(x, y_platy, 'g:', linewidth=3, label='Platykurtic (Flat, Light Tails)\nKurtosis < 3')

ax.set_title('Kurtosis: The Measure of "Tailedness"', fontsize=16, fontweight='bold')
ax.set_xlabel('Value (Centered)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 0.8)

# Annotations
ax.text(0, 0.72, 'Leptokurtic\nPeak', ha='center', color='red', fontweight='bold')
ax.text(2.5, 0.05, 'Heavy\nTail', ha='center', color='red', fontweight='bold')
ax.text(0, 0.35, 'Mesokurtic', ha='center', color='blue', fontweight='bold')
ax.text(0.8, 0.25, 'Platykurtic\nShoulder', ha='center', color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('module 2/images/kurtosis_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: kurtosis_comparison.png")

print("\nModule 2 complete: 4 visualizations created\n")

# ============================================================================
# MODULE 3 VISUALIZATIONS
# ============================================================================
print("### MODULE 3: Data Visualization ###\n")
os.makedirs('module 3/images', exist_ok=True)

# 1. Common Data Viz Types
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Histogram
data_hist = np.random.randn(1000)
axes[0, 0].hist(data_hist, bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Histogram\n(Distribution)', fontweight='bold')

# Scatter Plot
x_scatter = np.random.rand(50)
y_scatter = x_scatter * 2 + np.random.randn(50) * 0.2
axes[0, 1].scatter(x_scatter, y_scatter, color='coral', alpha=0.7)
axes[0, 1].set_title('Scatter Plot\n(Relationship)', fontweight='bold')

# Bar Plot
cats = ['A', 'B', 'C', 'D']
vals = [10, 25, 15, 30]
axes[0, 2].bar(cats, vals, color='lightgreen', edgecolor='black')
axes[0, 2].set_title('Bar Plot\n(Comparison)', fontweight='bold')

# Box Plot
data_box = [np.random.normal(0, 1, 100), np.random.normal(2, 1.5, 100)]
axes[1, 0].boxplot(data_box, labels=['Group 1', 'Group 2'], patch_artist=True)
axes[1, 0].set_title('Box Plot\n(Outliers & Spread)', fontweight='bold')

# Heatmap
data_heat = np.random.rand(5, 5)
im = axes[1, 1].imshow(data_heat, cmap='viridis')
axes[1, 1].set_title('Heatmap\n(Intensity/Matrix)', fontweight='bold')
plt.colorbar(im, ax=axes[1, 1])

# Line Plot
x_line = np.linspace(0, 10, 50)
y_line = np.sin(x_line)
axes[1, 2].plot(x_line, y_line, 'r-', linewidth=2)
axes[1, 2].set_title('Line Plot\n(Trend over Time/Seq)', fontweight='bold')

plt.suptitle('Common Data Visualization Types', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 3/images/data_viz_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: data_viz_types.png")

# 2. Anatomy of a Matplotlib Plot
fig, ax = plt.subplots(figsize=(10, 8))
x = np.linspace(0, 10, 100)
y = np.cos(x)
ax.plot(x, y, linewidth=3, label='Cosine Wave')
ax.set_title("Anatomy of a Plot (Title)", fontsize=14, fontweight='bold')
ax.set_xlabel("X Axis Label", fontsize=12)
ax.set_ylabel("Y Axis Label", fontsize=12)
ax.legend(loc='upper right')
ax.grid(True)

# Annotations pointing to elements
bbox_props = dict(boxstyle="round,pad=0.3", fc="wheat", ec="black", alpha=0.9)

ax.annotate('Title', xy=(0.5, 1.02), xytext=(0.5, 1.1), xycoords='axes fraction',
            ha='center', va='bottom', bbox=bbox_props,
            arrowprops=dict(arrowstyle='->', shrinkA=0))

ax.annotate('Y Axis', xy=(-0.05, 0.5), xytext=(-0.15, 0.5), xycoords='axes fraction',
            ha='center', va='center', bbox=bbox_props,
            arrowprops=dict(arrowstyle='->', shrinkA=0))

ax.annotate('X Axis', xy=(0.5, -0.08), xytext=(0.5, -0.15), xycoords='axes fraction',
            ha='center', va='top', bbox=bbox_props,
            arrowprops=dict(arrowstyle='->', shrinkA=0))

ax.annotate('Legend', xy=(0.9, 0.9), xytext=(0.7, 0.7), xycoords='axes fraction',
            ha='center', va='center', bbox=bbox_props,
            arrowprops=dict(arrowstyle='->', shrinkA=0))

ax.annotate('Grid', xy=(2, 0.5), xytext=(4, 0.5),
            ha='center', va='center', bbox=bbox_props,
            arrowprops=dict(arrowstyle='->', shrinkA=0))

ax.annotate('Spine', xy=(0, 0.5), xytext=(1, 0),
             ha='center', va='center', bbox=bbox_props,
             arrowprops=dict(arrowstyle='->', shrinkA=0))

plt.tight_layout()
plt.subplots_adjust(top=0.9, left=0.15, bottom=0.15)
plt.savefig('module 3/images/anatomy_of_a_plot.png', dpi=300)
plt.close()
print("Created: anatomy_of_a_plot.png")

print("\nModule 3 complete: 2 visualizations created\n")

# ============================================================================
# MODULE 4 VISUALIZATIONS
# ============================================================================
print("### MODULE 4: Data Analytics Primer ###\n")
os.makedirs('module 4/images', exist_ok=True)

# 1. Pandas Merge Types
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
    ax.annotate('', xy=(0.4, 0.4), xytext=(0.25, 0.65), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.6, 0.4), xytext=(0.75, 0.65), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

plt.suptitle('Pandas Merge/Join Types', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('module 4/images/pandas_merge_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: pandas_merge_types.png")

# 2. Normalization Comparison
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
np.random.seed(42)
data1 = np.random.randn(100) * 20 + 50
data2 = np.random.randn(100) * 5 + 10
# Original
axes[0, 0].scatter(data1, data2, alpha=0.6, s=30)
axes[0, 0].set_title('Original Data', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
# Min-Max
d1_mm = (data1 - data1.min()) / (data1.max() - data1.min())
d2_mm = (data2 - data2.min()) / (data2.max() - data2.min())
axes[0, 1].scatter(d1_mm, d2_mm, alpha=0.6, s=30, color='green')
axes[0, 1].set_title('Min-Max Scaling [0,1]', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
# Z-Score
d1_z = (data1 - data1.mean()) / data1.std()
d2_z = (data2 - data2.mean()) / data2.std()
axes[0, 2].scatter(d1_z, d2_z, alpha=0.6, s=30, color='purple')
axes[0, 2].set_title('Z-Score Standardization', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)
# Hists
axes[1, 0].hist(data1, alpha=0.7, label='F1', color='blue'); axes[1, 0].hist(data2, alpha=0.7, label='F2', color='orange')
axes[1, 0].set_title('Original Dist')
axes[1, 1].hist(d1_mm, alpha=0.7, color='blue'); axes[1, 1].hist(d2_mm, alpha=0.7, color='orange')
axes[1, 1].set_title('Min-Max Dist')
axes[1, 2].hist(d1_z, alpha=0.7, color='blue'); axes[1, 2].hist(d2_z, alpha=0.7, color='orange')
axes[1, 2].set_title('Z-Score Dist')

plt.suptitle('Feature Normalization Methods Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 4/images/normalization_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: normalization_comparison.png")

# 3. EDA Techniques
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Outlier Detection
data_out = np.concatenate([np.random.normal(0, 1, 50), [5, 6, -5]])
axes[0].boxplot(data_out, patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[0].set_title('1. Outlier Detection\n(Boxplots/IQR)', fontweight='bold')
axes[0].text(1.1, 5, 'Outlier!', color='red', fontweight='bold')

# Correlation
data_corr = pd.DataFrame(np.random.rand(5, 5), columns=list('ABCDE'))
sns.heatmap(data_corr.corr(), ax=axes[1], cmap='coolwarm', annot=False, cbar=False)
axes[1].set_title('2. Correlation Analysis\n(Heatmaps)', fontweight='bold')

# Missing Data
data_miss = pd.DataFrame(np.random.rand(10, 5) > 0.8)
sns.heatmap(data_miss, ax=axes[2], cmap='binary', cbar=False)
axes[2].set_title('3. Missing Data Patterns\n(Null Matrices)', fontweight='bold')

plt.suptitle('Key EDA Techniques', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 4/images/eda_techniques.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: eda_techniques.png")

print("\nModule 4 complete: 3 visualizations created\n")

# ============================================================================
# MODULE 6 VISUALIZATIONS
# ============================================================================
print("### MODULE 6: Unsupervised Learning ###\n")
os.makedirs('module 6/images', exist_ok=True)

# 1. PCA of Wine Dataset
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data_wine = load_wine()
X_wine = data_wine.data
y_wine = data_wine.target
target_names = data_wine.target_names

# Standardize
scaler = StandardScaler()
X_scaled_wine = scaler.fit_transform(X_wine)

# PCA
pca = PCA(n_components=2)
X_pca_wine = pca.fit_transform(X_scaled_wine)

# Plot
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca_wine[:, 0], X_pca_wine[:, 1], c=y_wine, cmap='viridis', edgecolor='k', s=70, alpha=0.8)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.title('PCA of Wine Dataset\n(13 Features -> 2 Dimensions)', fontsize=16, fontweight='bold')
plt.colorbar(scatter, label='Target Class', ticks=[0, 1, 2])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 6/images/pca_wine_dataset.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: pca_wine_dataset.png")

print("\nModule 6 complete: 1 visualization created\n")

print("="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*70)

# ============================================================================
# MODULE 7 VISUALIZATIONS
# ============================================================================
print("### MODULE 7: Optimization ###\n")
os.makedirs('module 7/images', exist_ok=True)

# 1. Loss Function Comparison (L1 vs L2)
x = np.linspace(-3, 3, 500)
mae = np.abs(x)
mse = x**2

plt.figure(figsize=(10, 6))
plt.plot(x, mae, 'b--', linewidth=2, label='L1 Loss (MAE) - Absolute Error')
plt.plot(x, mse, 'r-', linewidth=2, label='L2 Loss (MSE) - Squared Error')

plt.title('Loss Functions Comparison: L1 vs L2', fontsize=16, fontweight='bold')
plt.xlabel('Error (Residual)', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)

# Annotations
plt.text(2, 4.2, 'MSE penalizes\nlarge errors more', ha='center', color='red', fontweight='bold')
plt.text(2, 1.5, 'MAE is linear\n(Robust to outliers)', ha='center', color='blue', fontweight='bold')

plt.tight_layout()
plt.savefig('module 7/images/loss_function_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
plt.close()
print("Created: loss_function_comparison.png")

# 2. Real-World Example: California Housing (Median Income vs House Value)
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
X_cal = housing.data
y_cal = housing.target
feature_names = housing.feature_names
# Feature: MedInc (Median Income in block group) - Index 0
X_inc = X_cal[:, 0].reshape(-1, 1)

# Fit simple linear model to visualize
lr_cal = LinearRegression()
lr_cal.fit(X_inc, y_cal)
y_pred_cal = lr_cal.predict(X_inc)

plt.figure(figsize=(10, 6))
# Plot a subset of points for cleaner visualization
indices = np.random.choice(len(X_inc), 500, replace=False)
plt.scatter(X_inc[indices], y_cal[indices], alpha=0.4, color='teal', label='Data Points (Subset)')
plt.plot(X_inc, y_pred_cal, color='red', linewidth=3, label='Optimized Model (Gradient Descent)')

plt.title('California Housing: Income vs House Value (Optimization Goal)', fontweight='bold')
plt.xlabel('Median Income (Tens of Thousands)')
plt.ylabel('Median House Value (Hundreds of Thousands)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 7/images/california_housing_viz.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: california_housing_viz.png")

print("\nModule 7 complete: 2 visualizations created\n")

# ============================================================================
# MODULE 8 VISUALIZATIONS
# ============================================================================
print("### MODULE 8: Feature Engineering & Overfitting ###\n")
os.makedirs('module 8/images', exist_ok=True)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# 1. Polynomial Regression (Underfitting vs Optimal vs Overfitting)
np.random.seed(0)
n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = np.cos(1.5 * np.pi * X) + np.random.randn(n_samples) * 0.1

plt.figure(figsize=(14, 5))
degrees = [1, 4, 15]
titles = ['Underfitting (Degree 1)\nHigh Bias', 'Optimal Fit (Degree 4)\nGeneralizes Well', 'Overfitting (Degree 15)\nHigh Variance']

for i, degree in enumerate(degrees):
    ax = plt.subplot(1, 3, i + 1)
    
    # Fit Polynomial
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Convert features for plotting
    X_test = np.linspace(0, 1, 100)
    
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model", linewidth=2)
    plt.plot(X_test, np.cos(1.5 * np.pi * X_test), label="True Function", linestyle="--")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(titles[i], fontweight='bold')

plt.tight_layout()
plt.savefig('module 8/images/polynomial_regression_fit.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: polynomial_regression_fit.png")

# 2. Bias-Variance Tradeoff (Generalization Error)
complexity = np.linspace(1, 10, 100)
bias_squared = 1 / complexity  # Decreases with complexity
variance = 0.05 * np.exp(0.4 * complexity) # Increases with complexity
total_error = bias_squared + variance + 0.5 # Total Error + Irreducible noise

plt.figure(figsize=(8, 6))
plt.plot(complexity, total_error, 'k-', linewidth=3, label='Total Error (Generalization)')
plt.plot(complexity, bias_squared, 'b--', linewidth=2, label='Bias^2 (Underfitting)')
plt.plot(complexity, variance, 'r--', linewidth=2, label='Variance (Overfitting)')

# Optimal Point
min_idx = np.argmin(total_error)
plt.axvline(complexity[min_idx], color='green', linestyle=':', linewidth=2, label='Optimal Complexity')

plt.xlabel('Model Complexity', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Bias-Variance Tradeoff', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Annotations
plt.text(1.5, 3.5, 'High Bias\n(Underfitting)', ha='center', color='blue', fontweight='bold')
plt.text(9, 3.5, 'High Variance\n(Overfitting)', ha='center', color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('module 8/images/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: bias_variance_tradeoff.png")

plt.close()
print("Created: bias_variance_tradeoff.png")

# 3. Real-World Example: Auto MPG (Polynomial Regression)
# Demonstrate that relation between Horsepower and MPG is non-linear
try:
    mpg_data = sns.load_dataset('mpg').dropna()
    X_mpg = mpg_data['horsepower'].values.reshape(-1, 1)
    y_mpg = mpg_data['mpg'].values
    
    # Sort for plotting
    sorted_idx = np.argsort(X_mpg.flatten())
    X_mpg_sorted = X_mpg[sorted_idx]
    y_mpg_sorted = y_mpg[sorted_idx]
    
    # Linear Fit
    lr_mpg = LinearRegression()
    lr_mpg.fit(X_mpg, y_mpg)
    y_pred_lin = lr_mpg.predict(X_mpg_sorted)
    
    # Polynomial Fit (Degree 2)
    poly_mpg = PolynomialFeatures(degree=2)
    X_poly_mpg = poly_mpg.fit_transform(X_mpg)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly_mpg, y_mpg)
    y_pred_poly = lr_poly.predict(poly_mpg.transform(X_mpg_sorted))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_mpg, y_mpg, alpha=0.5, color='gray', label='Data Points')
    plt.plot(X_mpg_sorted, y_pred_lin, 'r--', linewidth=2, label='Linear Fit (Underfitting)')
    plt.plot(X_mpg_sorted, y_pred_poly, 'b-', linewidth=3, label='Polynomial Fit (Degree 2)')
    
    plt.title('Auto MPG: Horsepower vs MPG (Feature Engineering)', fontweight='bold')
    plt.xlabel('Horsepower')
    plt.ylabel('Miles Per Gallon (MPG)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('module 8/images/auto_mpg_poly.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: auto_mpg_poly.png")
except Exception as e:
    print(f"Skipping Auto MPG plot: {e}")

print("\nModule 8 complete: 3 visualizations created\n")

# ============================================================================
# MODULE 9 VISUALIZATIONS
# ============================================================================
print("### MODULE 9: Regularization & Feature Selection ###\n")
os.makedirs('module 9/images', exist_ok=True)

from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler

# 1. Regularization Path (Ridge vs Lasso)
# Demonstrate how coefficients shrink with alpha
np.random.seed(42)
n_samples, n_features = 50, 10
X = np.random.randn(n_samples, n_features)
# Some features are informative, others are noise
coef = 3 * np.random.randn(n_features)
inds = np.arange(n_features)
np.random.shuffle(inds)
coef[inds[5:]] = 0  # Sparsify coef
y = np.dot(X, coef) + 0.1 * np.random.normal(size=n_samples)

alphas = np.logspace(-4, 4, 100)

coefs_ridge = []
coefs_lasso = []

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X, y)
    coefs_ridge.append(ridge.coef_)
    
    lasso = Lasso(alpha=a)
    lasso.fit(X, y)
    coefs_lasso.append(lasso.coef_)

plt.figure(figsize=(14, 6))

ax1 = plt.subplot(1, 2, 1)
ax1.plot(alphas, coefs_ridge)
ax1.set_xscale('log')
ax1.set_xlabel('Alpha (Regularization Strength)')
ax1.set_ylabel('Coefficients')
ax1.set_title('Ridge Path (L2)\nCoefficients shrink but stay non-zero')
ax1.axis('tight')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 2, 2)
ax2.plot(alphas, coefs_lasso)
ax2.set_xscale('log')
ax2.set_xlabel('Alpha (Regularization Strength)')
ax2.set_title('Lasso Path (L1)\nCoefficients drop to zero (Feature Selection)')
ax2.axis('tight')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module 9/images/regularization_path.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: regularization_path.png")

# 2. Cross Validation Types Diagram
# Conceptual visualization
plt.figure(figsize=(12, 8))

# Data representation
data_len = 20
data_bar = np.ones((1, data_len))

# 1. Holdout
plt.subplot(4, 1, 1)
plt.title("Holdout Method (Train/Test Split)", fontsize=12, fontweight='bold', loc='left')
plt.barh(y=0, width=data_len*0.7, left=0, height=0.5, color='blue', label='Train')
plt.barh(y=0, width=data_len*0.3, left=data_len*0.7, height=0.5, color='orange', label='Test')
plt.xlim(0, data_len)
plt.axis('off')
plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0))

# 2. K-Fold (K=5)
plt.subplot(4, 1, 2)
plt.title("K-Fold Cross-Validation (K=5)", fontsize=12, fontweight='bold', loc='left')
for k in range(5):
    fold_size = data_len / 5
    start = k * fold_size
    plt.barh(y=k, width=data_len, left=0, height=0.6, color='blue', alpha=0.3) # Train background
    plt.barh(y=k, width=fold_size, left=start, height=0.6, color='orange', label='Validation' if k==0 else "") # Val fold
    plt.text(-1, k, f"Fold {k+1}", va='center', fontsize=10)

plt.xlim(0, data_len)
plt.axis('off')

# 3. Leave-One-Out (LOOCV)
plt.subplot(4, 1, 3)
plt.title("Leave-One-Out CV (N Iterations)", fontsize=12, fontweight='bold', loc='left')
# Show first 5 iterations
for i in range(5):
    plt.barh(y=i, width=data_len, left=0, height=0.6, color='blue', alpha=0.3)
    plt.barh(y=i, width=1, left=i, height=0.6, color='orange')
    plt.text(-1, i, f"Iter {i+1}", va='center', fontsize=10)

plt.text(data_len/2, 2, "...", fontsize=20, ha='center')
plt.xlim(0, data_len)
plt.axis('off')

plt.tight_layout()
plt.savefig('module 9/images/cv_types_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: cv_types_comparison.png")

plt.close()
print("Created: regularization_path.png")

# 2. Real-World Example: Diabetes (Lasso Feature Selection)
# Visualize which features are selected by Lasso as we increase regularization
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X_dia = diabetes.data
y_dia = diabetes.target
feature_names_dia = diabetes.feature_names

# Compute Lasso path
alphas_dia = np.logspace(-4, -0.5, 30)
coefs_dia = []
for a in alphas_dia:
    lasso_dia = Lasso(alpha=a, max_iter=10000)
    lasso_dia.fit(X_dia, y_dia)
    coefs_dia.append(lasso_dia.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas_dia, coefs_dia, linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Coefficients')
plt.title('Diabetes Dataset: Lasso Feature Selection', fontweight='bold')
plt.legend(feature_names_dia, loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True, alpha=0.3)
plt.axvline(0.01, color='red', linestyle='--', alpha=0.5, label='Strong Selection')
plt.tight_layout()
plt.savefig('module 9/images/diabetes_lasso.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: diabetes_lasso.png")

print("\nModule 9 complete: 3 visualizations created\n")

# ============================================================================
# MODULE 10 VISUALIZATIONS
# ============================================================================
print("### MODULE 10: Time Series Analysis ###\n")
os.makedirs('module 10/images', exist_ok=True)

# 1. Stationary vs Non-Stationary
np.random.seed(42)
t = np.arange(100)
# Non-Stationary (Trend + Seasonality)
trend = 0.5 * t
seasonality = 10 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 2, 100)
non_stationary = trend + seasonality + noise

# Stationary (Differenced)
stationary = np.diff(non_stationary)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, non_stationary, 'b-', label='Raw Data')
plt.title('Non-Stationary\n(Trend + Seasonality + Changing Mean)', fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t[1:], stationary, 'g-', label='Differenced')
plt.title('Stationary\n(Constant Mean & Variance)', fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Diff Value')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module 10/images/stationary_vs_nonstationary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: stationary_vs_nonstationary.png")

# 2. ACF / PACF Concept (Simulated)
# We will create a visual representation of what these look like
lags = np.arange(11)
acf_vals = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.0, -0.1, -0.05, 0.0, 0.0]) # Decay
pacf_vals = np.array([1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Cut-off at lag 1 (AR 1)

plt.figure(figsize=(12, 5))

# ACF
plt.subplot(1, 2, 1)
plt.bar(lags, acf_vals, width=0.3, color='blue')
plt.axhline(0, color='black', linewidth=0.8)
plt.axhspan(-0.2, 0.2, alpha=0.2, color='blue') # Significance bound
plt.title('ACF (Autocorrelation)\nGradual Decay = AR Process', fontweight='bold')
plt.xlabel('Lag')
plt.ylabel('Correlation')

# PACF
plt.subplot(1, 2, 2)
plt.bar(lags, pacf_vals, width=0.3, color='red')
plt.axhline(0, color='black', linewidth=0.8)
plt.axhspan(-0.2, 0.2, alpha=0.2, color='red') # Significance bound
plt.title('PACF (Partial Autocorrelation)\nSharp Cut-off = Order of AR Model', fontweight='bold')
plt.xlabel('Lag')

plt.tight_layout()
plt.savefig('module 10/images/acf_pacf_concept.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: acf_pacf_concept.png")

# 3. Forecasting & Uncertainty
# Visualize what a forecast looks like with confidence intervals
np.random.seed(101)
n_points = 50
t_train = np.arange(n_points)
y_train = 10 + 0.5 * t_train + np.random.normal(0, 2, n_points)

t_test = np.arange(n_points, n_points + 10)
y_test = 10 + 0.5 * t_test + np.random.normal(0, 2, 10)

# Forecast (Simple extrapolation with noise)
y_pred = 10 + 0.5 * t_test
# Confidence Interval (expanding over time)
uncertainty = np.arange(1, 11) * 0.5 + 2

plt.figure(figsize=(10, 6))
plt.plot(t_train, y_train, 'b.-', label='Historical Data')
plt.plot(t_test, y_test, 'g.-', label='Actual Future Data')
plt.plot(t_test, y_pred, 'r--', label='Forecast')
plt.fill_between(t_test, y_pred - uncertainty, y_pred + uncertainty, color='red', alpha=0.2, label='95% Confidence Interval')

plt.title('Forecasting & Uncertainty', fontsize=16, fontweight='bold')
plt.xlabel('Time')
plt.ylabel('Value')
plt.axvline(n_points - 0.5, color='k', linestyle=':', linewidth=1)
plt.text(n_points - 2, 25, 'Past', ha='right', fontsize=12)
plt.text(n_points + 1, 25, 'Future', ha='left', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('module 10/images/forecasting_uncertainty.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: forecasting_uncertainty.png")

print("Created: forecasting_uncertainty.png")

# 4. Real-World Example: Air Passengers (Trend + Seasonality)
print("Loading AirPassengers dataset...")
try:
    # Try loading from statsmodels (requires internet or local cache)
    import statsmodels.api as sm
    air_passengers = sm.datasets.get_rdataset("AirPassengers").data
    # Convert to datetime index
    air_passengers['time'] = pd.date_range(start='1949-01-01', periods=len(air_passengers), freq='MS')
    air_passengers.set_index('time', inplace=True)
    ts_air = air_passengers['value']

    plt.figure(figsize=(12, 10))
    result_air = sm.tsa.seasonal_decompose(ts_air, model='multiplicative')

    # Custom decomposition plot
    plt.subplot(411)
    plt.plot(ts_air, label='Original', color='blue')
    plt.legend(loc='upper left')
    plt.title('Air Passengers Data (1949-1960)\nMultiplicative Decomposition', fontweight='bold')

    plt.subplot(412)
    plt.plot(result_air.trend, label='Trend', color='orange')
    plt.legend(loc='upper left')

    plt.subplot(413)
    plt.plot(result_air.seasonal, label='Seasonality', color='green')
    plt.legend(loc='upper left')

    plt.subplot(414)
    plt.plot(result_air.resid, label='Residuals', color='red')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('module 10/images/air_passengers_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: air_passengers_analysis.png")
    
except Exception as e:
    print(f"Skipping AirPassengers plot due to load error: {e}")

print("\nModule 10 complete: 4 visualizations created\n")

# ============================================================================
# MODULE 11 VISUALIZATIONS
# ============================================================================
print("### MODULE 11: Used Car Price Analysis ###\n")
os.makedirs('module 11/images', exist_ok=True)

# Generate synthetic car data
np.random.seed(42)
n_cars = 1000
years = np.random.randint(2000, 2024, n_cars)
ages = 2024 - years
odometers = np.random.exponential(50000, n_cars) + (ages * 10000)
conditions = np.random.choice(['new', 'like new', 'excellent', 'good', 'fair', 'salvage'], n_cars, p=[0.05, 0.1, 0.3, 0.3, 0.2, 0.05])
condition_map = {'new': 5, 'like new': 4, 'excellent': 3, 'good': 2, 'fair': 1, 'salvage': 0}
condition_vals = np.array([condition_map[c] for c in conditions])

# Price model (Age, Odometer, Condition)
base_price = 45000
price = base_price * (0.9 ** ages) * (0.99999 ** odometers) * (0.8 + 0.1 * condition_vals) + np.random.normal(0, 2000, n_cars)
price = np.maximum(price, 500) # Minimum price

df_cars = pd.DataFrame({
    'price': price,
    'odometer': odometers,
    'year': years,
    'age': ages,
    'condition': conditions,
    'condition_val': condition_vals
})

# 1. Price Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_cars['price'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Used Car Prices', fontweight='bold')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('module 11/images/price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: price_distribution.png")

# 2. Correlation Matrix
plt.figure(figsize=(8, 6))
corr_matrix = df_cars[['price', 'year', 'odometer', 'condition_val']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix', fontweight='bold')
plt.savefig('module 11/images/feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: feature_correlation_matrix.png")

# 3. Price by Condition
plt.figure(figsize=(12, 6))
order = ['new', 'like new', 'excellent', 'good', 'fair', 'salvage']
sns.boxplot(x='condition', y='price', data=df_cars, order=order, palette='viridis')
plt.title('Price Distribution by Vehicle Condition', fontweight='bold')
plt.xlabel('Condition')
plt.ylabel('Price ($)')
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('module 11/images/price_by_condition.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: price_by_condition.png")

print("\nModule 11 complete: 3 visualizations created\n")

# ============================================================================
# MODULE 12 VISUALIZATIONS
# ============================================================================
print("### MODULE 12: K-Nearest Neighbors (KNN) ###\n")
os.makedirs('module 12/images', exist_ok=True)

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from matplotlib.colors import ListedColormap

# 1. Distance Metrics Visual
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
p1 = np.array([2, 2])
p2 = np.array([8, 6])

# Euclidean
ax = axes[0]
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=3, label='Euclidean (L2)')
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='red', s=100, zorder=5)
ax.set_title("Euclidean Distance (L2)\nShortest Path", fontweight='bold')
ax.grid(True, linestyle='--')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# Manhattan
ax = axes[1]
ax.plot([p1[0], p2[0]], [p1[1], p1[1]], 'g--', linewidth=3)
ax.plot([p2[0], p2[0]], [p1[1], p2[1]], 'g--', linewidth=3, label='Manhattan (L1)')
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='red', s=100, zorder=5)
ax.set_title("Manhattan Distance (L1)\nGrid/City Block", fontweight='bold')
ax.grid(True, linestyle='--')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# Chebyshev
ax = axes[2]
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k:', linewidth=1, alpha=0.3) # Layout
rect = plt.Rectangle((p1[0], p1[1]), p2[0]-p1[0], p2[1]-p1[1], fill=False, edgecolor='purple', linewidth=3, linestyle='-.')
ax.add_patch(rect)
ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], c='red', s=100, zorder=5)
ax.text(5, 1, "Max(|x2-x1|, |y2-y1|)", ha='center', color='purple')
ax.set_title("Chebyshev Distance (L∞)\nChessboard Move", fontweight='bold')
ax.grid(True, linestyle='--')
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

plt.suptitle("Common Distance Metrics", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 12/images/distance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: distance_metrics.png")

# 2. KNN Decision Boundaries (Effect of K)
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.3, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ks = [1, 15, 50]
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

for i, k in enumerate(ks):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    
    # Meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[i].pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
    axes[i].scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    axes[i].set_title(f"K = {k}\n{'Overfitting (High Variance)' if k==1 else ('Optimal?' if k==15 else 'Underfitting (High Bias)')}", fontweight='bold')

plt.suptitle("Effect of K on Decision Boundaries", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('module 12/images/knn_decision_boundaries.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: knn_decision_boundaries.png")

# 3. KNN Regression vs Linear Regression
np.random.seed(42)
X_reg = np.sort(5 * np.random.rand(40, 1), axis=0)
y_reg = np.sin(X_reg).ravel() + np.random.normal(0, 0.1, X_reg.shape[0])
T = np.linspace(0, 5, 500)[:, np.newaxis]

# Fit models
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_reg, y_reg)
y_knn = knn_reg.predict(T)

lin_reg = LinearRegression()
lin_reg.fit(X_reg, y_reg)
y_lin = lin_reg.predict(T)

plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='darkorange', label='Data')
plt.plot(T, y_knn, color='navy', label='KNN Regression (K=5)', linewidth=2)
plt.plot(T, y_lin, color='red', linestyle='--', label='Linear Regression', linewidth=2)
plt.plot(T, np.sin(T), color='green', alpha=0.3, label='True Function')
plt.title("KNN Regression (Non-Linear) vs Linear Regression", fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('module 12/images/knn_regression_vs_linear.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: knn_regression_vs_linear.png")

# 4. Error Rate vs K (Elbow Method)
from sklearn.model_selection import cross_val_score

X, y = make_moons(n_samples=500, noise=0.35, random_state=42)
k_values = range(1, 50)
error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    error_rates.append(1 - scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(k_values, error_rates, marker='o', linestyle='-', color='purple')
plt.title('Error Rate vs. K Value (The Elbow Method)', fontweight='bold')
plt.xlabel('Number of Neighbors K')
plt.ylabel('Error Rate')
plt.grid(True, alpha=0.3)
plt.savefig('module 12/images/error_vs_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: error_vs_k.png")

# 5. Real-World Example: Iris Dataset Visualization
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
target_names = iris.target_names

plt.figure(figsize=(10, 6))
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_iris[y_iris == i, 0], X_iris[y_iris == i, 1], color=color, alpha=0.8, lw=lw,
                label=target_name)

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Iris Dataset: Sepal Length vs Sepal Width', fontweight='bold')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.grid(True, alpha=0.3)
plt.savefig('module 12/images/iris_dataset.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: iris_dataset.png")


# 5. Precision-Recall vs Threshold
from sklearn.metrics import precision_recall_curve

# Train a model
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X, y)
y_scores = knn.predict_proba(X)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision & Recall vs. Threshold', fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.axvline(0.5, color='k', linestyle=':', label='Default Threshold (0.5)')

plt.tight_layout()
plt.savefig('module 12/images/precision_recall_threshold.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: precision_recall_threshold.png")



print("\nModule 12 complete: 5 visualizations created\n")

# ============================================================================
# MODULE 13 VISUALIZATIONS
# ============================================================================
print("### MODULE 13: Logistic Regression ###\n")
os.makedirs('module 13/images', exist_ok=True)

# 1. Sigmoid Function
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, 'b-', linewidth=3, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.axhline(0.5, color='red', linestyle='--', label='Decision Threshold (0.5)')
plt.axhline(1, color='gray', linestyle=':')
plt.xlabel('z (Log-Odds)', fontsize=12)
plt.ylabel('Probability $\sigma(z)$', fontsize=12)
plt.title('The Sigmoid Function', fontweight='bold', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/sigmoid_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: sigmoid_visualization.png")

# 2. Logistic Regression Fit (Toy Data)
np.random.seed(42)
X_toys = np.random.normal(0, 1, 100)
y_toys = (X_toys > 0).astype(int)
# Add noise (flip some labels)
flip_indices = np.random.choice(100, 10, replace=False)
y_toys[flip_indices] = 1 - y_toys[flip_indices]

clf = LogisticRegression()
clf.fit(X_toys.reshape(-1, 1), y_toys)
X_test_toys = np.linspace(-3, 3, 300).reshape(-1, 1)
y_prob_toys = clf.predict_proba(X_test_toys)[:, 1]

plt.figure(figsize=(10, 6))
plt.scatter(X_toys, y_toys, c=y_toys, cmap='bwr', alpha=0.6, edgecolors='k', label='Data')
plt.plot(X_test_toys, y_prob_toys, 'g-', linewidth=3, label='Logistic Model')
plt.axhline(0.5, color='gray', linestyle='--', label='Threshold')
plt.ylabel('Probability')
plt.xlabel('Feature Value')
plt.title('Logistic Regression Model Fit', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/lr.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: lr.png")

# 3. Decision Boundary (2D)
from sklearn.datasets import make_classification
X_2d, y_2d = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, random_state=42)
clf_2d = LogisticRegression()
clf_2d.fit(X_2d, y_2d)

# Meshgrid
x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))
Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, edgecolors='k', cmap='coolwarm', s=50)
plt.title('Linear Decision Boundary (2D)', fontweight='bold')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.tight_layout()
plt.savefig('module 13/images/dboundary.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: dboundary.png")

# 4. L1 Regularization Coefficients
# We'll calculate paths for a dataset
from sklearn.svm import l1_min_c
iris = load_iris()
X_iris = iris.data
y_iris = (iris.target == 0).astype(int) # Binary: Setosa vs Rest
X_iris_std = StandardScaler().fit_transform(X_iris)

cs = l1_min_c(X_iris_std, y_iris, loss='log') * np.logspace(0, 7, 16)
clf_l1 = LogisticRegression(penalty='l1', solver='liblinear', tol=1e-6, max_iter=int(1e6))
coefs_ = []
for c in cs:
    clf_l1.set_params(C=c)
    clf_l1.fit(X_iris_std, y_iris)
    coefs_.append(clf_l1.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.figure(figsize=(10, 6))
plt.plot(np.log10(cs), coefs_, marker='o')
plt.xlabel('log(C)')
plt.ylabel('Coefficients')
plt.title('L1 Regularization Path (Lasso Effect)', fontweight='bold')
plt.legend(iris.feature_names, loc='upper left')
plt.grid(True, alpha=0.3)
plt.axis('tight')
plt.tight_layout()
plt.savefig('module 13/images/coefl1.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: coefl1.png")

# 5. Threshold Analysis (Precision-Recall)
# Re-using the toy data model
y_scores_toys = clf.predict_proba(X_test_toys)[:, 1]
# We need true labels for the test set, let's just use the training set for this demo viz
y_scores_train = clf.predict_proba(X_toys.reshape(-1, 1))[:, 1]
prec, rec, thresh = precision_recall_curve(y_toys, y_scores_train)

plt.figure(figsize=(10, 6))
plt.plot(thresh, prec[:-1], 'b--', label='Precision')
plt.plot(thresh, rec[:-1], 'g-', label='Recall')
plt.title('Precision & Recall vs Threshold', fontweight='bold')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/thresh.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: thresh.png")

# 6. Performance Metrics Bar Chart
metrics = {'Accuracy': 0.85, 'Precision': 0.82, 'Recall': 0.88, 'F1': 0.85, 'AUC': 0.91}
plt.figure(figsize=(8, 5))
plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'purple', 'red'], alpha=0.7, edgecolor='k')
plt.ylim(0, 1.1)
plt.title('Model Performance Metrics', fontweight='bold')
plt.ylabel('Score')
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, str(v), ha='center', fontweight='bold')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/p3.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: p3.png")

# 7. Optimization Path (Conceptual)
# Contour plot of a convex function
delta = 0.025
x_opt = np.arange(-3.0, 3.0, delta)
y_opt = np.arange(-2.0, 2.0, delta)
X_opt, Y_opt = np.meshgrid(x_opt, y_opt)
Z_opt = X_opt**2 + Y_opt**2 # Simple convex bowl

plt.figure(figsize=(8, 6))
cs = plt.contour(X_opt, Y_opt, Z_opt, levels=20, cmap='viridis')
plt.clabel(cs, inline=1, fontsize=10)
# Mock path
path_x = [2.5, 2, 1.5, 1, 0.5, 0.2, 0]
path_y = [1.5, 1.2, 0.8, 0.5, 0.2, 0.1, 0]
plt.plot(path_x, path_y, 'ro-', linewidth=2, label='Gradient Descent Path')
plt.plot(0, 0, 'y*', markersize=15, label='Global Minimum')
plt.title('Cost Function Optimization (Gradient Descent)', fontweight='bold')
plt.xlabel('Parameter Beta 1')
plt.ylabel('Parameter Beta 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/betasopt.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: betasopt.png")

# 8. Feature Distribution (Penguins Flipper)
# Using seaborn built-in data or mock
# Assuming we can load penguins, otherwise mock
try:
    penguins = sns.load_dataset('penguins')
    plt.figure(figsize=(10, 6))
    sns.histplot(data=penguins, x='flipper_length_mm', hue='species', kde=True, element='step')
    plt.title('Feature Distribution by Class (Penguins)', fontweight='bold')
except:
    # Fallback
    d1 = np.random.normal(180, 10, 100)
    d2 = np.random.normal(210, 10, 100)
    plt.figure(figsize=(10, 6))
    plt.hist(d1, alpha=0.5, label='Class 0', bins=20)
    plt.hist(d2, alpha=0.5, label='Class 1', bins=20)
    plt.legend()
    plt.title('Feature Distribution by Class (Simulated)', fontweight='bold')
plt.tight_layout()
plt.savefig('module 13/images/flipperdist.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: flipperdist.png")

# 12. Real-World Example: Breast Cancer Wisconsin
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X_bc = data.data
y_bc = data.target
# Feature indices: 0=mean radius, 1=mean texture
plt.figure(figsize=(10, 6))
plt.scatter(X_bc[y_bc==0, 0], X_bc[y_bc==0, 1], c='red', label='Malignant', alpha=0.6)
plt.scatter(X_bc[y_bc==1, 0], X_bc[y_bc==1, 1], c='blue', label='Benign', alpha=0.6)
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Breast Cancer Wisconsin: Radius vs Texture', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('module 13/images/breast_cancer_viz.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: breast_cancer_viz.png")

print("\nModule 13 complete: 11 visualizations created\n")

# ============================================================================
# MODULE 14 VISUALIZATIONS
# ============================================================================
print("### MODULE 14: Decision Trees ###\n")
os.makedirs('module 14/images', exist_ok=True)

# 1. Gini vs Entropy
p = np.linspace(0.001, 0.999, 100)
gini = 1 - (p**2 + (1-p)**2)
entropy = -(p*np.log2(p) + (1-p)*np.log2(1-p))
# Scale entropy to match gini range for visual comparison (entropy max is 1, gini max is 0.5)
entropy_scaled = entropy * 0.5 

plt.figure(figsize=(10, 6))
plt.plot(p, gini, 'r-', linewidth=3, label='Gini Impurity')
plt.plot(p, entropy, 'b--', linewidth=3, label='Entropy (Base 2)')
plt.plot(p, entropy_scaled, 'g:', linewidth=3, label='Entropy (Scaled 0.5)')
plt.title('Impurity Measures: Gini vs Entropy', fontweight='bold')
plt.xlabel('Probability of Class 1')
plt.ylabel('Impurity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('module 14/images/gini_vs_entropy.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: gini_vs_entropy.png")

# 2. Overfitting (Depth vs Accuracy)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Generate complex moon data
from sklearn.datasets import make_moons
X_m, y_m = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_m, y_m, test_size=0.3, random_state=42)

depths = range(1, 16)
train_scores = []
test_scores = []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train_m, y_train_m)
    train_scores.append(clf.score(X_train_m, y_train_m))
    test_scores.append(clf.score(X_test_m, y_test_m))

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores, 'bo-', label='Training Accuracy')
plt.plot(depths, test_scores, 'rs-', label='Test Accuracy')
plt.title('Overfitting: Model Complexity (Depth) vs Accuracy', fontweight='bold')
plt.xlabel('Max Depth of Tree')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
# Highlight divergence
plt.annotate('Overfitting Starts Here', xy=(3, test_scores[2]), xytext=(5, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.tight_layout()
plt.savefig('module 14/images/overfitting_depth.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: overfitting_depth.png")

# 3. Decision Boundaries (Iris)
# Visualize splits on 2 features
iris = load_iris()
X_iris_2 = iris.data[:, [0, 2]] # Sepal length, Petal length
y_iris = iris.target

clf_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree.fit(X_iris_2, y_iris)

x_min, x_max = X_iris_2[:, 0].min() - 1, X_iris_2[:, 0].max() + 1
y_min, y_max = X_iris_2[:, 1].min() - 1, X_iris_2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))
Z = clf_tree.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X_iris_2[:, 0], X_iris_2[:, 1], c=y_iris, alpha=0.8, edgecolors='k', cmap='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.title('Decision Tree Boundaries (Depth=3)', fontweight='bold')
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.tight_layout()
plt.savefig('module 14/images/dt_boundaries_iris.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: dt_boundaries_iris.png")

# 4. Feature Importance
importances = clf_tree.feature_importances_
# Only used 2 features
feat_names = ['Sepal Length', 'Petal Length'] # Matches indices [0, 2]
plt.figure(figsize=(8, 5))
plt.bar(feat_names, importances, color=['skyblue', 'lightgreen'], edgecolor='k')
plt.title('Feature Importance (Iris Subset)', fontweight='bold')
plt.ylabel('Importance Score')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('module 14/images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: feature_importance.png")

plt.close()
print("Created: feature_importance.png")

# 5. Tree Structure Visualization
# Visualize the actual tree structure for Iris
from sklearn.tree import plot_tree
plt.figure(figsize=(20, 10))
# Train a slightly deeper tree for visualization
clf_tree_viz = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_tree_viz.fit(iris.data, iris.target)
plot_tree(clf_tree_viz, feature_names=iris.feature_names, class_names=iris.target_names, 
          filled=True, rounded=True, fontsize=12)
plt.title('Decision Tree Structure (Iris Dataset)', fontweight='bold', fontsize=16)
plt.tight_layout()
plt.savefig('module 14/images/tree_structure_viz.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: tree_structure_viz.png")

print("\nModule 14 complete: 5 visualizations created\n")
print("Created: breast_cancer_viz.png")

# 9. OVO vs OVR Conceptual Visualization
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import SVC # SVC exposes decision boundaries nicely for OVO/OVR illustration
# We use SVC linear because it makes the separation very clear for educational purposes

X_dummy, y_dummy = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                       n_informative=2, n_clusters_per_class=1, 
                                       n_classes=3, class_sep=2.0, random_state=42)

# OVR
clf_ovr = OneVsRestClassifier(SVC(kernel='linear', probability=True))
clf_ovr.fit(X_dummy, y_dummy)

# OVO
clf_ovo = OneVsOneClassifier(SVC(kernel='linear', probability=True))
clf_ovo.fit(X_dummy, y_dummy)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plotting function
def plot_boundaries_conceptual(ax, clf, title, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis', edgecolors='k')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    return scatter

# OVR Plot
_ = plot_boundaries_conceptual(axes[0], clf_ovr, 'One-vs-Rest (OVR)\nK Classifiers (Red vs Rest, etc.)', X_dummy, y_dummy)
# OVO Plot
scatter = plot_boundaries_conceptual(axes[1], clf_ovo, 'One-vs-One (OVO)\nK(K-1)/2 Classifiers (Red vs Blue, etc.)', X_dummy, y_dummy)

# Legend
handles, labels = scatter.legend_elements() if 'scatter' in locals() else ([], []) # Safe fallback
# Manually creating legend for classes 0, 1, 2
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=plt.cm.viridis(0.0), label='Class 0'),
                   Patch(facecolor=plt.cm.viridis(0.5), label='Class 1'),
                   Patch(facecolor=plt.cm.viridis(1.0), label='Class 2')]
axes[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('module 13/images/ovo_vs_ovr.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: ovo_vs_ovr.png")

# 10. Log Loss Function Concept
p = np.linspace(0.001, 0.999, 100)
cost_y1 = -np.log(p)
cost_y0 = -np.log(1 - p)

plt.figure(figsize=(10, 6))
plt.plot(p, cost_y1, 'b-', linewidth=3, label='y = 1 (Cost = -log(p))')
plt.plot(p, cost_y0, 'r-', linewidth=3, label='y = 0 (Cost = -log(1-p))')
plt.xlabel('Predicted Probability (p)', fontsize=12)
plt.ylabel('Cost (Loss)', fontsize=12)
plt.title('Log Loss (Binary Cross-Entropy) Function', fontweight='bold', fontsize=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 13/images/log_loss_concept.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: log_loss_concept.png")

# 11. Odds vs Probability vs Log-Odds
probs = np.linspace(0.01, 0.99, 100)
odds = probs / (1 - probs)
log_odds = np.log(odds)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Probability vs Odds
axes[0].plot(probs, odds, 'purple', linewidth=3)
axes[0].set_xlabel('Probability (p)')
axes[0].set_ylabel('Odds')
axes[0].set_title('Odds vs Probability\n(Exponential Growth)', fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 20) # Limit y-axis to see the curve better usually odds explode
axes[0].axvline(0.5, color='gray', linestyle='--')
axes[0].text(0.55, 10, 'p=0.5 -> Odds=1', fontsize=10)

# Plot 2: Probability vs Log-Odds (Sigmoid Inverse)
axes[1].plot(log_odds, probs, 'green', linewidth=3)
axes[1].set_xlabel('Log-Odds (Logit)')
axes[1].set_ylabel('Probability (p)')
axes[1].set_title('Probability vs Log-Odds\n(The Sigmoid)', fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(0, color='gray', linestyle='--')
axes[1].axhline(0.5, color='gray', linestyle='--')
axes[1].text(0.5, 0.4, 'Log-Odds=0 -> p=0.5', fontsize=10)

plt.tight_layout()
plt.savefig('module 13/images/odds_visualization.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: odds_visualization.png")

print("\nModule 13 complete: 11 visualizations created\n")

# ============================================================================
# MODULE 15 VISUALIZATIONS
# ============================================================================
print("### MODULE 15: Optimization & SGD ###\n")
os.makedirs('module 15/images', exist_ok=True)

# 1. Convex vs Non-Convex
x = np.linspace(-5, 5, 200)
y_convex = x**2
y_non_convex = x**2 + 10*np.sin(x) + 20 # Shift up for visibility

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(x, y_convex, 'b-', linewidth=3)
axes[0].set_title('Convex Function\n(Single Global Minimum)', fontweight='bold')
axes[0].set_xlabel('Parameter $\\theta$')
axes[0].set_ylabel('Cost $J(\\theta)$')
axes[0].grid(True, alpha=0.3)
axes[0].plot(0, 0, 'go', markersize=10, label='Global Minimum')
axes[0].legend()

axes[1].plot(x, y_non_convex, 'r-', linewidth=3)
axes[1].set_title('Non-Convex Function\n(Multiple Local Minima)', fontweight='bold')
axes[1].set_xlabel('Parameter $\\theta$')
axes[1].set_ylabel('Cost $J(\\theta)$')
axes[1].grid(True, alpha=0.3)
axes[1].plot(0, 20, 'go', markersize=10, label='Local Minimum')
# Approximate global min manually for viz
global_min_x = -1.3 
global_min_y = global_min_x**2 + 10*np.sin(global_min_x) + 20
axes[1].plot(global_min_x, global_min_y, 'k*', markersize=15, label='Global Minimum')
axes[1].legend()

plt.tight_layout()
plt.savefig('module 15/images/convex_vs_non_convex.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: convex_vs_non_convex.png")

# 2. Gradient Descent 1D Steps (Concept)
def f(x): return x**2
def df(x): return 2*x

theta = 4.0
history = [theta]
alpha = 0.1
for _ in range(5):
    theta = theta - alpha * df(theta)
    history.append(theta)

x_plot = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))
plt.plot(x_plot, f(x_plot), 'k-', alpha=0.6, label='$J(\\theta) = \\theta^2$')
history = np.array(history)
plt.plot(history, f(history), 'ro-', markersize=8, label='Gradient Descent Steps')

# Draw arrows for steps
for i in range(len(history)-1):
    plt.annotate('', xy=(history[i+1], f(history[i+1])), xytext=(history[i], f(history[i])),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=2))

plt.title('Gradient Descent Steps (1D)', fontweight='bold')
plt.xlabel('Parameter $\\theta$')
plt.ylabel('Cost $J(\\theta)$')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('module 15/images/gradient_descent_1d.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: gradient_descent_1d.png")

# 3. Learning Rate Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
alphas = [0.01, 0.1, 1.1] # Small, Good, Large (Diverging for x^2, df=2x -> x_new = x - 2.2x = -1.2x. Oscillates and diverges if > 1 for simple scaling? Actually for x^2, x_new = x(1-2alpha). Diverges if |1-2alpha| > 1 => 2alpha > 2 => alpha > 1. So 1.1 is diverging.)
titles = ['Too Small (Slow)', 'Just Right (Converges)', 'Too Large (Diverges)']

for ax, lr, title in zip(axes, alphas, titles):
    theta = 4.5
    path = [theta]
    for _ in range(10): # 10 steps
        theta = theta - lr * 2 * theta # derivative is 2*theta
        path.append(theta)
    
    path = np.array(path)
    ax.plot(x_plot, f(x_plot), 'k-', alpha=0.3)
    ax.plot(path, f(path), 'o-', markersize=6, linewidth=2, label=f'$\\alpha={lr}$')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('$\\theta$')
    if lr == 1.1:
        ax.set_ylim(0, 50) # Limit y-axis for diverging case
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
plt.savefig('module 15/images/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: learning_rate_comparison.png")

# 4. GD vs SGD Path (2D Contour)
# Generate a 2D loss surface (anisotropic bowl)
def loss_2d(x, y): return 0.5*x**2 + 2*y**2 # Gradient is [x, 4y]

# Batch GD Path (Smooth)
path_gd = []
x, y = 4.0, 3.0
lr = 0.05
for _ in range(20):
    path_gd.append([x, y])
    grad_x, grad_y = x, 4*y
    x -= lr * grad_x
    y -= lr * grad_y
path_gd = np.array(path_gd)

# SGD Path (Noisy - simulated with random noise added to gradient)
path_sgd = []
x, y = 4.0, 3.0
np.random.seed(42)
for _ in range(30):
    path_sgd.append([x, y])
    # True gradient + high noise to simulate "stochastic" mini-batch on full dataset
    noise = np.random.randn(2) * 2.0 
    grad_x, grad_y = x + noise[0], 4*y + noise[1]
    x -= lr * grad_x
    y -= lr * grad_y
path_sgd = np.array(path_sgd)

# Plot
x_grid = np.linspace(-5, 5, 100)
y_grid = np.linspace(-5, 5, 100)
X_g, Y_g = np.meshgrid(x_grid, y_grid)
Z_g = 0.5*X_g**2 + 2*Y_g**2

plt.figure(figsize=(10, 8))
plt.contour(X_g, Y_g, Z_g, levels=np.logspace(-1, 2, 20), cmap='gray_r', alpha=0.4)
plt.plot(path_gd[:, 0], path_gd[:, 1], 'b-o', label='Batch Gradient Descent', markersize=6, linewidth=2)
plt.plot(path_sgd[:, 0], path_sgd[:, 1], 'r-^', label='Stochastic Gradient Descent', markersize=6, linewidth=2, alpha=0.7)
plt.plot(0, 0, 'k*', markersize=15, label='Global Minimum')

plt.title('Optimization Paths: Batch GD vs SGD', fontweight='bold')
plt.xlabel('Parameter $\\theta_1$')
plt.ylabel('Parameter $\\theta_2$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('module 15/images/gd_vs_sgd_path.png', dpi=300, bbox_inches='tight')
plt.close()
print("Created: gd_vs_sgd_path.png")

print("\nModule 15 complete: 4 visualizations created\n")
