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

print("\nModule 2 complete: 3 visualizations created\n")

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

print("="*70)
print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
print("="*70)
