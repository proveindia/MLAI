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
print("Created: loss_function_comparison.png")

print("\nModule 7 complete: 1 visualization created\n")

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

print("\nModule 8 complete: 2 visualizations created\n")

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

print("\nModule 9 complete: 2 visualizations created\n")

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

print("\nModule 10 complete: 3 visualizations created\n")

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
