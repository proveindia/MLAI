import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import os

os.makedirs('images', exist_ok=True)
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

print('Creating Module 6 visualizations...')

# 1. SVD Matrix Decomposition
fig, ax = plt.subplots(figsize=(14, 6))

# Visual representation of SVD
ax.text(0.5, 0.85, 'Singular Value Decomposition (SVD)', ha='center', fontsize=18, fontweight='bold', transform=ax.transAxes)

# Matrix X
ax.add_patch(plt.Rectangle((0.05, 0.3), 0.15, 0.35, facecolor='#3498db', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.125, 0.48, 'X', ha='center', va='center', fontsize=24, fontweight='bold', color='white', transform=ax.transAxes)
ax.text(0.125, 0.15, 'm x n', ha='center', fontsize=12, style='italic', transform=ax.transAxes)

# Equals sign
ax.text(0.23, 0.48, '=', ha='center', va='center', fontsize=28, transform=ax.transAxes)

# Matrix U
ax.add_patch(plt.Rectangle((0.28, 0.3), 0.12, 0.35, facecolor='#e74c3c', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.34, 0.48, 'U', ha='center', va='center', fontsize=24, fontweight='bold', color='white', transform=ax.transAxes)
ax.text(0.34, 0.15, 'm x m', ha='center', fontsize=11, style='italic', transform=ax.transAxes)
ax.text(0.34, 0.72, 'Left Singular\\nVectors', ha='center', fontsize=9,transform=ax.transAxes)

# Matrix Sigma
ax.add_patch(plt.Rectangle((0.43, 0.3), 0.15, 0.35, facecolor='#2ecc71', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.505, 0.48, 'Σ', ha='center', va='center', fontsize=28, fontweight='bold', color='white', transform=ax.transAxes)
ax.text(0.505, 0.15, 'm x n', ha='center', fontsize=12, style='italic', transform=ax.transAxes)
ax.text(0.505, 0.72, 'Singular Values\\n(diagonal)', ha='center', fontsize=9, transform=ax.transAxes)

# Matrix V^T
ax.add_patch(plt.Rectangle((0.61, 0.3), 0.12, 0.35, facecolor='#9b59b6', alpha=0.6, edgecolor='black', linewidth=2))
ax.text(0.67, 0.48, 'V^T', ha='center', va='center', fontsize=22, fontweight='bold', color='white', transform=ax.transAxes)
ax.text(0.67, 0.15, 'n x n', ha='center', fontsize=11, style='italic', transform=ax.transAxes)
ax.text(0.67, 0.72, 'Right Singular\\nVectors (PCs)', ha='center', fontsize=9, transform=ax.transAxes)

# Visualization of diagonal matrix
for i in range(4):
    ax.plot([0.445 + i*0.04, 0.465 + i*0.04], [0.58 - i*0.06, 0.52 - i*0.06], 
            'k-', linewidth=3, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig('images/svd_matrix_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: svd_matrix_decomposition.png')

# 2. PCA Variance Explained (Scree Plot)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Generate sample data with known variance structure
np.random.seed(42)
n_samples = 200
n_features = 10

# Create correlated features
base = np.random.randn(n_samples, 3)
X = np.column_stack([
    base[:, 0] + np.random.randn(n_samples) * 0.1,
    base[:, 0] * 0.8 + np.random.randn(n_samples) * 0.2,
    base[:, 1] + np.random.randn(n_samples) * 0.1,
    base[:, 1] * 0.7 + np.random.randn(n_samples) * 0.3,
    base[:, 2] + np.random.randn(n_samples) * 0.2,
    np.random.randn(n_samples) * 0.5,
    np.random.randn(n_samples) * 0.4,
    np.random.randn(n_samples) * 0.3,
    np.random.randn(n_samples) * 0.2,
    np.random.randn(n_samples) * 0.1,
])

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA()
pca.fit(X_scaled)

# Scree plot
axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
             pca.explained_variance_ratio_, 'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Variance Explained Ratio', fontweight='bold', fontsize=12)
axes[0].set_title('Scree Plot', fontweight='bold', fontsize=14)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].set_xticks(range(1, 11))

# Mark elbow
axes[0].axvline(x=3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Elbow (keep 3 PCs)')
axes[0].legend()

# Cumulative variance
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 'bo-', linewidth=2.5, markersize=8)
axes[1].fill_between(range(1, len(cumulative_var) + 1), cumulative_var, alpha=0.3)
axes[1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Variance')
axes[1].axhline(y=0.90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='90% Variance')
axes[1].set_xlabel('Number of Components', fontweight='bold', fontsize=12)
axes[1].set_ylabel('Cumulative Variance Explained', fontweight='bold', fontsize=12)
axes[1].set_title('Cumulative Variance Explained', fontweight='bold', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(range(1, 11))
axes[1].set_ylim(0, 1.05)
axes[1].legend()

plt.suptitle('PCA Variance Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/pca_variance_explained.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: pca_variance_explained.png')

# 3. PCA 2D Projection
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Generate 3-class data in high dimensions
X_blob, y_blob = make_blobs(n_samples=300, n_features=10, centers=3,
                             cluster_std=1.5, random_state=42)

# Standardize
X_blob_scaled = StandardScaler().fit_transform(X_blob)

# PCA to 2D
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_blob_scaled)

# Original space (first 2 features)
scatter1 = axes[0].scatter(X_blob_scaled[:, 0], X_blob_scaled[:, 1], 
                           c=y_blob, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[0].set_xlabel('Original Feature 1 (standardized)', fontweight='bold')
axes[0].set_ylabel('Original Feature 2 (standardized)', fontweight='bold')
axes[0].set_title('Original Space (2 of 10 features)', fontweight='bold', fontsize=13)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Class')

# PCA space
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=y_blob, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% var)', fontweight='bold')
axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% var)', fontweight='bold')
axes[1].set_title('PCA Space (captures most variance)', fontweight='bold', fontsize=13)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Class')

# Add arrows for principal components
axes[1].arrow(0, 0, 3, 0, head_width=0.3, head_length=0.5, fc='red', ec='red', alpha=0.7, linewidth=2)
axes[1].arrow(0, 0, 0, 2, head_width=0.3, head_length=0.5, fc='blue', ec='blue', alpha=0.7, linewidth=2)
axes[1].text(3.5, 0, 'PC1', fontsize=12, color='red', fontweight='bold')
axes[1].text(0, 2.5, 'PC2', fontsize=12, color='blue', fontweight='bold')

plt.suptitle('High-Dimensional Data Projected to 2D via PCA', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('images/pca_2d_projection.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: pca_2d_projection.png')

# 4. K-Means Iterations
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Generate simple 2D data
X_km, y_true = make_blobs(n_samples=150, centers=3, n_features=2, 
                          cluster_std=0.8, random_state=42)

# Manual K-means iterations
np.random.seed(42)
n_clusters = 3

# Initialize random centroids
centroids = X_km[np.random.choice(X_km.shape[0], n_clusters, replace=False)]

iterations = [
    ('Initialization', centroids.copy(), None),
]

# Perform a few iterations manually
for iteration in range(5):
    # Assign points to nearest centroid
    distances = np.sqrt(((X_km - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    
    # Update centroids
    new_centroids = np.array([X_km[labels == k].mean(axis=0) for k in range(n_clusters)])
    
    iterations.append((f'Iteration {iteration +1}', new_centroids.copy(), labels.copy()))
    centroids = new_centroids

# Plot iterations
for idx, (title, cents, labs) in enumerate(iterations[:6]):
    ax = axes[idx//3, idx%3]
    
    if labs is not None:
        scatter = ax.scatter(X_km[:, 0], X_km[:, 1], c=labs, cmap='viridis', 
                            s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    else:
        ax.scatter(X_km[:, 0], X_km[:, 1], c='gray', s=50, alpha=0.4, edgecolors='k', linewidth=0.5)
    
    ax.scatter(cents[:, 0], cents[:, 1], c='red', s=300, alpha=0.8, 
              edgecolors='black', linewidth=2, marker='*', label='Centroids')
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.suptitle('K-Means Clustering: Convergence Process (K=3)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('images/kmeans_iterations.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: kmeans_iterations.png')

# 5. Elbow Method Detailed
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate inertia for different K values
inertias = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_km)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
ax.plot(k_range, inertias, 'bo-', linewidth=3, markersize=10, label='Inertia')
ax.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=13)
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontweight='bold', fontsize=13)
ax.set_title('Elbow Method for Optimal K', fontweight='bold', fontsize=15)
ax.grid(True, alpha=0.3)

# Highlight elbow
optimal_k = 3
ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.scatter([optimal_k], [inertias[optimal_k-1]], s=400, c='red', 
          marker='o', edgecolors='darkred', linewidth=3, zorder=5, label='Optimal K=3')

# Add annotation
ax.annotate('Elbow Point', xy=(optimal_k, inertias[optimal_k-1]), 
           xytext=(optimal_k + 1.5, inertias[optimal_k-1] + 20),
           fontsize=12, fontweight='bold',
           arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax.legend(fontsize=11)
ax.set_xticks(k_range)

plt.tight_layout()
plt.savefig('images/elbow_method_detailed.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: elbow_method_detailed.png')

# 6. DBSCAN Epsilon Effect
from sklearn.cluster import DBSCAN

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Generate data with noise
X_db, _ = make_blobs(n_samples=200, centers=3, n_features=2, 
                     cluster_std=0.6, random_state=42)
# Add noise points
noise = np.random.uniform(-8, 8, (20, 2))
X_db = np.vstack([X_db, noise])

eps_values = [0.3, 0.8, 1.5]

for idx, eps in enumerate(eps_values):
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(X_db)
    
    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Plot
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points
            class_member_mask = (labels == label)
            xy = X_db[class_member_mask]
            axes[idx].scatter(xy[:, 0], xy[:, 1], c='gray', marker='x', 
                            s=50, alpha=0.5, label='Noise')
        else:
            class_member_mask = (labels == label)
            xy = X_db[class_member_mask]
            axes[idx].scatter(xy[:, 0], xy[:, 1], c=[color], s=60, 
                            alpha=0.7, edgecolors='k', linewidth=0.5)
    
    axes[idx].set_title(f'eps={eps}\\nClusters: {n_clusters}, Noise: {n_noise}', 
                       fontweight='bold', fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlabel('Feature 1', fontweight='bold')
    if idx == 0:
        axes[idx].set_ylabel('Feature 2', fontweight='bold')

plt.suptitle('DBSCAN: Effect of Epsilon (ε) Parameter (min_samples=5)', 
            fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('images/dbscan_epsilon_effect.png', dpi=300, bbox_inches='tight')
plt.close()
print('Created: dbscan_epsilon_effect.png')

print('\\nAll Module 6 images created successfully!')
