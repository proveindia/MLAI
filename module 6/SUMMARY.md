# Module 6: Unsupervised Learning Summary

This module focuses on techniques to find hidden patterns in unlabeled data: Dimensionality Reduction (PCA) and Clustering.

## ⏱️ Quick Review (20 Mins)

### 1. Principal Component Analysis (PCA)
Reduces the number of features (dimensions) while retaining the most important information (variance).

**Why use it?**
- To reduce noise.
- To visualize high-dimensional data.
- To simplify datasets with many correlated variables.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Standardize the data (Vital for PCA!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 2. Apply PCA
pca = PCA(n_components=2) # Reduce to 2 dimensions
principal_components = pca.fit_transform(X_scaled)

# 3. Variance Explained
print(pca.explained_variance_ratio_) 
# Output: [0.45, 0.30] -> PC1 explains 45%, PC2 explains 30%
```

### 2. Clustering
Grouping similar data points together.

**K-Means Clustering:**
Partitions data into $K$ distinct clusters based on distance to centroids.
```python
from sklearn.cluster import KMeans

# Fit K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize
sns.scatterplot(data=df, x='PC1', y='PC2', hue='cluster')
plt.show()
```

**DBSCAN:**
Density-based clustering. Good for arbitrary shapes and outlier detection.
```python
from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
```

---
*Reference: Mod6_PCA_Clustering.ipynb*
