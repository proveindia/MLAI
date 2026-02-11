# Module 19: Collaborative Filtering and Recommendation Systems

## Overview
This module explored Recommendation Systems, focusing on Collaborative Filtering and the use of the `Surprise` library to predict user ratings.

## Key Concepts
*   **Collaborative Filtering:** Making predictions about the interests of a user by collecting preferences from many users.
*   **Latent Features:** Hidden features that explain the relationship between users and items.
*   **Alternating Least Squares (ALS):** Iterative optimization to find user and item factors.
*   **Matrix Factorization (SVD):** Decomposing the user-item interaction matrix into lower-dimensional matrices.
*   **Surprise Library:** A Python scikit for building and analyzing recommender systems.
*   **Hybrid Recommendations:** Combining multiple algorithms (e.g., SVD + SlopeOne) to improve prediction accuracy.

## Additional Algorithms Used
Beyond SVD and SlopeOne, the module utilized several other algorithms from the Surprise library:

1.  **NMF (Non-Negative Matrix Factorization):** similar to SVD but enforces non-negative factors, often leading to more interpretable results.
2.  **CoClustering:** A collaborative filtering algorithm based on co-clustering users and items (simultaneous clustering).
3.  **KNNBasic:** A basic collaborative filtering algorithm.
4.  **NormalPredictor:** A baseline algorithm that predicts a random rating based on the distribution of the training set (assumed to be normal), used for comparison.

## Hyperparameter Tuning

### Search Strategy: RandomizedSearchCV vs GridSearchCV
In the discussion assignment, **RandomizedSearchCV** was used instead of GridSearchCV.
*   **Why?** Recommender systems often have a large hyperparameter space (factors, learning rates, regularization). GridSearch tries every combination, which is computationally expensive and slow.
*   **RandomizedSearchCV** samples a fixed number of parameter settings (`n_iter`) from specified distributions, effectively finding good hyperparameters in a fraction of the time.

### Key Hyperparameters for SVD
The following parameters were tuned for the `SVD` algorithm:

*   **`n_factors`:** The number of latent factors (dimensions) to represent users and items. Higher values capture more complexity but risk overfitting.
    *   *Range searched:* 50 to 200.
*   **`n_epochs`:** The number of iterations of the SGD procedure.
    *   *Range searched:* 20 to 50.
*   **`lr_all`:** The learning rate for all parameters. Controls step size during optimization.
    *   *Distribution:* Uniform closer to 0 (e.g., 0.002 to 0.015).
*   **`reg_all`:** The regularization term for all parameters. Prevents overfitting by penalizing large coefficients.
    *   *Distribution:* Uniform (e.g., 0.02 to 0.1).

### Key Hyperparameters for KNNBasic
*   **`k`:** The maximum number of neighbors to take into account for aggregation.
    *   *Range searched:* 10 to 50.
*   **`min_k`:** The minimum number of neighbors to take into account for aggregation. If not enough neighbors are found, the neighbor aggregation is not performed.
    *   *Range searched:* 1 to 5.
*   **`sim_options`:** A dictionary of options for the similarity measure.
    *   **`name`:** The name of the similarity metric to use (MSD, Cosine, Pearson).
    *   **`user_based`:** Whether the similarity is computed between users (True) or items (False).

### Key Hyperparameters for NMF (Non-Negative Matrix Factorization)
*   **`n_factors`:** The number of latent factors.
    *   *Range searched:* 15 to 40.
*   **`n_epochs`:** The number of iterations of the SGD procedure.
    *   *Range searched:* 20 to 50.

## Key Formulas

### 1. Matrix Factorization (ALS Update Rule)
For Alternating Least Squares (ALS), the user factors $P$ are updated iteratively to minimize the error between predicted and actual ratings.

$$P_{a,b} := P_{a,b} - \alpha \sum_{j \in R_a}^N e_{a,j}Q_{b,j}$$

*   $P_{a,b}$: User factor matrix value.
*   $\alpha$: Learning rate.
*   $e_{a,j}$: Error term (difference between actual and predicted rating).
*   $Q_{b,j}$: Item factor matrix value.

### 2. Weighted Hybrid Prediction
A weighted hybrid model combines predictions from multiple algorithms (e.g., SVD and SlopeOne) using a linear combination.

$$ \hat{r}_{ui} = \alpha \cdot \hat{r}_{\text{SVD}} + (1-\alpha) \cdot \hat{r}_{\text{SlopeOne}} $$

In the assignment, an equal weight ($\alpha = 0.5$) was used:
$$ \hat{r}_{\text{hybrid}} = 0.5 \cdot \hat{r}_{\text{SVD}} + 0.5 \cdot \hat{r}_{\text{SlopeOne}} $$

### 3. Similarity Measures (KNN)
Memory-based collaborative filtering relies on similarity measures like Cosine Similarity or Pearson Correlation.

**Cosine Similarity:**

$$ \text{sim}(u, v) = \frac{\sum_{i} r_{ui} r_{vi}}{\sqrt{\sum_{i} r_{ui}^2} \sqrt{\sum_{i} r_{vi}^2}} $$

**Pearson Correlation:**

$$ \text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}} $$

**Mean Squared Difference (MSD) Similarity:**

$$ \text{msd}(u, v) = \frac{1}{|I_{uv}|} \sum_{i \in I_{uv}} (r_{ui} - r_{vi})^2 $$

$$ \text{sim}(u, v) = \frac{1}{\text{msd}(u, v) + 1} $$

### 4. Evaluation Metric (RMSE)
Root Mean Squared Error is the standard metric for evaluating rating predictions.

$$ \text{RMSE} = \sqrt{\frac{1}{|\hat{R}|} \sum_{\hat{r}_{ui} \in \hat{R}} (r_{ui} - \hat{r}_{ui})^2} $$

## Types of Hybrid Recommendation

While the module focused on **Weighted Hybrid**, there are several types of hybrid recommendation systems:

1.  **Weighted:** The scores of several recommendation techniques are combined together numerically (e.g., Linear Combination).
2.  **Switching:** The system switches between recommendation techniques depending on the heuristic or criteria (e.g., use Content-based if user profile is new, otherwise CF).
3.  **Mixed:** Recommendations from different referrers are presented together (e.g., "People who bought X also bought Y" next to "Recommended for you").
4.  **Feature Combination:** Features from different data sources are thrown together into a single recommendation algorithm.
5.  **Cascade:** One recommender refines the recommendations given by another.
6.  **Feature Augmentation:** Output from one technique is used as an input feature to another.
7.  **Meta-level:** The model generated by one recommender is used as the input for another.

## Implementation Details

### 1. Manual Collaborative Filtering (ALS)
We manually implemented an ALS approach using Linear Regression.
```python
# Iteratively solving for User and Item coefficients
lr = LinearRegression(fit_intercept=False).fit(X, y)
coefs = lr.coef_
```

### 2. Using the Surprise Library
We used `surpriselib` to streamline the recommendation process.

**Loading Data:**
```python
from surprise import Dataset, Reader
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['userId', 'title', 'rating']], reader)
train = data.build_full_trainset()
```

**SVD Model:**
```python
from surprise import SVD
model = SVD(n_factors=2, random_state=42)
model.fit(train)
test = train.build_testset()
predictions = model.test(test)
```

**SlopeOne Model:**
```python
from surprise import SlopeOne
slope_one = SlopeOne()
slope_one.fit(train)
slope_one_preds = slope_one.test(test)
```

**Hybrid Predictions:**
Combining predictions from SVD and SlopeOne can often yield better results.
```python
hybrid_preds = [0.5 * i.est + 0.5 * j.est for i, j in zip(slope_one_preds, svd_preds)]
```

**Cross Validation:**
Evaluating model performance using RMSE.
```python
from surprise.model_selection import cross_validate
cross_validate(model, data, measures=['RMSE'], cv=5)
```

## Assignment Highlights
*   **Data:** User ratings for artists/albums and MovieLens dataset.
*   **Goal:** Predict missing ratings for users.
*   **Process:**
    *   Implemented manual Matrix Factorization using `LinearRegression`.
    *   Utilized the `Surprise` library for standard algorithms like `SVD` and `SlopeOne`.
    *   Created a hybrid model by averaging predictions from different algorithms.
