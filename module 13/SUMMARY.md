# Module 13: Logistic Regression

## Overview
This module introduced Logistic Regression, a fundamental algorithm for classification tasks. Unlike linear regression which predicts continuous values, logistic regression predicts probabilities and is primarily used for binary and multiclass classification problems.

## Key Concepts
*   **Logistic Regression:** A statistical model that uses the logistic (sigmoid) function to model the probability of a binary outcome (e.g., pass/fail, yes/no, 0/1).
*   **Sigmoid Function:** The activation function that maps any real-valued input to a probability between 0 and 1.
*   **Decision Boundary:** The threshold (typically 0.5) used to convert predicted probabilities into discrete class predictions.
*   **Log Loss (Binary Cross-Entropy):** The cost function minimized during training to find optimal parameters.
*   **Odds and Odds Ratio:** Alternative ways to interpret logistic regression coefficients.
*   **Regularization:** Techniques (L1, L2) to prevent overfitting by penalizing large coefficients.
*   **Multiclass Classification:** Extending binary logistic regression to handle multiple classes using softmax (Multinomial), One-vs-Rest (OVR), or One-vs-One (OVO).
*   **Maximum Likelihood Estimation (MLE):** The statistical framework used to derive the Log Loss cost function.
*   **Cleanly Separable Data:** If classes are perfectly separable, MLE causes coefficients to grow to infinity (overfitting). Regularization (L2) prevents this.

### Multiclass Strategies

how do we adapt a binary classifier like Logistic Regression for multiple classes (e.g., Red, Blue, Green)?

**1. One-vs-Rest (OVR) / One-vs-All:**
*   **Strategy:** "Us vs The World".
*   **Mechanism:** Train $K$ separate classifiers.
    *   Classifier 1: Red vs [Blue + Green]
    *   Classifier 2: Blue vs [Red + Green]
    *   ...
*   **Prediction:** The classifier with the highest confidence score wins.
*   **Pros:** Efficient for large datasets, clear interpretation.
*   **Cons:** Class imbalance (1 vs K-1), ambiguous regions (where multiple classifiers claim the point).

**2. One-vs-One (OVO):**
*   **Strategy:** "Round-Robin Tournament".
*   **Mechanism:** Train a classifier for **every pair** of classes. Total classifiers = $\frac{K(K-1)}{2}$.
    *   Classifier 1: Red vs Blue (Ignore Green)
    *   Classifier 2: Red vs Green (Ignore Blue)
    *   Classifier 3: Blue vs Green (Ignore Red)
*   **Prediction:** Each classifier casts a vote. The class with the most votes wins.
*   **Pros:** Less sensitive to imbalance, generally accurate.
*   **Cons:** Computationally expensive for large $K$ (too many pairs).

**3. Multinomial (Softmax):**
*   **Strategy:** Direct probability estimation.
*   **Mechanism:** Optimizes a single cost function (Cross-Entropy) for all classes simultaneously using the Softmax function.
*   **Pros:** Theoretically most sound, calibrated probabilities.

![One-vs-One vs One-vs-Rest](images/ovo_vs_ovr.png)
*Figure: conceptual difference between OVR (Linear boundaries separating one class from rest) and OVO (Boundaries separating specific pairs).*

## Key Formulas

### 1. Sigmoid Function (Activation)
The sigmoid function maps any real-valued number to the range [0, 1].

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

*   **$\sigma(z)$** (Pronounced: *sigma of z*): Probability $P(y=1|x)$.
*   **$z$** (Pronounced: *z*): Log-odds (linear combination).
*   **$e$** (Pronounced: *e*): Euler's number ($\approx 2.718$).

![Sigmoid Function](images/sigmoid_visualization.png)
*Figure 1: The sigmoid function curve.*

### 2. Logistic Regression Model
Probability of positive class:

$$ P(y=1|x; \beta) = \sigma(\beta^T x) = \frac{1}{1 + e^{-(\beta_0 + \sum \beta_i x_i)}} $$

### 3. Odds and Odds Ratio
Odds represent the ratio of success to failure probability.

$$ \text{Odds} = \frac{P(y=1)}{1-P(y=1)} = e^z $$

*   **Interpretation:** $e^{\beta_i}$ is the multiplicative change in odds for a one-unit increase in $x_i$.
*   **Log-Odds (Logit):** $\ln(\text{Odds}) = z = \beta^T x$. This is the linear part of the model!

![Odds vs Probability](images/odds_visualization.png)
*Figure 2: Relationship between Probability, Odds, and Log-Odds.*

### 4. Log Loss (Cost Function)
We minimize the negative log-likelihood:

$$ J(\beta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right] $$

*   **$J(\beta)$** (Pronounced: *J of beta*): Cost function.
*   **$\log$**: Natural logarithm.

![Log Loss Function](images/log_loss_concept.png)
*Figure 3: The Log Loss function penalizes wrong predictions exponentially.*

### 5. Gradient (for Optimization)
Used in Gradient Descent. Same form as Linear Regression!

$$ \nabla J(\beta) = \frac{1}{m} X^T (\hat{y} - y) $$

![Cost Function Optimization](images/betasopt.png)
*Figure 4: Gradient descent finding the minimum of the cost function.*

### 6. Regularized Cost Functions
Penalties to prevent overfitting.

*   **Ridge (L2):** $J(\beta) + \lambda \sum \beta_j^2$
*   **Lasso (L1):** $J(\beta) + \lambda \sum |\beta_j|$

![L1 Regularization Path](images/coefl1.png)
*Figure 5: Lasso (L1) drives coefficients to zero, performing feature selection.*

### 7. Softmax (Multinomial Logistic Regression)
Generalization for $K$ classes:

$$ P(y=k|x) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} $$

### 8. Evaluation Metrics

**Confusion Matrix:**

```mermaid
graph TD
    subgraph Predicted Positive
    TP[True Positive<br>(Correct)]
    FP[False Positive<br>(Type I Error)]
    end
    subgraph Predicted Negative
    FN[False Negative<br>(Type II Error)]
    TN[True Negative<br>(Correct)]
    end
    Actual_Pos-->TP
    Actual_Pos-->FN
    Actual_Neg-->FP
    Actual_Neg-->TN
    style TP fill:#9f9,stroke:#333,stroke-width:2px
    style TN fill:#9f9,stroke:#333,stroke-width:2px
    style FP fill:#f99,stroke:#333,stroke-width:2px
    style FN fill:#f99,stroke:#333,stroke-width:2px
```

**Accuracy:** Overall correctness.
$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

**Precision:** Quality of positive predictions.
$$ \text{Precision} = \frac{TP}{TP + FP} $$

**Recall:** Quantity of positives found.
$$ \text{Recall} = \frac{TP}{TP + FN} $$

**F1 Score:** Harmonic mean.
$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

**ROC Curve & AUC:**
*   **TPR (Recall):** $TP/(TP+FN)$
*   **FPR:** $FP/(TN+FP)$
*   **AUC:** Area Under ROC Curve (1.0 = perfect, 0.5 = random).

![Decision Boundary](images/dboundary.png)
*Figure 6: Decision Boundary.*

![ROC Curve](images/p3.png)
*Figure 7: Performance metrics.*

### 9. Example Dataset: Breast Cancer Diagnosis
A classic binary classification problem.
*   **Goal:** Predict if a tumor is Malignant (0) or Benign (1).
*   **Features:** Radius, Texture, Smoothness, etc. (computed from cell images).
*   **Logic:** Logistic Regression estimates the probability $P(y=Benign|x)$. If $P > 0.5$, classify as Benign. The plot below shows two features where a linear decision boundary could separate the classes (though not perfectly).

![Breast Cancer Visualization](images/breast_cancer_viz.png)
*Figure 8: Scatter plot of Breast Cancer data (Radius vs Texture). Malignant tumors tend to be larger and more textured.*

## Business Applications

Logistic Regression is the workhorse of industrial classification:
*   **Credit Scoring:** Predicting default risk (Prob > Threshold -> Deny).
*   **Churn Prediction:** Identifying at-risk customers for retention campaigns.
*   **Ad Click-Through Rate (CTR):** Predicting probability of a click for ad ranking.
*   **Medical Diagnosis:** Estimating disease risk probabilities.
*   **Fraud Detection:** Flagging suspicious transactions.

## Hyperparameters

Logistic Regression in scikit-learn has several important hyperparameters:

### Regularization
*   **`penalty`**: Type of regularization to apply.
    *   Options: `'l1'`, `'l2'` (default), `'elasticnet'`, `None`
    *   **L2 (Ridge)**: Penalizes sum of squared coefficients, shrinks coefficients smoothly
    *   **L1 (Lasso)**: Penalizes sum of absolute values, can zero out coefficients (feature selection)
    *   **ElasticNet**: Combination of L1 and L2
    
*   **`C`**: Inverse of regularization strength (must be positive).
    *   Default: `1.0`
    *   *Smaller values* → stronger regularization → simpler model
    *   *Larger values* → weaker regularization → more complex model
    *   *Effect:* Controls the tradeoff between fitting training data and keeping coefficients small

![L1 Regularization Coefficients](images/coefl1.png)
*Figure 8: Effect of L1 regularization on coefficient values, showing how some coefficients become exactly zero.*

### Optimization
*   **`solver`**: Algorithm to use for optimization.
    *   Options: `'lbfgs'` (default), `'liblinear'`, `'newton-cg'`, `'sag'`, `'saga'`
    *   **lbfgs**: Good for small to medium datasets, supports L2 and None
    *   **liblinear**: Good for small datasets, supports L1 and L2
    *   **sag/saga**: Fast for large datasets, `saga` supports all penalties
    *   *Note:* Some solvers don't support all penalty types
    
*   **`max_iter`**: Maximum number of iterations for convergence.
    *   Default: `100`
    *   *Effect:* Increase if model doesn't converge

### Class Imbalance
*   **`class_weight`**: Weights associated with classes.
    *   Options: `None` (default), `'balanced'`, or dictionary
    *   **'balanced'**: Automatically adjusts weights inversely proportional to class frequencies: $\frac{n\_samples}{n\_classes \times n\_samples\_for\_class}$
    *   *Effect:* Helps model perform better on imbalanced datasets

### Multiclass Strategy
*   **`multi_class`**: Strategy for multiclass classification.
    *   Options: `'auto'`, `'ovr'`, `'multinomial'`
    *   **'ovr'** (One-vs-Rest): Trains one binary classifier per class
    *   **'multinomial'**: Uses softmax function for true multinomial loss
    *   *Effect:* Multinomial is generally better but requires solvers like 'lbfgs'

## Code for Learning

This section provides essential code for building, evaluating, and tuning Logistic Regression models.

### Setup and Imports

**Installation:**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

**Imports:**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, 
                             roc_curve, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

### Product Category Classification (PriceRunner Dataset)

In this module, we handle a dataset with **high-cardinality categorical features** (e.g., Product Titles, Merchant IDs). Standard One-Hot Encoding would create too many features, so we use advanced techniques.

#### 1. The Challenge: High Cardinality
A feature like "Product Name" might have thousands of unique values.
- **One-Hot Encoding:** Creates thousands of sparse columns (inefficient).
- **Target Encoding (James-Stein):** Replaces the category with a weighted average of the target variable for that category, shrinking it towards the global mean to prevent overfitting.

#### 2. Advanced Encoding Pipeline
We use `category_encoders` combined with `sklearn` pipelines.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import JamesSteinEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

# Feature Selection
# 'num' and 'cat' would be lists of column names
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', JamesSteinEncoder(), categorical_cols)
    ]
)

# Pipeline: Preprocess -> Select Features -> Classify
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear'))),
    ('classifier', LogisticRegression())
])

# Fit on training data
pipeline.fit(X_train, y_train)

# Evaluate
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.3f}")
```

#### 3. Why James-Stein?
It balances the **category's mean** with the **global mean**.
- If a category is reliable (many data points), use its own mean.
- If a category is rare (few data points), pull it towards the global average.
This is defined as:
$$ \hat{\mu}_i = (1 - B) y_i + B \bar{y} $$
Where $B$ is the shrinking factor.



![Logistic Regression Visualization](images/lr.png)
*Figure 9: Logistic regression model showing the sigmoid curve fitted to binary classification data.*

### 2. Visualizing the Sigmoid Curve

Plot the sigmoid function and predicted probabilities.

```python
# For single feature visualization
X_single = penguins_binary[['flipper_length_mm']]
logreg_single = LogisticRegression(random_state=42)
logreg_single.fit(X_single, y)

# Create range of values
x_range = np.linspace(X_single.min(), X_single.max(), 300)
# Predict probabilities
probs = logreg_single.predict_proba(x_range.reshape(-1, 1))[:, 1]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_single, y, alpha=0.5, label='Actual')
plt.plot(x_range, probs, color='red', linewidth=2, label='Sigmoid Curve')
plt.axhline(y=0.5, color='green', linestyle='--', label='Decision Boundary (0.5)')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Probability of Gentoo')
plt.title('Logistic Regression Sigmoid Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 3. Confusion Matrix and Classification Report

Comprehensive evaluation using confusion matrix.

```python
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Adelie', 'Gentoo'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Detailed classification report
print(classification_report(y_test, y_pred, target_names=['Adelie', 'Gentoo']))
```

![Threshold Analysis](images/thresh.png)
*Figure 10: Analysis of different classification thresholds showing precision-recall tradeoff.*

### 4. ROC Curve and AUC

Visualize model performance across different thresholds.

```python
# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'Logistic Regression (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"AUC Score: {auc:.3f}")
```

![Performance Comparison](images/p3.png)
*Figure 11: Model performance metrics comparison across different configurations.*

### 5. Threshold Tuning

Find optimal threshold for specific use case.

```python
# Calculate metrics for different thresholds
thresholds_to_test = np.linspace(0.1, 0.9, 50)
precision_scores = []
recall_scores = []
f1_scores = []

for threshold in thresholds_to_test:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    precision_scores.append(precision_score(y_test, y_pred_thresh))
    recall_scores.append(recall_score(y_test, y_pred_thresh))
    f1_scores.append(f1_score(y_test, y_pred_thresh))

# Plot threshold analysis
plt.figure(figsize=(10, 6))
plt.plot(thresholds_to_test, precision_scores, label='Precision', linewidth=2)
plt.plot(thresholds_to_test, recall_scores, label='Recall', linewidth=2)
plt.plot(thresholds_to_test, f1_scores, label='F1 Score', linewidth=2)
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Find threshold that maximizes F1
best_threshold = thresholds_to_test[np.argmax(f1_scores)]
print(f"Best Threshold for F1: {best_threshold:.3f}")
```

### 6. Regularization (L1 vs L2)

Compare different regularization techniques.

```python
from sklearn.preprocessing import StandardScaler

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with different regularization
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_scores = []
l2_scores = []

for C in C_values:
    # L1 regularization
    logreg_l1 = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
    logreg_l1.fit(X_train_scaled, y_train)
    l1_scores.append(logreg_l1.score(X_test_scaled, y_test))
    
    # L2 regularization
    logreg_l2 = LogisticRegression(penalty='l2', C=C, solver='lbfgs', random_state=42)
    logreg_l2.fit(X_train_scaled, y_train)
    l2_scores.append(logreg_l2.score(X_test_scaled, y_test))

# Plot comparison
plt.figure(figsize=(10, 6))
plt.semilogx(C_values, l1_scores, marker='o', label='L1 (Lasso)', linewidth=2)
plt.semilogx(C_values, l2_scores, marker='s', label='L2 (Ridge)', linewidth=2)
plt.xlabel('C (Inverse Regularization Strength)')
plt.ylabel('Accuracy')
plt.title('Regularization Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

![Beta Optimization](images/betasopt.png)
*Figure 12: Optimization path showing how coefficients converge during training.*

### 7. Multiclass Classification

Extend to multiple classes using multinomial logistic regression.

```python
# Use all penguin species
X_multi = penguins[['flipper_length_mm', 'bill_length_mm', 'bill_depth_mm']]
y_multi = penguins['species']

# Split data
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Train multinomial logistic regression
logreg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                                   max_iter=1000, random_state=42)
logreg_multi.fit(X_train_m, y_train_m)

# Predict
y_pred_m = logreg_multi.predict(X_test_m)

# Evaluate
print(f"Multiclass Accuracy: {accuracy_score(y_test_m, y_pred_m):.3f}")
print("\nClassification Report:")
print(classification_report(y_test_m, y_pred_m))

# Confusion matrix
cm_multi = confusion_matrix(y_test_m, y_pred_m)
disp_multi = ConfusionMatrixDisplay(confusion_matrix=cm_multi, 
                                     display_labels=logreg_multi.classes_)
disp_multi.plot(cmap='Blues')
plt.title('Multiclass Confusion Matrix')
plt.show()
```

![Feature Distribution](images/flipperdist.png)
*Figure 13: Distribution of flipper length feature across different penguin species.*

### 8. Handling Class Imbalance

Use class weights to handle imbalanced datasets.

```python
# Create imbalanced dataset (example)
# Simulate by undersampling one class
majority_class = y_train[y_train == 0]
minority_class = y_train[y_train == 1][:len(majority_class)//5]  # 1:5 ratio

y_imbalanced = pd.concat([majority_class, minority_class])
X_imbalanced = X_train.loc[y_imbalanced.index]

print(f"Class distribution: {y_imbalanced.value_counts().to_dict()}")

# Train without balancing
logreg_unbalanced = LogisticRegression(random_state=42)
logreg_unbalanced.fit(X_imbalanced, y_imbalanced)

# Train with balanced class weights
logreg_balanced = LogisticRegression(class_weight='balanced', random_state=42)
logreg_balanced.fit(X_imbalanced, y_imbalanced)

# Compare on test set
print("\nWithout Balancing:")
print(classification_report(y_test, logreg_unbalanced.predict(X_test)))

print("\nWith Balanced Weights:")
print(classification_report(y_test, logreg_balanced.predict(X_test)))
```

### 9. Pipeline with Preprocessing

Create a complete ML pipeline with preprocessing steps.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Example with mixed data types
penguins_full = sns.load_dataset('penguins')
X_full = penguins_full.drop('species', axis=1)
y_full = penguins_full['species']

# Define feature types
numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
categorical_features = ['island', 'sex']

# Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Create full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                     max_iter=1000, random_state=42))
])

# Train and evaluate
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

full_pipeline.fit(X_train_full, y_train_full)
y_pred_full = full_pipeline.predict(X_test_full)

print(f"Pipeline Accuracy: {accuracy_score(y_test_full, y_pred_full):.3f}")
print("\nClassification Report:")
print(classification_report(y_test_full, y_pred_full))
```

### 10. Hyperparameter Tuning with GridSearchCV

Find optimal hyperparameters using grid search.

```python
# Define parameter grid
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'saga'],
    'classifier__max_iter': [500, 1000]
}

# Grid search
grid_search = GridSearchCV(
    full_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_full, y_train_full)

# Best parameters
print("Best Parameters:", grid_search.best_params_)
print(f"Best CV Score: {grid_search.best_score_:.3f}")

# Test set performance
y_pred_best = grid_search.best_estimator_.predict(X_test_full)
print(f"Test Accuracy: {accuracy_score(y_test_full, y_pred_best):.3f}")
```

## Assignment Highlights
*   **Dataset:** Penguins dataset for binary and multiclass classification.
*   **Goal:** Classify penguin species based on physical measurements.
*   **Process:**
    *   Fitted Logistic Regression models for binary classification (Adelie vs Gentoo).
    *   Visualized the sigmoid curve to understand probability mapping.
    *   Evaluated models using accuracy, precision, recall, and F1 score.
    *   Created confusion matrices to analyze classification errors.
    *   Plotted ROC curves and calculated AUC scores.
    *   Experimented with different classification thresholds.
    *   Applied regularization (L1, L2) to prevent overfitting.
    *   Extended to multiclass classification using softmax.
    *   Built complete pipelines with preprocessing.
    *   Handled class imbalance using `class_weight='balanced'`.

## Advantages and Limitations

**Advantages:**
- Probabilistic output (not just class labels)
- Well-calibrated probabilities
- Interpretable coefficients (odds ratios)
- Efficient training and prediction
- Works well with linearly separable data
- Built-in regularization options
- Extends naturally to multiclass problems

**Limitations:**
- Assumes linear relationship between features and log-odds
- Sensitive to outliers (can be mitigated with regularization)
- Requires feature scaling for optimal performance
- May underperform with highly non-linear decision boundaries
- Can struggle with highly imbalanced datasets without adjustment
- Prone to overfitting with many features relative to samples
