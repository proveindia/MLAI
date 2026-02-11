# Module 13: Logistic Regression

## Overview
This module introduced Logistic Regression, a critical algorithm for classification tasks, and focused on evaluating classification models using various metrics.

## Key Concepts
*   **Logistic Regression:** A statistical model used for binary classification. It models the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick.
*   **Sigmoid Function:** The activation function used in logistic regression to map predictions to probabilities between 0 and 1.
*   **Decision Boundary:** The threshold (usually 0.5) used to classify the probability output into a discrete class.
*   **Evaluation Metrics:**
    *   **Accuracy:** The ratio of correctly predicted observations to the total observations.
    *   **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. High precision relates to the low false positive rate.
    *   **Recall (Sensitivity):** The ratio of correctly predicted positive observations to the all observations in actual class - yes.
    *   **Confusion Matrix:** A table used to describe the performance of a classification model.

## Assignment Highlights
*   **Dataset:** Penguins dataset.
*   **Goal:** Classify penguin species based on features like flipper length.
*   **Process:** 
    *   Fitted a Logistic Regression model.
    *   **ROC Curve (Receiver Operating Characteristic):** A plot of the True Positive Rate vs. False Positive Rate at different classification thresholds. The **AUC (Area Under the Curve)** represents the overall performance (1.0 is perfect, 0.5 is random guessing).
    *   Visualized the sigmoid curve and probabilities.
    *   Evaluated the model using accuracy, precision, and recall to understand the trade-offs between different metrics.

## Key Formulas

### 1. Sigmoid Function
Maps any real-valued number into the range [0, 1], interpreted as probability:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
where $z = \beta_0 + \beta_1 x_1 + ... + \beta_n x_n$.

### 2. Log Loss (Binary Cross-Entropy)
The cost function minimized in Logistic Regression:

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})] $$

## Implementation Details

### 1. Basic Logistic Regression
We implemented a basic Logistic Regression model to predict binary outcomes.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Fit the model
logreg = LogisticRegression().fit(X_train, y_train)

# Predict and Evaluate
preds = logreg.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, preds):.2f}')
print(f'Precision: {precision_score(y_test, preds):.2f}')
print(f'Recall: {recall_score(y_test, preds):.2f}')
```

### 2. Advanced Pipeline with Class Balancing
We used a `Pipeline` with `ColumnTransformer` for preprocessing and handled class imbalance using `class_weight='balanced'`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Preprocessing Pipelines
numerical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# Pipeline with Balanced Logistic Regression
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])

logistic_pipeline.fit(X_train, y_train)
```

### 3. Comprehensive Evaluation
We used `classification_report` to view precision, recall, and F1-score for each class.

```python
from sklearn.metrics import classification_report

y_pred = logistic_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Non-Purchase', 'Purchase'])
print(report)
```
