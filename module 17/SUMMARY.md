# Module 17: Term Deposit Subscription Prediction (CRISP-DM)

## Overview
This module involved a comprehensive project applying the CRISP-DM methodology to predict whether a bank client will subscribe to a term deposit.

## Key Concepts
*   **CRISP-DM Application:** End-to-end process from Business Understanding to Evaluation.
*   **Imbalanced Data:** Handling datasets where one class (non-subscribers) significantly outnumbers the other.
*   **Model Comparison:** Evaluating multiple classifiers (Logistic Regression, KNN, SVM, Decision Tree) to find the best performer.

## Key Findings
*   **Demographics:** Students and Retired individuals have the highest subscription rates.
*   **Age:** U-shaped relationship; young adults and seniors are more likely to subscribe.
*   **Seasonality:** March, September, October, and December are high-conversion months.
*   **Economic Indicators:** Employment variation rate and Euribor rates are strong predictors.
*   **Call Duration:** Long calls are strongly correlated with success (though a lagging indicator).

## Model Performance
*   **Decision Tree:** Achieved the highest accuracy (~91.62%) after hyperparameter tuning (max_depth=5).
*   **Logistic Regression & SVM:** Strong baselines with similar accuracy (~91.20%).
*   **KNN:** Slightly lower accuracy (~90.13%) and computationally expensive for inference.

## Recommendations
*   Target specific demographics (Students, Retired) and time campaigns during high-conversion months.
*   Focus on quality of conversation (duration) rather than just quantity.

## Implementation Details

### Model Comparison Pipeline
The project evaluated multiple classifiers (Logistic Regression, KNN, SVC, Decision Tree) using `GridSearchCV` related to key hyperparameters.

```python
models = {
    'logisticregression': (LogisticRegression(max_iter=1000), {'logisticregression__C': [0.1, 1, 10]}),
    'knn': (KNeighborsClassifier(), {'knn__n_neighbors': [3, 5, 7]}),
    'svc': (SVC(), {'svc__C': [0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}),
    'decisiontreeclassifier': (DecisionTreeClassifier(), {'decisiontreeclassifier__max_depth': [3, 5, 10]}),
}

results = []
for name, (model, param_grid) in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        (name, model)
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
```

### Feature Importance
Feature importance was analyzed using Logistic Regression coefficients.

```python
coeffs = lr_model.named_steps['classifier'].coef_[0]
coeff_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coeffs})
sns.barplot(x='Coefficient', y='Feature', data=coeff_df.head(10))
```

### Decision Tree Visualization
The Decision Tree structure was visualized to understand the decision rules.

```python
decision_tree = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_split=5)
plot_tree(decision_tree, feature_names=X.columns, class_names=['no', 'yes'], filled=True)
```

### Performance Evaluation
Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

```python
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
    disp.plot()
```
