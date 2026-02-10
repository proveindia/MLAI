# Term Deposit Subscription Prediction (CRISP-DM Analysis)

## Project Overview
This project applies the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology to analyze a Bank Marketing dataset. The primary business objective is to predict whether a client will subscribe to a term deposit (variable `y`).

The analysis identifies key customer demographics, temporal trends, and economic indicators that drive successful subscriptions, enabling the bank to optimize its marketing resources.

## Repository Contents
- **`TermDepositSubscription.ipynb`**: The main Jupyter Notebook containing:
  - Data Understanding & Cleaning
  - Exploratory Data Analysis (EDA)
  - Feature Engineering & Encoding
  - Model Building (Logistic Regression, KNN, SVM, Decision Tree)
  - Hyperparameter Tuning & Evaluation
- **`data/bank-additional-full.csv`**: The dataset used for analysis (sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)).

## Key Insights & Findings

### 1. Demographic Factors
- **Job Role**: **Students** and **Retired** individuals show the highest subscription rates (approx. 30% vs average 11%). This indicates distinct life stages with higher propensity to save.
- **Age**: The relationship is **U-shaped**. Young adults (<25 years) and seniors (>60 years) are significantly more likely to subscribe compared to middle-aged working adults (30-50).
- **Marital Status**: Single individuals have a slightly higher likelihood of subscribing compared to married or divorced clients.

### 2. Temporal Factors
- **Seasonality**: There is a strong seasonal trend. **March, September, October, and December** typically show conversion rates >40%, whereas May (the highest volume month) has the lowest conversion rate.
- **Recommendations**: Marketing campaigns should be strategically timed during these high-conversion months rather than the high-volume/low-success periods like May.

### 3. Economic Context
- **Macroeconomics**: Success is strongly correlated with economic indicators like **Employment Variation Rate** and **Euribor 3-month Rate**. Lower interest rates and employment variation generally correlate with higher subscription probability.

### 4. Key Predictors
- **Call Duration**: The duration of the last contact is the single strongest predictor of success (longer calls = higher success), though it is a lagging indicator.

## Methodology (CRISP-DM)
1.  **Business Understanding**: Objective is to increase efficiency of bank marketing campaigns.
2.  **Data Understanding**: Analyzed 41,188 records with 20 input features (demographic, history, economic).
3.  **Data Preparation**: Handled missing values (`unknown`), encoded categorical variables (One-Hot), and scaled numerical features.
4.  **Modeling**: Compared four classifiers:
    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Decision Tree
    - SGD Classifier
5.  **Evaluation**: Models were evaluated on Accuracy, Precision, Recall, and ROC-AUC.

## Model Performance (Hyperparameter Tuned)
| Model | Test Accuracy | Best Parameters | Key Observations |
| :--- | :--- | :--- | :--- |
| **Decision Tree** | **91.62%** | Max Depth: 5 | Top performer after tuning. Limiting depth prevented overfitting. |
| **Logistic Regression** | 91.20% | C: 1 | Strong baseline; fast training time (~1.2s). |
| **SVM** | 91.20% | Kernel: rbf, C: 1 | Matches Logistic Regression accuracy but much slower (~23.4s). |
| **SGD Classifier** | 91.03% | Max Iter: 2000 | Good efficiency, slightly lower accuracy than LogReg. |
| **KNN** | 90.13% | Neighbors: 7 | Lowest tuned accuracy; computationally equivalent to fit time but slow inference. |

## Recommendations / Next Steps
1.  **Targeted Campaigns**: focus on the "Student" and "Retired" demographics.
2.  **Seasonal Strategy**: Shift budget from May to the Fall (Sept/Oct) and Winter (Dec/Mar) months.
3.  **Sales Training**: Since call duration is critical, train agents to engage customers in meaningful conversations rather than aiming for short call times.

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
