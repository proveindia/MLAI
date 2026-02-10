# Capstone Project: Amazon Sales Prediction

## Research Question
Can future sales volumes of Amazon products be accurately predicted using historical data and market trends to proactively optimize inventory levels?

## Objectives
1. **Sales Velocity Prediction (Regression):** Predict the exact number of units sold based on price, marketing spend, and stock levels.
2.  **Demand Classification (Classification):** Classify days as "High Demand" or "Low Demand" to trigger inventory alerts.

## Expected Data Sources
The analysis will leverage a hybrid dataset combining proprietary sales records with public e-commerce benchmarks.
*   **Primary Source:** Internal historical data (Daily Unit Sales, Revenue, Inventory Levels, Pricing).
*   **External Source:** [Kaggle E-Commerce Sales Prediction Dataset](https://www.kaggle.com/datasets/nevildhinoja/e-commerce-sales-prediction-dataset) will be used to model the impact of environmental factors.
*   **Key Features:** based on the Nevil Dhinoja dataset structure, the model will incorporate:
    *   **Transactional:** `Price`, `Discount_Offered`, `Stock_Level`, `Previous_Day_Sales`.
    *   **Contextual:** `Time_of_Day`, `Day_of_Week`, `Week_of_Year`, `Month`, `Season`.
    *   **Environmental:** `Weather` conditions, `Promotion` status, `Holiday_Flag`.

## Techniques
1.  **Time Series Forecasting (ARIMA/SARIMA):** To model seasonality and long-term trends in sales volume.
2.  **Regression Analysis (Random Forest/XGBoost):** To predict specific sales counts based on `Price` and `Ad_Spend`.
3.  **Demand Classification (Logistic Regression/SVC):** Inspired by the Kaggle dataset's binary target, I will also train a model to classify days as "High Demand" vs "Low Demand" to trigger binary inventory alerts (Restock/Hold).
4.  **Evaluation:** Using RMSE for regression and F1-Score for demand classification.

## Expected Results
A **Sales Velocity Predictor** that provides a 30-day forecast of expected unit sales. This output will serve as a direct input for inventory panning, flagging when to reorder stock.

## Why This Question is Important
**If this question remains unanswered, businesses fly blind.**
Without accurate predictions, Amazon sellers face two expensive extremes:
1.  **Running Out of Stock:** If you can't predict a sales spike, you sell out. On Amazon, this doesn't just mean lost revenue today; it kills your algorithmic ranking, meaning you lose future sales even after you restock.
2.  **Hoarding Inventory:** If you overestimate demand, you tie up cash in unmovable products and pay growing storage fees to Amazon, which eats directly into profit margins.

**Benefit of Analysis:**
This project translates raw data into a "Capital Efficiency Engine." It allows business owners to move from "gut-feeling" ordering to precision supply chain management, ensuring every dollar spent on inventory generates maximum return.
