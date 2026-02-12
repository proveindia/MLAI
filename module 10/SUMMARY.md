# Module 10: Time Series Analysis

## Overview
This module explores **Time Series Analysis (TSA)**, the process of analyzing data points collected at regular time intervals to uncover underlying structures and predict future values. TSA is fundamental in finance, demand forecasting, and econometrics.

## Key Concepts

### 1. Stochastic Process vs. Time Series
*   **Stochastic Process:** A mathematical abstraction defining a sequence of random variables indexed by time. It represents *all possible* futures (the "multiverse" of outcomes).
*   **Time Series:** A single *realization* (sample path) of a stochastic process. The one history that actually happened.
*   **Goal:** We use the single observed Time Series to infer the properties (parameters) of the underlying Stochastic Process.

### 2. Stationarity
A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time.
*   **Why it matters:** Most classical models (like ARIMA) assume stationarity to make reliable predictions.
*   **Non-Stationary behaviors:** Trends (upward/downward movement), Seasonality (repeating cycles), and changing Variance (heteroscedasticity).

![Stationarity Comparison](images/stationary_vs_nonstationary.png)

*   **Fixes:**
    *   **Log Transformation:** Stabilizes variance (straightens out exponential growth).
    *   **Differencing:** Removes trends by subtracting the previous value ($y_t - y_{t-1}$).

### 3. Stationarity vs. Independence (Crucial Distinction)
*   **Stationarity:** The *rules* (statistical properties) don't change over time, but the data points are highly **dependent** on each other (today depends on yesterday).
*   **Independence:** Data points are completely unrelated (like flipping a coin).
*   **Goal of TSA:** We model the *dependency* (signal) so that what's left over (residuals) is *independent* (noise).
    *   *Raw Data:* Stationary but Dependent.
    *   *Residuals:* Stationary and Independent (White Noise).

### 4. Time Series Decomposition
Breaking a series into its constituent parts to understand it better:

![Time Series Decomposition](images/ts_decomposition_visual.png)

*   **Additive Model:** $y_t = \text{Trend} + \text{Seasonality} + \text{Residue}$
*   **Multiplicative Model:** $y_t = \text{Trend} \times \text{Seasonality} \times \text{Residue}$
    *   **Trend**: Long-term increase or decrease in the data.
    *   **Seasonality**: Repeating short-term cycle.
    *   **Residue (Noise)**: The random variation left over.

### 5. AR vs MA Models
*   **AR (AutoRegressive):** "Regressing on itself." Current value depends on its own past values. Like momentum.
    *   *Analogy:* If it was hot yesterday, it will likely be hot today.
*   **MA (Moving Average):** "Regressing on past errors." Current value depends on past forecast shocks/errors. Like a correction mechanism.
    *   *Analogy:* If I made a huge mistake yesterday, I will adjust today to compensate.

### 6. Autocorrelation (ACF & PACF)
*   **ACF (Autocorrelation Function):** Correlation between $y_t$ and its lags ($y_{t-1}, y_{t-2}, ...$). Shows direct and indirect effects.
*   **PACF (Partial Autocorrelation Function):** Direct correlation between $y_t$ and a lag, removing the influence of intermediate lags. Crucial for determining AR terms.

![ACF and PACF Concept](images/acf_pacf_concept.png)

## Key Formulas

### 1. Differencing (for Stationarity)
Used to remove trends.

$$ y'_t = y_t - y_{t-1} $$

### 2. AR (Autoregressive) Model
Predicting the current value based on past values.

$$ y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t $$

*   **$p$** (Pronounced: *p*): The order of the AR term (number of lags).
*   **$\phi$** (Pronounced: *phi*): Coefficients for past values (lags).
*   **$\epsilon_t$** (Pronounced: *epsilon sub t*): White noise error term.

### 3. MA (Moving Average) Model
Predicting the current value based on past forecast errors (shocks).

$$ y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q} $$

*   **$q$** (Pronounced: *q*): The order of the MA term.
*   **$\theta$** (Pronounced: *theta*): Coefficients for past error terms.

### 4. ARIMA Model
Combines AR, Integration (Differencing), and MA. Notation: **ARIMA(p, d, q)**.
*   **$p$**: AutoRegressive order (Lags).
*   **$d$**: Degree of Differencing (to make stationary).
*   **$q$**: Moving Average order (Errors).

### 5. SARIMA (Seasonal ARIMA)
When seasonality exists (e.g., repeating cycles every 12 months), plain ARIMA struggles. SARIMA adds seasonal parameters.

**Notation:** $ARIMA(p, d, q) \times (P, D, Q)_s$
*   **$(p, d, q)$**: Non-seasonal orders.
*   **$(P, D, Q)$**: Seasonal orders (applied to seasonal lags).
*   **$s$**: Seasonality period (e.g., 12 for monthly, 4 for quarterly).

**Key parameters:**
*   **$P$**: Seasonal Autoregressive order.
*   **$D$**: Seasonal Differencing order (e.g., $y_t - y_{t-12}$).
*   **$D$**: Seasonal Differencing order (e.g., $y_t - y_{t-12}$).
*   **$Q$**: Seasonal Moving Average order.

### 6. SARIMAX (Exogenous Regressors)
Extends SARIMA by adding external variables ($X$) that might influence the target.
*   **$X$**: Exogenous variables (e.g., Temperature, Holiday, Ad Spend).
*   **Use Case:** When past values of $y$ aren't enough, and external factors drive the trend.
*   **Usage:** In `statsmodels`, pass the external data to the `exog` parameter.

### 7. Error Metrics (Evaluation)
*   **MAE (Mean Absolute Error):** Average magnitude of errors.

$$ \text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i| $$

*   **RMSE (Root Mean Square Error):** Penalizes large errors heavily.

$$ \text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2} $$

## Code for Learning

### Setup and Import
```bash
pip install pandas numpy matplotlib statsmodels scipy
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
```

### 1. Visualizing Decomposition
Understanding the underlying structure of the data.

```python
# Create synthetic data
dates = pd.date_range(start='2020-01-01', periods=100, freq='M')
data = np.arange(100) + np.sin(np.arange(100) * 2 * np.pi / 12) * 10 + np.random.normal(0, 2, 100)
df = pd.Series(data, index=dates)

# Decompose
result = seasonal_decompose(df, model='additive')

# Plot
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(result.observed, label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(result.trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(result.resid, label='Residual')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
```

### 2. Testing for Stationarity (ADF Test)
The **Augmented Dickey-Fuller** test is the standard statistical test for stationarity.
*   **Null Hypothesis ($H_0$):** Series is Non-Stationary (has a unit root).
*   **Alternate Hypothesis ($H_1$):** Series is Stationary.
*   **Rule:** If p-value < 0.05, Reject $H_0$ (It is Stationary).

```python
def check_stationarity(series):
    result = adfuller(series.dropna())
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critial Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
        
    if result[1] < 0.05:
        print("✅ Result: Series is Stationary")
    else:
        print("❌ Result: Series is Non-Stationary (Diff needed)")

# Run check on raw data
check_stationarity(df)

# If non-stationary, difference it
df_diff = df.diff().dropna()
check_stationarity(df_diff)
```

### 3. The Forecasting Problem
*   **Goal:** Predict future values ($y_{t+1}, y_{t+2}$) based on historical data ($y_t, y_{t-1}$).
*   **Interpretation & Uncertainty:**
    *   A forecast is never a single number; it's a **probability distribution**.
    *   **Confidence Intervals:** The range where the true value is likely to fall (e.g., 95% probability).
    *   **Fan Chart:** Uncertainty typically grows as we forecast further into the future.

![Forecasting Uncertainty](images/forecasting_uncertainty.png)

### 4. ARMA Model Selection & Invertibility
*   **Stationarity:** Required for AR models (roots of characteristic equation outside unit circle).
*   **Invertibility:** Required for MA models (to express MA as an infinite AR process). Ensures unique model representation.
*   **Order Selection (Rule of Thumb):**
    *   **AR(p):** PACF cuts off at lag $p$, ACF decays gradually.
    *   **MA(q):** ACF cuts off at lag $q$, PACF decays gradually.
    *   **ARMA(p,q):** Both decay gradually (feature of mixed models).

### 5. ARIMA, SARIMA, and SARIMAX Forecasting
Building a model to predict future values.

## Forecasting Workflow

```mermaid
graph TD
    A[Raw Time Series] --> B{Stationary?}
    B -->|No| C[Differencing / Log Transform]
    C --> B
    B -->|Yes| D[Determine p, d, q via ACF/PACF]
    D --> E[Train ARIMA Model]
    E --> F[Check Residuals White Noise]
    F -->|Bad| D
    F -->|Good| G[Generate Forecasts]
```

```python
# Split data
train = df.iloc[:-12] # Train on all except last year
test = df.iloc[-12:]  # Test on last year

# Build Model (Order: p=1, d=1, q=1 is simpler example)
# In practice, use ACF/PACF plots to determine p and q
model = ARIMA(train, order=(1, 1, 1)) 

# For SARIMA (add seasonal_order)
# model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# For SARIMAX (add exog data)
# model = SARIMAX(train, order=(1, 1, 1), exog=train[['temperature', 'holiday']])

model_fit = model.fit()

# Forecast
forecast_result = model_fit.get_forecast(steps=12)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Evaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f'Test RMSE: {rmse:.3f}')

# Plot Results
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Training Data')
plt.plot(test.index, test, label='Actual Values')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ARIMA Forecast vs Actuals')
plt.legend()
plt.show()
```

### 6. Diagnosing Models with ACF/PACF
Helping to choose the right $p$ and $q$ for ARIMA.

```python
# Plot ACF and PACF
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(df.diff().dropna(), ax=ax1) # Use differenced (stationary) data
plot_pacf(df.diff().dropna(), ax=ax2)
plt.show()
```
*   **ACF cuts off after lag q:** Suggests MA(q) process.
*   **PACF cuts off after lag p:** Suggests AR(p) process.

### 7. Real-World Example: Air Passengers (Trend + Seasonality)
This classic dataset shows monthly totals of international airline passengers (1949-1960). It clearly demonstrates **Trend** (increasing travel) and **Seasonality** (summer peaks).

```python
import statsmodels.api as sm

# Load dataset (built-in)
dta = sm.datasets.get_rdataset("AirPassengers").data
dta['time'] = pd.date_range(start='1949-01-01', periods=len(dta), freq='MS')
dta.set_index('time', inplace=True)
ts = dta['value']

# Plot Decomposition
res = seasonal_decompose(ts, model='multiplicative')
res.plot()
plt.show()
```

![Air Passengers Decomposition](images/air_passengers_analysis.png)
*Figure: Decomposition of Air Passengers data showing clear trend and seasonality.*
