# Module 10: Time Series Analysis Summary

This module introduces Time Series forecasting, utilizing historical data points indexed in time order to predict future values.

## ⏱️ Quick Review (20 Mins)

### 1. Time Series Basics
Data collected at regular intervals (daily, monthly). 
- **Assumption**: The past contains patterns (Autocorrelation) useful for predicting the future.
- **Horizon**: How many steps into the future to predict.

### 2. Benchmarking Models
Before using complex models (ARIMA, LSTM), establish a baseline using simple statistical methods.

- **Naive**: Predict tomorrow = today.
- **Window Average**: Predict tomorrow = average of last $N$ days.
- **Seasonal Naive**: Predict tomorrow = same day last week/year.

```python
from statsforecast import StatsForecast
from statsforecast.models import Naive, WindowAverage, SeasonalNaive

# Define Models
models = [
    Naive(),
    WindowAverage(window_size=7),
    SeasonalNaive(season_length=7) # Weekly seasonality
]

# Forecast
sf = StatsForecast(models=models, freq='D')
sf.fit(df)
forecasts = sf.predict(h=7) # Predict next 7 days
```

### 3. Data Prep for Time Series
- **Datetime Index**: Ensure the time column is actually datetime type.
- **Regularity**: Data must be evenly spaced (e.g., every day). Missing days must be handled (filled).

```python
df['ds'] = pd.to_datetime(df['ds'])
# Filter short series
df = df.groupby('unique_id').filter(lambda x: len(x) >= 28)
```

---
*Reference: Mod10_Time_Series_simple.ipynb*
