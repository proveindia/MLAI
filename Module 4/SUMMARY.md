# Module 4: Data Analytics Primer Summary

This module introduces essential data cleaning and exploratory data analysis (EDA) techniques using Pandas.

## ⏱️ Quick Review (20 Mins)

### 1. Data Cleaning
Preparing raw data for analysis by handling missing values and inconsistent formatting.

**Cleaning Column Names:**
```python
# Normalize column names (lowercase, remove special chars, replace spaces)
df.columns = (df.columns
              .str.replace(r'[^\w\s-]', '', regex=True)
              .str.replace(r'[\n\s]+', '_', regex=True)
              .str.lower())
```

**Converting Data Types:**
```python
# Convert string numbers with commas to integers
df['population'] = df['population'].str.replace(',', '').astype(int)

# Convert to datetime
df['date'] = pd.to_datetime(df['date'], format="%d %B %Y")
```

### 2. Handling Missing Data
Identifying and treating NaNs.

```python
# Check for missing values
missing_pct = (df.isna().sum() / len(df) * 100).round(2)

# Handling strategies (drop or fill)
df_clean = df.dropna()  # Drop rows with missing values
# OR
df['col'] = df['col'].fillna(df['col'].mean()) # Impute with mean
```

### 3. Summary Statistics
Generating a "status report" of the dataset.

```python
# Custom summary stats function approach
stats = pd.DataFrame()
stats['mean'] = df.mean(numeric_only=True)
stats['null_pct'] = df.isna().mean() * 100
stats['skew'] = df.skew(numeric_only=True)
```

### 4. Outlier Detection
Identifying anomalies using statistical methods.

**Z-Score Method:**
Points lying more than 3 standard deviations from the mean.
```python
# Calculate Z-scores
z_scores = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()

# Filter outliers (keep rows within 3 std devs)
df_no_outliers = df[(z_scores.abs() <= 3).all(axis=1)]
```

---
*Reference: Mod4_Data_Analytics.ipynb*
