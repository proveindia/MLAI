# Module 8: Feature Engineering Summary

This module focuses on transforming raw data into features that Machine Learning algorithms can understand, particularly categorical and text data.

## ⏱️ Quick Review (20 Mins)

### 1. Ordinal Encoding
Used for categorical data with an **inherent order** (e.g., "Low", "Medium", "High").

- Preserves the ranking information.
- Converts categories to integers (0, 1, 2...).

```python
from sklearn.preprocessing import OrdinalEncoder

# Sample data
education = [['High School'], ['Bachelor'], ['Master'], ['PhD']]
enc = OrdinalEncoder(categories=[['High School', 'Bachelor', 'Master', 'PhD']])

encoded = enc.fit_transform(education)
# Result: [[0], [1], [2], [3]]
```

### 2. One-Hot Encoding
Used for categorical data **without** inherent order (e.g., "Red", "Blue", "Green").

- Creates a new binary column for each category.
- Avoids implying a false mathematical relationship (e.g., Red < Blue).

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sentences split into words
words = pd.DataFrame(['AI', 'Machine', 'Learning'], columns=['word'])

# Encode
oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = oh_encoder.fit_transform(words[['word']])

# Create DataFrame with new columns
encoded_df = pd.DataFrame(encoded_array, columns=oh_encoder.get_feature_names_out())
```

### 3. Text Processing
Preparing text for proper encoding.
- **Tokenization**: Splitting sentences into words.
- **Cleaning**: Removing punctuation, lowercase conversion.

```python
sentence = "I love AI!"
tokens = sentence.lower().split()
# ['i', 'love', 'ai!']
```

---
*Reference: Mod8_Feat_Engineering.ipynb*
