# Module 1: Python Fundamentals Summary

This module covers the core building blocks of Python programming, essential for data science and machine learning.

## ⏱️ Quick Review (20 Mins)

### 1. Basic Data Types
Understanding the fundamental types of data in Python.

| Type | Description | Example |
| :--- | :--- | :--- |
| `int` | Integer numbers | `x = 10` |
| `float` | Decimal numbers | `y = 10.5` |
| `str` | Text strings | `s = "Hello"` |
| `bool` | Boolean values | `True`, `False` |

**Type Inspection & Conversion:**
```python
x = 10
print(type(x))  # <class 'int'>

y = float(x)    # Convert int to float -> 10.0
z = str(x)      # Convert int to string -> "10"
```

### 2. Lists & Indexing
Lists are ordered, mutable collections of items.

```python
# Creating a list
temperatures = [20, 22, 21, 19]

# Indexing (0-based)
first = temperatures[0]  # 20
last = temperatures[-1]  # 19

# Mixed types allowed
mixed_list = ["Dec 10", 20, True]
```

### 3. Dictionaries
Key-value pairs for storing data values. Keys must be unique and immutable.

```python
# Creating a dictionary
daily_temps = {
    "Dec 10": 20,
    "Dec 11": 22
}

# Accessing values
print(daily_temps["Dec 10"])  # Output: 20
```

### 4. Control Flow: Loops
Iterating over sequences (lists, strings, ranges).

**For Loop:**
```python
test_scores = [40, 65, 91]

# Iterate over list
for score in test_scores:
    print(score)

# Iterate with range()
for i in range(5):  # 0, 1, 2, 3, 4
    print(i)
```

**Conditional Logic inside Loop:**
```python
# Curve scores only if they are below 80
for score in test_scores:
    if score < 80:
        score += 7
    print(score)
```

### 5. Functions
Reusable blocks of code.

**Built-in Functions:**
- `print()`, `type()`, `len()`, `max()`, `min()`, `abs()`, `round()`

**User-Defined Functions:**
```python
def score_curve(scores):
    """Adds 7 points to scores below 80"""
    updated_scores = []
    for score in scores:
        if score < 80:
            updated_scores.append(score + 7)
        else:
            updated_scores.append(score)
    return updated_scores

new_scores = score_curve([70, 85, 90])
print(new_scores)  # [77, 85, 90]
```

---
*Reference: Module 1 Notebooks (Basic Data Types, Lists, Dictionaries, Loops, Functions)*
