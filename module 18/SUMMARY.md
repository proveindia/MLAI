# Module 18: Natural Language Processing - Tokenization

## Overview
This module introduced Natural Language Processing (NLP) techniques, specifically focusing on tokenization using the `nltk` library.

## Key Concepts
*   **Tokenization:** The process of breaking down text into smaller units (tokens).
    *   **Word Tokenization:** Splitting text into individual words.
    *   **Sentence Tokenization:** Splitting text into sentences.
*   **NLTK (Natural Language Toolkit):** A leading Python library for working with human language data.
*   **Lexical Diversity:** The ratio of unique words to the total number of words in a text. A measure of vocabulary richness.
*   **Bag of Words (implied context):** Representing text as a collection of its words, disregarding grammar and word order but keeping multiplicity.

## Assignment Highlights
*   **Data:** Excerpt from Isaac Newton's *Principia* and WhatsApp status dataset.
*   **Goal:** Tokenize text and analyze basic statistics.
*   **Process:**
    *   Used `word_tokenize` and `sent_tokenize`.
    *   Calculated the number of unique words using `set()`.
    *   Computed lexical diversity.
    *   Applied tokenization to a Pandas DataFrame column to analyze a corpus of text.

## Implementation Details

### Tokenization
Tokenization is performed using `nltk.word_tokenize` and `nltk.sent_tokenize`.

```python
from nltk import word_tokenize, sent_tokenize

# Word Tokenization
ans1 = word_tokenize(principia)

# Sentence Tokenization
ans2 = sent_tokenize(principia)
```

### Vocabulary Analysis
We can calculate the vocabulary size (unique tokens) and lexical diversity.

```python
# Unique words
ans3 = set(word_tokenize(principia))

# Lexical Diversity
ans5 = len(set(word_tokenize(principia)))/len(word_tokenize(principia))
```

### Tokenization on DataFrame
Applying tokenization to a DataFrame column.

```python
ans6 = len(set(happy_df['content'].apply(word_tokenize).sum()))
```

### Stemming
Using `PorterStemmer` to reduce words to their root form.

```python
def stemmer(text):
    stem = PorterStemmer()
    return ' '.join([stem.stem(w) for w in word_tokenize(text)])
```
