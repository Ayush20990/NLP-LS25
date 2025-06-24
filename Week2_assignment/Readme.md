# Assignment 2.1: Text Vectorization Implementation

## Objective

The goal of this assignment is to manually implement the TF-IDF algorithm and compare the results with scikit-learn's `CountVectorizer` and `TfidfVectorizer`.

---

## Corpus

```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
```

---

## Manual TF-IDF Implementation Summary

1. **Tokenization and Vocabulary Building**
   - Created a set of unique words from the corpus.
   - Calculated the term frequency (TF) for each word in every sentence.
   - Computed the document frequency (DF) for each word.

2. **TF-IDF Calculation**
   - For each word in each document, the TF-IDF value was computed using:
      **TF-IDF(w, d) = TF(w, d) × log(N / DF(w))**
     where \( N \) is the total number of documents.

3. **Libraries Used**
   - `math` for logarithmic computation.

---

## scikit-learn Comparison

### CountVectorizer

- Converts the corpus into a document-term matrix of raw word counts.
- Ignores the importance of words based on rarity across documents.

### TfidfVectorizer

- Converts the corpus into a TF-IDF weighted document-term matrix.
- Downweights frequent/common words like "the", "is", and "a".
- **Important Note:** Scikit-learn uses a slightly different (smoothed) formula for IDF by default:

**IDF(t) = log((1 + n) / (1 + DF(t))) + 1**


  This prevents division by zero and ensures a non-zero weight for all terms. Our manual implementation does not apply this smoothing, so differences are observed in the implementation
---



## Explanation of Differences

- **Stop Words Like 'the'**:
  - Appear in all three sentences.
  - Result: high DF and low IDF which gives low TF-IDF in both manual and sklearn's implementation.
  - `CountVectorizer` gives them high counts because it doesn’t penalize for commonality.

- **Informative Words (e.g., 'celestial', 'star')**:
  - Appear only once.
  - Result: low DF and high IDF which gives higher TF-IDF scores.

- **Differences in TF-IDF Scores**:
  - Due to scikit-learn’s IDF smoothing (`log((1 + N)/(1 + DF)) + 1`), its TF-IDF values differ from our unsmoothed manual version.

---

## Conclusion

The manual implementation of TF-IDF closely aligns with scikit-learn's `TfidfVectorizer` output, with slight expected variations due to smoothing. Understanding how each component (TF, DF, IDF) contributes helps better interpret vectorized data and choose the right model inputs.
