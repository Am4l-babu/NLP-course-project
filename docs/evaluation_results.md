Dataset: **built-in synthetic demo dataset** - 300 labelled rows. Random seed 42; library versions: scikit-learn 1.9.1, pandas 3.0.6.

Split: 225 training / 75 test rows (test fraction 0.25, stratified by `sentiment+category`).

### Sentiment classification

| Model | CV macro-F1 (train split, mean +/- sd) | Test accuracy | Test precision (macro) | Test recall (macro) | Test macro-F1 |
|---|---|---|---|---|---|
| Logistic Regression **(selected)** | 0.723 +/- 0.063 (5-fold) | 0.800 | 0.801 | 0.801 | 0.800 |
| Multinomial Naive Bayes | 0.679 +/- 0.053 (5-fold) | 0.773 | 0.777 | 0.777 | 0.771 |
| Linear SVM | 0.717 +/- 0.076 (5-fold) | 0.800 | 0.803 | 0.802 | 0.800 |

### Category classification

| Model | CV macro-F1 (train split, mean +/- sd) | Test accuracy | Test precision (macro) | Test recall (macro) | Test macro-F1 |
|---|---|---|---|---|---|
| Logistic Regression **(selected)** | 0.636 +/- 0.086 (5-fold) | 0.680 | 0.750 | 0.675 | 0.678 |
| Multinomial Naive Bayes | 0.550 +/- 0.034 (5-fold) | 0.587 | 0.622 | 0.584 | 0.587 |
| Linear SVM | 0.618 +/- 0.046 (5-fold) | 0.720 | 0.769 | 0.716 | 0.711 |

