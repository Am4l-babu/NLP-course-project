# Likely viva questions and accurate answers

Answers describe *this* implementation. Numbers come from `docs/evaluation_results.md`; re-run training to refresh them.

**1. What does your system do?**
It classifies each feedback text by sentiment (Positive/Neutral/Negative) and topic (10 categories), estimates a High/Medium/Low priority with transparent rules, and presents counts, charts and searchable tables in a Streamlit app. It supports staff; it does not make decisions about people.

**2. What is TF-IDF?**
Term Frequency x Inverse Document Frequency. TF is how often a term occurs in one feedback; IDF = log(N / number of feedbacks containing the term) (scikit-learn uses a smoothed version). Words common to every feedback get low weight; words characteristic of a few feedbacks get high weight. Each feedback becomes a sparse vector over the vocabulary. I use word unigrams and bigrams with sublinear TF (1 + log tf).

**3. Why bigrams?**
They capture short phrases such as "not working" or "too slow" whose meaning differs from the separate words.

**4. Why didn't you remove stop words?**
Standard lists contain "not", "no", "never", which carry sentiment. Removing them would make "not working" look like "working". The vectoriser's IDF already down-weights very common words. Stop words are removed only for the exploratory word-frequency charts, with negations kept.

**5. Which algorithms did you compare and which won?**
Logistic Regression, Multinomial Naive Bayes and Linear SVM. By cross-validated macro-F1 on the training split, Logistic Regression won for both sentiment (0.723) and topic (0.636) on the demo data.

**6. How do Naive Bayes, Logistic Regression and SVM work?**
*NB:* applies Bayes' rule assuming terms are independent given the class; multiplies class prior by per-term likelihoods. *Logistic Regression:* learns a weight per term per class; softmax of the weighted sums gives probabilities; trained by minimising regularised log-loss. *Linear SVM:* finds the separating hyperplane with the largest margin; outputs a distance, not a probability.

**7. Why macro-F1 and not accuracy?**
Macro-F1 averages the F1 of each class with equal weight, so a small class cannot be hidden by a large one when classes are imbalanced. Accuracy can look high while a rare class is ignored.

**8. Explain precision, recall and F1.**
Precision = TP/(TP+FP): of items predicted as the class, how many were right. Recall = TP/(TP+FN): of the true items, how many were found. F1 is their harmonic mean. "Macro" averages over classes.

**9. How did you prevent data leakage?**
The TF-IDF vectoriser is a step inside the scikit-learn Pipeline, so its vocabulary and IDF are learned only from the training portion - in the final fit and again inside every cross-validation fold. The test set is used neither for fitting nor for model selection. Exact duplicates are removed before splitting, and the same text with conflicting labels is rejected.

**10. Why stratified splitting and a fixed seed?**
Stratification keeps class proportions the same in train and test (here by sentiment x category). The fixed seed (42) makes results reproducible.

**11. Is 80 % accuracy good?**
Not as a claim about real feedback. The test set is 75 synthetic, tidy sentences, so the estimate is noisy (one sentence = 1.3 points) and optimistic about real data. The result shows that the pipeline works; real-world quality needs a larger, representative, carefully labelled dataset.

**12. What is a confusion matrix?**
A table of true class vs predicted class. The diagonal holds correct predictions; off-diagonal cells show which classes are confused (e.g. other topics being mistaken for Infrastructure).

**13. Are your confidence scores reliable?**
They are raw `predict_proba` values and are **not calibrated** - 0.8 does not mean 80 % chance of being right. Calibration (e.g. `CalibratedClassifierCV`) was not implemented or tested, so the UI says "model probability (uncalibrated)". Linear SVM gives none at all.

**14. How is priority computed? Is it machine learning?**
No. There is no labelled priority data, so a trained model would be fabricated. It is a transparent score: negative sentiment +1, safety/health/conduct cue +3 (Negative) / +2 (Neutral) / +1 (Positive), urgency wording +2, repeated/unresolved +1, "not working/broken" +1. High >= 3, Medium 1-2, Low 0. The reasons are displayed. It is not a validated emergency detector and High items need human verification.

**15. Does negative sentiment mean urgent?**
No. An ordinary complaint ("the projector is not working") is Medium. A neutral sentence can describe a serious problem, and a positive one can mention a problem, so priority combines text cues with sentiment instead of copying it.

**16. What about sarcasm, mixed sentiment, spelling mistakes and Malayalam-English?**
A bag-of-words model cannot detect sarcasm ("Great, another cancelled class" came out Positive in my sample). Mixed sentiment gets a single label. I normalise Wi-Fi/wifi variants and shorten elongated words, but arbitrary typos are unseen words. Malayalam script is preserved by the normaliser, but the model only learns it if the training data contains it; mine does not. Fixes: more data, character n-grams, embeddings, aspect-based models.

**17. How do you handle bad input?**
Empty/whitespace/symbol-only text is rejected with a message; very long text is truncated at 2,000 characters with a notice; control characters are removed; in batch mode unusable rows are kept in the output and marked `skipped`. Uploaded training CSVs are validated (columns, labels, empty text, conflicting and exact duplicates, class counts) and training is blocked with line-numbered errors. Nothing is silently discarded - duplicates removed are reported.

**18. What privacy measures did you implement?**
No personal fields are required; e-mails/URLs/phone numbers are removed before modelling and masked in the explorer; feedback text is not logged or stored in model metadata; uploads stay in memory; telemetry is disabled in `.streamlit/config.toml`; downloads are explicit.

**19. Is loading a joblib file safe?**
Joblib files are pickles and can execute code when loaded. So the app never loads a user-supplied model. It loads only the two fixed files in `models/`, checks their SHA-256 against `model_metadata.json` before unpickling and checks that the object is a scikit-learn Pipeline. This protects against corruption and accidental swapping, not against an attacker with write access to the folder.

**20. Is the training page secure?**
By default no - the course version runs in open local mode and says so. If the `SFC_ADMIN_PASSCODE` environment variable is set, the page requires it and compares it on the server (constant-time), so it is real access control and not just a hidden button. A production system needs proper authentication, roles and HTTPS.

**21. Where does your dataset come from?**
It is synthetic: 300 sentences I wrote (balanced: 10 categories x 3 sentiments x 10), with `DEMO-` IDs and random dates. It is not real student feedback and is flagged as such in the UI and README.

**22. How would you improve the system?**
Collect and label real, anonymised data with agreement checks; calibrate probabilities; multi-label topics; aspect-based sentiment; subword/character features and embeddings for typos and code-mixed text; learn priority from labelled data; add authentication and storage policy.

**23. Why not use BERT or a large language model?**
The brief asks for a lightweight CPU-only system with no paid APIs, and with 300 examples a classical model is easier to train, explain and reproduce. Transformers are listed as a future enhancement.

**24. What is the time/space complexity?**
Training TF-IDF + linear model is roughly linear in the number of documents x average tokens (times iterations for Logistic Regression). Prediction is linear in text length. The demo trains in seconds on a CPU.

**25. How do you test the system?**
102 pytest tests: normalisation and priority rules, CSV and label validation, class-count checks, training/selection/reproducibility, prediction on unseen text, batch order/column preservation, empty batches, CSV export, missing/corrupted/swapped model files, and headless Streamlit tests that render each page and run the upload flows. Run `python -m pytest -q`.
