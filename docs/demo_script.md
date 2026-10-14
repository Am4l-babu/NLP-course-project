# Demonstration script (about 10 minutes)

**Before you start:** activate the virtual environment, run `streamlit run app.py`, open http://localhost:8501 in a 1280-px-wide window, and have `samples/sample_input.csv` ready. Optional: run `python -m src.train` first so the app loads a saved model.

| Min | Say | Do |
|---|---|---|
| 0:00 | "Colleges receive far more feedback than anyone can read. My system turns free text into sentiment, topic and a priority hint, so staff can find what matters first. It supports people; it never decides about people." | Show the **Dashboard**. |
| 0:45 | "This data is **synthetic demo data** - 300 sentences I wrote for testing, not real students' feedback. And these are model predictions on data it was trained on, so they look better than reality - the banner says so." | Point at the two notes; show KPI cards, sentiment-by-topic chart. Open **Filters**, filter to Negative + Infrastructure. |
| 2:00 | "Single feedback. The model has never seen this sentence." | **Analyze Feedback**: type *The classroom projector is not working.* -> Analyze. Show sentiment badge (text + colour), topic, priority "Medium" and the *Why this priority?* reasons. |
| 3:00 | "Now a safety complaint." | Enter *The staircase near the lab is slippery and someone could get injured. Please fix this urgently.* -> priority **High**; "a person must verify High items". Note the probability bars say *uncalibrated*. |
| 4:00 | "And a case where it fails - sarcasm." | Enter *Great. Another cancelled class. Exactly what I wanted.* -> Positive. "A bag-of-words model has no sense of irony; that's why every screen carries the limitations note." |
| 5:00 | "Whole files at once." | **Bulk Classification**: upload `sample_input.csv` -> validation warns that 2 rows are unusable -> **Classify feedback**. Show *24 uploaded / 22 classified / 2 skipped*, filter the table, click **Download classified CSV**, open it: original columns first, `predicted_*` columns appended, skipped rows kept. |
| 6:30 | "How was the model built?" | **Model Training** -> expand *How it works* (TF-IDF, three algorithms). Click **Train models**. Walk through: stratified 225/75 split, 5-fold CV, selection by macro-F1, comparison chart, confusion matrix. |
| 8:00 | "Be honest about numbers: about 0.80 macro-F1 for sentiment and 0.68 for topic on 75 test rows from synthetic data. That proves the pipeline works, not that it will be 80 % accurate on real feedback." | Show *Incorrect test predictions*, top terms (note the caveat: association is not causation). |
| 9:00 | "Safeguards." | Briefly: upload validation errors (upload a CSV missing `sentiment` on the training page to show the red messages and disabled button); no personal data needed; models only loaded from `models/` with checksum; optional admin passcode. |
| 9:30 | "Next steps: real labelled data, calibrated probabilities, aspect-based sentiment, Malayalam-English support." | **Feedback Explorer** (term charts labelled *exploratory*) as a closing view. |

## Backup plan
* If the app does not start: `python -m pytest -q` (shows all 102 tests passing) and `python -m src.train` in the terminal.
* If training is slow on the demo laptop: skip the *Train models* click and show the saved results in `docs/evaluation_results.md`.
