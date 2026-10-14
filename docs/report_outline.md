# Course project report - outline

Suggested length: 25-35 pages. Fill every section with *your own* results (re-run `python -m src.train` and the app); do not copy numbers you did not produce.

1. **Title page, certificate, declaration, acknowledgement**
2. **Abstract** (150-250 words): problem, method (TF-IDF + classical classifiers), result you measured, caveat about the synthetic data.
3. **Introduction**
   1.1 Background: volume of feedback in colleges / services
   1.2 Problem statement
   1.3 Objectives and scope (decision *support*, not decision making)
   1.4 Organisation of the report
4. **Literature / background survey** - text classification, sentiment analysis, TF-IDF, Naive Bayes, Logistic Regression, SVM. Cite sources you actually read; do not invent citations.
5. **System analysis**
   3.1 Existing (manual) system and its drawbacks
   3.2 Proposed system
   3.3 Functional and non-functional requirements (README sections 7-8)
   3.4 Feasibility: technical (CPU only), economic (no paid APIs), operational
6. **System design**
   4.1 Architecture diagram (README Mermaid diagram)
   4.2 Module description (`src/` files)
   4.3 Data design: CSV schemas, label set, validation rules
   4.4 UI design: pages and navigation, colour/label policy
   4.5 Privacy and security design
7. **Implementation**
   5.1 Dataset (synthetic demo set; how it was built; its limits)
   5.2 Preprocessing (normalisation choices, why negations are kept)
   5.3 Feature extraction: TF-IDF with unigrams + bigrams
   5.4 Algorithms and hyper-parameters
   5.5 Training pipeline: stratified split, cross-validation, selection by macro-F1
   5.6 Priority rules
   5.7 Prediction, batch export, model persistence and integrity check
   5.8 Streamlit application (screenshots from your own run)
8. **Testing**
   6.1 Test strategy (unit, integration, headless UI)
   6.2 Test cases table (empty input, invalid CSV, missing columns, invalid labels, class counts, unseen text, corrupted model, export)
   6.3 Results of `python -m pytest -q`
9. **Results and discussion**
   7.1 Metrics tables and confusion matrices (README section 16, regenerated)
   7.2 Error analysis: correct vs incorrect examples
   7.3 Effect of data size and quality (optional experiment: train on 50 % of the demo data and compare)
   7.4 Reliability of probabilities
   7.5 Sarcasm, mixed sentiment, spelling, Malayalam-English code-mixing
10. **Ethics, privacy and responsible use**
11. **Limitations**
12. **Conclusion and future work**
13. **References** (only sources you used)
14. **Appendices**: sample input/output CSV, key code listings, user manual, test log
