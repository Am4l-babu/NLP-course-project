"""Model training, comparison, selection and evaluation.

Methodology (also documented in the README):

1. Validate the data (``validation.validate_training_df``) - training refuses to
   start on invalid data.
2. One stratified train/test split with a fixed seed. The test set is **never**
   used to choose a model or fit any component (TF-IDF vocabulary included).
3. For each task (sentiment, category) three candidate pipelines
   ``TF-IDF(1-2 grams) -> classifier`` are compared with stratified k-fold
   cross-validation *inside the training split*; the vectoriser is re-fitted in
   every fold, so there is no leakage.
4. The candidate with the best mean CV macro-F1 is selected (ties prefer a model
   that outputs probabilities). Test-set metrics are then reported for every
   candidate, for transparency, but they did not influence the selection.

Run from the command line:  python -m src.train [--data CSV] [--no-save] [--report FILE]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings as _warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import config
from src.evaluation import collect_examples, compute_metrics, top_features
from src.preprocessing import normalize_text
from src.validation import validate_training_df

logger = logging.getLogger("sfc.train")


class TrainingError(Exception):
    """Training could not be completed; the message is safe to show to the user."""


def build_pipeline(estimator) -> Pipeline:
    """TF-IDF (word uni+bigrams) followed by ``estimator``.

    Everything learned from text (vocabulary, IDF weights) is part of the
    pipeline, so it is fitted on training data only and re-used identically at
    prediction time.
    """
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    lowercase=False,  # normalize_text already lower-cases
                    token_pattern=r"\S+",  # normalize_text leaves space-separated tokens
                    ngram_range=config.TFIDF_NGRAM_RANGE,
                    sublinear_tf=True,
                    min_df=1,
                ),
            ),
            ("clf", estimator),
        ]
    )


# Order doubles as the tie-break preference (probability-capable models first).
CANDIDATE_FACTORIES = {
    "Logistic Regression": lambda: LogisticRegression(C=10.0, max_iter=2000, class_weight="balanced"),
    "Multinomial Naive Bayes": lambda: MultinomialNB(alpha=0.5),
    "Linear SVM": lambda: LinearSVC(C=1.0, class_weight="balanced"),
}


@dataclass
class CandidateResult:
    name: str
    cv_folds: int
    cv_f1_macro_mean: float
    cv_f1_macro_std: float
    cv_accuracy_mean: float
    test_metrics: dict
    supports_probabilities: bool
    pipeline: Pipeline = field(repr=False, default=None)


@dataclass
class TaskResult:
    task: str
    classes: list
    candidates: list
    selected_name: str
    examples: dict
    features: dict

    @property
    def selected(self) -> CandidateResult:
        return next(c for c in self.candidates if c.name == self.selected_name)

    @property
    def pipeline(self) -> Pipeline:
        return self.selected.pipeline


@dataclass
class TrainingResult:
    tasks: dict
    dataset: dict
    split: dict
    seed: int
    trained_at: str
    warnings: list
    versions: dict
    status: str = "success"

    def to_dict(self) -> dict:
        """JSON-serialisable summary (no fitted objects, no feedback text)."""
        return {
            "status": self.status,
            "trained_at": self.trained_at,
            "seed": self.seed,
            "versions": self.versions,
            "dataset": self.dataset,
            "split": self.split,
            "warnings": self.warnings,
            "selection_metric": config.SELECTION_METRIC,
            "tasks": {
                name: {
                    "classes": t.classes,
                    "selected": t.selected_name,
                    "candidates": [
                        {
                            "name": c.name,
                            "cv_folds": c.cv_folds,
                            "cv_f1_macro_mean": c.cv_f1_macro_mean,
                            "cv_f1_macro_std": c.cv_f1_macro_std,
                            "cv_accuracy_mean": c.cv_accuracy_mean,
                            "supports_probabilities": c.supports_probabilities,
                            "test_metrics": c.test_metrics,
                        }
                        for c in t.candidates
                    ],
                }
                for name, t in self.tasks.items()
            },
        }


def _make_split(df: pd.DataFrame, seed: int, test_size: float):
    """Stratified split, trying the most specific stratification that the class
    counts allow: sentiment x category, then category, then sentiment, then none."""
    strategies = [
        ("sentiment+category", df[config.SENTIMENT_COLUMN] + "|" + df[config.CATEGORY_COLUMN]),
        ("category", df[config.CATEGORY_COLUMN]),
        ("sentiment", df[config.SENTIMENT_COLUMN]),
        ("none (random)", None),
    ]
    for name, strata in strategies:
        try:
            tr, te = train_test_split(
                df.index, test_size=test_size, random_state=seed, stratify=strata
            )
            return tr, te, name
        except ValueError:
            continue
    raise TrainingError("The dataset is too small to create a train/test split.")


def _run_task(task, labels_order, X_train, y_train, X_test, y_test, seed, notes) -> TaskResult:
    classes = [c for c in labels_order if c in set(y_train)]
    missing_train = [c for c in set(y_test) if c not in set(y_train)]
    if missing_train:
        raise TrainingError(
            f"{task}: class(es) {sorted(missing_train)} appear in the test set but not in the "
            f"training set. Add more examples of these classes."
        )
    absent_test = [c for c in classes if c not in set(y_test)]
    if absent_test:
        notes.append(
            f"{task}: class(es) {absent_test} are absent from the test set, so their "
            f"test metrics are 0 / undefined. Add more examples to evaluate them."
        )

    min_count = int(pd.Series(y_train).value_counts().min())
    folds = min(config.CV_FOLDS, min_count)
    if folds < 2:
        raise TrainingError(
            f"{task}: the smallest class has only {min_count} training example(s); "
            f"cross-validation needs at least 2. Add more examples."
        )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    candidates = []
    for name, factory in CANDIDATE_FACTORIES.items():
        pipe = build_pipeline(factory())
        try:
            cvr = cross_validate(
                pipe, X_train, y_train, cv=cv, error_score="raise",
                scoring={"f1": "f1_macro", "acc": "accuracy"},
            )
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            metrics = compute_metrics(y_test, pred, labels=classes)
        except Exception as exc:  # evaluation failure must never look like success
            raise TrainingError(f"{task}: training/evaluation of '{name}' failed: {exc}") from exc
        candidates.append(
            CandidateResult(
                name=name,
                cv_folds=folds,
                cv_f1_macro_mean=float(np.mean(cvr["test_f1"])),
                cv_f1_macro_std=float(np.std(cvr["test_f1"])),
                cv_accuracy_mean=float(np.mean(cvr["test_acc"])),
                test_metrics=metrics,
                supports_probabilities=hasattr(pipe, "predict_proba"),
                pipeline=pipe,
            )
        )

    # Selection: best CV macro-F1; tie-break by preference order (stable sort).
    order = list(CANDIDATE_FACTORIES)
    best = sorted(candidates, key=lambda c: (-round(c.cv_f1_macro_mean, 6), order.index(c.name)))[0]

    pipe = best.pipeline
    pred = pipe.predict(X_test)
    scores = pipe.predict_proba(X_test).max(axis=1) if best.supports_probabilities else None
    return TaskResult(
        task=task,
        classes=classes,
        candidates=candidates,
        selected_name=best.name,
        examples=collect_examples(X_test, y_test, pred, scores),
        features=top_features(pipe),
    )


def train_models(
    df: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
    test_size: float = config.TEST_SIZE,
    dataset_name: str = "uploaded data",
) -> TrainingResult:
    """Train and evaluate the sentiment and category classifiers."""
    validation = validate_training_df(df)  # defence in depth: never train on unvalidated data
    if not validation.ok:
        raise TrainingError("Data validation failed: " + " ".join(validation.errors))
    data = validation.df.reset_index(drop=True)

    tr_idx, te_idx, strategy = _make_split(data, seed, test_size)
    train, test = data.loc[tr_idx], data.loc[te_idx]
    notes = list(validation.warnings)
    if strategy != "sentiment+category":
        notes.append(f"Split was stratified by '{strategy}' because class counts did not allow finer stratification.")

    tasks = {}
    for task, col, order in (
        ("sentiment", config.SENTIMENT_COLUMN, config.SENTIMENTS),
        ("category", config.CATEGORY_COLUMN, config.CATEGORIES),
    ):
        tasks[task] = _run_task(
            task, order,
            train[config.TEXT_COLUMN].tolist(), train[col].tolist(),
            test[config.TEXT_COLUMN].tolist(), test[col].tolist(),
            seed, notes,
        )

    result = TrainingResult(
        tasks=tasks,
        dataset={
            "name": dataset_name,
            "n_rows": int(len(data)),
            "sentiment_counts": validation.info.get("sentiment_counts", {}),
            "category_counts": validation.info.get("category_counts", {}),
        },
        split={
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_size": test_size,
            "stratified_by": strategy,
        },
        seed=seed,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        warnings=notes,
        versions={"scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    )
    # Counts only - never log the feedback text itself.
    logger.info("Trained models on %d rows (%d train / %d test)", len(data), len(train), len(test))
    return result


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    from src.data_loader import DataLoadError, load_demo_data, read_csv_bytes
    from src.evaluation import training_report_markdown
    from src.predict import FeedbackClassifier

    parser = argparse.ArgumentParser(description="Train the feedback classifiers.")
    parser.add_argument("--data", help="Labelled CSV (default: built-in demo dataset)")
    parser.add_argument("--no-save", action="store_true", help="Do not write models/ files")
    parser.add_argument("--report", help="Write a Markdown evaluation report to this file")
    args = parser.parse_args(argv)

    try:
        if args.data:
            path = Path(args.data)
            raw = read_csv_bytes(path.read_bytes(), path.name)
            name = path.name
            df = raw
        else:
            df = load_demo_data()
            name = "built-in synthetic demo dataset"
        result = train_models(df, dataset_name=name)
    except (DataLoadError, TrainingError, OSError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    summary = result.to_dict()
    report = training_report_markdown(summary)
    print(report)
    for w in result.warnings:
        print("WARNING:", w)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report + "\n", encoding="utf-8")
        print(f"Report written to {args.report}")
    if not args.no_save:
        FeedbackClassifier.from_training_result(result).save()
        print(f"Models saved to {config.MODELS_DIR}")
    return 0


if __name__ == "__main__":
    _warnings.filterwarnings("ignore", category=UserWarning)
    sys.exit(main())
