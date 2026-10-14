"""Metrics, model diagnostics and example selection."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred, labels: list) -> dict:
    """Accuracy, macro precision/recall/F1, per-class report, confusion matrix.

    ``labels`` fixes the class order. ``zero_division=0`` is used so a class that
    is never predicted scores 0 instead of raising a warning.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "n_test": len(y_true),
    }


def top_features(pipeline, n: int = 8) -> dict:
    """Most influential terms per class for a fitted TF-IDF + linear/NB pipeline.

    Diagnostic only: a large weight means "this term co-occurs with this class in
    the training data", NOT that the term causes the feeling or topic.
    """
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    classes = list(clf.classes_)

    if hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        if coef.shape[0] == 1 and len(classes) == 2:  # binary: one row
            coef = np.vstack([-coef[0], coef[0]])
        scores = coef
    elif hasattr(clf, "feature_log_prob_"):
        flp = np.asarray(clf.feature_log_prob_)
        scores = flp - flp.mean(axis=0, keepdims=True)  # how much more likely than average
    else:
        return {}

    out = {}
    for i, cls in enumerate(classes):
        idx = np.argsort(scores[i])[::-1][:n]
        out[cls] = [(str(names[j]), float(scores[i][j])) for j in idx]
    return out


def collect_examples(texts, y_true, y_pred, scores: Optional[np.ndarray], k: int = 6) -> dict:
    """Return DataFrames of correct and incorrect test-set predictions."""
    df = pd.DataFrame(
        {
            "feedback_text": list(texts),
            "true_label": list(y_true),
            "predicted_label": list(y_pred),
            "model_score": list(scores) if scores is not None else [np.nan] * len(y_true),
        }
    )
    df["correct"] = df["true_label"] == df["predicted_label"]
    correct = df[df["correct"]].sort_values("model_score", ascending=False).head(k)
    wrong = df[~df["correct"]].sort_values("model_score", ascending=False).head(k)
    return {
        "correct": correct.drop(columns="correct").reset_index(drop=True),
        "incorrect": wrong.drop(columns="correct").reset_index(drop=True),
        "n_correct": int(df["correct"].sum()),
        "n_incorrect": int((~df["correct"]).sum()),
    }


def training_report_markdown(summary: dict) -> str:
    """Render a ``TrainingResult.to_dict()`` as Markdown (used for docs and README)."""
    lines = []
    ds, sp = summary["dataset"], summary["split"]
    lines.append(f"Dataset: **{ds['name']}** - {ds['n_rows']} labelled rows. "
                 f"Random seed {summary['seed']}; library versions: scikit-learn "
                 f"{summary['versions']['scikit_learn']}, pandas {summary['versions']['pandas']}.\n")
    lines.append(f"Split: {sp['n_train']} training / {sp['n_test']} test rows "
                 f"(test fraction {sp['test_size']}, stratified by `{sp['stratified_by']}`).\n")
    for task_name, task in summary["tasks"].items():
        lines.append(f"### {task_name.title()} classification\n")
        lines.append("| Model | CV macro-F1 (train split, mean +/- sd) | Test accuracy | Test precision (macro) | Test recall (macro) | Test macro-F1 |")
        lines.append("|---|---|---|---|---|---|")
        for c in task["candidates"]:
            m = c["test_metrics"]
            mark = " **(selected)**" if c["name"] == task["selected"] else ""
            lines.append(
                f"| {c['name']}{mark} | {c['cv_f1_macro_mean']:.3f} +/- {c['cv_f1_macro_std']:.3f} "
                f"({c['cv_folds']}-fold) | {m['accuracy']:.3f} | {m['precision_macro']:.3f} | "
                f"{m['recall_macro']:.3f} | {m['f1_macro']:.3f} |"
            )
        lines.append("")
    return "\n".join(lines)
