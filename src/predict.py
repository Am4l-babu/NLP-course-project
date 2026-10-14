"""Inference: single-text and batch prediction, plus safe model persistence.

Model files and security
------------------------
``joblib`` files are Python pickles; loading an untrusted one can execute
arbitrary code. Therefore:

* the app never loads a model from a user-supplied path or upload - only the two
  fixed file names in ``config.MODELS_DIR``;
* ``model_metadata.json`` stores a SHA-256 of each model file, which is verified
  *before* unpickling, and the unpickled object's type is checked afterwards.

This protects against corruption and accidental file swaps. It cannot protect
against someone who can already write to the ``models/`` folder (they could also
rewrite the metadata) - that needs OS-level access control.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

import config
from src.preprocessing import STATUS_OK, STATUS_TRUNCATED, prepare_text
from src.priority import estimate_priority
from src.validation import validate_unlabeled_df

logger = logging.getLogger("sfc.predict")

METADATA_FORMAT_VERSION = 1


class ModelLoadError(Exception):
    """A saved model is missing, corrupted, tampered with or of the wrong type."""


class PredictionError(Exception):
    """Prediction could not be performed (bad input data)."""


@dataclass
class Prediction:
    """Result for one feedback entry. ``scores`` are raw model probabilities
    (NOT calibrated) and are ``None`` when the selected model has no
    ``predict_proba`` (e.g. Linear SVM)."""

    original_text: str
    status: str  # ok | truncated | empty | invalid
    message: str = ""
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_scores: dict = field(default_factory=dict)
    category: Optional[str] = None
    category_score: Optional[float] = None
    category_scores: dict = field(default_factory=dict)
    priority: Optional[str] = None
    priority_reasons: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (STATUS_OK, STATUS_TRUNCATED)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _proba(pipeline: Pipeline, texts: list):
    """Return (labels, class_order, probability matrix or None)."""
    labels = pipeline.predict(texts)
    if hasattr(pipeline, "predict_proba"):
        return labels, list(pipeline.classes_), pipeline.predict_proba(texts)
    return labels, list(pipeline.classes_), None


class FeedbackClassifier:
    """A pair of fitted pipelines (sentiment, category) plus metadata."""

    def __init__(self, sentiment_pipeline: Pipeline, category_pipeline: Pipeline, metadata: dict = None):
        for p in (sentiment_pipeline, category_pipeline):
            if not isinstance(p, Pipeline) or not hasattr(p, "predict"):
                raise ModelLoadError("Model object is not a fitted scikit-learn Pipeline.")
        self.sentiment_pipeline = sentiment_pipeline
        self.category_pipeline = category_pipeline
        self.metadata = metadata or {}

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_training_result(cls, result) -> "FeedbackClassifier":
        if getattr(result, "status", None) != "success":
            raise ModelLoadError("Refusing to use a training result that did not succeed.")
        return cls(
            result.tasks["sentiment"].pipeline,
            result.tasks["category"].pipeline,
            metadata=result.to_dict(),
        )

    @property
    def model_id(self) -> str:
        return str(self.metadata.get("trained_at", "unknown"))

    @property
    def supports_probabilities(self) -> dict:
        return {
            "sentiment": hasattr(self.sentiment_pipeline, "predict_proba"),
            "category": hasattr(self.category_pipeline, "predict_proba"),
        }

    # -- persistence ------------------------------------------------------- #
    def save(self, models_dir: Path = None) -> Path:
        models_dir = Path(models_dir or config.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        s_path = models_dir / config.SENTIMENT_MODEL_FILE
        c_path = models_dir / config.CATEGORY_MODEL_FILE
        joblib.dump(self.sentiment_pipeline, s_path)
        joblib.dump(self.category_pipeline, c_path)
        meta = dict(self.metadata)
        meta["format_version"] = METADATA_FORMAT_VERSION
        meta["files"] = {
            "sentiment": {"name": s_path.name, "sha256": _sha256(s_path)},
            "category": {"name": c_path.name, "sha256": _sha256(c_path)},
        }
        (models_dir / config.MODEL_METADATA_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        logger.info("Saved models to %s", models_dir)
        return models_dir

    @classmethod
    def load(cls, models_dir: Path = None) -> "FeedbackClassifier":
        models_dir = Path(models_dir or config.MODELS_DIR)
        meta_path = models_dir / config.MODEL_METADATA_FILE
        s_path = models_dir / config.SENTIMENT_MODEL_FILE
        c_path = models_dir / config.CATEGORY_MODEL_FILE
        for p in (meta_path, s_path, c_path):
            if not p.is_file():
                raise ModelLoadError(
                    f"Saved model file '{p.name}' not found. Train a model first "
                    f"(Model Training page, or `python -m src.train`)."
                )
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            expected = {k: meta["files"][k]["sha256"] for k in ("sentiment", "category")}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise ModelLoadError(f"Model metadata is unreadable or incomplete: {exc}") from exc

        if _sha256(s_path) != expected["sentiment"] or _sha256(c_path) != expected["category"]:
            raise ModelLoadError(
                "A saved model file does not match its recorded checksum (corrupted or "
                "modified). It was not loaded. Retrain the models."
            )
        try:
            sentiment = joblib.load(s_path)
            category = joblib.load(c_path)
        except Exception as exc:  # unpickling can fail in many ways
            raise ModelLoadError(
                f"A saved model could not be loaded (possibly saved with a different "
                f"scikit-learn version): {exc}"
            ) from exc
        return cls(sentiment, category, meta)

    def to_joblib_bytes(self, which: str) -> bytes:
        """Serialise one pipeline for download (outgoing only; never re-uploaded)."""
        pipe = {"sentiment": self.sentiment_pipeline, "category": self.category_pipeline}[which]
        buf = io.BytesIO()
        joblib.dump(pipe, buf)
        return buf.getvalue()

    # -- prediction -------------------------------------------------------- #
    def predict_texts(self, texts: list) -> list:
        """Predict a list of raw strings; never raises on bad individual entries."""
        prepared = [prepare_text(t) for t in texts]
        results = [
            Prediction(original_text=t if isinstance(t, str) else ("" if t is None else str(t)),
                       status=p.status, message=p.message)
            for t, p in zip(texts, prepared)
        ]
        usable = [i for i, p in enumerate(prepared) if p.usable]
        if not usable:
            return results

        batch = [prepared[i].text for i in usable]
        try:
            s_labels, s_classes, s_proba = _proba(self.sentiment_pipeline, batch)
            c_labels, c_classes, c_proba = _proba(self.category_pipeline, batch)
        except Exception as exc:
            raise PredictionError(f"The model failed to score the input: {exc}") from exc

        for j, i in enumerate(usable):
            r = results[i]
            r.sentiment = str(s_labels[j])
            r.category = str(c_labels[j])
            if s_proba is not None:
                r.sentiment_scores = {c: float(s_proba[j][k]) for k, c in enumerate(s_classes)}
                r.sentiment_score = float(s_proba[j].max())
            if c_proba is not None:
                r.category_scores = {c: float(c_proba[j][k]) for k, c in enumerate(c_classes)}
                r.category_score = float(c_proba[j].max())
            pr = estimate_priority(prepared[i].text, r.sentiment)
            r.priority = pr.label
            r.priority_reasons = pr.reasons
        logger.info("Classified %d of %d entries", len(usable), len(texts))
        return results

    def predict_one(self, text) -> Prediction:
        return self.predict_texts([text])[0]

    def predict_batch(self, df: pd.DataFrame, text_column: str = config.TEXT_COLUMN):
        """Classify every row of ``df``.

        Returns ``(result_df, summary)``. ``result_df`` has all original columns in
        their original order and row order, followed by ``config.PREDICTION_COLUMNS``.
        Rows with unusable text are kept and marked ``skipped: ...``.
        """
        check = validate_unlabeled_df(df, text_column)
        if not check.ok:
            raise PredictionError(" ".join(check.errors))

        preds = self.predict_texts(df[text_column].tolist())
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        out = df.copy()
        out["predicted_sentiment"] = [p.sentiment or "" for p in preds]
        out["predicted_category"] = [p.category or "" for p in preds]
        out["predicted_priority"] = [p.priority or "" for p in preds]
        out["sentiment_confidence"] = [
            round(p.sentiment_score, 4) if p.sentiment_score is not None else np.nan for p in preds
        ]
        out["category_confidence"] = [
            round(p.category_score, 4) if p.category_score is not None else np.nan for p in preds
        ]
        out["classification_timestamp"] = [stamp if p.ok else "" for p in preds]
        out["classification_status"] = [
            ("skipped: " + p.message) if not p.ok else ("ok; " + p.message if p.message else "ok")
            for p in preds
        ]
        n_ok = sum(p.ok for p in preds)
        summary = {
            "uploaded": len(df),
            "classified": n_ok,
            "skipped": len(df) - n_ok,
            "truncated": sum(p.status == STATUS_TRUNCATED for p in preds),
            "text_column": text_column,
            "timestamp": stamp,
            "warnings": check.warnings,
        }
        return out, summary


# --------------------------------------------------------------------------- #
# Command line:  python -m src.predict --input in.csv --output out.csv
# --------------------------------------------------------------------------- #
def main(argv: Optional[list] = None) -> int:
    from src.data_loader import DataLoadError, read_csv_bytes, to_csv_bytes

    parser = argparse.ArgumentParser(description="Classify a CSV of unlabelled feedback.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-column", default=config.TEXT_COLUMN)
    args = parser.parse_args(argv)
    try:
        src = Path(args.input)
        df = read_csv_bytes(src.read_bytes(), src.name)
        clf = FeedbackClassifier.load()
        out, summary = clf.predict_batch(df, args.text_column)
    except (DataLoadError, ModelLoadError, PredictionError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    Path(args.output).write_bytes(to_csv_bytes(out))
    print(f"Classified {summary['classified']} of {summary['uploaded']} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
