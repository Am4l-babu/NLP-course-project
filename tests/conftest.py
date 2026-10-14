"""Shared fixtures. Models are trained once per test session on the synthetic demo data."""
import pandas as pd
import pytest

from src.data_loader import load_demo_data
from src.predict import FeedbackClassifier
from src.train import train_models


@pytest.fixture(scope="session")
def demo_df():
    return load_demo_data()


@pytest.fixture(scope="session")
def training_result(demo_df):
    return train_models(demo_df, dataset_name="test demo data")


@pytest.fixture(scope="session")
def classifier(training_result):
    return FeedbackClassifier.from_training_result(training_result)


def make_training_frame(n_per_cell: int = 3, categories=("Academics", "Library"), **overrides) -> pd.DataFrame:
    """Small synthetic labelled frame for validation tests."""
    rows = []
    for cat in categories:
        for sent in ("Positive", "Neutral", "Negative"):
            for i in range(n_per_cell):
                rows.append({"feedback_text": f"synthetic {sent.lower()} {cat.lower()} comment number {i}",
                             "sentiment": sent, "category": cat})
    return pd.DataFrame(rows)
