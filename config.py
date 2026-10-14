"""Central configuration for the Smart Feedback Classification System.

Everything that a course-project user might reasonably want to tweak (labels,
limits, random seed, colours) lives here so the rest of the code has no magic
constants.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
SAMPLES_DIR = BASE_DIR / "samples"
DEMO_DATA_PATH = DATA_DIR / "demo_feedback.csv"

# Fixed file names. The app only ever loads models from these exact names inside
# MODELS_DIR - it never loads a user-supplied path (see README, "Privacy and security").
SENTIMENT_MODEL_FILE = "sentiment_model.joblib"
CATEGORY_MODEL_FILE = "category_model.joblib"
MODEL_METADATA_FILE = "model_metadata.json"

# --------------------------------------------------------------------------- #
# Labels (configurable)
# --------------------------------------------------------------------------- #
SENTIMENTS = ["Positive", "Neutral", "Negative"]
CATEGORIES = [
    "Academics",
    "Faculty",
    "Infrastructure",
    "Laboratory",
    "Administration",
    "Library",
    "Canteen",
    "Internet and Wi-Fi",
    "Transport",
    "Other",
]
PRIORITIES = ["High", "Medium", "Low"]

# Sentiment colours are paired with text labels everywhere so that meaning never
# depends on colour alone.
SENTIMENT_COLORS = {
    "Positive": "#2A9D8F",
    "Neutral": "#8D99AE",
    "Negative": "#E76F51",
}
PRIORITY_COLORS = {"High": "#C1121F", "Medium": "#F4A261", "Low": "#8D99AE"}

# --------------------------------------------------------------------------- #
# Dataset schema
# --------------------------------------------------------------------------- #
TEXT_COLUMN = "feedback_text"
SENTIMENT_COLUMN = "sentiment"
CATEGORY_COLUMN = "category"
ID_COLUMN = "feedback_id"
DATE_COLUMN = "created_at"
REQUIRED_TRAINING_COLUMNS = [TEXT_COLUMN, SENTIMENT_COLUMN, CATEGORY_COLUMN]

# Columns appended by the classifier. Input files must not already contain them.
PREDICTION_COLUMNS = [
    "predicted_sentiment",
    "predicted_category",
    "predicted_priority",
    "sentiment_confidence",
    "category_confidence",
    "classification_timestamp",
    "classification_status",
]

# --------------------------------------------------------------------------- #
# Limits (input validation)
# --------------------------------------------------------------------------- #
MAX_UPLOAD_MB = 5
MAX_UPLOAD_ROWS = 50_000
MAX_TEXT_CHARS = 2_000
ALLOWED_UPLOAD_EXTENSIONS = (".csv",)
MIN_TOTAL_TRAINING_ROWS = 30
MIN_EXAMPLES_PER_CLASS = 5

# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
TFIDF_NGRAM_RANGE = (1, 2)
SELECTION_METRIC = "macro F1-score (mean of cross-validation on the training split)"

# Raw model scores below this value are flagged "low score - review manually".
LOW_CONFIDENCE_THRESHOLD = 0.50

# --------------------------------------------------------------------------- #
# Access control for the training page
# --------------------------------------------------------------------------- #
# If this environment variable is set, the Model Training page asks for it
# (checked on the server with a constant-time comparison). If it is not set the
# page runs in open "local course project" mode and says so on screen.
ADMIN_PASSCODE_ENV = "SFC_ADMIN_PASSCODE"
