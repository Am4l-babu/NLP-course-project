"""Text validation and normalisation.

Two layers are kept deliberately separate:

* ``prepare_text`` - *input hygiene*: reject empty / symbol-only input, strip
  control characters, cap the length. Returns the cleaned (but still human
  readable) text plus a status the UI can show.
* ``normalize_text`` - *model normalisation*: the function plugged into the
  TF-IDF vectoriser as its ``preprocessor``. Because it lives inside the
  scikit-learn ``Pipeline``, training and prediction are guaranteed to use the
  exact same transformation.

Design choices worth knowing for a viva:

* Negations ("not", "no", "never") are **kept**. Standard stop-word lists remove
  them, which would turn "not working" into "working".
* Contractions are expanded ("isn't" -> "is not") so negation survives.
* "!" and "?" are kept as separate tokens - they carry a little sentiment signal.
* Emails, URLs and phone numbers are removed before vectorisation, so personal
  contact details never become model features.
* Characters of Indic scripts (including Malayalam) are preserved rather than
  stripped, although the model can only learn them if the training data
  contains them.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config

# --------------------------------------------------------------------------- #
# PII scrubbing
# --------------------------------------------------------------------------- #
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_RE = re.compile(r"(?<![\w])\+?\d[\d\s().-]{7,}\d(?![\w])")


def _phone_sub(replacement: str):
    def _sub(match: re.Match) -> str:
        digits = sum(ch.isdigit() for ch in match.group(0))
        return replacement if digits >= 9 else match.group(0)

    return _sub


def mask_pii(text: str, replacement: str = None) -> str:
    """Replace e-mail addresses, URLs and phone-like numbers.

    With ``replacement=None`` readable tags (``[email]`` ...) are used - this is
    what the UI shows. ``normalize_text`` passes a space instead.
    """
    if not isinstance(text, str):
        return ""
    url_r = replacement if replacement is not None else "[url]"
    email_r = replacement if replacement is not None else "[email]"
    phone_r = replacement if replacement is not None else "[phone]"
    text = _URL_RE.sub(url_r, text)
    text = _EMAIL_RE.sub(email_r, text)
    text = _PHONE_RE.sub(_phone_sub(phone_r), text)
    return text


# --------------------------------------------------------------------------- #
# Normalisation used inside the scikit-learn pipelines
# --------------------------------------------------------------------------- #
_SPECIAL_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bcannot\b"), "can not"),
    (re.compile(r"\b(it|that|there|here|what|he|she|who)'s\b"), r"\1 is"),
    (re.compile(r"n't\b"), " not"),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ve\b"), " have"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]
_WIFI_RE = re.compile(r"\bwi[\s-]?fi\b")
# Keep word characters, whitespace, ! ? and combining marks / Indic scripts
# (U+0900-U+0DFF covers Devanagari ... Malayalam ... Sinhala). Everything else
# (punctuation, emoji, symbols) becomes a space. "_" is a \w char, so drop it.
_STRIP_RE = re.compile(r"[^\w\s!?̀-ͯऀ-෿]|_")
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1+")
_PUNCT_SPACE_RE = re.compile(r"([!?])")
_ELONGATION_RE = re.compile(r"([^\W\d_])\1{2,}")
_WS_RE = re.compile(r"\s+")


def normalize_text(text) -> str:
    """Lower-case, scrub PII, expand contractions, tidy punctuation.

    Never raises: non-string input (``None``, ``NaN``, numbers) becomes ``""``
    so a single bad cell cannot crash a batch.
    """
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = mask_pii(t, replacement=" ")
    t = t.lower()
    t = _WIFI_RE.sub("wifi", t)  # spelling variants of one domain term -> one token
    for pattern, repl in _SPECIAL_CONTRACTIONS:
        t = pattern.sub(repl, t)
    t = _STRIP_RE.sub(" ", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1", t)
    t = _PUNCT_SPACE_RE.sub(r" \1 ", t)
    t = _ELONGATION_RE.sub(r"\1\1", t)  # "sooooo" -> "soo"
    return _WS_RE.sub(" ", t).strip()


# --------------------------------------------------------------------------- #
# Input validation (what the user typed / what a CSV cell contains)
# --------------------------------------------------------------------------- #
STATUS_OK = "ok"
STATUS_TRUNCATED = "truncated"
STATUS_EMPTY = "empty"
STATUS_INVALID = "invalid"


@dataclass(frozen=True)
class PreparedText:
    text: str
    status: str
    message: str = ""

    @property
    def usable(self) -> bool:
        return self.status in (STATUS_OK, STATUS_TRUNCATED)


def prepare_text(raw, max_chars: int = config.MAX_TEXT_CHARS) -> PreparedText:
    """Validate and clean one feedback entry. Never raises."""
    if raw is None:
        return PreparedText("", STATUS_EMPTY, "Feedback is empty.")
    if not isinstance(raw, str):
        try:
            if raw != raw:  # NaN
                return PreparedText("", STATUS_EMPTY, "Feedback is empty.")
        except Exception:  # pragma: no cover - exotic objects
            pass
        raw = str(raw)

    # Drop control / invisible format characters but keep ordinary whitespace.
    cleaned = "".join(
        ch
        for ch in raw
        if ch in "\n\t " or unicodedata.category(ch) not in ("Cc", "Cf", "Cs")
    )
    cleaned = _WS_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return PreparedText("", STATUS_EMPTY, "Feedback is empty.")
    if not any(ch.isalnum() for ch in cleaned):
        return PreparedText(
            "", STATUS_INVALID, "Feedback contains no letters or digits."
        )
    if not normalize_text(cleaned):
        return PreparedText(
            "", STATUS_INVALID, "Feedback has no usable text after cleaning."
        )
    if len(cleaned) > max_chars:
        return PreparedText(
            cleaned[:max_chars],
            STATUS_TRUNCATED,
            f"Feedback was longer than {max_chars:,} characters and was truncated.",
        )
    return PreparedText(cleaned, STATUS_OK)


# --------------------------------------------------------------------------- #
# Stop words for the *exploratory* term charts (never used by the classifier)
# --------------------------------------------------------------------------- #
NEGATION_WORDS = frozenset(
    {"not", "no", "nor", "never", "none", "nothing", "nobody", "cannot", "neither"}
)
STOP_WORDS = sorted(
    (set(ENGLISH_STOP_WORDS) - NEGATION_WORDS)
    | {"also", "us", "get", "got", "would", "could", "should"}
)
