"""Transparent, rule-based priority indicator (High / Medium / Low).

This is **not** a trained model and **not** a validated emergency detector.
There is no labelled priority data in the project, so a learned model would be
fabricated. Instead a small, readable scoring scheme combines:

* the predicted sentiment,
* "severe" cues (safety, health, harassment ...),
* urgency words ("urgent", "immediately" ...),
* persistence cues ("still", "again", "for weeks" ...),
* functional-failure cues ("not working", "broken", "leaking" ...).

Every contribution is returned as a human-readable reason so reviewers can see
why an item was flagged. High-priority items always need human verification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.preprocessing import normalize_text

SEVERE_RE = re.compile(
    r"\b(unsafe|not safe|danger\w*|hazard\w*|fire|electric shock|short circuit|"
    r"injur\w*|accident\w*|harass\w*|ragging|bully\w*|bullied|threat\w*|abus\w*|"
    r"assault\w*|discriminat\w*|theft|stolen|food poisoning|sick|vomit\w*|"
    r"fainted|hospital|suicid\w*|self harm|strangers?|rash\w*|insect|toxic)\b"
)
URGENT_RE = re.compile(
    r"\b(urgent\w*|immediate\w*|asap|emergency|critical\w*|right away|at once|serious\w*)\b"
)
PERSISTENT_RE = re.compile(
    r"\b(still|again|repeated\w*|every day|daily|for (?:days|weeks|months)|"
    r"weeks|months|no response|never respond\w*|never repl\w*|nobody|no one|"
    r"ignor\w*|not resolved|unresolved|pending|complain\w*)\b"
)
FAILURE_RE = re.compile(
    r"\b(not working|does not work|do not work|did not work|broken|out of order|"
    r"leak\w*|not available|unavailable|crash\w*|is down|are down|failed|failing|"
    r"dirty|unhygienic|missing|stuck|torn)\b"
)
_NEGATORS = {"no", "not", "never", "without", "zero", "nor"}


@dataclass(frozen=True)
class PriorityResult:
    label: str
    score: int
    reasons: list = field(default_factory=list)


def _matches(pattern: re.Pattern, text: str, guard_negation: bool) -> list:
    """Return cue strings found in ``text``; optionally skip negated ones
    (e.g. "no safety issues", "never any harassment")."""
    found = []
    for m in pattern.finditer(text):
        if guard_negation:
            before = text[: m.start()].split()[-2:]
            # "not safe" / "not available" already contain the negation themselves
            if any(tok in _NEGATORS for tok in before) and not m.group(0).startswith(
                "not "
            ):
                continue
        found.append(m.group(0))
    return found


def estimate_priority(text: str, sentiment: str) -> PriorityResult:
    """Score one feedback entry. ``sentiment`` is the predicted label."""
    norm = normalize_text(text)
    score = 0
    reasons: list = []

    if sentiment == "Negative":
        score += 1
        reasons.append("negative sentiment (+1)")
    elif sentiment == "Positive":
        reasons.append("positive sentiment (reduces the score)")

    severe = _matches(SEVERE_RE, norm, guard_negation=True)
    if severe:
        weight = {"Negative": 3, "Neutral": 2}.get(sentiment, 1)
        score += weight
        reasons.append(f"safety / health / conduct cue: '{severe[0]}' (+{weight})")

    urgent = _matches(URGENT_RE, norm, guard_negation=True)
    if urgent:
        score += 2
        reasons.append(f"urgency wording: '{urgent[0]}' (+2)")

    persistent = _matches(PERSISTENT_RE, norm, guard_negation=False)
    if persistent:
        score += 1
        reasons.append(f"repeated / unresolved problem cue: '{persistent[0]}' (+1)")

    failure = _matches(FAILURE_RE, norm, guard_negation=False)
    if failure:
        score += 1
        reasons.append(f"something not working / unavailable: '{failure[0]}' (+1)")

    if sentiment == "Positive" and not severe and not urgent:
        score = max(0, score - 1)

    label = "High" if score >= 3 else "Medium" if score >= 1 else "Low"
    if not reasons or reasons == ["positive sentiment (reduces the score)"]:
        reasons = ["no priority cues found"]
    return PriorityResult(label=label, score=score, reasons=reasons)
