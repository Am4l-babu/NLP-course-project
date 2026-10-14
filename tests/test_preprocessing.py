import numpy as np
import pytest

import config
from src.preprocessing import (
    STATUS_EMPTY, STATUS_INVALID, STATUS_OK, STATUS_TRUNCATED,
    mask_pii, normalize_text, prepare_text,
)
from src.priority import estimate_priority


@pytest.mark.parametrize("value", ["", "   ", "\n\t  ", None, np.nan])
def test_empty_or_whitespace_feedback_is_rejected(value):
    result = prepare_text(value)
    assert result.status == STATUS_EMPTY
    assert not result.usable


@pytest.mark.parametrize("value", ["!!!", "...", "\U0001F600\U0001F600", "-- -- --"])
def test_symbol_only_feedback_is_invalid(value):
    assert prepare_text(value).status == STATUS_INVALID


def test_valid_feedback_passes_through():
    r = prepare_text("The Wi-Fi is slow.")
    assert r.status == STATUS_OK and r.text == "The Wi-Fi is slow."


def test_very_long_feedback_is_truncated_not_crashed():
    r = prepare_text("word " * 5000)
    assert r.status == STATUS_TRUNCATED
    assert len(r.text) <= config.MAX_TEXT_CHARS
    assert r.usable


def test_control_characters_are_removed():
    r = prepare_text("bad\x00 text\x07 here")
    assert "\x00" not in r.text and "\x07" not in r.text
    assert r.usable


def test_unsupported_and_non_latin_characters_do_not_crash():
    for text in ["ഫാക്കൽറ്റി നല്ലതാണ്", "良い先生です", "café \U0001F600 ok", "‮evil"]:
        assert isinstance(normalize_text(text), str)
        assert prepare_text(text).status in (STATUS_OK, STATUS_INVALID)


def test_malayalam_script_is_preserved():
    assert "ഫാക്കൽറ്റി" in normalize_text("ഫാക്കൽറ്റി നല്ലതാണ്")


def test_normalize_text_basics():
    assert normalize_text("  The   PROJECTOR,  is NOT working!!! ") == "the projector is not working !"


def test_negation_survives_contraction_expansion():
    assert "not" in normalize_text("The lab isn't working").split()
    assert "will" in normalize_text("It won't start").split()


def test_wifi_spellings_share_one_token():
    assert normalize_text("Wi-Fi") == normalize_text("wifi") == normalize_text("WI FI") == "wifi"


def test_elongation_and_repeated_punctuation_are_tamed():
    assert normalize_text("sooooo good???") == "soo good ?"


def test_normalize_never_raises_on_non_strings():
    assert normalize_text(None) == "" and normalize_text(12) == "" and normalize_text(np.nan) == ""


def test_personal_contact_details_are_removed_before_modelling():
    out = normalize_text("Mail me at jane.doe@example.com or call +91 98765 43210 or see https://x.io/a")
    assert "@" not in out and "98765" not in out and "http" not in out


def test_mask_pii_for_display():
    masked = mask_pii("write to a.b@c.com or 9876543210")
    assert "[email]" in masked and "[phone]" in masked
    assert mask_pii("Room 204 at 4:30") == "Room 204 at 4:30"  # short numbers are not phones


# ---- priority rules ---------------------------------------------------------- #
def test_priority_high_for_safety_issue():
    assert estimate_priority("The staircase railing is loose and it feels unsafe.", "Negative").label == "High"


def test_priority_medium_for_ordinary_complaint():
    assert estimate_priority("The classroom projector is not working.", "Negative").label == "Medium"


def test_priority_low_for_plain_positive_or_neutral():
    assert estimate_priority("The library is open until 6 PM.", "Neutral").label == "Low"
    assert estimate_priority("Great teaching this semester.", "Positive").label == "Low"


def test_neutral_text_can_still_be_flagged():
    assert estimate_priority("A student was injured in the lab, this is urgent.", "Neutral").label == "High"


def test_negated_safety_cue_is_not_flagged():
    assert estimate_priority("There were no safety issues, everything is fine.", "Positive").label == "Low"


def test_priority_reports_reasons():
    result = estimate_priority("Fix the broken lift urgently", "Negative")
    assert result.reasons and any("urgen" in r for r in result.reasons)


def test_health_cue_is_detected_in_various_phrasings():
    assert estimate_priority("Hostel mess food made several of us sick last night.", "Negative").label == "High"
    assert estimate_priority("Students were vomiting after lunch.", "Negative").label == "High"
