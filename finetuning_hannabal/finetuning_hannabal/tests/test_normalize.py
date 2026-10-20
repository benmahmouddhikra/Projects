from __future__ import annotations

from normalize import (
    CANONICAL_PLACEHOLDER_RE,
    detect_placeholders,
    normalize_text,
    stable_id,
)


def test_normalization_and_ids_are_deterministic():
    raw = "Cafe\u0301\r\nالعربية\rfin"
    expected = "Café\nالعربية\nfin"
    assert normalize_text(raw) == expected
    assert normalize_text(raw) == normalize_text(raw)
    assert stable_id("x", raw) == stable_id("x", raw)


def test_placeholder_grammar_and_arabic_fallback():
    occurrences = detect_placeholders("[Nom client] [اسم العميل]")
    assert occurrences[0]["variable_id"] == "nom_client"
    assert occurrences[1]["variable_id"] == "source_field_002"
    assert all(CANONICAL_PLACEHOLDER_RE.fullmatch(item["canonical_token"]) for item in occurrences)

