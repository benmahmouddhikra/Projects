"""Deterministic text normalization, identifiers, and placeholder handling."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable


CANONICAL_PLACEHOLDER_RE = re.compile(r"^\[[A-Z][A-Z0-9_]*\]$")
SOURCE_PLACEHOLDER_RE = re.compile(r"\[([^\[\]\r\n]{1,160})\]")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{sha256_text(payload)[:length]}"


def normalize_text(value: str) -> str:
    """Normalize line endings and Unicode without stripping bidi marks or accents."""
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", value)


def normalization_map(raw: str, normalized: str) -> list[dict[str, Any]]:
    """Return deterministic coordinate runs mapping normalized text to raw text."""
    matcher = SequenceMatcher(a=raw, b=normalized, autojunk=False)
    return [
        {
            "operation": operation,
            "raw_start": raw_start,
            "raw_end": raw_end,
            "normalized_start": normalized_start,
            "normalized_end": normalized_end,
        }
        for operation, raw_start, raw_end, normalized_start, normalized_end
        in matcher.get_opcodes()
    ]


def ascii_snake_case(label: str) -> str:
    decomposed = unicodedata.normalize("NFKD", label)
    ascii_value = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_value = ascii_value.encode("ascii", "ignore").decode("ascii").lower()
    ascii_value = re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")
    if ascii_value and ascii_value[0].isdigit():
        ascii_value = f"field_{ascii_value}"
    return ascii_value


def canonical_variable_id(label: str, ordinal: int) -> str:
    candidate = ascii_snake_case(label)
    return candidate or f"source_field_{ordinal:03d}"


def canonical_token(variable_id: str) -> str:
    token = f"[{variable_id.upper()}]"
    if not CANONICAL_PLACEHOLDER_RE.fullmatch(token):
        raise ValueError(f"Invalid variable identifier: {variable_id}")
    return token


def detect_placeholders(text: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    labels_to_ids: dict[str, str] = {}
    used_ids: set[str] = set()
    for match in SOURCE_PLACEHOLDER_RE.finditer(text):
        source_token = match.group(0)
        label = normalize_text(match.group(1)).strip()
        key = label.casefold()
        if key not in labels_to_ids:
            base = canonical_variable_id(label, len(labels_to_ids) + 1)
            variable_id = base
            suffix = 2
            while variable_id in used_ids:
                variable_id = f"{base}_{suffix}"
                suffix += 1
            labels_to_ids[key] = variable_id
            used_ids.add(variable_id)
        variable_id = labels_to_ids[key]
        results.append(
            {
                "variable_id": variable_id,
                "canonical_token": canonical_token(variable_id),
                "source_token": source_token,
                "start": match.start(),
                "end": match.end(),
            }
        )
    return results


def replace_placeholders(text: str, occurrences: Iterable[dict[str, Any]]) -> str:
    rendered = text
    ordered = sorted(occurrences, key=lambda item: item["start"], reverse=True)
    for item in ordered:
        rendered = (
            rendered[: item["start"]]
            + item["canonical_token"]
            + rendered[item["end"] :]
        )
    return rendered


def detect_languages(text: str) -> list[dict[str, Any]]:
    """Conservative script/stop-word labels; every result remains review-required."""
    sample = text[:100_000]
    arabic = len(re.findall(r"[\u0600-\u06ff]", sample))
    latin_words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", sample.casefold())
    latin = sum(len(word) for word in latin_words)
    results: list[dict[str, Any]] = []
    total_letters = arabic + latin
    if arabic:
        results.append({"tag": "ar", "basis": "script_detection", "confidence": round(arabic / max(total_letters, 1), 4)})
    if latin:
        french_markers = {"le", "la", "les", "des", "contrat", "entre", "parties", "article", "française", "france"}
        english_markers = {"the", "and", "agreement", "contract", "between", "party", "shall", "whereas"}
        fr_score = sum(word in french_markers for word in latin_words)
        en_score = sum(word in english_markers for word in latin_words)
        if fr_score > en_score and fr_score >= 2:
            tag = "fr"
        elif en_score > fr_score and en_score >= 2:
            tag = "en"
        else:
            tag = "und-Latn"
        results.append({"tag": tag, "basis": "deterministic_text_heuristic", "confidence": round(max(fr_score, en_score) / max(len(latin_words), 1), 4)})
    return results or [{"tag": "und", "basis": "unknown", "confidence": 0.0}]

