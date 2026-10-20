"""Strict JSON Schema and semantic grounding validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from normalize import CANONICAL_PLACEHOLDER_RE, sha256_text, stable_id


FORBIDDEN_PROVIDER_KEYS = {
    "embedding", "embeddings", "vector", "vectors", "embedding_model",
    "vector_database", "index_provider", "database_provider", "publication_api",
}


def schema_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "schemas"


def load_schema(name: str) -> dict[str, Any]:
    return json.loads((schema_directory() / name).read_text(encoding="utf-8"))


def schema_errors(instance: dict[str, Any], schema_name: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(instance), key=lambda error: [str(item) for item in error.absolute_path])
    return [f"schema:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}" for error in errors]


def _walk(value: Any, path: str = "$") -> Iterable[tuple[str, Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def _span_text(span: dict[str, Any], blocks: dict[str, dict[str, Any]]) -> str | None:
    block = blocks.get(span.get("block_id"))
    if block is None:
        return None
    start, end = span.get("normalized_start"), span.get("normalized_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(block["normalized_text"]):
        return None
    return block["normalized_text"][start:end]


def semantic_errors(record: dict[str, Any], extraction: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    blocks = {block["block_id"]: block for block in extraction["blocks"]}
    source_ids = {source["source_id"] for source in record.get("source_files", [])}

    for path, value in _walk(record):
        if isinstance(value, dict):
            forbidden = FORBIDDEN_PROVIDER_KEYS.intersection(key.casefold() for key in value)
            if forbidden:
                errors.append(f"semantic:{path}:provider-specific or embedding keys forbidden: {sorted(forbidden)}")
            span_keys = {"source_id", "extraction_id", "block_id", "raw_start", "raw_end", "normalized_start", "normalized_end", "text_sha256"}
            if span_keys.issubset(value):
                text = _span_text(value, blocks)
                if value["source_id"] not in source_ids:
                    errors.append(f"semantic:{path}:source_id is not declared")
                if value["extraction_id"] != extraction["extraction_id"]:
                    errors.append(f"semantic:{path}:extraction_id mismatch")
                if text is None:
                    errors.append(f"semantic:{path}:invalid source span")
                elif sha256_text(text) != value["text_sha256"]:
                    errors.append(f"semantic:{path}:source span hash mismatch")

    def unique(field: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            errors.append(f"semantic:{field}:IDs must be unique")

    unique("source_files", [item["source_id"] for item in record.get("source_files", [])])
    unique("variables", [item["variable_id"] for item in record.get("variables", [])])
    unique("parties", [item["party_id"] for item in record.get("parties", [])])
    unique("sections", [item["section_id"] for item in record.get("sections", [])])
    clause_ids = [clause["clause_id"] for section in record.get("sections", []) for clause in section["clauses"]]
    unique("clauses", clause_ids)

    section_orders = [section["order"] for section in record.get("sections", [])]
    if section_orders != list(range(1, len(section_orders) + 1)):
        errors.append("semantic:sections:orders must be contiguous from 1")
    variable_ids = {item["variable_id"] for item in record.get("variables", [])}
    party_ids = {item["party_id"] for item in record.get("parties", [])}
    token_by_variable = {item["variable_id"]: item["canonical_token"] for item in record.get("variables", [])}
    for variable in record.get("variables", []):
        if variable["canonical_token"] != f"[{variable['variable_id'].upper()}]":
            errors.append(f"semantic:variables:{variable['variable_id']}:canonical token mismatch")
        if variable["party_id"] is not None and variable["party_id"] not in party_ids:
            errors.append(f"semantic:variables:{variable['variable_id']}:unknown party reference")
    for party in record.get("parties", []):
        for variable_id in party["variable_ids"]:
            if variable_id not in variable_ids:
                errors.append(f"semantic:parties:{party['party_id']}:unknown variable {variable_id}")
    for section in record.get("sections", []):
        clause_orders = [clause["order"] for clause in section["clauses"]]
        if clause_orders != list(range(1, len(clause_orders) + 1)):
            errors.append(f"semantic:sections:{section['section_id']}:clause orders must be contiguous")
        for clause in section["clauses"]:
            for variable_id in clause["variable_ids"]:
                if variable_id not in variable_ids:
                    errors.append(f"semantic:clauses:{clause['clause_id']}:unknown variable {variable_id}")
            source_text = _span_text(clause["source_span"], blocks)
            if source_text is None:
                continue
            expected = source_text
            clause_span = clause["source_span"]
            replacements: list[tuple[int, int, str]] = []
            for variable in record.get("variables", []):
                for occurrence in variable["occurrences"]:
                    if occurrence["block_id"] != clause_span["block_id"]:
                        continue
                    start = occurrence["normalized_start"]
                    end = occurrence["normalized_end"]
                    if start >= clause_span["normalized_start"] and end <= clause_span["normalized_end"]:
                        replacements.append((start - clause_span["normalized_start"], end - clause_span["normalized_start"], variable["canonical_token"]))
            for start, end, token in sorted(replacements, reverse=True):
                expected = expected[:start] + token + expected[end:]
            if clause["text"] != expected:
                errors.append(f"semantic:clauses:{clause['clause_id']}:text is not an exact source slice plus deterministic placeholder normalization")
            for token in re.findall(r"\[[^\[\]\r\n]+\]", clause["text"]):
                if not CANONICAL_PLACEHOLDER_RE.fullmatch(token):
                    errors.append(f"semantic:clauses:{clause['clause_id']}:malformed placeholder {token}")
                elif token not in token_by_variable.values():
                    errors.append(f"semantic:clauses:{clause['clause_id']}:undeclared placeholder {token}")

    title = record.get("document", {}).get("title")
    title_evidence = record.get("document", {}).get("title_evidence", [])
    if title is not None:
        if not title_evidence:
            errors.append("semantic:document.title:title requires source evidence")
        elif _span_text(title_evidence[0], blocks) != title:
            errors.append("semantic:document.title:title differs from source span")

    document = record.get("document", {})
    quality = record.get("quality", {})
    if quality.get("training_eligibility") == "eligible":
        if document.get("document_kind") != "contract_template":
            errors.append("semantic:quality:only contract_template can be eligible")
        required_quality_reviews = ["structure_review_status", "legal_review_status"]
        if document.get("initial_document_kind") == "completed_contract" or document.get("template_state") in {"completed", "mixed"}:
            required_quality_reviews.append("privacy_review_status")
        for field in required_quality_reviews:
            if quality.get(field) != "approved":
                errors.append(f"semantic:quality:{field} must be approved for eligibility")
    if document.get("initial_document_kind") == "completed_contract" and document.get("document_kind") == "contract_template":
        if quality.get("sanitization_status") != "completed" or not document.get("reclassification_history"):
            errors.append("semantic:document:completed contract reclassification lacks sanitization/audit history")
    return errors


def validate_canonical(record: dict[str, Any], extraction: dict[str, Any]) -> list[str]:
    return schema_errors(record, "contract-template-v1.schema.json") + semantic_errors(record, extraction)


def validate_extraction(record: dict[str, Any]) -> list[str]:
    return schema_errors(record, "extraction-record-v1.schema.json")


def validate_manifest(record: dict[str, Any]) -> list[str]:
    return schema_errors(record, "manifest-record-v1.schema.json")


def validate_labeling(record: dict[str, Any]) -> list[str]:
    return schema_errors(record, "labeling-response-v1.schema.json")


def build_review_record(
    source_ids: list[str], *, template_id: str | None, reasons: list[str], document_kind: str | None = None,
) -> dict[str, Any]:
    reasons = sorted(set(reasons)) or ["human_review_required"]
    reviews = {"classification", "structure", "legal"}
    if template_id is None:
        reviews.add("extraction")
    if document_kind == "completed_contract":
        reviews.update({"privacy", "sanitization"})
    if document_kind not in {None, "contract_template", "completed_contract"}:
        reviews.add("classification")
    review_id = stable_id("review", *(sorted(source_ids) + [template_id or "no-template"]))
    return {
        "schema_version": "review-record-v1",
        "review_record_id": review_id,
        "template_id": template_id,
        "source_ids": sorted(source_ids),
        "queue_status": "pending",
        "reasons": reasons,
        "required_reviews": sorted(reviews),
        "decisions": [],
        "versions": {"pipeline": "1.0.0", "schema": "review-record-v1"},
    }
