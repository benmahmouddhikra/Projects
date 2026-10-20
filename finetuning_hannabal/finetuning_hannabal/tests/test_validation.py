from __future__ import annotations

import copy

import pytest

from assemble import assemble_canonical, reclassify_completed_record
from extract import extract_document
from inventory import MEDIA_TYPES, sha256_file
from normalize import sha256_text, stable_id
from validate import (
    FORBIDDEN_PROVIDER_KEYS,
    build_review_record,
    schema_errors,
    validate_canonical,
    validate_labeling,
)


def _manifest(path):
    return {
        "schema_version": "manifest-record-v1",
        "source_id": stable_id("src", path.as_posix()),
        "relative_path": "templates/test-contract.docx",
        "sha256": sha256_file(path),
        "media_type": MEDIA_TYPES[".docx"],
        "extension": "docx",
        "size_bytes": path.stat().st_size,
        "folder_hint": "test",
        "exact_duplicate": {"group_id": None, "is_primary": True, "member_paths": ["templates/test-contract.docx"]},
        "inventory_status": "inventoried",
        "pipeline_version": "1.0.0",
    }


def _canonical(minimal_docx, taxonomy):
    manifest = _manifest(minimal_docx)
    extraction = extract_document(minimal_docx, manifest)
    canonical = assemble_canonical(extraction, [manifest], taxonomy, model_reason="test boundary")
    return extraction, canonical


def test_source_spans_cross_references_and_unknown_properties(minimal_docx, taxonomy):
    extraction, canonical = _canonical(minimal_docx, taxonomy)
    assert validate_canonical(canonical, extraction) == []

    bad_span = copy.deepcopy(canonical)
    bad_span["document"]["title_evidence"][0]["text_sha256"] = "0" * 64
    assert any("hash mismatch" in error for error in validate_canonical(bad_span, extraction))

    bad_ref = copy.deepcopy(canonical)
    bad_ref["sections"][0]["clauses"][0]["variable_ids"].append("missing_variable")
    assert any("unknown variable" in error for error in validate_canonical(bad_ref, extraction))

    unknown = copy.deepcopy(canonical)
    unknown["unexpected"] = True
    assert any("Additional properties" in error for error in validate_canonical(unknown, extraction))


def test_model_text_injection_is_rejected():
    span = {"block_id": "block_000001", "start": 0, "end": 1, "text_sha256": sha256_text("A")}
    response = {
        "schema_version": "labeling-response-v1", "source_id": "src_test",
        "document": {"title_span": span, "document_kind": "unknown", "template_state": "unknown", "contract_family": "unknown", "contract_type_label": None, "languages": ["en"]},
        "jurisdictions": [], "parties": [],
        "sections": [{"kind": "unclassified", "heading_span": None, "body_spans": [span], "requirement_status": "unspecified", "condition_span": None,
                      "clauses": [{"clause_type": "unclassified", "heading_span": None, "body_spans": [span], "requirement_status": "unspecified", "condition_span": None, "citation_spans": [], "placeholder_indexes": [], "clause_text": "injected"}]}],
        "placeholders": [], "unparsed_spans": [], "warnings": [],
        "versions": {"model": "qwen", "prompt": "p", "taxonomy": "1", "schema": "labeling-response-v1", "pipeline": "1"}
    }
    assert any("Additional properties" in error for error in validate_labeling(response))


def test_review_queue_and_completed_reclassification_safeguards(tmp_path, taxonomy):
    from conftest import write_minimal_docx

    path = tmp_path / "completed-contract.docx"
    write_minimal_docx(path, "CONTRACT\nArticle 1\nFilled value")
    manifest = _manifest(path)
    extraction = extract_document(path, manifest)
    canonical = assemble_canonical(extraction, [manifest], taxonomy, model_reason="test")
    assert canonical["document"]["initial_document_kind"] == "completed_contract"
    review = build_review_record([manifest["source_id"]], template_id=canonical["template_id"], reasons=["human_review_required"], document_kind="completed_contract")
    assert review["queue_status"] == "pending"
    assert {"privacy", "sanitization", "legal", "structure"}.issubset(review["required_reviews"])
    with pytest.raises(ValueError, match="sanitization_status"):
        reclassify_completed_record(canonical, actor="reviewer", review_record_id=review["review_record_id"], reason="review")

    canonical["quality"].update({
        "sanitization_status": "completed", "privacy_review_status": "approved",
        "structure_review_status": "approved", "legal_review_status": "approved",
    })
    canonical["sanitization_history"].append({
        "action_id": "action_sanitized", "action": "human_sanitization", "actor_type": "human", "actor": "reviewer", "details": "Fixture review"
    })
    reclassified = reclassify_completed_record(canonical, actor="reviewer", review_record_id=review["review_record_id"], reason="sanitized and approved")
    assert reclassified["document"]["document_kind"] == "contract_template"
    assert reclassified["document"]["initial_document_kind"] == "completed_contract"
    assert reclassified["document"]["reclassification_history"]
    assert reclassified["quality"]["training_eligibility"] == "eligible"


def test_provider_neutral_output(minimal_docx, taxonomy):
    extraction, canonical = _canonical(minimal_docx, taxonomy)
    forbidden_seen = []

    def walk(value):
        if isinstance(value, dict):
            forbidden_seen.extend(FORBIDDEN_PROVIDER_KEYS.intersection(key.casefold() for key in value))
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(canonical)
    assert forbidden_seen == []
    assert "embedding" not in str(canonical).casefold()

