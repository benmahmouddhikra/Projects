from __future__ import annotations

import copy
import json

from assemble import assemble_canonical
from extract import extract_document
from inventory import MEDIA_TYPES, sha256_file
from normalize import stable_id
from qwen_enrichment import build_packet_artifacts, build_packets, merge_contract_labels, validate_response_for_packet
from runtime_templates import runtime_from_canonical, validate_runtime
from validate import validate_canonical


def _manifest(path):
    return {
        "schema_version": "manifest-record-v1", "source_id": stable_id("src", path.as_posix()),
        "relative_path": "templates/test-contract.docx", "sha256": sha256_file(path),
        "media_type": MEDIA_TYPES[".docx"], "extension": "docx", "size_bytes": path.stat().st_size,
        "folder_hint": "test", "exact_duplicate": {"group_id": None, "is_primary": True, "member_paths": ["templates/test-contract.docx"]},
        "inventory_status": "inventoried", "pipeline_version": "1.0.0",
    }


def _canonical(minimal_docx, taxonomy):
    manifest = _manifest(minimal_docx)
    extraction = extract_document(minimal_docx, manifest)
    return extraction, assemble_canonical(extraction, [manifest], taxonomy, model_reason="test")


def _response(packet, *, section_kind="unclassified", variable_type="unknown"):
    return {
        "schema_version": "qwen-contract-label-response-v1", "packet_id": packet["packet_id"],
        "template_id": packet["template_id"], "section_id": packet["section"]["section_id"],
        "section_kind": section_kind, "section_requirement_status": "unspecified",
        "clause_labels": [{"clause_id": clause["clause_id"], "requirement_status": "unspecified"} for clause in packet["clauses"]],
        "variable_labels": [{"variable_id": variable["variable_id"], "value_type": variable_type, "requirement_status": "unspecified"} for variable in packet["variables"]],
        "warnings": [],
    }


def test_packets_are_contract_only_and_fall_back_to_clauses(minimal_docx, taxonomy, tmp_path):
    _, canonical = _canonical(minimal_docx, taxonomy)
    packets, failures = build_packets(canonical, max_chars=1)
    assert packets
    assert all(packet["scope"] == "clause" for packet in packets)
    assert any(failure["code"] == "clause_packet_exceeds_max_chars" for failure in failures)

    non_contract = copy.deepcopy(canonical)
    non_contract["document"]["document_kind"] = "policy"
    skipped, reasons = build_packets(non_contract, max_chars=6000)
    assert skipped == []
    assert reasons[0]["code"] == "non_contract_document_kind"

    root = tmp_path / "canonical"
    target = root / "one"
    target.mkdir(parents=True)
    (target / "canonical.json").write_text(json.dumps(non_contract), encoding="utf-8")
    summary = build_packet_artifacts(root, tmp_path / "packets", max_chars=6000)
    assert summary["contract_records"] == 0
    assert summary["non_contract_skipped"] == 1


def test_response_rejects_injection_and_merges_only_existing_labels(minimal_docx, taxonomy):
    extraction, canonical = _canonical(minimal_docx, taxonomy)
    packets, _ = build_packets(canonical, max_chars=6000)
    response = _response(packets[0], section_kind="definitions", variable_type="text")
    assert validate_response_for_packet(response, packets[0]) == []
    injected = copy.deepcopy(response)
    injected["invented_clause_text"] = "new law"
    assert validate_response_for_packet(injected, packets[0])

    enriched = merge_contract_labels(canonical, packets, [_response(packet, section_kind="definitions", variable_type="text") for packet in packets], model_tag="qwen-test", prompt_sha256="prompt")
    assert validate_canonical(enriched, extraction) == []
    assert enriched["audit_history"][-1]["action"] == "qwen_packet_enrichment"
    assert [clause["text"] for section in enriched["sections"] for clause in section["clauses"]] == [clause["text"] for section in canonical["sections"] for clause in section["clauses"]]
    assert all(variable["value_type"] == "text" for variable in enriched["variables"])


def test_merge_rejects_conflicting_labels(minimal_docx, taxonomy):
    _, canonical = _canonical(minimal_docx, taxonomy)
    packets, _ = build_packets(canonical, max_chars=6000)
    packet = packets[0]
    duplicate_packet = copy.deepcopy(packet)
    duplicate_packet["packet_id"] = stable_id("pkt", packet["packet_id"], "duplicate")
    try:
        merge_contract_labels(canonical, [packet, duplicate_packet], [_response(packet, section_kind="definitions"), _response(duplicate_packet, section_kind="termination")], model_tag="qwen-test", prompt_sha256="prompt")
    except ValueError as exc:
        assert "Conflicting labels" in str(exc)
    else:
        raise AssertionError("Expected conflicting labels to be rejected")


def test_runtime_requires_reviewed_eligible_blank_template(minimal_docx, taxonomy):
    _, canonical = _canonical(minimal_docx, taxonomy)
    try:
        runtime_from_canonical(canonical)
    except ValueError as exc:
        assert "training_eligibility" in str(exc)
    else:
        raise AssertionError("Unreviewed record must be rejected")

    eligible = copy.deepcopy(canonical)
    eligible["quality"].update({
        "training_eligibility": "eligible",
        "structure_review_status": "approved",
        "legal_review_status": "approved",
    })
    runtime = runtime_from_canonical(eligible)
    assert validate_runtime(runtime) == []
    assert runtime["template_id"] == eligible["template_id"]
    assert runtime["canonical_ref"]["source_ids"] == [eligible["source_files"][0]["source_id"]]

    completed = copy.deepcopy(eligible)
    completed["document"]["document_kind"] = "completed_contract"
    try:
        runtime_from_canonical(completed)
    except ValueError as exc:
        assert "document_kind" in str(exc)
    else:
        raise AssertionError("Completed contract must be rejected")

    demo = copy.deepcopy(canonical)
    demo["document"].update({"document_kind": "completed_contract", "template_state": "completed"})
    runtime_demo = runtime_from_canonical(demo, demo_unreviewed=True)
    assert runtime_demo["runtime_status"] == "demo_unreviewed"
    assert runtime_demo["document"]["document_kind"] == "completed_contract"
    assert validate_runtime(runtime_demo) == []
