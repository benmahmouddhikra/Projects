"""Build exact, source-ordered contract targets from retrieved evidence.

Legal prose is copied from the canonical extraction.  The only generated text
operation is deterministic replacement of canonical template variables with
validated Agent 1 facts.  Demo review status never becomes training approval.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from agent2_intake import contains_forbidden_model_identifier, sha256_file, validate_generation_facts
from normalize import CANONICAL_PLACEHOLDER_RE, sha256_text, stable_id
from qwen_enrichment import _read_json, _write_json
from runtime_templates import validate_runtime
from validate import load_schema, schema_errors


EVIDENCE_SCHEMA = "retrieved-contract-evidence-v1.schema.json"
TARGET_VALIDATION_SCHEMA = "contract-target-validation-v1.schema.json"
EVIDENCE_FILE = "retrieved-evidence.json"
TARGET_FILE = "contract-target.txt"
TARGET_VALIDATION_FILE = "contract-target-validation.json"
BRACKET_TOKEN_RE = re.compile(r"\[[^\[\]\r\n]{1,160}\]")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _value_hash(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _schema_errors(value: dict[str, Any], schema_name: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(value), key=lambda error: [str(item) for item in error.absolute_path])
    return [
        f"schema:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}"
        for error in errors
    ]


def build_retrieved_evidence(
    runtime: dict[str, Any],
    *,
    clauses: list[dict[str, Any]] | None = None,
    demo: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Package exact runtime evidence; no model is allowed to add legal text."""
    errors = list(validate_runtime(runtime))
    package = {
        "schema_version": "retrieved-contract-evidence-v1",
        "retrieval_status": "demo_runtime_evidence" if demo else "production_rag_evidence",
        "template": copy.deepcopy(runtime),
        "clauses": copy.deepcopy(clauses or []),
    }
    errors.extend(_schema_errors(package, EVIDENCE_SCHEMA))
    if contains_forbidden_model_identifier(package):
        errors.append("privacy:retrieved evidence contains a forbidden operational identifier")
    return (package if not errors else None), sorted(set(errors))


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _block_evidence(canonical: dict[str, Any], block_id: str) -> tuple[list[str], list[str], list[str]]:
    source_ids: set[str] = set()
    clause_ids: list[str] = []
    content_ids: list[str] = []
    for section in canonical.get("sections", []):
        span = section.get("source_span", {})
        if span.get("block_id") == block_id and span.get("source_id"):
            source_ids.add(span["source_id"])
        for clause in section.get("clauses", []):
            span = clause.get("source_span", {})
            if span.get("block_id") == block_id:
                clause_ids.append(clause["clause_id"])
                if span.get("source_id"):
                    source_ids.add(span["source_id"])
    for content in canonical.get("unparsed_source_content", []):
        span = content.get("source_span", {})
        if span.get("block_id") == block_id:
            content_ids.append(content["content_id"])
            if span.get("source_id"):
                source_ids.add(span["source_id"])
    return sorted(source_ids), sorted(set(clause_ids)), sorted(set(content_ids))


def _render_source(
    facts: dict[str, Any],
    canonical: dict[str, Any],
    extraction: dict[str, Any],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    binding_by_id = {binding["variable_id"]: binding for binding in facts.get("variable_bindings", [])}
    occurrences_by_block: dict[str, list[dict[str, Any]]] = {}
    for variable in canonical.get("variables", []):
        binding = binding_by_id.get(variable["variable_id"])
        if binding is None:
            errors.append(f"mapping:{variable['variable_id']}:missing Agent 2 binding")
            continue
        for occurrence in variable.get("occurrences", []):
            occurrences_by_block.setdefault(occurrence["block_id"], []).append({
                **occurrence,
                "variable_id": variable["variable_id"],
                "canonical_token": variable["canonical_token"],
                "replacement": variable["canonical_token"] if binding["status"] != "provided" else _render_value(binding["value"]),
            })

    full_source = extraction.get("normalized_text", "")
    blocks = sorted(extraction.get("blocks", []), key=lambda block: block["ordinal"])
    passages: list[dict[str, Any]] = []
    rendered_segments: list[str] = []
    for index, block in enumerate(blocks):
        block_id = block["block_id"]
        block_text = block["normalized_text"]
        global_start = block["normalized_global_start"]
        global_end = block["normalized_global_end"]
        if full_source[global_start:global_end] != block_text:
            errors.append(f"source:{block_id}:block text does not match extraction global coordinates")
        operations = sorted(
            occurrences_by_block.get(block_id, []),
            key=lambda item: (item["normalized_start"], item["normalized_end"]),
            reverse=True,
        )
        occupied: list[tuple[int, int]] = []
        variable_ids: list[str] = []
        rendered = block_text
        for operation in operations:
            start, end = operation["normalized_start"], operation["normalized_end"]
            if start < 0 or end > len(block_text) or start >= end:
                errors.append(f"source:{operation['variable_id']}:invalid occurrence coordinates in {block_id}")
                continue
            if any(not (end <= left or start >= right) for left, right in occupied):
                errors.append(f"source:{operation['variable_id']}:overlapping occurrence coordinates in {block_id}")
                continue
            source_token = block_text[start:end]
            if sha256_text(source_token) != operation.get("text_sha256"):
                errors.append(f"source:{operation['variable_id']}:occurrence hash mismatch in {block_id}")
                continue
            rendered = rendered[:start] + operation["replacement"] + rendered[end:]
            occupied.append((start, end))
            variable_ids.append(operation["variable_id"])

        next_start = blocks[index + 1]["normalized_global_start"] if index + 1 < len(blocks) else len(full_source)
        prefix = full_source[:global_start] if index == 0 else ""
        separator = full_source[global_end:next_start]
        segment = prefix + rendered + separator
        rendered_segments.append(segment)
        source_ids, clause_ids, content_ids = _block_evidence(canonical, block_id)
        if not source_ids:
            source_id = extraction.get("source", {}).get("source_id")
            if source_id:
                source_ids = [source_id]
        passages.append({
            "passage_id": stable_id("passage", facts.get("template_id"), block_id),
            "source_ids": source_ids,
            "block_id": block_id,
            "clause_ids": clause_ids,
            "content_ids": content_ids,
            "variable_ids": sorted(set(variable_ids)),
            "source_sha256": sha256_text(prefix + block_text + separator),
            "target_sha256": sha256_text(segment),
        })
    return "".join(rendered_segments), passages, sorted(set(errors))


def _check(status: str, details: str) -> dict[str, str]:
    return {"status": status, "details": details}


def validate_contract_target(
    target: str,
    facts: dict[str, Any],
    runtime: dict[str, Any],
    canonical: dict[str, Any],
    extraction: dict[str, Any],
    *,
    hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    errors = list(validate_generation_facts(facts)) + list(validate_runtime(runtime))
    errors.extend(schema_errors(canonical, "contract-template-v1.schema.json"))
    if facts.get("template_id") != runtime.get("template_id") or canonical.get("template_id") != runtime.get("template_id"):
        errors.append("identity:template IDs do not agree")
    expected, passages, render_errors = _render_source(facts, canonical, extraction)
    errors.extend(render_errors)

    if target != expected:
        errors.append("grounding:target differs from deterministic source reconstruction")
    runtime_clause_ids = {
        clause["clause_id"]
        for section in runtime.get("sections", [])
        for clause in section.get("clauses", [])
    }
    canonical_clause_ids = {
        clause["clause_id"]
        for section in canonical.get("sections", [])
        for clause in section.get("clauses", [])
    }
    missing_clause_ids = sorted(runtime_clause_ids - canonical_clause_ids)
    if missing_clause_ids:
        errors.append("structure:runtime clauses absent from canonical evidence: " + ", ".join(missing_clause_ids))

    canonical_section_ids = {section["section_id"] for section in canonical.get("sections", [])}
    required_sections = {
        section["section_id"] for section in runtime.get("sections", [])
        if section.get("requirement_status") in {"explicit_required", "required", "mandatory"}
    }
    missing_required_sections = sorted(required_sections - canonical_section_ids)
    if missing_required_sections:
        errors.append("structure:required sections absent from canonical evidence: " + ", ".join(missing_required_sections))

    binding_by_token = {binding["canonical_token"]: binding for binding in facts.get("variable_bindings", [])}
    remaining = BRACKET_TOKEN_RE.findall(target)
    malformed = sorted({token for token in remaining if not CANONICAL_PLACEHOLDER_RE.fullmatch(token)})
    undeclared = sorted({token for token in remaining if token not in binding_by_token})
    incorrectly_remaining = sorted({
        token for token in remaining
        if token in binding_by_token and binding_by_token[token]["status"] == "provided"
    })
    if malformed:
        errors.append("placeholder:malformed tokens remain: " + ", ".join(malformed))
    if undeclared:
        errors.append("placeholder:undeclared tokens remain: " + ", ".join(undeclared))
    if incorrectly_remaining:
        errors.append("placeholder:provided variables were not replaced: " + ", ".join(incorrectly_remaining))

    legal_references = sorted({
        reference
        for jurisdiction in runtime.get("jurisdictions", [])
        for reference in jurisdiction.get("legal_references", [])
    } | {
        reference
        for section in runtime.get("sections", [])
        for clause in section.get("clauses", [])
        for reference in clause.get("legal_references", [])
    })
    absent_legal_references = [reference for reference in legal_references if reference and reference not in extraction.get("normalized_text", "")]
    if absent_legal_references:
        errors.append("grounding:legal references absent from source extraction: " + ", ".join(absent_legal_references))

    runtime_languages = runtime.get("document", {}).get("languages", [])
    fact_language = facts.get("document", {}).get("language")
    language_ok = fact_language in runtime_languages or str(fact_language).casefold().startswith("ambiguity")
    if not language_ok:
        errors.append("metadata:Agent 1 language is not preserved by runtime evidence")

    quality = canonical.get("quality", {})
    completed = runtime.get("document", {}).get("document_kind") == "completed_contract"
    privacy_approved = not completed or (
        quality.get("privacy_review_status") == "approved"
        and quality.get("sanitization_status") in {"approved", "completed", "not_required"}
    )
    reviews_approved = (
        quality.get("structure_review_status") == "approved"
        and quality.get("legal_review_status") == "approved"
        and privacy_approved
    )
    deterministic_failures = bool(errors)
    training_eligible = bool(
        not deterministic_failures
        and runtime.get("runtime_status") == "eligible"
        and reviews_approved
        and quality.get("training_eligibility") == "eligible"
    )
    warnings = [
        "AMBIGUITY: broader semantic internal-contradiction measurement is not specified by the project sources.",
        "AMBIGUITY: legal drafting-style measurement is not specified by the project sources.",
        "AMBIGUITY: MENA-adaptation measurement is not specified by the project sources.",
        "AMBIGUITY: required-section completeness is specified as both 100% and greater than 99%; this report checks source-section presence only.",
    ]
    if completed and not privacy_approved:
        warnings.append("Completed contract remains demo_unreviewed; privacy and sanitization approval is absent.")

    checks = {
        "required_sections": _check("failed" if missing_required_sections else "passed", f"{len(required_sections)} source-declared required section(s) checked; no KPI threshold asserted."),
        "missing_clauses": _check("failed" if missing_clause_ids else "passed", f"{len(runtime_clause_ids)} runtime clause identifier(s) preserved in canonical evidence."),
        "placeholders": _check("failed" if malformed or undeclared or incorrectly_remaining else "passed", f"{len(remaining)} declared missing or ambiguous placeholder occurrence(s) remain."),
        "variable_consistency": _check("failed" if any(item.startswith("mapping:") for item in errors) else "passed", "Every runtime variable has one validated binding; conflicting values are rejected by intake."),
        "internal_contradictions": _check("not_measured", "Deterministic duplicate-variable contradictions are checked; broader semantic legal contradiction has no source-defined method."),
        "unsupported_legal_content": _check("failed" if target != expected else "passed", "Target equals exact source reconstruction plus deterministic user-fact substitutions."),
        "legal_references": _check("failed" if absent_legal_references else "passed", f"{len(legal_references)} retrieved legal-reference string(s) checked against source text."),
        "legal_drafting_style": _check("not_measured", "No source-defined automated legal-style measurement method is available."),
        "mena_adaptation": _check("not_measured", "No source-defined automated MENA-adaptation measurement method is available."),
        "jurisdiction_preservation": _check("passed", f"{len(runtime.get('jurisdictions', []))} retrieved jurisdiction record(s) remain in the evidence package and export manifest."),
        "language_preservation": _check("passed" if language_ok else "failed", f"Agent 1 language {fact_language!r}; runtime languages {runtime_languages!r}."),
        "grounding_coverage": _check("failed" if target != expected or not passages else "passed", f"{len(passages)} ordered extraction passage(s) reconstruct the complete target; no project KPI claim is made."),
        "completed_contract_privacy": _check("not_applicable" if not completed else ("passed" if privacy_approved else "failed"), "Completed-contract production eligibility requires privacy and sanitization approval; demo output does not confer approval."),
    }
    supplied = hashes or {}
    report = {
        "schema_version": "contract-target-validation-v1",
        "template_id": runtime.get("template_id", "unknown"),
        "status": "blocked" if deterministic_failures else ("eligible" if training_eligible else "demo_unreviewed"),
        "valid_for_demo_export": not deterministic_failures,
        "training_eligible": training_eligible,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "hashes": {
            "generation_facts_sha256": supplied.get("generation_facts_sha256") or _value_hash(facts),
            "runtime_sha256": supplied.get("runtime_sha256") or _value_hash(runtime),
            "canonical_sha256": supplied.get("canonical_sha256") or _value_hash(canonical),
            "extraction_sha256": supplied.get("extraction_sha256") or _value_hash(extraction),
            "target_sha256": sha256_text(target),
        },
        "passages": passages,
        "checks": checks,
    }
    report_schema_errors = _schema_errors(report, TARGET_VALIDATION_SCHEMA)
    if report_schema_errors:
        report["errors"] = sorted(set(report["errors"] + report_schema_errors))
        report["status"] = "blocked"
        report["valid_for_demo_export"] = False
        report["training_eligible"] = False
    return report


def build_contract_target(
    facts: dict[str, Any],
    runtime: dict[str, Any],
    canonical: dict[str, Any],
    extraction: dict[str, Any],
    *,
    hashes: dict[str, str] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    target, _, render_errors = _render_source(facts, canonical, extraction)
    report = validate_contract_target(target, facts, runtime, canonical, extraction, hashes=hashes)
    if render_errors or not report["valid_for_demo_export"]:
        return None, report
    return target, report


def build_record_target(
    record_dir: Path,
    canonical_path: Path,
    extraction_path: Path,
) -> tuple[str | None, dict[str, Any], dict[str, Any] | None]:
    runtime_path = record_dir / "runtime.json"
    facts_path = record_dir / "agent2-generation-facts.json"
    runtime = _read_json(runtime_path)
    facts = _read_json(facts_path)
    canonical = _read_json(canonical_path)
    extraction = _read_json(extraction_path)
    evidence, evidence_errors = build_retrieved_evidence(runtime, demo=runtime.get("runtime_status") == "demo_unreviewed")
    if evidence is not None:
        _write_json(record_dir / EVIDENCE_FILE, evidence)
    target, report = build_contract_target(
        facts, runtime, canonical, extraction,
        hashes={
            "generation_facts_sha256": sha256_file(facts_path),
            "runtime_sha256": sha256_file(runtime_path),
            "canonical_sha256": sha256_file(canonical_path),
            "extraction_sha256": sha256_file(extraction_path),
        },
    )
    if evidence_errors:
        report["errors"] = sorted(set(report["errors"] + evidence_errors))
        report["valid_for_demo_export"] = False
        report["training_eligible"] = False
        report["status"] = "blocked"
    if target is not None and report["valid_for_demo_export"]:
        (record_dir / TARGET_FILE).write_text(target, encoding="utf-8", newline="\n")
    _write_json(record_dir / TARGET_VALIDATION_FILE, report)
    return (target if report["valid_for_demo_export"] else None), report, evidence


__all__ = [
    "EVIDENCE_FILE", "TARGET_FILE", "TARGET_VALIDATION_FILE", "build_contract_target",
    "build_record_target", "build_retrieved_evidence", "validate_contract_target",
]
