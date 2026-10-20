"""Build compact runtime template records from reviewed enriched canonical records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_enrichment import _canonical_paths, _read_json, _write_json
from validate import schema_errors


RUNTIME_SCHEMA = "contract-template-runtime-v1.schema.json"


def runtime_from_canonical(canonical: dict[str, Any], *, demo_unreviewed: bool = False) -> dict[str, Any]:
    """Project an enriched canonical contract; demo output remains explicitly unreviewed."""
    document = canonical.get("document", {})
    quality = canonical.get("quality", {})
    if document.get("document_kind") not in {"contract_template", "completed_contract"}:
        raise ValueError("Runtime records require a contract_template or completed_contract")
    if not demo_unreviewed:
        if document.get("document_kind") != "contract_template":
            raise ValueError("Production runtime records require document_kind contract_template")
        if document.get("template_state") != "blank":
            raise ValueError("Production runtime records require template_state blank")
        if quality.get("training_eligibility") != "eligible":
            raise ValueError("Production runtime records require training_eligibility eligible")
        for field in ("structure_review_status", "legal_review_status"):
            if quality.get(field) != "approved":
                raise ValueError(f"Production runtime records require {field} approved")
        if document.get("initial_document_kind") == "completed_contract" or document.get("template_state") in {"completed", "mixed"}:
            if quality.get("privacy_review_status") != "approved":
                raise ValueError("Production runtime records require privacy_review_status approved")
    if schema_errors(canonical, "contract-template-v1.schema.json"):
        raise ValueError("Runtime records require a schema-valid canonical record")
    runtime = {
        "schema_version": "contract-template-runtime-v1",
        "runtime_status": "demo_unreviewed" if demo_unreviewed else "eligible",
        "template_id": canonical["template_id"],
        "canonical_ref": {
            "schema_version": "contract-template-v1",
            "template_id": canonical["template_id"],
            "source_ids": [source["source_id"] for source in canonical["source_files"]],
        },
        "document": {
            "document_kind": document["document_kind"],
            "template_state": document["template_state"],
            "title": document["title"],
            "contract_family": document["contract_family"],
            "contract_type": document["contract_type"],
            "languages": [language["tag"] for language in document["languages"]],
        },
        "jurisdictions": [
            {
                "country_code": jurisdiction["country_code"],
                "governing_law_text": jurisdiction["governing_law_text"],
                "legal_references": [reference["text"] for reference in jurisdiction["legal_citations"]],
            }
            for jurisdiction in canonical["jurisdictions"]
        ],
        "parties": [
            {
                "party_id": party["party_id"], "role": party["role"],
                "original_role_label": party["original_role_label"], "entity_type": party["entity_type"],
                "variable_ids": party["variable_ids"],
            }
            for party in canonical["parties"]
        ],
        "variables": [
            {
                "variable_id": variable["variable_id"], "canonical_token": variable["canonical_token"],
                "value_type": variable["value_type"], "party_id": variable["party_id"],
                "requirement_status": variable["requirement_status"], "allowed_choices": variable["allowed_choices"],
            }
            for variable in canonical["variables"]
        ],
        "sections": [
            {
                "section_id": section["section_id"], "order": section["order"], "kind": section["kind"],
                "heading": section["heading"], "requirement_status": section["requirement_status"],
                "condition": section["condition"],
                "clauses": [
                    {
                        "clause_id": clause["clause_id"], "order": clause["order"],
                        "clause_type": clause["clause_type"], "heading": clause["heading"], "text": clause["text"],
                        "variable_ids": clause["variable_ids"],
                        "legal_references": [reference["text"] for reference in clause["legal_references"]],
                        "requirement_status": clause["requirement_status"], "condition": clause["condition"],
                    }
                    for clause in section["clauses"]
                ],
            }
            for section in canonical["sections"]
        ],
        "unparsed_content": [
            {"content_id": content["content_id"], "text": content["text"], "reason": content["reason"]}
            for content in canonical["unparsed_source_content"]
        ],
    }
    errors = validate_runtime(runtime)
    if errors:
        raise ValueError("Runtime validation failed: " + "; ".join(errors))
    return runtime


def validate_runtime(record: dict[str, Any]) -> list[str]:
    errors = schema_errors(record, RUNTIME_SCHEMA)
    if errors:
        return errors
    if record["template_id"] != record["canonical_ref"]["template_id"]:
        errors.append("semantic:canonical_ref:template_id mismatch")

    def unique(field: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            errors.append(f"semantic:{field}:IDs must be unique")

    unique("canonical_ref.source_ids", record["canonical_ref"]["source_ids"])
    unique("parties", [party["party_id"] for party in record["parties"]])
    unique("variables", [variable["variable_id"] for variable in record["variables"]])
    unique("sections", [section["section_id"] for section in record["sections"]])
    variable_ids = {variable["variable_id"] for variable in record["variables"]}
    party_ids = {party["party_id"] for party in record["parties"]}
    for party in record["parties"]:
        for variable_id in party["variable_ids"]:
            if variable_id not in variable_ids:
                errors.append(f"semantic:parties:{party['party_id']}:unknown variable {variable_id}")
    for variable in record["variables"]:
        if variable["party_id"] is not None and variable["party_id"] not in party_ids:
            errors.append(f"semantic:variables:{variable['variable_id']}:unknown party reference")
    section_orders = [section["order"] for section in record["sections"]]
    if section_orders != list(range(1, len(section_orders) + 1)):
        errors.append("semantic:sections:orders must be contiguous from 1")
    clause_ids: list[str] = []
    for section in record["sections"]:
        orders = [clause["order"] for clause in section["clauses"]]
        if orders != list(range(1, len(orders) + 1)):
            errors.append(f"semantic:sections:{section['section_id']}:clause orders must be contiguous")
        for clause in section["clauses"]:
            clause_ids.append(clause["clause_id"])
            for variable_id in clause["variable_ids"]:
                if variable_id not in variable_ids:
                    errors.append(f"semantic:clauses:{clause['clause_id']}:unknown variable {variable_id}")
    unique("clauses", clause_ids)
    return errors


def generate_runtime_artifacts(canonical_dir: Path, output_dir: Path, *, demo_unreviewed: bool = False) -> dict[str, int]:
    """Write reviewed production records, or explicitly flagged demo contract records."""
    generated = rejected = 0
    index: list[dict[str, str]] = []
    seen_templates: set[str] = set()
    for canonical_path in _canonical_paths(canonical_dir):
        canonical = _read_json(canonical_path)
        artifact_key = canonical_path.parent.relative_to(canonical_dir).as_posix() if not canonical_dir.is_file() else canonical_path.parent.name
        report_dir = output_dir / "records" / artifact_key
        template_id = canonical.get("template_id", "unknown")
        if template_id in seen_templates:
            rejected += 1
            _write_json(report_dir / "validation.json", {"runtime_valid": False, "errors": ["duplicate_template_id"]})
            continue
        seen_templates.add(template_id)
        try:
            merge_report = _read_json(canonical_path.parent / "merge.json")
            source_validation = _read_json(canonical_path.parent / "validation.json")
            if merge_report.get("status") != "merged" or not source_validation.get("canonical_schema_and_semantics_valid"):
                raise ValueError("Runtime records require a validated enriched canonical artifact")
            runtime = runtime_from_canonical(canonical, demo_unreviewed=demo_unreviewed)
            _write_json(report_dir / "runtime.json", runtime)
            _write_json(report_dir / "validation.json", {"runtime_valid": True, "errors": []})
            index.append({"artifact_key": artifact_key, "template_id": runtime["template_id"], "runtime_status": runtime["runtime_status"], "runtime": f"records/{artifact_key}/runtime.json"})
            generated += 1
        except Exception as exc:
            _write_json(report_dir / "validation.json", {"runtime_valid": False, "errors": [f"{type(exc).__name__}: {exc}"]})
            rejected += 1
    _write_json(output_dir / "index.json", index)
    summary = {"generated": generated, "rejected": rejected, "runtime_status": "demo_unreviewed" if demo_unreviewed else "eligible"}
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runtime"))
    parser.add_argument("--demo-unreviewed", action="store_true", help="Export enriched contract records for demo use only; never assert training eligibility.")
    args = parser.parse_args()
    print(json.dumps(generate_runtime_artifacts(args.canonical_dir, args.output_dir, demo_unreviewed=args.demo_unreviewed), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
