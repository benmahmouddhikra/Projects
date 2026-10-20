"""Run Agent 2 demo preparation through strict three-field Alpaca export."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from agent2_intake import (
    FACTS_FILE,
    VALIDATION_FILE as INTAKE_VALIDATION_FILE,
    adapt_record_directory,
    contains_forbidden_model_identifier,
    sha256_file,
)
from contract_target_builder import (
    EVIDENCE_FILE,
    TARGET_FILE,
    TARGET_VALIDATION_FILE,
    build_record_target,
)
from normalize import sha256_text, stable_id
from qwen_enrichment import _read_json, _write_json
from runtime_templates import validate_runtime
from validate import load_schema


ALPACA_SCHEMA = "alpaca-record-v1.schema.json"
MANIFEST_SCHEMA = "alpaca-export-manifest-v1.schema.json"
INSTRUCTION = (
    "Generate the requested contract using only the supplied user information and "
    "retrieved documentary evidence. Do not add unsupported legal content."
)
EXPECTED_DEMO_RECORDS = 90
PROVIDER_OR_WORKFLOW_KEYS = {
    "session_id", "user_id", "embedding", "embeddings", "vector", "vectors",
    "provider", "model_tag", "review", "reviewer", "source_path", "raw_input_ref",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_errors(value: dict[str, Any], schema_name: str) -> list[str]:
    validator = Draft202012Validator(load_schema(schema_name))
    errors = sorted(validator.iter_errors(value), key=lambda error: [str(item) for item in error.absolute_path])
    return [
        f"schema:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}"
        for error in errors
    ]


def _disallowed_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).casefold() in PROVIDER_OR_WORKFLOW_KEYS:
                found.append(child)
            found.extend(_disallowed_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_disallowed_paths(item, f"{path}[{index}]"))
    return found


def validate_alpaca_row(row: dict[str, Any]) -> list[str]:
    errors = _schema_errors(row, ALPACA_SCHEMA)
    parsed: dict[str, Any] | None = None
    try:
        parsed_value = json.loads(row.get("input", ""))
        if not isinstance(parsed_value, dict):
            errors.append("alpaca:input must serialize a JSON object")
        else:
            parsed = parsed_value
    except (json.JSONDecodeError, TypeError):
        errors.append("alpaca:input is not valid JSON")
    if contains_forbidden_model_identifier(row):
        errors.append("privacy:Alpaca row contains session_id or user_id")
    if parsed is not None:
        paths = _disallowed_paths(parsed)
        if paths:
            errors.append("alpaca:model input contains provider, workflow, embedding, vector, or source-path keys: " + ", ".join(paths))
        if set(parsed) != {"agent2_generation_facts", "retrieval"}:
            errors.append("alpaca:model input must contain only agent2_generation_facts and retrieval")
        retrieval = parsed.get("retrieval")
        if not isinstance(retrieval, dict) or set(retrieval) != {"template", "clauses"}:
            errors.append("alpaca:retrieval must contain only template and clauses")
    return sorted(set(errors))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = "".join(_canonical_json(row) + "\n" for row in rows)
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _rejection(artifact_key: str, template_id: str | None, stage: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "alpaca-rejection-v1",
        "artifact_key": artifact_key,
        "template_id": template_id,
        "stage": stage,
        "reasons": sorted(set(reasons)),
    }


def _read_index(runtime_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    index_path = runtime_dir / "index.json"
    if not index_path.is_file():
        return [], ["audit:runtime index.json is missing"]
    try:
        index = _read_json(index_path)
    except Exception as exc:
        return [], [f"audit:runtime index is malformed: {type(exc).__name__}: {exc}"]
    if not isinstance(index, list):
        return [], ["audit:runtime index must be an array"]
    template_ids = [entry.get("template_id") for entry in index]
    artifact_keys = [entry.get("artifact_key") for entry in index]
    if len(template_ids) != len(set(template_ids)):
        errors.append("audit:runtime index contains duplicate template IDs")
    if len(artifact_keys) != len(set(artifact_keys)):
        errors.append("audit:runtime index contains duplicate artifact keys")
    indexed_dirs = {str(key) for key in artifact_keys}
    filesystem_dirs = {
        path.name for path in (runtime_dir / "records").iterdir()
        if path.is_dir() and (path / "runtime.json").is_file()
    } if (runtime_dir / "records").is_dir() else set()
    if indexed_dirs != filesystem_dirs:
        errors.append(
            "audit:runtime index/filesystem mismatch; "
            f"index_only={sorted(indexed_dirs - filesystem_dirs)}, filesystem_only={sorted(filesystem_dirs - indexed_dirs)}"
        )
    return index, errors


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _review_status(canonical: dict[str, Any]) -> dict[str, str]:
    quality = canonical.get("quality", {})
    return {
        "structure": quality.get("structure_review_status", "missing"),
        "legal": quality.get("legal_review_status", "missing"),
        "privacy": quality.get("privacy_review_status", "missing"),
        "sanitization": quality.get("sanitization_status", "missing"),
    }


def _target_review_errors(record_dir: Path, target_report: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    """Validate the minimal production gate without creating or approving reviews."""
    review_path = record_dir / "agent2-target-review.json"
    if not review_path.is_file():
        return ["production target review is missing"]
    try:
        review = _read_json(review_path)
    except Exception as exc:
        return [f"production target review is malformed: {type(exc).__name__}: {exc}"]
    errors: list[str] = []
    if review.get("target_sha256") != target_report.get("hashes", {}).get("target_sha256"):
        errors.append("production target review does not reference the current target hash")
    for field in ("structure_review_status", "legal_review_status"):
        if review.get(field) != "approved":
            errors.append(f"production target review requires {field}=approved")
    if runtime.get("document", {}).get("document_kind") == "completed_contract":
        if review.get("privacy_review_status") != "approved":
            errors.append("completed-contract production target review requires privacy_review_status=approved")
        if review.get("sanitization_status") != "completed":
            errors.append("completed-contract production target review requires sanitization_status=completed")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        errors.append("production target review requires a named reviewer")
    return errors


def _manifest_entry(
    *,
    row: dict[str, Any],
    row_index: int,
    artifact_key: str,
    runtime: dict[str, Any],
    canonical: dict[str, Any],
    record_dir: Path,
    runtime_dir: Path,
    output_dir: Path,
    mode: str,
) -> dict[str, Any]:
    target_path = record_dir / TARGET_FILE
    facts_path = record_dir / FACTS_FILE
    evidence_path = record_dir / EVIDENCE_FILE
    intake_validation_path = record_dir / INTAKE_VALIDATION_FILE
    target_validation_path = record_dir / TARGET_VALIDATION_FILE
    source_ids = runtime["canonical_ref"]["source_ids"]
    clause_ids = sorted({
        clause["clause_id"]
        for section in runtime.get("sections", [])
        for clause in section.get("clauses", [])
    })
    row_hash = sha256_text(_canonical_json(row))
    entry = {
        "schema_version": "alpaca-export-manifest-v1",
        "row_id": stable_id("alpaca", runtime["template_id"], row_hash),
        "row_index": row_index,
        "status": "demo_unreviewed" if mode == "demo" else "eligible",
        "training_eligible": mode == "production",
        "artifact_key": artifact_key,
        "template_id": runtime["template_id"],
        "source_ids": source_ids,
        "clause_ids": clause_ids,
        "jurisdictions": [item.get("country_code") for item in runtime.get("jurisdictions", [])],
        "languages": runtime.get("document", {}).get("languages", []),
        "paths": {
            "raw_agent1_input": _relative(record_dir / "agent1-input.json", runtime_dir),
            "runtime": _relative(record_dir / "runtime.json", runtime_dir),
            "generation_facts": _relative(facts_path, runtime_dir),
            "generation_validation": _relative(intake_validation_path, runtime_dir),
            "retrieved_evidence": _relative(evidence_path, runtime_dir),
            "contract_target": _relative(target_path, runtime_dir),
            "contract_target_validation": _relative(target_validation_path, runtime_dir),
            "alpaca_dataset": _relative(output_dir / "alpaca.jsonl", output_dir.parent),
        },
        "hashes": {
            "row_sha256": row_hash,
            "agent1_input_sha256": sha256_file(record_dir / "agent1-input.json"),
            "runtime_sha256": sha256_file(record_dir / "runtime.json"),
            "generation_facts_sha256": sha256_file(facts_path),
            "retrieved_evidence_sha256": sha256_file(evidence_path),
            "contract_target_sha256": sha256_file(target_path),
        },
        "review": _review_status(canonical),
    }
    return entry


def export_alpaca_dataset(
    runtime_dir: Path,
    output_dir: Path,
    *,
    canonical_dir: Path,
    extraction_dir: Path,
    mode: str = "demo",
    expected_records: int | None = None,
) -> dict[str, Any]:
    """Audit and process every indexed record, then write deterministic JSONL."""
    if mode not in {"demo", "production"}:
        raise ValueError("mode must be demo or production")
    if mode == "demo" and "demo" not in output_dir.name.casefold():
        raise ValueError("Demo exports require an output directory name containing 'demo'")
    if mode == "production" and "demo" in output_dir.name.casefold():
        raise ValueError("Production exports must not use a directory name containing 'demo'")

    index, audit_errors = _read_index(runtime_dir)
    expected = expected_records
    if expected is not None and len(index) != expected:
        audit_errors.append(f"audit:expected {expected} runtime records, found {len(index)}")
    upstream_summary_path = runtime_dir / "agent1-input-summary.json"
    audit_warnings: list[str] = []
    upstream_summary_audit: dict[str, Any] | None = None
    if not upstream_summary_path.is_file():
        audit_warnings.append("audit:agent1-input-summary.json is not present; per-record Agent 1 validations were audited directly")
    else:
        try:
            upstream_summary = _read_json(upstream_summary_path)
            upstream_summary_audit = {
                key: upstream_summary.get(key)
                for key in (
                    "runtime_records", "blank_templates", "completed_contracts", "generated",
                    "blocked", "qwen_failed", "status", "model",
                )
            }
            if upstream_summary.get("runtime_records") != len(index):
                audit_errors.append("audit:Agent 1 summary runtime_records does not match runtime index")
            accounted = sum(
                int(upstream_summary.get(key, 0) or 0)
                for key in ("generated", "blocked", "qwen_failed")
            )
            if accounted != len(index):
                audit_errors.append("audit:Agent 1 summary outcome counts do not reconcile to runtime index")
        except Exception as exc:
            audit_errors.append(f"audit:agent1-input-summary.json is malformed: {type(exc).__name__}: {exc}")

    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    jurisdiction_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    agent1_valid_count = 0

    for entry in index:
        artifact_key = str(entry.get("artifact_key", "unknown"))
        template_id = entry.get("template_id")
        record_dir = runtime_dir / "records" / artifact_key
        runtime_path = record_dir / "runtime.json"
        upstream_runtime_validation_path = record_dir / "validation.json"
        agent1_path = record_dir / "agent1-input.json"
        agent1_validation_path = record_dir / "agent1-input-validation.json"

        missing = [
            path.name for path in (runtime_path, upstream_runtime_validation_path, agent1_path, agent1_validation_path)
            if not path.is_file()
        ]
        if missing:
            rejected.append(_rejection(artifact_key, template_id, "stage_0_upstream_audit", ["missing required file: " + name for name in missing]))
            continue
        try:
            runtime = _read_json(runtime_path)
            upstream_runtime_validation = _read_json(upstream_runtime_validation_path)
            upstream_agent1_validation = _read_json(agent1_validation_path)
        except Exception as exc:
            rejected.append(_rejection(artifact_key, template_id, "stage_0_upstream_audit", [f"malformed JSON: {type(exc).__name__}: {exc}"]))
            continue

        runtime_errors = validate_runtime(runtime)
        if runtime.get("template_id") != template_id:
            runtime_errors.append("identity:index template_id does not match runtime")
        if entry.get("runtime_status") != runtime.get("runtime_status"):
            runtime_errors.append("identity:index runtime_status does not match runtime")
        if entry.get("runtime") != f"records/{artifact_key}/runtime.json":
            runtime_errors.append("identity:index runtime path does not match artifact key")
        if not upstream_runtime_validation.get("runtime_valid") or upstream_runtime_validation.get("errors"):
            runtime_errors.append("upstream:runtime validation did not pass")
        if upstream_agent1_validation.get("status") in {"blocked", "qwen_failed"} or upstream_agent1_validation.get("errors"):
            runtime_errors.append("upstream:Agent 1 validation did not pass")
        if runtime.get("runtime_status") == "demo_unreviewed" and upstream_agent1_validation.get("status") != "demo_unreviewed":
            runtime_errors.append("identity:demo runtime and Agent 1 validation statuses do not agree")
        if runtime_errors:
            rejected.append(_rejection(artifact_key, template_id, "stage_0_upstream_audit", runtime_errors))
            continue

        kind_counts[runtime["document"]["document_kind"]] += 1
        family_counts[runtime["document"]["contract_family"]] += 1
        language_counts.update(runtime["document"].get("languages", []))
        jurisdiction_counts.update(str(item.get("country_code")) for item in runtime.get("jurisdictions", []))
        agent1_valid_count += 1

        try:
            facts, intake_report = adapt_record_directory(record_dir, artifact_key=artifact_key)
        except Exception as exc:
            rejected.append(_rejection(artifact_key, template_id, "stage_1_agent2_intake", [f"{type(exc).__name__}: {exc}"]))
            continue
        if facts is None or not intake_report.get("valid"):
            rejected.append(_rejection(artifact_key, template_id, "stage_1_agent2_intake", intake_report.get("errors", ["intake validation failed"])))
            continue

        canonical_path = canonical_dir / artifact_key / "canonical.json"
        extraction_path = extraction_dir / artifact_key / "extraction.json"
        missing_evidence = [path.as_posix() for path in (canonical_path, extraction_path) if not path.is_file()]
        if missing_evidence:
            rejected.append(_rejection(artifact_key, template_id, "stage_2_retrieved_evidence", ["missing evidence artifact: " + path for path in missing_evidence]))
            continue
        try:
            target, target_report, evidence = build_record_target(record_dir, canonical_path, extraction_path)
            canonical = _read_json(canonical_path)
        except Exception as exc:
            rejected.append(_rejection(artifact_key, template_id, "stage_3_target_builder", [f"{type(exc).__name__}: {exc}"]))
            continue
        if target is None or evidence is None or not target_report.get("valid_for_demo_export"):
            rejected.append(_rejection(artifact_key, template_id, "stage_4_contract_validation", target_report.get("errors", ["target validation failed"])))
            continue
        if mode == "production":
            review_errors = _target_review_errors(record_dir, target_report, runtime)
            if not target_report.get("training_eligible"):
                review_errors.append("production export requires approved source structure, legal, privacy/sanitization when applicable, grounding, and training eligibility")
            if review_errors:
                rejected.append(_rejection(artifact_key, template_id, "stage_5_review_eligibility", review_errors))
                continue

        model_input = {
            "agent2_generation_facts": facts,
            "retrieval": {"template": evidence["template"], "clauses": evidence["clauses"]},
        }
        row = {
            "instruction": INSTRUCTION,
            "input": _canonical_json(model_input),
            "output": target,
        }
        row_errors = validate_alpaca_row(row)
        if row_errors:
            rejected.append(_rejection(artifact_key, template_id, "stage_6_alpaca_export", row_errors))
            continue
        manifest = _manifest_entry(
            row=row, row_index=len(rows), artifact_key=artifact_key, runtime=runtime,
            canonical=canonical, record_dir=record_dir, runtime_dir=runtime_dir,
            output_dir=output_dir, mode=mode,
        )
        manifest_errors = _schema_errors(manifest, MANIFEST_SCHEMA)
        if manifest_errors:
            rejected.append(_rejection(artifact_key, template_id, "stage_6_alpaca_export", manifest_errors))
            continue
        rows.append(row)
        manifests.append(manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "alpaca.jsonl", rows)
    _write_jsonl(output_dir / "manifest.jsonl", manifests)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    summary = {
        "schema_version": "alpaca-export-summary-v1",
        "mode": mode,
        "status": "demo_unreviewed" if mode == "demo" else "eligible",
        "training_eligible": mode == "production" and bool(rows),
        "expected_runtime_records": expected,
        "runtime_records_audited": len(index),
        "agent1_valid_records": agent1_valid_count,
        "exported_rows": len(rows),
        "rejected_records": len(rejected),
        "audit_errors": sorted(set(audit_errors)),
        "audit_warnings": sorted(set(audit_warnings)),
        "upstream_agent1_summary": upstream_summary_audit,
        "distributions": {
            "document_kind": dict(sorted(kind_counts.items())),
            "contract_family": dict(sorted(family_counts.items())),
            "language": dict(sorted(language_counts.items())),
            "jurisdiction": dict(sorted(jurisdiction_counts.items())),
        },
        "minimum_eligible_examples_required": 200,
        "eligible_examples": len(rows) if mode == "production" else 0,
        "minimum_volume_met": mode == "production" and len(rows) >= 200,
        "kpi_claims": [],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("artifacts/runtime-demo"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/alpaca-demo"))
    parser.add_argument("--canonical-dir", type=Path, default=Path("artifacts/enriched-canonical/records"))
    parser.add_argument("--extraction-dir", type=Path, default=Path("artifacts/full/records"))
    parser.add_argument("--mode", choices=("demo", "production"), default="demo")
    parser.add_argument("--expected-records", type=int, default=EXPECTED_DEMO_RECORDS)
    args = parser.parse_args()
    expected = args.expected_records if args.expected_records > 0 else None
    summary = export_alpaca_dataset(
        args.runtime_dir,
        args.output_dir,
        canonical_dir=args.canonical_dir,
        extraction_dir=args.extraction_dir,
        mode=args.mode,
        expected_records=expected,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
