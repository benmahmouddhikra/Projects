"""Run the fixed 36-file deterministic pilot and write auditable artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from assemble import assemble_canonical
from extract import extract_document
from inventory import assert_expected_counts, inventory_corpus, sha256_file, write_jsonl
from ollama_extract import OllamaConfig, OllamaStructuredLabeler, installed_models
from validate import (
    build_review_record,
    schema_errors,
    validate_canonical,
    validate_extraction,
    validate_manifest,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def validate_fixed_pilot(pilot: dict[str, Any], repository_root: Path) -> None:
    entries = pilot.get("entries", [])
    expected = pilot.get("expected_file_count")
    if len(entries) != expected or expected != 36:
        raise ValueError(f"Pilot manifest must contain exactly 36 entries; got {len(entries)}")
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("Pilot manifest paths must be unique")
    if [entry["order"] for entry in entries] != list(range(1, 37)):
        raise ValueError("Pilot orders must be contiguous 1..36")
    missing = [path for path in paths if not (repository_root / Path(path)).is_file()]
    if missing:
        raise FileNotFoundError("Pilot paths are missing: " + ", ".join(missing))
    folders = {Path(path).parts[1] for path in paths}
    corpus_folders = {path.name for path in (repository_root / "templates").iterdir() if path.is_dir()}
    if folders != corpus_folders:
        raise ValueError(f"Pilot folder coverage differs: missing={sorted(corpus_folders-folders)}, extra={sorted(folders-corpus_folders)}")
    extensions = {Path(path).suffix.casefold() for path in paths}
    if extensions != {".pdf", ".doc", ".docx"}:
        raise ValueError(f"Pilot must cover PDF, DOC, DOCX; got {sorted(extensions)}")


def _source_files_unchanged(before: list[dict[str, Any]], repository_root: Path) -> tuple[bool, list[str]]:
    changed: list[str] = []
    for record in before:
        path = repository_root / Path(record["relative_path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            changed.append(record["relative_path"])
    return not changed, changed


def run_pilot(repository_root: Path, feature_root: Path, configured_model: str, num_ctx: int) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    feature_root = feature_root.resolve()
    templates_dir = repository_root / "templates"
    artifacts_dir = feature_root / "artifacts"
    pilot_artifacts_dir = artifacts_dir / "pilot"
    review_dir = artifacts_dir / "review-queue"
    taxonomy = _read_json(feature_root / "config" / "taxonomy.json")
    pilot_manifest = _read_json(feature_root / "config" / "pilot-manifest.json")
    validate_fixed_pilot(pilot_manifest, repository_root)

    manifest = inventory_corpus(templates_dir, repository_root)
    assert_expected_counts(manifest)
    manifest_errors = [error for record in manifest for error in validate_manifest(record)]
    if manifest_errors:
        raise ValueError("Manifest validation failed: " + "; ".join(manifest_errors[:10]))
    write_jsonl(manifest, artifacts_dir / "manifest.jsonl")
    by_path = {record["relative_path"]: record for record in manifest}
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        by_hash[record["sha256"]].append(record)

    available_models = installed_models()
    model_is_available = configured_model in available_models
    labeler: OllamaStructuredLabeler | None = None
    model_blocker: str | None = None
    if model_is_available:
        labeler = OllamaStructuredLabeler(
            feature_root / "schemas" / "labeling-response-v1.schema.json",
            feature_root / "prompts" / "extract-structure.txt",
            OllamaConfig(model_tag=configured_model, num_ctx=num_ctx),
        )
    else:
        model_blocker = (
            f"Configured model {configured_model!r} is not installed/reachable. "
            f"Installed Ollama models: {available_models or 'none'}."
        )

    selected_entries = pilot_manifest["entries"]
    processed_hashes: set[str] = set()
    index_entries: list[dict[str, Any]] = []
    extraction_statuses: Counter[str] = Counter()
    initial_kinds: Counter[str] = Counter()
    review_count = 0
    canonical_count = 0
    model_used_count = 0
    validation_failure_count = 0

    for entry in selected_entries:
        manifest_record = by_path[entry["path"]]
        digest = manifest_record["sha256"]
        if digest in processed_hashes:
            existing = next(item for item in index_entries if item["sha256"] == digest)
            existing["selected_paths"].append(entry["path"])
            existing["selection_reasons"].append(entry["selection_reason"])
            continue
        processed_hashes.add(digest)
        selected_group_paths = {candidate["path"] for candidate in selected_entries if by_path[candidate["path"]]["sha256"] == digest}
        members = [member for member in by_hash[digest] if member["relative_path"] in selected_group_paths]
        members.sort(key=lambda row: row["relative_path"].casefold())
        primary = next((member for member in members if member["exact_duplicate"]["is_primary"]), members[0])
        source_path = repository_root / Path(primary["relative_path"])
        extraction = extract_document(source_path, primary)
        extraction_errors = validate_extraction(extraction)
        extraction_statuses[extraction["extraction_status"]] += 1
        artifact_key = primary["exact_duplicate"]["group_id"] or primary["source_id"]
        artifact_dir = pilot_artifacts_dir / artifact_key
        _write_json(artifact_dir / "extraction.json", extraction)

        canonical = None
        canonical_errors: list[str] = []
        labeling = None
        model_status = "unavailable" if labeler is None else "failed"
        model_reason = model_blocker
        if extraction["status"] != "failed":
            if labeler is not None:
                try:
                    labeling = labeler.label(extraction, taxonomy)
                    _write_json(artifact_dir / "labeling-response.json", labeling)
                    model_status = "used"
                    model_reason = None
                    model_used_count += 1
                except Exception as exc:
                    model_status = "failed"
                    model_reason = f"{type(exc).__name__}: {exc}"
            canonical = assemble_canonical(
                extraction,
                members,
                taxonomy,
                labeling_response=labeling,
                model_status=model_status,
                model_tag=configured_model,
                model_reason=model_reason,
            )
            canonical_errors = validate_canonical(canonical, extraction)
            if canonical_errors:
                validation_failure_count += 1
            canonical_count += 1
            initial_kinds[canonical["document"]["initial_document_kind"]] += 1
            _write_json(artifact_dir / "canonical.json", canonical)
        validation_report = {
            "source_ids": [member["source_id"] for member in members],
            "extraction_schema_valid": not extraction_errors,
            "extraction_errors": extraction_errors,
            "canonical_schema_and_semantics_valid": canonical is not None and not canonical_errors,
            "canonical_errors": canonical_errors,
            "human_reviews_complete": False,
            "training_eligible": False,
        }
        _write_json(artifact_dir / "validation.json", validation_report)
        reasons = ["human_review_required"]
        reasons.extend(extraction_errors)
        reasons.extend(canonical_errors)
        if extraction["status"] == "failed":
            reasons.append(f"extraction_failed:{extraction['extraction_status']}")
        if model_status != "used":
            reasons.append(f"model_assistance_{model_status}")
        document_kind = canonical["document"]["document_kind"] if canonical else None
        if document_kind not in {None, "contract_template", "completed_contract"}:
            reasons.append("AMBIGUITY-023_non_contract_destination_unresolved")
        review = build_review_record(
            [member["source_id"] for member in members],
            template_id=canonical["template_id"] if canonical else None,
            reasons=reasons,
            document_kind=document_kind,
        )
        review_schema_errors = schema_errors(review, "review-record-v1.schema.json")
        if review_schema_errors:
            raise ValueError("Review schema validation failed: " + "; ".join(review_schema_errors))
        _write_json(review_dir / f"{review['review_record_id']}.json", review)
        review_count += 1
        index_entries.append({
            "artifact_key": artifact_key,
            "sha256": digest,
            "selected_paths": [entry["path"]],
            "selection_reasons": [entry["selection_reason"]],
            "source_ids": [member["source_id"] for member in members],
            "extraction_status": extraction["extraction_status"],
            "template_id": canonical["template_id"] if canonical else None,
            "initial_document_kind": canonical["document"]["initial_document_kind"] if canonical else None,
            "model_assistance": model_status,
            "review_record_id": review["review_record_id"],
            "canonical_valid": canonical is not None and not canonical_errors,
        })

    unchanged, changed_paths = _source_files_unchanged(manifest, repository_root)
    index_entries.sort(key=lambda item: min(pilot["order"] for pilot in selected_entries if pilot["path"] in item["selected_paths"]))
    pilot_index = {
        "pilot_version": pilot_manifest["pilot_version"],
        "selected_physical_files": 36,
        "processed_content_groups": len(index_entries),
        "entries": index_entries,
    }
    _write_json(pilot_artifacts_dir / "index.json", pilot_index)
    summary = {
        "pipeline_version": "1.0.0",
        "inventory": {"total": len(manifest), "pdf": 955, "doc": 297, "docx": 24},
        "exact_duplicate_groups": sum(1 for members in by_hash.values() if len(members) > 1),
        "pilot": {
            "selected_physical_files": 36,
            "processed_content_groups": len(index_entries),
            "canonical_records": canonical_count,
            "review_queue_records": review_count,
            "extraction_statuses": dict(sorted(extraction_statuses.items())),
            "initial_document_kinds": dict(sorted(initial_kinds.items())),
            "canonical_validation_failures": validation_failure_count,
            "human_reviewed_records": 0,
            "legally_approved_records": 0,
            "training_eligible_records": 0,
        },
        "ollama": {
            "configured_model": configured_model,
            "num_ctx": num_ctx,
            "temperature": 0,
            "concurrency": 1,
            "installed_models": available_models,
            "model_assisted_records": model_used_count,
            "blocker": model_blocker,
            "runtime_measurements": []
        },
        "source_integrity": {"all_templates_unchanged": unchanged, "changed_paths": changed_paths},
        "ambiguities": [
            "AMBIGUITY-023: non-contract knowledge-base/training destination remains unresolved.",
            "Agent 2 intake, Alpaca export, fine-tuning, embeddings, provider publication, and runtime integration remain outside this feature."
        ]
    }
    _write_json(artifacts_dir / "pilot-run-summary.json", summary)
    if not unchanged:
        raise RuntimeError("Source integrity failure: " + ", ".join(changed_paths))
    return summary


def main() -> int:
    feature_root = Path(__file__).resolve().parents[1]
    default_repository_root = feature_root
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=default_repository_root)
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--num-ctx", type=int, default=4096)
    args = parser.parse_args()
    summary = run_pilot(args.repository_root, feature_root, args.model, args.num_ctx)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
