"""Process the complete source corpus into auditable canonical demo candidates.

This runner deliberately keeps the fixed 36-file pilot unchanged. Exact duplicate
files are processed once while every physical source path remains attached to the
canonical record. Unreviewed outputs are never promoted to training-eligible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from assemble import assemble_canonical
from extract import extract_document, failed_extraction_record
from inventory import assert_expected_counts, inventory_corpus, sha256_file, write_jsonl
from normalize import sha256_text
from ollama_extract import OllamaConfig, OllamaStructuredLabeler, installed_models
from validate import (
    build_review_record,
    schema_errors,
    validate_canonical,
    validate_extraction,
    validate_manifest,
)


FULL_CORPUS_VERSION = "1.0.0"
CANDIDATE_DOCUMENT_KINDS = {"contract_template", "completed_contract"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _source_files_unchanged(
    before: list[dict[str, Any]], repository_root: Path
) -> tuple[bool, list[str]]:
    changed: list[str] = []
    for record in before:
        path = repository_root / Path(record["relative_path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            changed.append(record["relative_path"])
    return not changed, changed


def _content_groups(
    manifest: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]], dict[str, Any]]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest:
        by_hash[record["sha256"]].append(record)

    groups: list[tuple[str, list[dict[str, Any]], dict[str, Any]]] = []
    for digest, members in by_hash.items():
        members.sort(key=lambda row: row["relative_path"].casefold())
        primary = next(
            (member for member in members if member["exact_duplicate"]["is_primary"]),
            members[0],
        )
        artifact_key = primary["exact_duplicate"]["group_id"] or primary["source_id"]
        groups.append((artifact_key, members, primary))
    groups.sort(key=lambda group: group[2]["relative_path"].casefold())
    return groups


def _run_signature(
    *,
    digest: str,
    source_ids: list[str],
    configured_model: str,
    num_ctx: int,
    model_is_available: bool,
    use_model: bool,
    taxonomy: dict[str, Any],
    taxonomy_path: Path,
    prompt_path: Path,
) -> str:
    payload = {
        "full_corpus_version": FULL_CORPUS_VERSION,
        "source_sha256": digest,
        "source_ids": sorted(source_ids),
        "configured_model": configured_model,
        "num_ctx": num_ctx,
        "use_model": use_model,
        "model_is_available": model_is_available,
        "taxonomy_version": taxonomy["taxonomy_version"],
        "taxonomy_sha256": sha256_file(taxonomy_path),
        "prompt_sha256": sha256_file(prompt_path),
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(serialized)


def _resume_artifacts(
    artifact_dir: Path, signature: str, digest: str
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]] | None:
    state_path = artifact_dir / "state.json"
    extraction_path = artifact_dir / "extraction.json"
    if not state_path.is_file() or not extraction_path.is_file():
        return None
    try:
        state = _read_json(state_path)
        if state.get("status") != "completed" or state.get("run_signature") != signature:
            return None
        extraction = _read_json(extraction_path)
        if extraction.get("source", {}).get("sha256") != digest:
            return None
        canonical_path = artifact_dir / "canonical.json"
        canonical = (
            _read_json(canonical_path)
            if extraction.get("status") != "failed" and canonical_path.is_file()
            else None
        )
        if extraction.get("status") != "failed" and canonical is None:
            return None
        return extraction, canonical, state
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _candidate_decision(
    extraction: dict[str, Any] | None,
    extraction_errors: list[str],
    canonical: dict[str, Any] | None,
    canonical_errors: list[str],
    processing_error: str | None,
) -> tuple[bool, str]:
    if processing_error:
        return False, "pipeline_error"
    if extraction is None:
        return False, "missing_extraction"
    if extraction_errors:
        return False, "extraction_validation_failed"
    if extraction["extraction_status"] != "success":
        return False, f"extraction_{extraction['extraction_status']}"
    if canonical is None:
        return False, "canonical_not_generated"
    if canonical_errors:
        return False, "canonical_validation_failed"
    kind = canonical["document"]["document_kind"]
    if kind not in CANDIDATE_DOCUMENT_KINDS:
        return False, f"excluded_document_kind:{kind}"
    return True, "unreviewed_contract_demo_candidate"


def _extract_with_legacy_doc_timeout(
    source_path: Path,
    manifest_record: dict[str, Any],
    artifact_dir: Path,
    feature_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Isolate Word COM so one legacy DOC cannot block the corpus indefinitely."""
    if source_path.suffix.casefold() != ".doc":
        return extract_document(source_path, manifest_record)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    worker_output = artifact_dir / "extraction-worker.json"
    worker_manifest = artifact_dir / "manifest-worker.json"
    _write_json(worker_manifest, manifest_record)
    command = [
        sys.executable,
        str(feature_root / "src" / "extract.py"),
        str(source_path),
        "--output",
        str(worker_output),
        "--manifest-record",
        str(worker_manifest),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if worker_output.is_file():
            return _read_json(worker_output)
        return failed_extraction_record(
            source_path,
            manifest_record,
            warning=f"legacy_doc_worker_failed:exit_{completed.returncode}",
        )
    except subprocess.TimeoutExpired:
        return failed_extraction_record(
            source_path,
            manifest_record,
            warning=f"legacy_doc_extraction_timeout:{timeout_seconds}s",
        )
    finally:
        worker_output.unlink(missing_ok=True)
        worker_manifest.unlink(missing_ok=True)


def _review_reasons(
    entry: dict[str, Any], extraction_errors: list[str], canonical_errors: list[str]
) -> list[str]:
    reasons = ["human_review_required", *extraction_errors, *canonical_errors]
    if entry["extraction_status"] != "success":
        reasons.append(f"extraction_not_successful:{entry['extraction_status']}")
    if entry["model_assistance"] != "used":
        reasons.append(f"model_assistance_{entry['model_assistance']}")
    if entry["document_kind"] not in {None, *CANDIDATE_DOCUMENT_KINDS}:
        reasons.append("AMBIGUITY-023_non_contract_destination_unresolved")
    return sorted(set(reasons))


def run_full_corpus(
    repository_root: Path,
    feature_root: Path,
    configured_model: str,
    num_ctx: int,
    *,
    output_dir: Path | None = None,
    skip_review: bool = False,
    resume: bool = True,
    strict_counts: bool = True,
    use_model: bool = True,
    doc_timeout_seconds: int = 60,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    feature_root = feature_root.resolve()
    output_dir = (output_dir or feature_root / "artifacts" / "full").resolve()
    records_dir = output_dir / "records"
    review_dir = output_dir / "review-queue"
    templates_dir = repository_root / "templates"
    taxonomy_path = feature_root / "config" / "taxonomy.json"
    prompt_path = feature_root / "prompts" / "extract-structure.txt"
    taxonomy = _read_json(taxonomy_path)

    manifest = inventory_corpus(templates_dir, repository_root)
    if strict_counts:
        assert_expected_counts(manifest)
    manifest_errors = [error for record in manifest for error in validate_manifest(record)]
    if manifest_errors:
        raise ValueError("Manifest validation failed: " + "; ".join(manifest_errors[:10]))
    write_jsonl(manifest, output_dir / "manifest.jsonl")

    available_models = installed_models() if use_model else []
    model_is_available = use_model and configured_model in available_models
    labeler: OllamaStructuredLabeler | None = None
    if model_is_available:
        labeler = OllamaStructuredLabeler(
            feature_root / "schemas" / "labeling-response-v1.schema.json",
            prompt_path,
            OllamaConfig(model_tag=configured_model, num_ctx=num_ctx),
        )
        model_blocker = None
    elif not use_model:
        model_blocker = "Model assistance was explicitly disabled."
    else:
        model_blocker = (
            f"Configured model {configured_model!r} is not installed/reachable. "
            f"Installed Ollama models: {available_models or 'none'}."
        )

    groups = _content_groups(manifest)
    _write_json(
        output_dir / "run-config.json",
        {
            "schema_version": "full-corpus-run-config-v1",
            "full_corpus_version": FULL_CORPUS_VERSION,
            "physical_source_files": len(manifest),
            "content_groups": len(groups),
            "configured_model": configured_model,
            "num_ctx": num_ctx,
            "use_model": use_model,
            "model_is_available": model_is_available,
            "skip_review": skip_review,
            "resume": resume,
            "strict_counts": strict_counts,
            "doc_timeout_seconds": doc_timeout_seconds,
            "output_dir": str(output_dir),
        },
    )

    index_entries: list[dict[str, Any]] = []
    extraction_statuses: Counter[str] = Counter()
    document_kinds: Counter[str] = Counter()
    template_states: Counter[str] = Counter()
    model_statuses: Counter[str] = Counter()
    resumed_groups = 0
    processing_failures = 0

    for position, (artifact_key, members, primary) in enumerate(groups, 1):
        artifact_dir = records_dir / artifact_key
        source_path = repository_root / Path(primary["relative_path"])
        source_ids = [member["source_id"] for member in members]
        source_paths = [member["relative_path"] for member in members]
        digest = primary["sha256"]
        signature = _run_signature(
            digest=digest,
            source_ids=source_ids,
            configured_model=configured_model,
            num_ctx=num_ctx,
            model_is_available=model_is_available,
            use_model=use_model,
            taxonomy=taxonomy,
            taxonomy_path=taxonomy_path,
            prompt_path=prompt_path,
        )
        print(
            f"[{position}/{len(groups)}] {primary['relative_path']}",
            file=sys.stderr,
            flush=True,
        )

        resumed = False
        processing_error: str | None = None
        state: dict[str, Any] = {}
        loaded = _resume_artifacts(artifact_dir, signature, digest) if resume else None
        if loaded is not None:
            extraction, canonical, state = loaded
            resumed = True
            resumed_groups += 1
        else:
            extraction = None
            canonical = None
            model_status = "unavailable" if labeler is None else "failed"
            model_reason = model_blocker
            try:
                extraction = _extract_with_legacy_doc_timeout(
                    source_path,
                    primary,
                    artifact_dir,
                    feature_root,
                    doc_timeout_seconds,
                )
                _write_json(artifact_dir / "extraction.json", extraction)
                labeling = None
                if extraction["status"] != "failed" and labeler is not None:
                    try:
                        labeling = labeler.label(extraction, taxonomy)
                        _write_json(artifact_dir / "labeling-response.json", labeling)
                        model_status = "used"
                        model_reason = None
                    except Exception as exc:
                        model_status = "failed"
                        model_reason = f"{type(exc).__name__}: {exc}"
                if extraction["status"] != "failed":
                    canonical = assemble_canonical(
                        extraction,
                        members,
                        taxonomy,
                        labeling_response=labeling,
                        model_status=model_status,
                        model_tag=configured_model,
                        model_reason=model_reason,
                    )
                    _write_json(artifact_dir / "canonical.json", canonical)
                state = {
                    "schema_version": "full-corpus-record-state-v1",
                    "status": "completed",
                    "run_signature": signature,
                    "source_sha256": digest,
                    "source_ids": source_ids,
                    "model_status": model_status,
                    "model_reason": model_reason,
                }
                _write_json(artifact_dir / "state.json", state)
            except Exception as exc:
                processing_error = f"{type(exc).__name__}: {exc}"
                processing_failures += 1
                _write_json(
                    artifact_dir / "error.json",
                    {
                        "schema_version": "full-corpus-processing-error-v1",
                        "source_ids": source_ids,
                        "error": processing_error,
                    },
                )

        extraction_errors = validate_extraction(extraction) if extraction else []
        canonical_errors = (
            validate_canonical(canonical, extraction)
            if canonical is not None and extraction is not None
            else []
        )
        model_status = state.get(
            "model_status",
            canonical.get("model_assistance", {}).get("status", "not_run")
            if canonical
            else "not_run",
        )
        extraction_status = extraction["extraction_status"] if extraction else "pipeline_error"
        document_kind = canonical["document"]["document_kind"] if canonical else None
        template_state = canonical["document"]["template_state"] if canonical else None
        demo_candidate, decision_reason = _candidate_decision(
            extraction,
            extraction_errors,
            canonical,
            canonical_errors,
            processing_error,
        )
        validation_report = {
            "source_ids": source_ids,
            "extraction_schema_valid": extraction is not None and not extraction_errors,
            "extraction_errors": extraction_errors,
            "canonical_schema_and_semantics_valid": canonical is not None and not canonical_errors,
            "canonical_errors": canonical_errors,
            "human_reviews_complete": False,
            "training_eligible": False,
            "demo_candidate": demo_candidate,
            "demo_candidate_reason": decision_reason,
        }
        _write_json(artifact_dir / "validation.json", validation_report)

        entry = {
            "artifact_key": artifact_key,
            "sha256": digest,
            "source_ids": source_ids,
            "source_paths": source_paths,
            "primary_source_path": primary["relative_path"],
            "extraction_status": extraction_status,
            "template_id": canonical["template_id"] if canonical else None,
            "document_kind": document_kind,
            "template_state": template_state,
            "model_assistance": model_status,
            "canonical_valid": canonical is not None and not canonical_errors,
            "demo_candidate": demo_candidate,
            "decision_reason": decision_reason,
            "training_eligibility": (
                canonical["quality"]["training_eligibility"] if canonical else "excluded"
            ),
            "canonical_path": (
                f"records/{artifact_key}/canonical.json" if canonical else None
            ),
            "resumed": resumed,
            "processing_error": processing_error,
        }
        index_entries.append(entry)
        extraction_statuses[extraction_status] += 1
        document_kinds[document_kind or "none"] += 1
        template_states[template_state or "none"] += 1
        model_statuses[model_status] += 1

        if not skip_review:
            review = build_review_record(
                source_ids,
                template_id=canonical["template_id"] if canonical else None,
                reasons=_review_reasons(entry, extraction_errors, canonical_errors),
                document_kind=document_kind,
            )
            review_schema_errors = schema_errors(review, "review-record-v1.schema.json")
            if review_schema_errors:
                raise ValueError(
                    "Review schema validation failed: " + "; ".join(review_schema_errors)
                )
            _write_json(review_dir / f"{review['review_record_id']}.json", review)
            entry["review_record_id"] = review["review_record_id"]

    candidates = [entry for entry in index_entries if entry["demo_candidate"]]
    excluded = [entry for entry in index_entries if not entry["demo_candidate"]]
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": "full-corpus-index-v1",
            "full_corpus_version": FULL_CORPUS_VERSION,
            "physical_source_files": len(manifest),
            "processed_content_groups": len(index_entries),
            "demo_candidate_groups": len(candidates),
            "excluded_groups": len(excluded),
            "review_queue_generated": not skip_review,
            "entries": index_entries,
        },
    )
    _write_jsonl(output_dir / "demo-candidates.jsonl", candidates)
    _write_jsonl(output_dir / "excluded.jsonl", excluded)

    unchanged, changed_paths = _source_files_unchanged(manifest, repository_root)
    summary = {
        "schema_version": "full-corpus-run-summary-v1",
        "full_corpus_version": FULL_CORPUS_VERSION,
        "inventory": {
            "physical_source_files": len(manifest),
            "content_groups": len(groups),
            "exact_duplicate_groups": sum(
                1 for _, members, _ in groups if len(members) > 1
            ),
        },
        "results": {
            "processed_content_groups": len(index_entries),
            "resumed_content_groups": resumed_groups,
            "processing_failures": processing_failures,
            "demo_candidate_groups": len(candidates),
            "excluded_groups": len(excluded),
            "extraction_statuses": dict(sorted(extraction_statuses.items())),
            "document_kinds": dict(sorted(document_kinds.items())),
            "template_states": dict(sorted(template_states.items())),
            "model_statuses": dict(sorted(model_statuses.items())),
            "human_reviewed_records": 0,
            "training_eligible_records": 0,
        },
        "ollama": {
            "configured_model": configured_model,
            "num_ctx": num_ctx,
            "temperature": 0,
            "concurrency": 1,
            "installed_models": available_models,
            "blocker": model_blocker,
        },
        "review": {
            "skipped": skip_review,
            "queue_records": 0 if skip_review else len(index_entries),
        },
        "source_integrity": {
            "all_templates_unchanged": unchanged,
            "changed_paths": changed_paths,
        },
        "output_files": {
            "index": "index.json",
            "demo_candidates": "demo-candidates.jsonl",
            "excluded": "excluded.jsonl",
        },
        "ambiguities": [
            "AMBIGUITY: Unreviewed demo candidates are not approved Alpaca or fine-tuning records.",
            "AMBIGUITY-023: non-contract knowledge-base/training destination remains unresolved.",
        ],
    }
    _write_json(output_dir / "run-summary.json", summary)
    if not unchanged:
        raise RuntimeError("Source integrity failure: " + ", ".join(changed_paths))
    return summary


def main() -> int:
    feature_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Extract and classify every PDF, DOC, and DOCX source in templates/."
    )
    parser.add_argument("--repository-root", type=Path, default=feature_root)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default="qwen3.5:4b-q4_K_M")
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Do not create review records; outputs remain unreviewed demo candidates.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every content group instead of resuming compatible completed groups.",
    )
    parser.add_argument(
        "--allow-count-drift",
        action="store_true",
        help="Allow a corpus whose extension counts differ from the recorded 1,276-file corpus.",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Use only deterministic extraction and filename routing.",
    )
    parser.add_argument(
        "--doc-timeout",
        type=int,
        default=60,
        help="Maximum seconds for one legacy .doc extraction before excluding it and continuing.",
    )
    args = parser.parse_args()
    summary = run_full_corpus(
        args.repository_root,
        feature_root,
        args.model,
        args.num_ctx,
        output_dir=args.output_dir,
        skip_review=args.skip_review,
        resume=not args.force,
        strict_counts=not args.allow_count_drift,
        use_model=not args.no_model,
        doc_timeout_seconds=args.doc_timeout,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
