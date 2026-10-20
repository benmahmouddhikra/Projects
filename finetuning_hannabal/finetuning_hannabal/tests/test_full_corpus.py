from __future__ import annotations

import json
import shutil
import subprocess

from conftest import write_minimal_docx
from full_corpus import (
    _candidate_decision,
    _extract_with_legacy_doc_timeout,
    run_full_corpus,
)
from inventory import MEDIA_TYPES, sha256_file
from normalize import stable_id
from validate import validate_extraction


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_full_corpus_deduplicates_filters_skips_review_and_resumes(
    tmp_path, feature_root
):
    repository_root = tmp_path / "repository"
    templates = repository_root / "templates" / "test"
    templates.mkdir(parents=True)

    blank = templates / "blank-contract.docx"
    duplicate = templates / "copy-agreement.docx"
    completed = templates / "service-contract.docx"
    policy = templates / "privacy-policy.docx"
    write_minimal_docx(blank, "CONTRACT\nArticle 1\nClient: [CLIENT_NAME]")
    shutil.copyfile(blank, duplicate)
    write_minimal_docx(completed, "SERVICE CONTRACT\nArticle 1\nFilled client name")
    write_minimal_docx(policy, "PRIVACY POLICY\nData processing rules")
    source_hashes = {path: sha256_file(path) for path in templates.iterdir()}

    output_dir = tmp_path / "full-output"
    summary = run_full_corpus(
        repository_root,
        feature_root,
        "unused-model",
        4096,
        output_dir=output_dir,
        skip_review=True,
        resume=True,
        strict_counts=False,
        use_model=False,
    )

    assert summary["inventory"] == {
        "physical_source_files": 4,
        "content_groups": 3,
        "exact_duplicate_groups": 1,
    }
    assert summary["results"]["demo_candidate_groups"] == 2
    assert summary["results"]["excluded_groups"] == 1
    assert summary["results"]["training_eligible_records"] == 0
    assert summary["review"] == {"skipped": True, "queue_records": 0}
    assert not (output_dir / "review-queue").exists()

    candidates = _jsonl(output_dir / "demo-candidates.jsonl")
    excluded = _jsonl(output_dir / "excluded.jsonl")
    assert {row["document_kind"] for row in candidates} == {
        "contract_template",
        "completed_contract",
    }
    assert all(row["training_eligibility"] == "review_required" for row in candidates)
    assert excluded[0]["document_kind"] == "policy"
    duplicate_group = next(row for row in candidates if len(row["source_paths"]) == 2)
    assert sorted(duplicate_group["source_paths"]) == [
        "templates/test/blank-contract.docx",
        "templates/test/copy-agreement.docx",
    ]
    assert len(_jsonl(output_dir / "manifest.jsonl")) == 4
    assert all(sha256_file(path) == digest for path, digest in source_hashes.items())

    resumed = run_full_corpus(
        repository_root,
        feature_root,
        "unused-model",
        4096,
        output_dir=output_dir,
        skip_review=True,
        resume=True,
        strict_counts=False,
        use_model=False,
    )
    assert resumed["results"]["resumed_content_groups"] == 3
    assert all(row["resumed"] for row in _jsonl(output_dir / "demo-candidates.jsonl"))


def test_mixed_contract_is_a_demo_candidate_but_not_approved():
    extraction = {"extraction_status": "success"}
    canonical = {
        "document": {"document_kind": "contract_template", "template_state": "mixed"}
    }
    included, reason = _candidate_decision(extraction, [], canonical, [], None)
    assert included is True
    assert reason == "unreviewed_contract_demo_candidate"


def test_legacy_doc_timeout_becomes_failed_extraction(monkeypatch, tmp_path, feature_root):
    path = tmp_path / "stuck-contract.doc"
    path.write_bytes(b"legacy-doc-fixture")
    relative_path = "templates/test/stuck-contract.doc"
    manifest = {
        "source_id": stable_id("src", relative_path),
        "relative_path": relative_path,
        "sha256": sha256_file(path),
        "media_type": MEDIA_TYPES[".doc"],
        "size_bytes": path.stat().st_size,
    }

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    extraction = _extract_with_legacy_doc_timeout(
        path, manifest, tmp_path / "artifact", feature_root, 7
    )
    assert extraction["status"] == "failed"
    assert extraction["extraction_status"] == "unreadable"
    assert extraction["warnings"] == ["legacy_doc_extraction_timeout:7s"]
    assert extraction["source"]["source_id"] == manifest["source_id"]
    assert validate_extraction(extraction) == []
