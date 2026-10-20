from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from review_workspace import ReviewError, ReviewWorkspace


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def workspace(repository_root: Path, tmp_path: Path) -> ReviewWorkspace:
    value = ReviewWorkspace(repository_root, workspace_root=tmp_path / "review-workspace")
    assert value.initialize() == 35
    assert value.initialize() == 0
    return value


def find_record(workspace: ReviewWorkspace, kind: str) -> str:
    return next(row["review_record_id"] for row in workspace.list_records() if row["document_kind"] == kind)


def test_all_pilot_groups_duplicate_encrypted_sampling_and_resume(workspace: ReviewWorkspace):
    rows = workspace.list_records()
    assert len(rows) == 35
    assert sum(row["duplicate_count"] == 2 for row in rows) == 1
    encrypted = next(row for row in rows if row["extraction_status"] == "encrypted")
    assert encrypted["review_coverage"] == "automated_triage"
    assert encrypted["final_disposition"] == "excluded"
    assert any(row["review_coverage"] == "sampled_audit" for row in rows)
    assert all(
        row["review_coverage"] == "full"
        for row in rows if row["document_kind"] in {"contract_template", "completed_contract"}
    )
    state_path = workspace.records_root / rows[0]["review_record_id"] / "state.json"
    before = state_path.read_bytes()
    assert workspace.initialize() == 0
    assert state_path.read_bytes() == before


def test_invalid_corrections_do_not_mutate_workspace(workspace: ReviewWorkspace):
    review_id = find_record(workspace, "contract_template")
    state_path, canonical_path, revisions = workspace._paths(review_id)
    before = (state_path.read_bytes(), canonical_path.read_bytes(), list(revisions.glob("*.json")))
    with pytest.raises(ReviewError, match="Unsupported correction"):
        workspace.apply_correction(review_id, "Reviewer", {"clause_text": "injected legal text"})
    with pytest.raises(ReviewError, match="unknown variable"):
        state, canonical, _ = workspace.load(review_id)
        clause = next(clause for section in canonical["sections"] for clause in section["clauses"])
        workspace.apply_correction(review_id, "Reviewer", {
            "clauses": [{"clause_id": clause["clause_id"], "variable_ids": ["missing_variable"]}],
        })
    after = (state_path.read_bytes(), canonical_path.read_bytes(), list(revisions.glob("*.json")))
    assert after == before


def test_atomic_revision_and_unicode_round_trip(workspace: ReviewWorkspace):
    review_id = find_record(workspace, "unknown")
    state = workspace.decide(
        review_id, "المراجعة Élodie", "classification", "approved",
        "مراجعة conforme à l’évidence",
    )
    assert state["revision"] == 1
    revision_path = workspace.records_root / review_id / "revisions" / "000001.json"
    assert revision_path.is_file()
    revision = revision_path.read_text(encoding="utf-8")
    assert "المراجعة Élodie" in revision
    reloaded, _, _ = workspace.load(review_id)
    assert reloaded["decisions"]["classification"]["notes"] == "مراجعة conforme à l’évidence"
    assert reloaded["revision_history"][0]["action"] == "decision:classification:approved"


def _sanitizable_occurrence(canonical: dict, extraction: dict) -> tuple[str, int, int]:
    blocks = {block["block_id"]: block for block in extraction["blocks"]}
    occupied = [
        (span["block_id"], span["normalized_start"], span["normalized_end"])
        for variable in canonical["variables"] for span in variable["occurrences"]
    ]
    containers = [
        clause["source_span"] for section in canonical["sections"] for clause in section["clauses"]
    ] + [item["source_span"] for item in canonical["unparsed_source_content"]]
    for span in containers:
        text = blocks[span["block_id"]]["normalized_text"][span["normalized_start"]:span["normalized_end"]]
        for match in re.finditer(r"\w{3,}", text, re.UNICODE):
            start = span["normalized_start"] + match.start()
            end = span["normalized_start"] + match.end()
            if not any(block == span["block_id"] and not (end <= left or start >= right) for block, left, right in occupied):
                return span["block_id"], start, end
    raise AssertionError("No sanitizable occurrence found")


def test_completed_contract_requires_sanitization_then_becomes_eligible(workspace: ReviewWorkspace):
    review_id = find_record(workspace, "completed_contract")
    state, canonical, extraction = workspace.load(review_id)
    with pytest.raises(ReviewError, match="Required approvals are incomplete"):
        workspace.finalize(review_id, "Reviewer", "approved", "Premature approval")
    with pytest.raises(ReviewError, match="guarded final approval"):
        workspace.apply_correction(review_id, "Reviewer", {"document": {"document_kind": "contract_template"}})
    block_id, start, end = _sanitizable_occurrence(canonical, extraction)
    workspace.sanitize(
        review_id, "Reviewer", block_id=block_id, start=start, end=end,
        variable_id="private_value", value_type="text",
    )
    state, canonical, _ = workspace.load(review_id)
    assert "[PRIVATE_VALUE]" in "\n".join(
        [clause["text"] for section in canonical["sections"] for clause in section["clauses"]]
        + [item["text"] for item in canonical["unparsed_source_content"]]
    )
    for category in state["required_reviews"]:
        workspace.decide(review_id, "Reviewer", category, "approved", f"Approved {category} evidence")
    final = workspace.finalize(review_id, "Reviewer", "approved", "Sanitized source is a blank reusable template")
    _, canonical, _ = workspace.load(review_id)
    assert final["training_eligibility"] == "eligible"
    assert canonical["document"]["initial_document_kind"] == "completed_contract"
    assert canonical["document"]["document_kind"] == "contract_template"
    assert canonical["quality"]["training_eligibility"] == "eligible"
    assert canonical["document"]["reclassification_history"]


def test_blank_template_full_review_becomes_eligible(workspace: ReviewWorkspace):
    review_id = find_record(workspace, "contract_template")
    state, canonical, _ = workspace.load(review_id)
    assert canonical["document"]["template_state"] == "blank"
    assert set(state["required_reviews"]) == {"extraction", "classification", "structure", "legal"}
    for category in state["required_reviews"]:
        workspace.decide(review_id, "Reviewer", category, "approved", f"Approved {category} evidence")
    final = workspace.finalize(review_id, "Reviewer", "approved", "Blank reusable template approved")
    _, canonical, _ = workspace.load(review_id)
    assert final["training_eligibility"] == "eligible"
    assert canonical["quality"]["training_eligibility"] == "eligible"


def test_non_contract_review_never_grants_template_eligibility(workspace: ReviewWorkspace):
    review_id = find_record(workspace, "policy")
    state, _, _ = workspace.load(review_id)
    for category in state["required_reviews"]:
        workspace.decide(review_id, "Reviewer", category, "approved", f"Audit approved: {category}")
    final = workspace.finalize(review_id, "Reviewer", "approved", "Truthful policy classification confirmed")
    _, canonical, _ = workspace.load(review_id)
    assert final["training_eligibility"] == "excluded"
    assert canonical["document"]["document_kind"] == "policy"
    assert canonical["quality"]["training_eligibility"] == "excluded"


def test_source_and_generated_artifact_hashes_remain_unchanged(workspace: ReviewWorkspace, repository_root: Path):
    protected = [path for path in (repository_root / "artifacts").rglob("*") if path.is_file()]
    protected.extend(
        repository_root / Path(path)
        for entry in workspace.index["entries"] for path in entry["selected_paths"]
    )
    before = {path: file_digest(path) for path in protected}
    review_id = find_record(workspace, "unknown")
    workspace.decide(review_id, "Reviewer", "classification", "approved", "Classification evidence checked")
    assert {path: file_digest(path) for path in protected} == before


def test_source_resolver_accepts_only_inventoried_ids(workspace: ReviewWorkspace):
    state, _, _ = workspace.load(next(iter(workspace.entries)))
    path, extension = workspace.source_path(state["sources"][0]["source_id"])
    assert path.is_file()
    assert extension in {".pdf", ".doc", ".docx"}
    with pytest.raises(ReviewError, match="Unknown source"):
        workspace.source_path("../../README.md")
