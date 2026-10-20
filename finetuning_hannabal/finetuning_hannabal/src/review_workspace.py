"""Persistent, auditable review state for immutable pilot artifacts."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from assemble import reclassify_completed_record, source_span
from inventory import sha256_file
from normalize import sha256_text, stable_id
from validate import schema_errors, validate_canonical


REVIEW_CATEGORIES = ("extraction", "classification", "structure", "privacy", "sanitization", "legal")
EXCLUDED_KINDS = {"policy", "form", "letter", "resolution", "court_filing", "guidance", "other"}
DOCUMENT_KINDS = {"contract_template", "completed_contract", *EXCLUDED_KINDS, "unknown"}
TEMPLATE_STATES = {"blank", "completed", "mixed", "unknown"}
CONTRACT_FAMILIES = {"employment", "commercial", "saas", "nda", "investment", "real_estate", "procurement", "unknown"}
VALUE_TYPES = {"text", "date", "number", "money", "duration", "address", "identifier", "signature", "choice", "unknown"}
REQUIREMENT_STATUSES = {"explicit_required", "explicit_conditional", "explicit_optional", "unspecified"}
LANGUAGE_RE = re.compile(r"^(und|[a-z]{2,3})(-[A-Za-z0-9]{2,8})*$")
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ReviewError(ValueError):
    """A rejected review operation. No workspace files have been changed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _source_slice(span: dict[str, Any], blocks: dict[str, dict[str, Any]]) -> str:
    block = blocks.get(span["block_id"])
    if block is None:
        raise ReviewError(f"Unknown evidence block {span['block_id']}")
    start, end = span["normalized_start"], span["normalized_end"]
    return block["normalized_text"][start:end]


def _render_evidence_text(
    evidence_span: dict[str, Any], variables: list[dict[str, Any]], blocks: dict[str, dict[str, Any]],
) -> str:
    text = _source_slice(evidence_span, blocks)
    replacements: list[tuple[int, int, str]] = []
    for variable in variables:
        for occurrence in variable["occurrences"]:
            if occurrence["block_id"] != evidence_span["block_id"]:
                continue
            if (occurrence["normalized_start"] >= evidence_span["normalized_start"] and
                    occurrence["normalized_end"] <= evidence_span["normalized_end"]):
                replacements.append((
                    occurrence["normalized_start"] - evidence_span["normalized_start"],
                    occurrence["normalized_end"] - evidence_span["normalized_start"],
                    variable["canonical_token"],
                ))
    for start, end, token in sorted(replacements, reverse=True):
        text = text[:start] + token + text[end:]
    return text


class ReviewWorkspace:
    """Load pilot groups and store all human changes outside generated artifacts."""

    def __init__(
        self, repository_root: Path, *, artifacts_root: Path | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.artifacts_root = (artifacts_root or self.repository_root / "artifacts").resolve()
        self.workspace_root = (workspace_root or self.repository_root / "review-workspace").resolve()
        self.records_root = self.workspace_root / "records"
        self.index = _read_json(self.artifacts_root / "pilot" / "index.json")
        self.entries = {entry["review_record_id"]: entry for entry in self.index["entries"]}
        self._source_lookup: dict[str, tuple[str, str]] = {}
        self._sampled_ids = self._deterministic_audit_sample()
        for review_id in self.entries:
            canonical = self._artifact_canonical(review_id)
            if canonical is not None:
                for source in canonical["source_files"]:
                    self._source_lookup[source["source_id"]] = (source["relative_path"], source["sha256"])
            else:
                extraction = self._artifact_extraction(review_id)
                source = extraction["source"]
                self._source_lookup[source["source_id"]] = (source["relative_path"], source["sha256"])

    def _artifact_dir(self, review_id: str) -> Path:
        return self.artifacts_root / "pilot" / self.entries[review_id]["artifact_key"]

    def _artifact_review(self, review_id: str) -> dict[str, Any]:
        return _read_json(self.artifacts_root / "review-queue" / f"{review_id}.json")

    def _artifact_extraction(self, review_id: str) -> dict[str, Any]:
        return _read_json(self._artifact_dir(review_id) / "extraction.json")

    def _artifact_canonical(self, review_id: str) -> dict[str, Any] | None:
        path = self._artifact_dir(review_id) / "canonical.json"
        return _read_json(path) if path.is_file() else None

    def _deterministic_audit_sample(self) -> set[str]:
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for entry in self.entries.values():
            kind = entry["initial_document_kind"]
            if kind not in EXCLUDED_KINDS:
                continue
            folder = Path(entry["selected_paths"][0]).parts[1]
            strata[(folder, kind)].append(entry)
        selected: set[str] = set()
        for candidates in strata.values():
            count = max(1, math.ceil(len(candidates) * 0.10))
            candidates.sort(key=lambda item: (item["sha256"], item["review_record_id"]))
            selected.update(item["review_record_id"] for item in candidates[:count])
        return selected

    def _coverage(self, review_id: str, canonical: dict[str, Any] | None) -> str:
        if canonical is None:
            return "automated_triage"
        kind = canonical["document"]["document_kind"]
        if kind in {"contract_template", "completed_contract"}:
            return "full"
        if kind == "unknown":
            return "classification_only"
        return "sampled_audit" if review_id in self._sampled_ids else "automated_triage"

    @staticmethod
    def _required_reviews(canonical: dict[str, Any] | None, coverage: str) -> list[str]:
        if canonical is None or coverage == "automated_triage":
            return []
        if coverage == "classification_only":
            return ["classification"]
        if coverage == "sampled_audit":
            return ["extraction", "classification"]
        document = canonical["document"]
        required = {"extraction", "classification", "structure", "legal"}
        if document["document_kind"] == "completed_contract" or document["template_state"] in {"completed", "mixed"}:
            required.update({"privacy", "sanitization"})
        return sorted(required)

    def initialize(self) -> int:
        """Create missing working copies. Existing review work is never overwritten."""
        created = 0
        for review_id, entry in self.entries.items():
            record_dir = self.records_root / review_id
            state_path = record_dir / "state.json"
            if state_path.is_file():
                # Validation is derived state. A never-reviewed initialization may be
                # refreshed after validator upgrades without manufacturing a human revision.
                existing = _read_json(state_path)
                if existing.get("revision") == 0 and existing.get("reviewer") is None:
                    existing_canonical_path = record_dir / "canonical-working.json"
                    existing_canonical = _read_json(existing_canonical_path) if existing_canonical_path.is_file() else None
                    existing_errors = self._validation_errors(
                        existing, existing_canonical, self._artifact_extraction(review_id),
                    )
                    if existing["validation"]["valid"] != (not existing_errors) or existing["validation"]["errors"] != existing_errors:
                        existing["validation"] = {"valid": not existing_errors, "errors": existing_errors, "checked_at": _now()}
                        _atomic_json(state_path, existing)
                continue
            canonical = self._artifact_canonical(review_id)
            extraction = self._artifact_extraction(review_id)
            review = self._artifact_review(review_id)
            coverage = self._coverage(review_id, canonical)
            now = _now()
            paths = entry["selected_paths"]
            source_ids = review["source_ids"]
            if canonical:
                by_id = {source["source_id"]: source for source in canonical["source_files"]}
                sources = [{
                    "source_id": source_id,
                    "relative_path": by_id[source_id]["relative_path"],
                    "sha256": by_id[source_id]["sha256"],
                } for source_id in source_ids]
            else:
                sources = [{"source_id": source_ids[0], "relative_path": paths[0], "sha256": entry["sha256"]}]
            decisions = {
                category: {"status": "pending", "reviewer": None, "notes": "", "decided_at": None}
                for category in REVIEW_CATEGORIES
            }
            required = self._required_reviews(canonical, coverage)
            for category in REVIEW_CATEGORIES:
                if category not in required:
                    decisions[category]["status"] = "not_applicable"
            excluded = canonical is None or canonical["document"]["document_kind"] in EXCLUDED_KINDS
            state = {
                "schema_version": "review-workspace-record-v1",
                "review_record_id": review_id,
                "template_id": entry["template_id"],
                "artifact_key": entry["artifact_key"],
                "artifact_paths": {
                    "review": f"artifacts/review-queue/{review_id}.json",
                    "extraction": f"artifacts/pilot/{entry['artifact_key']}/extraction.json",
                    "canonical": f"artifacts/pilot/{entry['artifact_key']}/canonical.json" if canonical else None,
                    "validation": f"artifacts/pilot/{entry['artifact_key']}/validation.json",
                },
                "sources": sources,
                "reviewer": None,
                "revision": 0,
                "review_coverage": coverage,
                "initial_classification": entry["initial_document_kind"],
                "required_reviews": required,
                "decisions": decisions,
                "final_disposition": "excluded" if canonical is None else "pending",
                "training_eligibility": "excluded" if excluded else "review_required",
                "validation": {"valid": False, "errors": [], "checked_at": now},
                "revision_history": [],
                "created_at": now,
                "updated_at": now,
            }
            errors = self._validation_errors(state, canonical, extraction)
            state["validation"] = {"valid": not errors, "errors": errors, "checked_at": now}
            errors = schema_errors(state, "review-workspace-record-v1.schema.json")
            if errors:
                raise ReviewError("Cannot initialize review state: " + "; ".join(errors))
            record_dir.mkdir(parents=True, exist_ok=True)
            if canonical is not None:
                _atomic_json(record_dir / "canonical-working.json", canonical)
            _atomic_json(state_path, state)
            created += 1
        return created

    def _paths(self, review_id: str) -> tuple[Path, Path, Path]:
        if review_id not in self.entries:
            raise ReviewError("Unknown review record")
        directory = self.records_root / review_id
        return directory / "state.json", directory / "canonical-working.json", directory / "revisions"

    def load(self, review_id: str) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        state_path, canonical_path, _ = self._paths(review_id)
        if not state_path.is_file():
            self.initialize()
        state = _read_json(state_path)
        canonical = _read_json(canonical_path) if canonical_path.is_file() else None
        return state, canonical, self._artifact_extraction(review_id)

    def list_records(self) -> list[dict[str, Any]]:
        self.initialize()
        rows = []
        for review_id, entry in self.entries.items():
            state, canonical, extraction = self.load(review_id)
            rows.append({
                "review_record_id": review_id,
                "source_paths": [source["relative_path"] for source in state["sources"]],
                "duplicate_count": len(state["sources"]),
                "extraction_status": extraction["extraction_status"],
                "document_kind": canonical["document"]["document_kind"] if canonical else None,
                "review_coverage": state["review_coverage"],
                "required_reviews": state["required_reviews"],
                "completed_reviews": sum(
                    state["decisions"][category]["status"] == "approved"
                    for category in state["required_reviews"]
                ),
                "final_disposition": state["final_disposition"],
                "training_eligibility": state["training_eligibility"],
                "validation_valid": state["validation"]["valid"],
                "revision": state["revision"],
                "sha256": entry["sha256"],
            })
        return rows

    def _validation_errors(
        self, state: dict[str, Any], canonical: dict[str, Any] | None, extraction: dict[str, Any],
    ) -> list[str]:
        errors = schema_errors(state, "review-workspace-record-v1.schema.json")
        if canonical is not None:
            errors.extend(validate_canonical(canonical, extraction))
            source_pairs = {(item["source_id"], item["relative_path"], item["sha256"]) for item in canonical["source_files"]}
            for source in state["sources"]:
                if (source["source_id"], source["relative_path"], source["sha256"]) not in source_pairs:
                    errors.append(f"workspace:sources:{source['source_id']}:does not match immutable canonical provenance")
        if state["revision"] != len(state["revision_history"]):
            errors.append("workspace:revision:revision number and history length differ")
        return errors

    def _commit(
        self, review_id: str, reviewer: str, action: str,
        mutate: Callable[[dict[str, Any], dict[str, Any] | None, dict[str, Any]], None],
    ) -> dict[str, Any]:
        reviewer = reviewer.strip()
        if not reviewer:
            raise ReviewError("Reviewer name is required")
        old_state, old_canonical, extraction = self.load(review_id)
        state, canonical = copy.deepcopy(old_state), copy.deepcopy(old_canonical)
        mutate(state, canonical, extraction)
        revision = old_state["revision"] + 1
        now = _now()
        state["reviewer"] = reviewer
        state["revision"] = revision
        state["updated_at"] = now
        state["revision_history"].append({
            "revision": revision, "action": action, "reviewer": reviewer, "created_at": now,
        })
        errors = self._validation_errors(state, canonical, extraction)
        state["validation"] = {"valid": not errors, "errors": errors, "checked_at": now}
        final_errors = schema_errors(state, "review-workspace-record-v1.schema.json")
        if final_errors or errors:
            raise ReviewError("Invalid change: " + "; ".join(final_errors + errors))
        state_path, canonical_path, revisions_path = self._paths(review_id)
        snapshot = {
            "schema_version": "review-revision-v1", "revision": revision,
            "action": action, "reviewer": reviewer, "created_at": now,
            "state_before": old_state, "canonical_before": old_canonical,
        }
        _atomic_json(revisions_path / f"{revision:06d}.json", snapshot)
        if canonical is not None:
            _atomic_json(canonical_path, canonical)
        _atomic_json(state_path, state)
        return state

    def apply_correction(self, review_id: str, reviewer: str, correction: dict[str, Any]) -> dict[str, Any]:
        """Apply only explicitly editable metadata; evidence coordinates and source text stay locked."""
        if not isinstance(correction, dict):
            raise ReviewError("Correction payload must be a JSON object")
        allowed = {"document", "languages", "sections", "clauses", "variables"}
        unknown = set(correction) - allowed
        if unknown:
            raise ReviewError(f"Unsupported correction properties: {sorted(unknown)}")

        def mutate(state: dict[str, Any], canonical: dict[str, Any] | None, extraction: dict[str, Any]) -> None:
            if canonical is None:
                raise ReviewError("Unreadable records have no canonical working copy to correct")
            document_changes = correction.get("document", {})
            if not isinstance(document_changes, dict):
                raise ReviewError("Document correction must be an object")
            allowed_document = {"document_kind", "template_state", "contract_family", "contract_type"}
            if set(document_changes) - allowed_document:
                raise ReviewError("Unsupported document correction")
            for field, value in document_changes.items():
                if field == "document_kind" and value not in DOCUMENT_KINDS:
                    raise ReviewError("Invalid document kind")
                if field == "template_state" and value not in TEMPLATE_STATES:
                    raise ReviewError("Invalid template state")
                if field == "contract_family" and value not in CONTRACT_FAMILIES:
                    raise ReviewError("Invalid contract family")
                if field == "contract_type" and value is not None and not isinstance(value, str):
                    raise ReviewError("Contract type must be text or null")
                if (field == "document_kind" and value == "contract_template" and
                        canonical["document"]["initial_document_kind"] == "completed_contract"):
                    raise ReviewError("Completed contracts can be reclassified only by the guarded final approval gate")
                canonical["document"][field] = value

            if "languages" in correction:
                tags = correction["languages"]
                if not isinstance(tags, list) or not tags or any(not isinstance(tag, str) or not LANGUAGE_RE.fullmatch(tag) for tag in tags):
                    raise ReviewError("Languages must be a non-empty list of valid BCP-47-like tags")
                if len(tags) != len(set(tags)):
                    raise ReviewError("Language tags must be unique")
                originals = canonical["document"]["languages"]
                basis = originals[0] if originals else None
                if basis is None:
                    raise ReviewError("Language labels cannot be created without existing source evidence")
                by_tag = {item["tag"]: item for item in originals}
                canonical["document"]["languages"] = [
                    copy.deepcopy(by_tag.get(tag, {**basis, "tag": tag})) for tag in tags
                ]

            section_by_id = {item["section_id"]: item for item in canonical["sections"]}
            clause_by_id = {
                clause["clause_id"]: clause
                for section in canonical["sections"] for clause in section["clauses"]
            }
            section_changes = correction.get("sections", [])
            if not isinstance(section_changes, list) or any(not isinstance(item, dict) for item in section_changes):
                raise ReviewError("Section corrections must be a list of objects")
            for change in section_changes:
                permitted = {"section_id", "order", "kind", "requirement_status", "condition"}
                if set(change) - permitted or "section_id" not in change or change["section_id"] not in section_by_id:
                    raise ReviewError("Section corrections require an existing section_id and editable fields only")
                section = section_by_id[change["section_id"]]
                for field, value in change.items():
                    if field == "section_id":
                        continue
                    if field == "order" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                        raise ReviewError("Section order must be a positive integer")
                    if field == "kind" and (not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value)):
                        raise ReviewError("Section kind must be lower_snake_case")
                    if field == "requirement_status" and value not in REQUIREMENT_STATUSES:
                        raise ReviewError("Invalid section requirement status")
                    if field == "condition" and value is not None and not isinstance(value, str):
                        raise ReviewError("Section condition must be text or null")
                    section[field] = value
            clause_changes = correction.get("clauses", [])
            if not isinstance(clause_changes, list) or any(not isinstance(item, dict) for item in clause_changes):
                raise ReviewError("Clause corrections must be a list of objects")
            for change in clause_changes:
                permitted = {"clause_id", "order", "clause_type", "requirement_status", "condition", "variable_ids"}
                if set(change) - permitted or "clause_id" not in change or change["clause_id"] not in clause_by_id:
                    raise ReviewError("Clause corrections require an existing clause_id and editable fields only")
                clause = clause_by_id[change["clause_id"]]
                for field, value in change.items():
                    if field == "clause_id":
                        continue
                    if field == "order" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                        raise ReviewError("Clause order must be a positive integer")
                    if field == "clause_type" and (not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value)):
                        raise ReviewError("Clause type must be lower_snake_case")
                    if field == "requirement_status" and value not in REQUIREMENT_STATUSES:
                        raise ReviewError("Invalid clause requirement status")
                    if field == "condition" and value is not None and not isinstance(value, str):
                        raise ReviewError("Clause condition must be text or null")
                    if field == "variable_ids" and (
                        not isinstance(value, list)
                        or any(not isinstance(item, str) or not IDENTIFIER_RE.fullmatch(item) for item in value)
                    ):
                        raise ReviewError("Clause variable_ids must be a list of lower_snake_case IDs")
                    clause[field] = value

            variable_changes = correction.get("variables", [])
            if not isinstance(variable_changes, list) or any(not isinstance(item, dict) for item in variable_changes):
                raise ReviewError("Variable corrections must be a list of objects")
            variable_by_id = {item["variable_id"]: item for item in canonical["variables"]}
            rename_map: dict[str, str] = {}
            for change in variable_changes:
                permitted = {"variable_id", "new_variable_id", "value_type", "party_id", "requirement_status", "allowed_choices"}
                if set(change) - permitted or "variable_id" not in change or change["variable_id"] not in variable_by_id:
                    raise ReviewError("Variable corrections require an existing variable_id and editable fields only")
                new_id = change.get("new_variable_id", change["variable_id"])
                if not isinstance(new_id, str) or not IDENTIFIER_RE.fullmatch(new_id):
                    raise ReviewError("Variable IDs must be lower_snake_case")
                rename_map[change["variable_id"]] = new_id
            final_ids = [rename_map.get(item["variable_id"], item["variable_id"]) for item in canonical["variables"]]
            if len(final_ids) != len(set(final_ids)):
                raise ReviewError("Variable correction creates duplicate IDs")
            party_ids = {item["party_id"] for item in canonical["parties"]}
            for change in variable_changes:
                variable = variable_by_id[change["variable_id"]]
                if "value_type" in change and change["value_type"] not in VALUE_TYPES:
                    raise ReviewError("Invalid variable value type")
                if "party_id" in change and change["party_id"] is not None and change["party_id"] not in party_ids:
                    raise ReviewError("Unknown variable party association")
                if "requirement_status" in change and change["requirement_status"] not in REQUIREMENT_STATUSES:
                    raise ReviewError("Invalid variable requirement status")
                if "allowed_choices" in change and (not isinstance(change["allowed_choices"], list) or any(not isinstance(x, str) for x in change["allowed_choices"])):
                    raise ReviewError("Allowed choices must be text values")
                old_id = variable["variable_id"]
                for field in ("value_type", "party_id", "requirement_status", "allowed_choices"):
                    if field in change:
                        variable[field] = change[field]
                variable["variable_id"] = rename_map.get(old_id, old_id)
                variable["canonical_token"] = f"[{variable['variable_id'].upper()}]"
            if rename_map:
                for party in canonical["parties"]:
                    party["variable_ids"] = [rename_map.get(item, item) for item in party["variable_ids"]]
                for clause in clause_by_id.values():
                    clause["variable_ids"] = [rename_map.get(item, item) for item in clause["variable_ids"]]

            if sorted(section["order"] for section in canonical["sections"]) != list(range(1, len(canonical["sections"]) + 1)):
                raise ReviewError("Section orders must remain contiguous")
            canonical["sections"].sort(key=lambda item: item["order"])
            known_variables = {item["variable_id"] for item in canonical["variables"]}
            blocks = {block["block_id"]: block for block in extraction["blocks"]}
            for section in canonical["sections"]:
                if sorted(clause["order"] for clause in section["clauses"]) != list(range(1, len(section["clauses"]) + 1)):
                    raise ReviewError("Clause orders must remain contiguous within each section")
                section["clauses"].sort(key=lambda item: item["order"])
                for clause in section["clauses"]:
                    if any(item not in known_variables for item in clause["variable_ids"]):
                        raise ReviewError("Clause references an unknown variable")
                    clause["text"] = _render_evidence_text(clause["source_span"], canonical["variables"], blocks)
            for old_id, new_id in rename_map.items():
                if old_id == new_id:
                    continue
                for item in canonical["unparsed_source_content"]:
                    item["text"] = item["text"].replace(f"[{old_id.upper()}]", f"[{new_id.upper()}]")

            new_coverage = self._coverage(review_id, canonical)
            new_required = self._required_reviews(canonical, new_coverage)
            state["review_coverage"] = new_coverage
            state["required_reviews"] = new_required
            for category in REVIEW_CATEGORIES:
                if category in new_required and state["decisions"][category]["status"] == "not_applicable":
                    state["decisions"][category] = {"status": "pending", "reviewer": None, "notes": "", "decided_at": None}
                elif category not in new_required and state["decisions"][category]["status"] == "pending":
                    state["decisions"][category]["status"] = "not_applicable"
            candidate = canonical["document"]["document_kind"] in {"contract_template", "completed_contract"}
            state["training_eligibility"] = "review_required" if candidate else "excluded"
            canonical["quality"]["training_eligibility"] = state["training_eligibility"]

        return self._commit(review_id, reviewer, "structured_correction", mutate)

    def sanitize(
        self, review_id: str, reviewer: str, *, block_id: str, start: int, end: int,
        variable_id: str, value_type: str = "text",
    ) -> dict[str, Any]:
        if not IDENTIFIER_RE.fullmatch(variable_id):
            raise ReviewError("Variable ID must be lower_snake_case")
        if value_type not in VALUE_TYPES:
            raise ReviewError("Invalid variable value type")

        def mutate(state: dict[str, Any], canonical: dict[str, Any] | None, extraction: dict[str, Any]) -> None:
            if canonical is None:
                raise ReviewError("Unreadable records cannot be sanitized")
            if canonical["document"]["template_state"] not in {"completed", "mixed"} and canonical["document"]["document_kind"] != "completed_contract":
                raise ReviewError("Sanitization is allowed only for completed or mixed records")
            blocks = {block["block_id"]: block for block in extraction["blocks"]}
            block = blocks.get(block_id)
            if block is None or not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(block["normalized_text"]):
                raise ReviewError("Sanitization must select a valid exact extracted-text occurrence")
            selected_text = block["normalized_text"][start:end]
            if not selected_text.strip():
                raise ReviewError("Sanitization cannot replace blank text")
            for variable in canonical["variables"]:
                for occurrence in variable["occurrences"]:
                    if occurrence["block_id"] == block_id and not (end <= occurrence["normalized_start"] or start >= occurrence["normalized_end"]):
                        raise ReviewError("Sanitization occurrence overlaps an existing variable")
            occurrence = source_span(extraction, block, start, end)
            variable = next((item for item in canonical["variables"] if item["variable_id"] == variable_id), None)
            if variable is None:
                variable = {
                    "variable_id": variable_id,
                    "canonical_token": f"[{variable_id.upper()}]",
                    "original_tokens": [selected_text],
                    "value_type": value_type,
                    "party_id": None,
                    "requirement_status": "unspecified",
                    "allowed_choices": [],
                    "occurrences": [occurrence],
                }
                canonical["variables"].append(variable)
            else:
                variable["occurrences"].append(occurrence)
                if selected_text not in variable["original_tokens"]:
                    variable["original_tokens"].append(selected_text)
            containers = [
                clause for section in canonical["sections"] for clause in section["clauses"]
                if clause["source_span"]["block_id"] == block_id
                and start >= clause["source_span"]["normalized_start"]
                and end <= clause["source_span"]["normalized_end"]
            ]
            containers.extend(
                item for item in canonical["unparsed_source_content"]
                if item["source_span"]["block_id"] == block_id
                and start >= item["source_span"]["normalized_start"]
                and end <= item["source_span"]["normalized_end"]
            )
            if not containers:
                raise ReviewError("Selected occurrence is outside canonical clause and unparsed evidence")
            for item in containers:
                item["text"] = _render_evidence_text(item["source_span"], canonical["variables"], blocks)
                if "variable_ids" in item and variable_id not in item["variable_ids"]:
                    item["variable_ids"].append(variable_id)
            canonical["quality"]["sanitization_status"] = "required"
            action_id = stable_id("action", canonical["template_id"], variable_id, block_id, start, end, state["revision"] + 1)
            canonical["sanitization_history"].append({
                "action_id": action_id, "action": "evidence_bound_sanitization",
                "actor_type": "human", "actor": reviewer.strip(),
                "details": f"Replaced exact {block_id}[{start}:{end}] evidence with [{variable_id.upper()}].",
            })
            canonical["audit_history"].append(copy.deepcopy(canonical["sanitization_history"][-1]))

        return self._commit(review_id, reviewer, "evidence_bound_sanitization", mutate)

    def decide(self, review_id: str, reviewer: str, category: str, status: str, notes: str) -> dict[str, Any]:
        if category not in REVIEW_CATEGORIES:
            raise ReviewError("Unknown review category")
        if status not in {"approved", "rejected", "changes_required"}:
            raise ReviewError("Invalid decision status")
        if not notes.strip():
            raise ReviewError("Decision notes are required")

        def mutate(state: dict[str, Any], canonical: dict[str, Any] | None, extraction: dict[str, Any]) -> None:
            if category not in state["required_reviews"]:
                raise ReviewError(f"{category} review is not required for this record")
            if canonical is None and status == "approved":
                raise ReviewError("Encrypted or unreadable records cannot be approved without a readable authorized source")
            state["decisions"][category] = {
                "status": status, "reviewer": reviewer.strip(), "notes": notes.strip(), "decided_at": _now(),
            }
            if canonical is not None:
                quality_field = {
                    "structure": "structure_review_status", "legal": "legal_review_status", "privacy": "privacy_review_status",
                }.get(category)
                if quality_field:
                    canonical["quality"][quality_field] = "approved" if status == "approved" else "rejected"
                if category == "sanitization":
                    if status == "approved" and not canonical["sanitization_history"]:
                        raise ReviewError("Sanitization cannot be approved before at least one evidence-bound action")
                    canonical["quality"]["sanitization_status"] = "completed" if status == "approved" else "failed"
            if status != "approved":
                state["final_disposition"] = "changes_required" if status == "changes_required" else "rejected"
                state["training_eligibility"] = "excluded"
                if canonical is not None:
                    canonical["quality"]["training_eligibility"] = "excluded"

        return self._commit(review_id, reviewer, f"decision:{category}:{status}", mutate)

    def finalize(self, review_id: str, reviewer: str, disposition: str, notes: str) -> dict[str, Any]:
        if disposition not in {"approved", "rejected", "changes_required"}:
            raise ReviewError("Invalid final disposition")
        if not notes.strip():
            raise ReviewError("Final disposition notes are required")

        def mutate(state: dict[str, Any], canonical: dict[str, Any] | None, extraction: dict[str, Any]) -> None:
            if disposition == "approved":
                if canonical is None or extraction["extraction_status"] in {"encrypted", "unreadable", "empty", "malformed", "likely_ocr_required"}:
                    raise ReviewError("Encrypted or unreadable records cannot be approved without a readable authorized source")
                missing = [category for category in state["required_reviews"] if state["decisions"][category]["status"] != "approved"]
                if missing:
                    raise ReviewError("Required approvals are incomplete: " + ", ".join(missing))
                document = canonical["document"]
                is_candidate = document["document_kind"] in {"contract_template", "completed_contract"}
                if is_candidate and state["review_coverage"] != "full":
                    raise ReviewError("Contract-template candidates require full human review")
                if document["document_kind"] == "completed_contract":
                    updated = reclassify_completed_record(
                        canonical, actor=reviewer.strip(), review_record_id=review_id, reason=notes.strip(),
                    )
                    canonical.clear()
                    canonical.update(updated)
                    is_candidate = True
                elif document["document_kind"] == "contract_template":
                    if document["template_state"] != "blank":
                        raise ReviewError("A contract_template must be blank at final approval")
                    canonical["quality"]["training_eligibility"] = "eligible"
                else:
                    canonical["quality"]["training_eligibility"] = "excluded"
                state["training_eligibility"] = "eligible" if is_candidate else "excluded"
            else:
                state["training_eligibility"] = "excluded"
                if canonical is not None:
                    canonical["quality"]["training_eligibility"] = "excluded"
            state["final_disposition"] = disposition

        return self._commit(review_id, reviewer, f"final:{disposition}", mutate)

    def source_path(self, source_id: str) -> tuple[Path, str]:
        """Resolve only an inventoried source ID, never an arbitrary user path."""
        if source_id not in self._source_lookup:
            raise ReviewError("Unknown source")
        relative_path, expected_hash = self._source_lookup[source_id]
        candidate = (self.repository_root / Path(relative_path)).resolve()
        templates_root = (self.repository_root / "templates").resolve()
        try:
            candidate.relative_to(templates_root)
        except ValueError as exc:
            raise ReviewError("Source path escapes templates root") from exc
        if not candidate.is_file() or sha256_file(candidate) != expected_hash:
            raise ReviewError("Source is missing or its immutable hash has changed")
        return candidate, candidate.suffix.casefold()
