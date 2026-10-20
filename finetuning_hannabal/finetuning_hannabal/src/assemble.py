"""Grounded canonical assembly and completed-record reclassification safeguards."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Iterable

from normalize import (
    canonical_token,
    detect_languages,
    detect_placeholders,
    replace_placeholders,
    sha256_text,
    stable_id,
)


HEADING_PREFIX_RE = re.compile(
    r"^(?:(?:article|section|clause|titre|chapitre)\s+[0-9ivxlcdm]+|[0-9]+(?:\.[0-9]+)*[.)-]?|المادة|البند|الفصل|أولاً|اولا)",
    re.IGNORECASE,
)


def _normalized_to_raw(block: dict[str, Any], start: int, end: int) -> tuple[int, int]:
    raw_points: list[int] = []
    for mapping in block["normalization_map"]:
        ns, ne = mapping["normalized_start"], mapping["normalized_end"]
        rs, re_ = mapping["raw_start"], mapping["raw_end"]
        if ne < start or ns > end:
            continue
        if mapping["operation"] == "equal":
            overlap_start = max(start, ns)
            overlap_end = min(end, ne)
            raw_points.extend([rs + overlap_start - ns, rs + overlap_end - ns])
        else:
            raw_points.extend([rs, re_])
    if not raw_points:
        return start, end
    return min(raw_points), max(raw_points)


def source_span(extraction: dict[str, Any], block: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    raw_start, raw_end = _normalized_to_raw(block, start, end)
    value = block["normalized_text"][start:end]
    return {
        "source_id": extraction["source"]["source_id"],
        "extraction_id": extraction["extraction_id"],
        "block_id": block["block_id"],
        "raw_start": raw_start,
        "raw_end": raw_end,
        "normalized_start": start,
        "normalized_end": end,
        "text_sha256": sha256_text(value),
    }


def _line_spans(text: str) -> Iterable[tuple[int, int, str]]:
    cursor = 0
    for line in text.splitlines(keepends=True):
        visible = line.rstrip("\n")
        yield cursor, cursor + len(visible), visible
        cursor += len(line)
    if cursor < len(text):
        yield cursor, len(text), text[cursor:]


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not 2 <= len(stripped) <= 180:
        return False
    if HEADING_PREFIX_RE.match(stripped):
        return True
    letters = [char for char in stripped if char.isalpha()]
    if len(letters) >= 4 and sum(char.isupper() for char in letters) / len(letters) >= 0.82:
        return True
    return False


def deterministic_structure(
    extraction: dict[str, Any], token_map: dict[str, tuple[str, str]] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    unparsed: list[dict[str, Any]] = []
    for block in extraction["blocks"]:
        text = block["normalized_text"]
        if not text.strip():
            continue
        headings = [(start, end, line.strip()) for start, end, line in _line_spans(text) if _looks_like_heading(line)]
        if not headings:
            start = len(text) - len(text.lstrip())
            end = len(text.rstrip())
            span = source_span(extraction, block, start, end)
            unparsed.append({
                "content_id": stable_id("unparsed", extraction["extraction_id"], block["block_id"], start, end),
                "text": text[start:end],
                "reason": "no_deterministic_heading_boundary",
                "source_span": span,
            })
            continue
        first_heading_start = headings[0][0]
        if text[:first_heading_start].strip():
            prefix_start = len(text[:first_heading_start]) - len(text[:first_heading_start].lstrip())
            prefix_end = len(text[:first_heading_start].rstrip())
            unparsed.append({
                "content_id": stable_id("unparsed", extraction["extraction_id"], block["block_id"], prefix_start, prefix_end),
                "text": text[prefix_start:prefix_end],
                "reason": "content_before_first_detected_heading",
                "source_span": source_span(extraction, block, prefix_start, prefix_end),
            })
        for heading_index, (heading_start, heading_end, heading) in enumerate(headings):
            section_end = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(text)
            while section_end > heading_start and text[section_end - 1].isspace():
                section_end -= 1
            if section_end <= heading_start:
                continue
            body_start = heading_end
            while body_start < section_end and text[body_start].isspace():
                body_start += 1
            clause_start = body_start if body_start < section_end else heading_start
            section_order = len(sections) + 1
            section_id = stable_id("sec", extraction["extraction_id"], block["block_id"], heading_start, section_end)
            clause_span = source_span(extraction, block, clause_start, section_end)
            clause_text = text[clause_start:section_end]
            local_occurrences = detect_placeholders(clause_text)
            if token_map:
                for occurrence in local_occurrences:
                    mapped = token_map.get(occurrence["source_token"].casefold())
                    if mapped:
                        occurrence["variable_id"], occurrence["canonical_token"] = mapped
            clause_text = replace_placeholders(clause_text, local_occurrences)
            variable_ids = sorted({item["variable_id"] for item in local_occurrences})
            clause = {
                "clause_id": stable_id("clause", section_id, clause_start, section_end),
                "order": 1,
                "clause_type": "unclassified",
                "heading": heading,
                "text": clause_text,
                "variable_ids": variable_ids,
                "legal_references": [],
                "source_span": clause_span,
                "requirement_status": "unspecified",
                "condition": None,
            }
            sections.append({
                "section_id": section_id,
                "order": section_order,
                "kind": "unclassified",
                "heading": heading,
                "heading_span": source_span(extraction, block, heading_start, heading_end),
                "source_span": source_span(extraction, block, heading_start, section_end),
                "requirement_status": "unspecified",
                "condition": None,
                "clauses": [clause],
            })
    return sections, unparsed


def deterministic_variables(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    variables: dict[str, dict[str, Any]] = {}
    label_ids: dict[str, str] = {}
    used_ids: set[str] = set()
    ordinal = 0
    for block in extraction["blocks"]:
        occurrences = detect_placeholders(block["normalized_text"])
        for occurrence in occurrences:
            key = occurrence["source_token"].casefold()
            if key not in label_ids:
                ordinal += 1
                base = occurrence["variable_id"]
                variable_id = base
                suffix = 2
                while variable_id in used_ids:
                    variable_id = f"{base}_{suffix}"
                    suffix += 1
                label_ids[key] = variable_id
                used_ids.add(variable_id)
            variable_id = label_ids[key]
            record = variables.setdefault(variable_id, {
                "variable_id": variable_id,
                "canonical_token": canonical_token(variable_id),
                "original_tokens": [],
                "value_type": "unknown",
                "party_id": None,
                "requirement_status": "unspecified",
                "allowed_choices": [],
                "occurrences": [],
            })
            if occurrence["source_token"] not in record["original_tokens"]:
                record["original_tokens"].append(occurrence["source_token"])
            record["occurrences"].append(source_span(extraction, block, occurrence["start"], occurrence["end"]))
    return list(variables.values())


def _filename_kind(relative_path: str, taxonomy: dict[str, Any], has_placeholders: bool) -> tuple[str, str, list[dict[str, str]]]:
    filename = Path(relative_path).stem.casefold()
    patterns = taxonomy["deterministic_filename_classification"]
    issues: list[dict[str, str]] = []
    for kind in ("policy", "form", "letter", "resolution", "court_filing", "guidance"):
        if any(pattern.casefold() in filename for pattern in patterns[kind]):
            return kind, "unknown", issues
    if any(pattern.casefold() in filename for pattern in patterns["contract_candidate"]):
        if has_placeholders:
            return "contract_template", "blank", issues
        issues.append({"code": "completed_contract_heuristic_requires_review", "severity": "warning", "message": "Contract-like filename without detected square-bracket placeholders was initially classified completed and requires human confirmation."})
        return "completed_contract", "completed", issues
    issues.append({"code": "document_kind_unresolved", "severity": "warning", "message": "Deterministic filename rules did not resolve document kind."})
    return "unknown", "unknown", issues


def _first_text_span(extraction: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]]]:
    for block in extraction["blocks"]:
        for start, end, line in _line_spans(block["normalized_text"]):
            title = line.strip()
            if title:
                left = start + len(line) - len(line.lstrip())
                right = left + len(title)
                return title, [source_span(extraction, block, left, right)]
    return None, []


def _language_records(extraction: dict[str, Any]) -> list[dict[str, Any]]:
    first_evidence: list[dict[str, Any]] = []
    for block in extraction["blocks"]:
        if block["normalized_text"].strip():
            start = len(block["normalized_text"]) - len(block["normalized_text"].lstrip())
            end = len(block["normalized_text"].rstrip())
            first_evidence = [source_span(extraction, block, start, end)]
            break
    return [
        {"tag": result["tag"], "basis": result["basis"], "confidence": result["confidence"], "evidence": first_evidence}
        for result in detect_languages(extraction["normalized_text"])
    ]


def _label_span(extraction: dict[str, Any], label_span: dict[str, Any]) -> dict[str, Any]:
    block = next(block for block in extraction["blocks"] if block["block_id"] == label_span["block_id"])
    return source_span(extraction, block, label_span["start"], label_span["end"])


def _label_text(extraction: dict[str, Any], label_span: dict[str, Any]) -> str:
    block = next(block for block in extraction["blocks"] if block["block_id"] == label_span["block_id"])
    return block["normalized_text"][label_span["start"]:label_span["end"]]


def _canonicalize_labeled_text(
    extraction: dict[str, Any], label_span: dict[str, Any], variables: list[dict[str, Any]]
) -> tuple[str, list[str]]:
    text = _label_text(extraction, label_span)
    replacements: list[tuple[int, int, str, str]] = []
    for variable in variables:
        for occurrence in variable["occurrences"]:
            if occurrence["block_id"] != label_span["block_id"]:
                continue
            start, end = occurrence["normalized_start"], occurrence["normalized_end"]
            if start >= label_span["start"] and end <= label_span["end"]:
                replacements.append((start - label_span["start"], end - label_span["start"], variable["canonical_token"], variable["variable_id"]))
    variable_ids: set[str] = set()
    for start, end, token, variable_id in sorted(replacements, reverse=True):
        text = text[:start] + token + text[end:]
        variable_ids.add(variable_id)
    return text, sorted(variable_ids)


def _model_structure(
    extraction: dict[str, Any], labeling: dict[str, Any], variables: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    covered: dict[str, list[tuple[int, int]]] = {block["block_id"]: [] for block in extraction["blocks"]}
    for section_label in labeling["sections"]:
        for body_index, body_span in enumerate(section_label["body_spans"]):
            covered[body_span["block_id"]].append((body_span["start"], body_span["end"]))
            section_id = stable_id("sec", extraction["extraction_id"], body_span["block_id"], body_span["start"], body_span["end"])
            heading_label = section_label["heading_span"] if body_index == 0 else None
            heading = _label_text(extraction, heading_label) if heading_label else None
            clauses: list[dict[str, Any]] = []
            for clause_label in section_label["clauses"]:
                for clause_body in clause_label["body_spans"]:
                    if clause_body["block_id"] != body_span["block_id"] or clause_body["start"] < body_span["start"] or clause_body["end"] > body_span["end"]:
                        continue
                    clause_text, variable_ids = _canonicalize_labeled_text(extraction, clause_body, variables)
                    if not clause_text:
                        continue
                    legal_references = [
                        {"text": _label_text(extraction, citation), "source_span": _label_span(extraction, citation)}
                        for citation in clause_label["citation_spans"]
                    ]
                    clauses.append({
                        "clause_id": stable_id("clause", section_id, clause_body["block_id"], clause_body["start"], clause_body["end"]),
                        "order": len(clauses) + 1,
                        "clause_type": clause_label["clause_type"],
                        "heading": _label_text(extraction, clause_label["heading_span"]) if clause_label["heading_span"] else None,
                        "text": clause_text,
                        "variable_ids": variable_ids,
                        "legal_references": legal_references,
                        "source_span": _label_span(extraction, clause_body),
                        "requirement_status": clause_label["requirement_status"],
                        "condition": _label_text(extraction, clause_label["condition_span"]) if clause_label["condition_span"] else None,
                    })
            if not clauses:
                clause_text, variable_ids = _canonicalize_labeled_text(extraction, body_span, variables)
                if clause_text:
                    clauses.append({
                        "clause_id": stable_id("clause", section_id, body_span["block_id"], body_span["start"], body_span["end"]),
                        "order": 1, "clause_type": "unclassified", "heading": heading, "text": clause_text,
                        "variable_ids": variable_ids, "legal_references": [], "source_span": _label_span(extraction, body_span),
                        "requirement_status": section_label["requirement_status"],
                        "condition": _label_text(extraction, section_label["condition_span"]) if section_label["condition_span"] else None,
                    })
            sections.append({
                "section_id": section_id,
                "order": len(sections) + 1,
                "kind": section_label["kind"],
                "heading": heading,
                "heading_span": _label_span(extraction, heading_label) if heading_label else None,
                "source_span": _label_span(extraction, body_span),
                "requirement_status": section_label["requirement_status"],
                "condition": _label_text(extraction, section_label["condition_span"]) if section_label["condition_span"] else None,
                "clauses": clauses,
            })
    unparsed: list[dict[str, Any]] = []
    for span in labeling["unparsed_spans"]:
        covered[span["block_id"]].append((span["start"], span["end"]))
        text = _label_text(extraction, span)
        if text.strip():
            unparsed.append({
                "content_id": stable_id("unparsed", extraction["extraction_id"], span["block_id"], span["start"], span["end"]),
                "text": text,
                "reason": "model_labeled_unparsed",
                "source_span": _label_span(extraction, span),
            })
    for block in extraction["blocks"]:
        intervals = sorted(covered[block["block_id"]])
        merged: list[list[int]] = []
        for start, end in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        cursor = 0
        for start, end in [*merged, [len(block["normalized_text"]), len(block["normalized_text"])]]:
            gap = block["normalized_text"][cursor:start]
            if gap.strip():
                left = cursor + len(gap) - len(gap.lstrip())
                right = start - (len(gap) - len(gap.rstrip()))
                unparsed.append({
                    "content_id": stable_id("unparsed", extraction["extraction_id"], block["block_id"], left, right),
                    "text": block["normalized_text"][left:right],
                    "reason": "meaningful_content_not_covered_by_model_spans",
                    "source_span": source_span(extraction, block, left, right),
                })
            cursor = max(cursor, end)
    return sections, unparsed


def assemble_canonical(
    extraction: dict[str, Any],
    manifest_members: list[dict[str, Any]],
    taxonomy: dict[str, Any],
    *,
    labeling_response: dict[str, Any] | None = None,
    model_status: str = "unavailable",
    model_tag: str | None = None,
    model_reason: str | None = None,
) -> dict[str, Any]:
    if extraction["status"] == "failed":
        raise ValueError("Cannot assemble a canonical template from failed extraction")
    variables = deterministic_variables(extraction)
    token_map = {
        token.casefold(): (variable["variable_id"], variable["canonical_token"])
        for variable in variables for token in variable["original_tokens"]
    }
    if labeling_response is None:
        sections, unparsed = deterministic_structure(extraction, token_map)
    else:
        sections, unparsed = _model_structure(extraction, labeling_response, variables)
    has_placeholders = bool(variables)
    initial_kind, template_state, classification_issues = _filename_kind(extraction["source"]["relative_path"], taxonomy, has_placeholders)
    title, title_evidence = _first_text_span(extraction)
    contract_family = "unknown"
    contract_type = None
    languages = _language_records(extraction)
    jurisdictions: list[dict[str, Any]] = []
    parties: list[dict[str, Any]] = []
    if labeling_response is not None:
        labels = labeling_response["document"]
        initial_kind = labels["document_kind"]
        template_state = labels["template_state"]
        contract_family = labels["contract_family"]
        contract_type = labels["contract_type_label"]
        classification_issues = []
        if labels["title_span"]:
            title = _label_text(extraction, labels["title_span"])
            title_evidence = [_label_span(extraction, labels["title_span"])]
        else:
            title, title_evidence = None, []
        language_evidence = title_evidence or _language_records(extraction)[0]["evidence"]
        languages = [{"tag": tag, "basis": "model_label", "confidence": 0.0, "evidence": language_evidence} for tag in labels["languages"]] or _language_records(extraction)
        jurisdictions = []
        for jurisdiction in labeling_response["jurisdictions"]:
            governing = _label_text(extraction, jurisdiction["governing_law_spans"][0]) if jurisdiction["governing_law_spans"] else None
            jurisdictions.append({
                "country_code": jurisdiction["country_code"], "basis": jurisdiction["basis"],
                "governing_law_text": governing,
                "legal_citations": [{"text": _label_text(extraction, span), "source_span": _label_span(extraction, span)} for span in jurisdiction["citation_spans"]],
                "evidence": [_label_span(extraction, span) for span in jurisdiction["evidence_spans"]],
                "review_status": "pending",
            })
        parties = [{
            "party_id": stable_id("party", extraction["extraction_id"], index, party["role"]),
            "role": party["role"],
            "original_role_label": _label_text(extraction, party["role_spans"][0]),
            "entity_type": party["entity_type"], "variable_ids": [],
            "occurrences": [_label_span(extraction, span) for span in party["role_spans"]],
        } for index, party in enumerate(labeling_response["parties"], 1)]
    issues = [
        {"code": "human_review_required", "severity": "warning", "message": "No human structure, privacy, or legal approval has been recorded."},
        *classification_issues,
    ]
    if labeling_response is None:
        issues.append({"code": "model_assistance_unavailable", "severity": "warning", "message": model_reason or "No suitable local Ollama model was available."})
    if not sections:
        issues.append({"code": "no_sections_confirmed", "severity": "warning", "message": "No section boundaries were confirmed; meaningful text remains in unparsed_source_content."})
    if initial_kind not in {"contract_template", "completed_contract"}:
        issues.append({"code": "ambiguity_023_non_contract_destination", "severity": "warning", "message": "AMBIGUITY-023: the non-contract knowledge-base/training destination is unresolved."})
    template_id = stable_id("tpl", extraction["normalized_text_sha256"])
    source_files = []
    for member in sorted(manifest_members, key=lambda row: row["relative_path"].casefold()):
        source_files.append({
            "source_id": member["source_id"],
            "relative_path": member["relative_path"],
            "sha256": member["sha256"],
            "media_type": member["media_type"],
            "size_bytes": member["size_bytes"],
            "page_count": extraction["counts"]["pages"] or None,
            "extraction_method": extraction["method"],
            "extraction_status": extraction["extraction_status"],
            "extraction_warnings": extraction["warnings"],
            "extracted_text_sha256": extraction["normalized_text_sha256"],
        })
    training_eligibility = "excluded" if initial_kind not in {"contract_template", "completed_contract"} else "review_required"
    action_id = stable_id("action", template_id, "deterministic_assembly", "1.0.0")
    return {
        "schema_version": "contract-template-v1",
        "template_id": template_id,
        "source_files": source_files,
        "document": {
            "title": title,
            "title_evidence": title_evidence,
            "initial_document_kind": initial_kind,
            "document_kind": initial_kind,
            "reclassification_history": [],
            "template_state": template_state,
            "contract_family": contract_family,
            "contract_type": contract_type,
            "languages": languages,
        },
        "jurisdictions": jurisdictions,
        "parties": parties,
        "variables": variables,
        "sections": sections,
        "unparsed_source_content": unparsed,
        "quality": {
            "extraction_status": extraction["extraction_status"],
            "structure_review_status": "pending",
            "legal_review_status": "pending",
            "privacy_review_status": "pending",
            "training_eligibility": training_eligibility,
            "sanitization_status": "required" if initial_kind == "completed_contract" else "not_required",
            "issues": issues,
        },
        "sanitization_history": [],
        "audit_history": [{
            "action_id": action_id,
            "action": "deterministic_assembly",
            "actor_type": "pipeline",
            "actor": "contract-template-dataset-pipeline-1.0.0",
            "details": "Assembled only from validated source spans; no legal approval or training eligibility asserted.",
        }],
        "model_assistance": {
            "status": model_status,
            "provider": "ollama" if model_status in {"used", "failed", "unavailable"} else "none",
            "model_tag": model_tag,
            "reason": model_reason,
        },
        "versions": {
            "pipeline": "1.0.0", "schema": "contract-template-v1", "taxonomy": taxonomy["taxonomy_version"],
            "normalizer": "1.0.0", "prompt": labeling_response["versions"]["prompt"] if labeling_response else None,
            "model": model_tag if labeling_response else None,
        },
    }


def reclassify_completed_record(
    record: dict[str, Any], *, actor: str, review_record_id: str, reason: str
) -> dict[str, Any]:
    result = copy.deepcopy(record)
    if result["document"]["initial_document_kind"] != "completed_contract":
        raise ValueError("Only records initially classified completed_contract may use this direct reclassification")
    if result["document"]["document_kind"] != "completed_contract":
        raise ValueError("Record is not currently a completed_contract")
    quality = result["quality"]
    required = {
        "sanitization_status": "completed",
        "privacy_review_status": "approved",
        "structure_review_status": "approved",
        "legal_review_status": "approved",
    }
    failures = [f"{field} must be {expected}" for field, expected in required.items() if quality.get(field) != expected]
    if failures:
        raise ValueError("; ".join(failures))
    if not actor.strip() or not review_record_id.strip() or not reason.strip():
        raise ValueError("Human actor, review record, and reason are required")
    result["document"]["document_kind"] = "contract_template"
    result["document"]["template_state"] = "blank"
    result["document"]["reclassification_history"].append({
        "from_kind": "completed_contract", "to_kind": "contract_template", "actor": actor,
        "review_record_id": review_record_id, "reason": reason, "sanitization_status": "completed",
    })
    result["audit_history"].append({
        "action_id": stable_id("action", result["template_id"], review_record_id, actor),
        "action": "human_reclassification_after_sanitization",
        "actor_type": "human", "actor": actor,
        "details": f"Direct reclassification recorded by {review_record_id}: {reason}",
    })
    result["quality"]["training_eligibility"] = "eligible"
    return result
