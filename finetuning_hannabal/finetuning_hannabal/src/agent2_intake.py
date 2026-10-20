"""Deterministic Agent 2 intake for grounded fine-tuning examples.

The raw Agent 1 payload remains an audit artifact.  This module creates a
sanitized model-facing projection, maps every declared runtime variable once,
and rejects conflicting mappings instead of choosing a precedence that the
project sources do not define.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from agent1_demo_inputs import agent1_schema_errors
from normalize import ascii_snake_case, normalize_text, sha256_text
from qwen_enrichment import _read_json, _write_json
from runtime_templates import validate_runtime
from validate import load_schema


FACTS_SCHEMA = "agent2-generation-facts-v1.schema.json"
VALIDATION_SCHEMA = "agent2-generation-validation-v1.schema.json"
FACTS_FILE = "agent2-generation-facts.json"
VALIDATION_FILE = "agent2-generation-validation.json"
FORBIDDEN_MODEL_KEYS = {"session_id", "user_id"}


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


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value).strip()
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]).casefold())}
    return value


def _forbidden_paths(value: Any, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if str(key).casefold() in FORBIDDEN_MODEL_KEYS:
                found.append(child)
            found.extend(_forbidden_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_forbidden_paths(item, f"{path}[{index}]"))
    return found


def contains_forbidden_model_identifier(value: Any) -> bool:
    """Reject forbidden keys even if they are nested in an unexpected object."""
    if _forbidden_paths(value):
        return True
    serialized = _canonical_json(value).casefold()
    return any(f'"{name}"' in serialized for name in FORBIDDEN_MODEL_KEYS)


def _dynamic_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for container in ("informations_principales", "autres_informations"):
        for key, value in payload.get(container, {}).items():
            facts.append({"key": ascii_snake_case(str(key)), "path": f"{container}.{key}", "value": _normalize_value(value)})
    for index, party in enumerate(payload.get("parties", [])):
        for key, value in party.get("informations_contact", {}).items():
            facts.append({
                "key": ascii_snake_case(str(key)),
                "path": f"parties[{index}].informations_contact.{key}",
                "value": _normalize_value(value),
            })
    return facts


def _proposal_sources(
    proposal: dict[str, Any] | None,
    payload: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Recover exact Agent 1 paths from the upstream mapping proposal.

    The proposal supplies coordinates only.  Values are always read back from
    the validated raw Agent 1 payload.
    """
    if not proposal:
        return {}
    party_indexes = {
        party.get("party_key"): index for index, party in enumerate(proposal.get("parties", []))
    }
    candidates: dict[str, list[dict[str, Any]]] = {}
    for binding in proposal.get("variable_bindings", []):
        variable_id = binding.get("variable_id")
        destination = binding.get("destination")
        party_index = party_indexes.get(binding.get("party_key"))
        path: str | None = None
        value: Any = None
        if destination == "party_name" and party_index is not None and party_index < len(payload.get("parties", [])):
            path = f"parties[{party_index}].nom"
            value = payload["parties"][party_index].get("nom")
        elif destination == "party_contact" and party_index is not None and party_index < len(payload.get("parties", [])):
            path = f"parties[{party_index}].informations_contact.{variable_id}"
            value = payload["parties"][party_index].get("informations_contact", {}).get(variable_id)
        elif destination in {"informations_principales", "autres_informations"}:
            path = f"{destination}.{variable_id}"
            value = payload.get(destination, {}).get(variable_id)
        if path is not None and value is not None:
            candidates.setdefault(str(variable_id), []).append({"path": path, "value": _normalize_value(value)})
    return candidates


def _normalization_entry(path: str, raw_value: Any, normalized_value: Any) -> dict[str, Any]:
    return {"source_path": path, "raw_value": copy.deepcopy(raw_value), "normalized_value": copy.deepcopy(normalized_value)}


def adapt_agent2_input(
    payload: dict[str, Any],
    runtime: dict[str, Any],
    agent1_validation: dict[str, Any],
    *,
    artifact_key: str,
    raw_input_ref: str,
    proposal: dict[str, Any] | None = None,
    hashes: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Create model-facing facts and a non-model validation/audit sidecar."""
    errors = list(agent1_schema_errors(payload)) + list(validate_runtime(runtime))
    warnings = list(agent1_validation.get("warnings", []))
    template_id = runtime.get("template_id", "unknown")

    if agent1_validation.get("template_id") != template_id:
        errors.append("identity:agent1 validation template_id does not match runtime")
    if agent1_validation.get("status") in {"blocked", "qwen_failed"}:
        errors.append(f"upstream:agent1 validation status is {agent1_validation.get('status')}")
    if agent1_validation.get("errors"):
        errors.append("upstream:agent1 validation contains errors")
    source_ids = runtime.get("canonical_ref", {}).get("source_ids", [])
    if artifact_key not in source_ids:
        errors.append("identity:artifact key is not one of the runtime source IDs")

    normalization_audit: list[dict[str, Any]] = []

    def normalized(path: str, raw: Any) -> Any:
        value = _normalize_value(raw)
        normalization_audit.append(_normalization_entry(path, raw, value))
        return value

    document = {
        "contract_type": normalized("type_contrat", payload.get("type_contrat", "")),
        "contract_subtype": normalized("sous_type", payload.get("sous_type", "")),
        "country": normalized("metadata.pays", payload.get("metadata", {}).get("pays", "")),
        "jurisdiction": normalized("metadata.juridiction", payload.get("metadata", {}).get("juridiction", "")),
        "language": normalized(
            "informations_generales.langue_contrat",
            payload.get("informations_generales", {}).get("langue_contrat", ""),
        ),
    }
    parties: list[dict[str, Any]] = []
    for index, party in enumerate(payload.get("parties", []), start=1):
        base = f"parties[{index - 1}]"
        parties.append({
            "party_key": f"agent1_party_{index:03d}",
            "name": normalized(f"{base}.nom", party.get("nom", "")),
            "entity_type": normalized(f"{base}.type", party.get("type", "")),
            "role": normalized(f"{base}.role", party.get("role", "")),
            "contact_information": normalized(f"{base}.informations_contact", party.get("informations_contact", {})),
        })

    dynamic = _dynamic_facts(payload)
    proposal_candidates = _proposal_sources(proposal, payload)
    missing_ids = {ascii_snake_case(str(item)) for item in payload.get("informations_manquantes", [])}
    used_paths: set[str] = set()
    bindings: list[dict[str, Any]] = []
    status_counts = {"provided": 0, "missing": 0, "ambiguous": 0}

    for variable in runtime.get("variables", []):
        variable_id = variable["variable_id"]
        candidates = [
            {"path": fact["path"], "value": fact["value"]}
            for fact in dynamic if fact["key"] == variable_id
        ]
        candidates.extend(proposal_candidates.get(variable_id, []))
        unique: dict[str, dict[str, Any]] = {}
        for candidate in candidates:
            unique.setdefault(_canonical_json(candidate["value"]), candidate)
            used_paths.add(candidate["path"])
        if len(unique) > 1:
            conflict_paths = sorted(candidate["path"] for candidate in candidates)
            errors.append(f"mapping:{variable_id}:conflicting values at {', '.join(conflict_paths)}")
            value = variable["canonical_token"]
            status = "ambiguous"
            source_path = None
        elif unique:
            candidate = next(iter(unique.values()))
            raw_value = candidate["value"]
            source_path = candidate["path"]
            if isinstance(raw_value, str) and raw_value.strip().casefold().startswith("ambiguity"):
                value = variable["canonical_token"]
                status = "ambiguous"
            else:
                value = raw_value
                status = "provided"
        else:
            value = variable["canonical_token"]
            status = "missing"
            source_path = None
            if variable_id not in missing_ids:
                warnings.append(f"mapping:{variable_id}:no Agent 1 value; preserved canonical placeholder")
        status_counts[status] += 1
        bindings.append({
            "variable_id": variable_id,
            "canonical_token": variable["canonical_token"],
            "value": value,
            "status": status,
            "agent1_source_path": source_path,
        })

    unmapped = [
        {"agent1_source_path": fact["path"], "value": fact["value"]}
        for fact in dynamic if fact["path"] not in used_paths
    ]
    if unmapped:
        warnings.append(f"mapping:{len(unmapped)} Agent 1 fact(s) are not mapped to runtime variables")

    facts = {
        "schema_version": "agent2-generation-facts-v1",
        "record_status": runtime.get("runtime_status", "demo_unreviewed"),
        "template_id": template_id,
        "document": document,
        "parties": parties,
        "variable_bindings": bindings,
        "unmapped_facts": unmapped,
        "warnings": sorted(set(warnings)),
    }
    forbidden = _forbidden_paths(facts)
    if contains_forbidden_model_identifier(facts):
        errors.append("privacy:model-facing artifact contains session_id or user_id")
    errors.extend(_schema_errors(facts, FACTS_SCHEMA))
    if len(bindings) != len(runtime.get("variables", [])):
        errors.append("mapping:not every runtime variable was represented exactly once")
    variable_ids = [binding["variable_id"] for binding in bindings]
    if len(variable_ids) != len(set(variable_ids)):
        errors.append("mapping:runtime variables were represented more than once")

    supplied_hashes = hashes or {}
    report = {
        "schema_version": "agent2-generation-validation-v1",
        "template_id": template_id,
        "status": "blocked" if errors else runtime.get("runtime_status", "demo_unreviewed"),
        "valid": not errors,
        "training_eligible": bool(not errors and runtime.get("runtime_status") == "eligible"),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "forbidden_identifier_paths": forbidden,
        "hashes": {
            "agent1_input_sha256": supplied_hashes.get("agent1_input_sha256") or _value_hash(payload),
            "runtime_sha256": supplied_hashes.get("runtime_sha256") or _value_hash(runtime),
            "agent1_validation_sha256": supplied_hashes.get("agent1_validation_sha256") or _value_hash(agent1_validation),
            "agent1_proposal_sha256": supplied_hashes.get("agent1_proposal_sha256") or (_value_hash(proposal) if proposal else None),
        },
        "variable_counts": {
            "total": len(bindings), **status_counts, "unmapped_facts": len(unmapped),
        },
        "normalization_audit": normalization_audit,
        "raw_input_ref": raw_input_ref,
    }
    report_errors = _schema_errors(report, VALIDATION_SCHEMA)
    if report_errors:
        report["errors"] = sorted(set(report["errors"] + report_errors))
        report["valid"] = False
        report["status"] = "blocked"
        report["training_eligible"] = False
    return (facts if report["valid"] else None), report


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapt_record_directory(record_dir: Path, *, artifact_key: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Read one runtime record directory and write deterministic intake artifacts."""
    runtime_path = record_dir / "runtime.json"
    input_path = record_dir / "agent1-input.json"
    upstream_validation_path = record_dir / "agent1-input-validation.json"
    proposal_path = record_dir / "agent1-qwen-proposal.json"
    runtime = _read_json(runtime_path)
    payload = _read_json(input_path)
    upstream_validation = _read_json(upstream_validation_path)
    proposal = _read_json(proposal_path) if proposal_path.is_file() else None
    facts, report = adapt_agent2_input(
        payload,
        runtime,
        upstream_validation,
        artifact_key=artifact_key or record_dir.name,
        raw_input_ref="agent1-input.json",
        proposal=proposal,
        hashes={
            "agent1_input_sha256": sha256_file(input_path),
            "runtime_sha256": sha256_file(runtime_path),
            "agent1_validation_sha256": sha256_file(upstream_validation_path),
            "agent1_proposal_sha256": sha256_file(proposal_path) if proposal_path.is_file() else None,
        },
    )
    if facts is not None:
        _write_json(record_dir / FACTS_FILE, facts)
    _write_json(record_dir / VALIDATION_FILE, report)
    return facts, report


def validate_generation_facts(facts: dict[str, Any]) -> list[str]:
    errors = _schema_errors(facts, FACTS_SCHEMA)
    if contains_forbidden_model_identifier(facts):
        errors.append("privacy:model-facing artifact contains session_id or user_id")
    return sorted(set(errors))


__all__: Iterable[str] = (
    "FACTS_FILE", "VALIDATION_FILE", "adapt_agent2_input", "adapt_record_directory",
    "contains_forbidden_model_identifier", "sha256_file", "validate_generation_facts",
)
