"""Generate unreviewed Agent 1 demo inputs from compact runtime contracts.

Qwen proposes source-grounded classifications and mappings.  Deterministic code
validates those proposals and assembles the unchanged Agent 1 payload shape.
Nothing produced here is reviewed, privacy-approved, or training-eligible.
"""

from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft7Validator, Draft202012Validator, FormatChecker

from normalize import sha256_text
from ollama_extract import installed_models
from qwen_enrichment import _read_json, _write_json
from validate import load_schema


AGENT1_SCHEMA = "agent1-raw-payload-v1.schema.json"
PROPOSAL_SCHEMA = "qwen-agent1-demo-proposal-v1.schema.json"
PROMPT_NAME = "qwen-agent1-demo-generation-v1.txt"
PROPOSAL_FILE = "agent1-qwen-proposal.json"
RAW_RESPONSE_FILE = "agent1-qwen-response.json"
INPUT_FILE = "agent1-input.json"
VALIDATION_FILE = "agent1-input-validation.json"
SUMMARY_FILE = "agent1-input-summary.json"

VALUE_TYPES = {
    "text", "date", "number", "money", "duration", "address",
    "identifier", "signature", "choice", "unknown",
}


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def runtime_paths(runtime_dir: Path) -> list[Path]:
    index_path = runtime_dir / "index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        if not isinstance(index, list):
            raise ValueError("runtime index.json must be an array")
        paths = [runtime_dir / entry["runtime"] for entry in index]
    else:
        paths = sorted(runtime_dir.rglob("runtime.json"), key=lambda item: item.as_posix().casefold())
    if len(paths) != len(set(paths)):
        raise ValueError("runtime inputs contain duplicate paths")
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise ValueError("runtime index references missing files: " + ", ".join(missing))
    return paths


def agent1_schema_errors(payload: dict[str, Any]) -> list[str]:
    validator = Draft7Validator(load_schema(AGENT1_SCHEMA), format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: [str(item) for item in error.absolute_path])
    return [f"schema:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}" for error in errors]


def proposal_schema_errors(proposal: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(load_schema(PROPOSAL_SCHEMA))
    errors = sorted(validator.iter_errors(proposal), key=lambda error: [str(item) for item in error.absolute_path])
    return [f"schema:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}" for error in errors]


def _runtime_ids(runtime: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    clause_ids = {
        clause["clause_id"]
        for section in runtime.get("sections", [])
        for clause in section.get("clauses", [])
    }
    content_ids = {content["content_id"] for content in runtime.get("unparsed_content", [])}
    variable_ids = {variable["variable_id"] for variable in runtime.get("variables", [])}
    return clause_ids, content_ids, variable_ids


def validate_proposal_for_runtime(proposal: dict[str, Any], runtime: dict[str, Any]) -> list[str]:
    errors = proposal_schema_errors(proposal)
    if errors:
        return errors
    if proposal["template_id"] != runtime["template_id"]:
        errors.append("semantic:template_id:does not match runtime")
    kind = runtime["document"]["document_kind"]
    expected_mode = "source_extracted" if kind == "completed_contract" else "synthetic"
    if proposal["fact_mode"] != expected_mode:
        errors.append(f"semantic:fact_mode:expected {expected_mode} for {kind}")

    clause_ids, content_ids, variable_ids = _runtime_ids(runtime)
    proposed_variable_ids = [binding["variable_id"] for binding in proposal["variable_bindings"]]
    if len(proposed_variable_ids) != len(set(proposed_variable_ids)):
        errors.append("semantic:variable_bindings:variable IDs must be unique")
    if set(proposed_variable_ids) != variable_ids:
        errors.append("semantic:variable_bindings:must map every runtime variable exactly once")

    party_keys = [party["party_key"] for party in proposal["parties"]]
    if len(party_keys) != len(set(party_keys)):
        errors.append("semantic:parties:party keys must be unique")
    party_key_set = set(party_keys)
    if expected_mode == "synthetic" and any(party["name"] is not None for party in proposal["parties"]):
        errors.append("semantic:parties:blank-template proposals must not generate names")
    if expected_mode == "synthetic" and proposal["additional_facts"]:
        errors.append("semantic:additional_facts:blank-template proposals must not add facts")

    evidence_objects: list[dict[str, Any]] = [proposal["document"], *proposal["parties"], *proposal["variable_bindings"], *proposal["additional_facts"]]
    for item in evidence_objects:
        unknown_clauses = set(item["evidence_clause_ids"]) - clause_ids
        unknown_content = set(item["evidence_content_ids"]) - content_ids
        if unknown_clauses:
            errors.append(f"semantic:evidence:unknown clause IDs {sorted(unknown_clauses)}")
        if unknown_content:
            errors.append(f"semantic:evidence:unknown content IDs {sorted(unknown_content)}")

    runtime_variables = {variable["variable_id"]: variable for variable in runtime.get("variables", [])}
    for binding in proposal["variable_bindings"]:
        party_destination = binding["destination"] in {"party_name", "party_contact"}
        if party_destination and binding["party_key"] not in party_key_set:
            errors.append(f"semantic:variables:{binding['variable_id']}:party destination requires a known party_key")
        if not party_destination and binding["party_key"] is not None:
            errors.append(f"semantic:variables:{binding['variable_id']}:non-party destination requires null party_key")
        if binding["destination"] == "missing" and binding["status"] == "provided":
            errors.append(f"semantic:variables:{binding['variable_id']}:missing destination cannot be provided")
        if binding["variable_id"] not in runtime_variables:
            errors.append(f"semantic:variables:{binding['variable_id']}:unknown runtime variable")
            continue
        current_type = runtime_variables[binding["variable_id"]]["value_type"]
        if current_type != "unknown" and binding["value_type"] != current_type:
            errors.append(f"semantic:variables:{binding['variable_id']}:cannot contradict reviewed value_type {current_type}")
        if binding["value_type"] not in VALUE_TYPES:
            errors.append(f"semantic:variables:{binding['variable_id']}:unknown value_type")

    seen_fact_keys: set[tuple[str, str | None, str]] = set()
    for fact in proposal["additional_facts"]:
        party_destination = fact["destination"] == "party_contact"
        if party_destination and fact["party_key"] not in party_key_set:
            errors.append(f"semantic:additional_facts:{fact['key']}:party_contact requires a known party_key")
        if not party_destination and fact["party_key"] is not None:
            errors.append(f"semantic:additional_facts:{fact['key']}:non-party destination requires null party_key")
        identity = (fact["destination"], fact["party_key"], fact["key"])
        if identity in seen_fact_keys:
            errors.append(f"semantic:additional_facts:{fact['key']}:duplicate destination key")
        seen_fact_keys.add(identity)
        if expected_mode == "source_extracted" and not (fact["evidence_clause_ids"] or fact["evidence_content_ids"]):
            errors.append(f"semantic:additional_facts:{fact['key']}:source-extracted fact requires evidence")

    language = proposal["document"]["langue_contrat"]
    if language is not None and language not in runtime["document"]["languages"]:
        errors.append("semantic:document.langue_contrat:must be one of runtime document.languages")
    return sorted(set(errors))


def normalize_proposal_for_runtime(proposal: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Enforce document-kind boundaries while recording every removed proposal."""
    result = copy.deepcopy(proposal)
    warnings = list(result.get("warnings", []))
    runtime_languages = runtime.get("document", {}).get("languages", [])
    proposed_language = result.get("document", {}).get("langue_contrat")
    if proposed_language not in runtime_languages:
        if len(runtime_languages) == 1:
            result["document"]["langue_contrat"] = runtime_languages[0]
            warnings.append(
                f"Normalization: replaced unsupported language {proposed_language!r} with runtime language {runtime_languages[0]!r}."
            )
        else:
            result["document"]["langue_contrat"] = None
            warnings.append("AMBIGUITY: Qwen language is not supported by runtime document.languages.")

    party_occurrences: dict[str, int] = {}
    for party in result.get("parties", []):
        original_key = party["party_key"]
        occurrence = party_occurrences.get(original_key, 0) + 1
        party_occurrences[original_key] = occurrence
        if occurrence > 1:
            party["party_key"] = f"{original_key}_{occurrence:03d}"
            warnings.append(
                f"Normalization: renamed duplicate party_key {original_key} to {party['party_key']}; existing variable references remain attached to the first party."
            )
    if runtime["document"]["document_kind"] == "contract_template":
        additional = result.get("additional_facts", [])
        if additional:
            warnings.append(
                f"Normalization: dropped {len(additional)} Qwen additional_facts item(s); blank templates may use only declared variables."
            )
            result["additional_facts"] = []
        generated_names = [party.get("party_key") for party in result.get("parties", []) if party.get("name") is not None]
        if generated_names:
            warnings.append(
                "Normalization: removed Qwen-generated blank-template party names for " + ", ".join(generated_names) + "."
            )
            for party in result["parties"]:
                party["name"] = None
    else:
        runtime_variable_ids = {variable["variable_id"] for variable in runtime.get("variables", [])}
        unexpected = [
            binding["variable_id"] for binding in result.get("variable_bindings", [])
            if binding["variable_id"] not in runtime_variable_ids
        ]
        if unexpected:
            warnings.append(
                "Normalization: dropped Qwen-invented completed-contract variable IDs: " + ", ".join(unexpected) + "."
            )
            result["variable_bindings"] = [
                binding for binding in result["variable_bindings"]
                if binding["variable_id"] in runtime_variable_ids
            ]

    for binding in result.get("variable_bindings", []):
        if binding["destination"] not in {"party_name", "party_contact"} and binding.get("party_key") is not None:
            warnings.append(
                f"Normalization: removed party_key from non-party variable destination {binding['variable_id']}."
            )
            binding["party_key"] = None

    seen_fact_keys: dict[tuple[str, str | None, str], int] = {}
    for fact in result.get("additional_facts", []):
        if fact["destination"] != "party_contact" and fact.get("party_key") is not None:
            warnings.append(f"Normalization: removed party_key from non-party fact {fact['key']}.")
            fact["party_key"] = None
        identity = (fact["destination"], fact.get("party_key"), fact["key"])
        occurrence = seen_fact_keys.get(identity, 0) + 1
        seen_fact_keys[identity] = occurrence
        if occurrence > 1:
            original_key = fact["key"]
            fact["key"] = f"{original_key}_{occurrence:03d}"
            warnings.append(
                f"Normalization: renamed duplicate extracted fact key {original_key} to {fact['key']}."
            )
    result["warnings"] = warnings
    return result


def _party_names(parties: list[dict[str, Any]], *, source_extracted: bool) -> dict[str, str]:
    company_index = person_index = 0
    names: dict[str, str] = {}
    for party in parties:
        if party["name"]:
            names[party["party_key"]] = party["name"]
        elif source_extracted:
            names[party["party_key"]] = f"AMBIGUITY_PARTY_{len(names) + 1:03d}"
        elif party["type"] == "entreprise":
            company_index += 1
            names[party["party_key"]] = f"DEMO_COMPANY_{company_index:03d}"
        else:
            person_index += 1
            names[party["party_key"]] = f"DEMO_PERSON_{person_index:03d}"
    return names


def _demo_value(binding: dict[str, Any], variable: dict[str, Any], case_date: str, ordinal: int) -> Any | None:
    value_type = binding["value_type"]
    variable_id = binding["variable_id"]
    if binding["status"] != "provided" or binding["destination"] == "missing":
        return None
    if value_type == "date":
        return case_date
    if value_type == "number":
        return 100 + ordinal
    if value_type == "money":
        return f"DEMO_AMOUNT_{ordinal:03d}"
    if value_type == "duration":
        return f"DEMO_DURATION_{ordinal:03d}"
    if value_type == "address":
        return f"DEMO_ADDRESS_{ordinal:03d}"
    if value_type == "identifier":
        return f"DEMO_ID_{ordinal:03d}"
    if value_type == "choice":
        choices = variable.get("allowed_choices", [])
        return choices[0] if choices else None
    if value_type in {"signature", "unknown"}:
        return None
    return f"DEMO_{variable_id.upper()}"


def assemble_agent1_payload(proposal: dict[str, Any], runtime: dict[str, Any], *, case_date: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors = validate_proposal_for_runtime(proposal, runtime)
    try:
        date.fromisoformat(case_date)
    except ValueError:
        errors.append("semantic:case_date:must be YYYY-MM-DD")
    document = proposal.get("document", {})
    if errors:
        return None, sorted(set(errors))

    party_names = _party_names(proposal["parties"], source_extracted=proposal["fact_mode"] == "source_extracted")
    parties_by_key: dict[str, dict[str, Any]] = {}
    for party in proposal["parties"]:
        parties_by_key[party["party_key"]] = {
            "nom": party_names[party["party_key"]],
            "type": party["type"],
            "role": party["role"],
            "informations_contact": {},
        }

    variables = {variable["variable_id"]: variable for variable in runtime.get("variables", [])}
    principales: dict[str, Any] = {}
    autres: dict[str, Any] = {}
    missing: list[str] = []
    for ordinal, binding in enumerate(proposal["variable_bindings"], start=1):
        if binding["destination"] == "party_name":
            value = party_names.get(binding["party_key"])
        else:
            value = _demo_value(binding, variables[binding["variable_id"]], case_date, ordinal)
        if value is None:
            missing.append(binding["variable_id"])
        elif binding["destination"] == "party_contact":
            parties_by_key[binding["party_key"]]["informations_contact"][binding["variable_id"]] = value
        elif binding["destination"] == "autres_informations":
            autres[binding["variable_id"]] = value
        elif binding["destination"] == "informations_principales":
            principales[binding["variable_id"]] = value

    for fact in proposal["additional_facts"]:
        if fact["destination"] == "party_contact":
            parties_by_key[fact["party_key"]]["informations_contact"][fact["key"]] = fact["value"]
        elif fact["destination"] == "autres_informations":
            autres[fact["key"]] = fact["value"]
        else:
            principales[fact["key"]] = fact["value"]

    resolved_document = {
        field: value.strip() if isinstance(value, str) and value.strip() else "AMBIGUITY"
        for field, value in document.items()
        if field in {"type_contrat", "sous_type", "pays", "juridiction", "langue_contrat"}
    }
    payload = {
        "metadata": {
            "session_id": f"demo_{runtime['template_id']}",
            "user_id": "demo_user",
            "date_generation": f"{case_date}T00:00:00Z",
            "pays": resolved_document["pays"],
            "juridiction": resolved_document["juridiction"],
            "type_document": "contrat",
        },
        "type_contrat": resolved_document["type_contrat"],
        "sous_type": resolved_document["sous_type"],
        "parties": list(parties_by_key.values()),
        "informations_principales": principales,
        "informations_generales": {
            "date_creation_contrat": case_date,
            "langue_contrat": resolved_document["langue_contrat"],
        },
        "informations_manquantes": sorted(set(missing)),
        "autres_informations": autres,
    }
    errors = agent1_schema_errors(payload)
    return (None, errors) if errors else (payload, [])


def _compact_runtime(runtime: dict[str, Any]) -> dict[str, Any]:
    """Overflow fallback retaining IDs, metadata, and bounded source contexts."""
    sections: list[dict[str, Any]] = []
    for section in runtime.get("sections", []):
        clauses = []
        for clause in section.get("clauses", []):
            text = clause["text"]
            clauses.append({
                **{key: clause[key] for key in ("clause_id", "heading", "variable_ids", "legal_references")},
                "text": text if len(text) <= 1200 else text[:1200] + " [TRUNCATED_FOR_CONTEXT]",
            })
        sections.append({
            "section_id": section["section_id"], "heading": section["heading"],
            "kind": section["kind"], "clauses": clauses,
        })
    return {
        "schema_version": runtime["schema_version"], "runtime_status": runtime["runtime_status"],
        "template_id": runtime["template_id"], "document": runtime["document"],
        "jurisdictions": runtime["jurisdictions"], "parties": runtime["parties"],
        "variables": runtime["variables"], "sections": sections,
        "unparsed_content": [
            {**content, "text": content["text"] if len(content["text"]) <= 800 else content["text"][:800] + " [TRUNCATED_FOR_CONTEXT]"}
            for content in runtime.get("unparsed_content", [])[:40]
        ],
    }


def _call_qwen(
    runtime: dict[str, Any], *, model_tag: str, num_ctx: int, num_predict: int,
    timeout_seconds: int, base_url: str, compact: bool = False,
) -> dict[str, Any]:
    prompt = (_root() / "prompts" / PROMPT_NAME).read_text(encoding="utf-8")
    supplied = _compact_runtime(runtime) if compact else runtime
    full_prompt = prompt + json.dumps(supplied, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    body = {
        "model": model_tag, "prompt": full_prompt, "stream": False,
        "format": load_schema(PROPOSAL_SCHEMA),
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            envelope = json.load(response)
        return json.loads(envelope["response"])
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Qwen request failed: {type(exc).__name__}: {exc}") from exc


def _is_context_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return any(fragment in message for fragment in ("context", "token", "too long", "length"))


def generate_agent1_inputs(
    runtime_dir: Path, *, model_tag: str, num_ctx: int, num_predict: int,
    timeout_seconds: int, case_date: str, base_url: str = "http://127.0.0.1:11434",
    overwrite_proposals: bool = False,
    qwen_call: Callable[..., dict[str, Any]] | None = None,
    progress: bool = False,
) -> dict[str, int | str]:
    paths = runtime_paths(runtime_dir)
    available = installed_models(base_url)
    if qwen_call is None and model_tag not in available:
        raise RuntimeError(f"Configured model {model_tag!r} is not installed/reachable. Installed models: {available or 'none'}")
    call = qwen_call or _call_qwen
    prompt = (_root() / "prompts" / PROMPT_NAME).read_text(encoding="utf-8")
    counts = {
        "runtime_records": len(paths), "blank_templates": 0, "completed_contracts": 0,
        "generated": 0, "blocked": 0, "qwen_failed": 0, "resumed_proposals": 0,
        "compact_fallbacks": 0,
    }

    for position, runtime_path in enumerate(paths, start=1):
        runtime = _read_json(runtime_path)
        record_dir = runtime_path.parent
        kind = runtime["document"]["document_kind"]
        if progress:
            print(f"[{position}/{len(paths)}] {runtime['template_id']} {kind}", flush=True)
        if kind == "contract_template":
            counts["blank_templates"] += 1
        elif kind == "completed_contract":
            counts["completed_contracts"] += 1
        else:
            counts["blocked"] += 1
            _write_json(record_dir / VALIDATION_FILE, {
                "schema_version": "agent1-input-validation-v1", "template_id": runtime["template_id"],
                "status": "blocked", "runtime_document_kind": kind, "eligible_for_alpaca_export": False,
                "errors": ["unsupported_runtime_document_kind"], "warnings": [],
            })
            continue

        proposal_path = record_dir / PROPOSAL_FILE
        raw_response_path = record_dir / RAW_RESPONSE_FILE
        used_compact = False
        try:
            proposal: dict[str, Any] | None = None
            if proposal_path.is_file() and not overwrite_proposals:
                candidate = _read_json(proposal_path)
                if not validate_proposal_for_runtime(candidate, runtime):
                    proposal = candidate
                    counts["resumed_proposals"] += 1
                elif raw_response_path.is_file():
                    normalized = normalize_proposal_for_runtime(_read_json(raw_response_path), runtime)
                    if not validate_proposal_for_runtime(normalized, runtime):
                        proposal = normalized
                        _write_json(proposal_path, normalized)
                        counts["resumed_proposals"] += 1
            if proposal is None:
                try:
                    raw_proposal = call(
                        runtime, model_tag=model_tag, num_ctx=num_ctx, num_predict=num_predict,
                        timeout_seconds=timeout_seconds, base_url=base_url, compact=False,
                    )
                except Exception as exc:
                    if not _is_context_error(exc):
                        raise
                    raw_proposal = call(
                        runtime, model_tag=model_tag, num_ctx=num_ctx, num_predict=num_predict,
                        timeout_seconds=timeout_seconds, base_url=base_url, compact=True,
                    )
                    used_compact = True
                    counts["compact_fallbacks"] += 1
                _write_json(raw_response_path, raw_proposal)
                proposal = normalize_proposal_for_runtime(raw_proposal, runtime)
                _write_json(proposal_path, proposal)

            proposal_errors = validate_proposal_for_runtime(proposal, runtime)
            if proposal_errors:
                counts["blocked"] += 1
                _write_json(record_dir / VALIDATION_FILE, {
                    "schema_version": "agent1-input-validation-v1", "template_id": runtime["template_id"],
                    "status": "blocked", "runtime_document_kind": kind, "fact_mode": proposal.get("fact_mode"),
                    "eligible_for_alpaca_export": False, "errors": proposal_errors,
                    "warnings": proposal.get("warnings", []), "used_compact_fallback": used_compact,
                })
                continue
            payload, errors = assemble_agent1_payload(proposal, runtime, case_date=case_date)
            if errors or payload is None:
                counts["blocked"] += 1
                _write_json(record_dir / VALIDATION_FILE, {
                    "schema_version": "agent1-input-validation-v1", "template_id": runtime["template_id"],
                    "status": "blocked", "runtime_document_kind": kind, "fact_mode": proposal["fact_mode"],
                    "eligible_for_alpaca_export": False, "errors": errors,
                    "warnings": proposal["warnings"], "used_compact_fallback": used_compact,
                })
                continue
            _write_json(record_dir / INPUT_FILE, payload)
            unresolved_fields = [
                field for field in ("type_contrat", "sous_type", "pays", "juridiction", "langue_contrat")
                if not isinstance(proposal["document"].get(field), str)
                or not proposal["document"][field].strip()
                or proposal["document"][field].strip().casefold() == "ambiguity"
            ]
            ambiguity_warnings = [f"AMBIGUITY: required Agent 1 field {field} is unresolved" for field in unresolved_fields]
            unresolved_parties = [party["party_key"] for party in proposal["parties"] if not party.get("name")]
            if proposal["fact_mode"] == "source_extracted":
                ambiguity_warnings.extend(
                    f"AMBIGUITY: source party name is unresolved for {party_key}" for party_key in unresolved_parties
                )
            _write_json(record_dir / VALIDATION_FILE, {
                "schema_version": "agent1-input-validation-v1", "template_id": runtime["template_id"],
                "status": "demo_unreviewed", "runtime_document_kind": kind, "fact_mode": proposal["fact_mode"],
                "agent1_schema_valid": True, "all_template_variables_accounted_for": True,
                "eligible_for_alpaca_export": False, "privacy_review_required": kind == "completed_contract",
                "qwen_model": model_tag, "num_ctx": num_ctx, "num_predict": num_predict,
                "prompt_sha256": sha256_text(prompt),
                "runtime_sha256": sha256_text(json.dumps(runtime, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
                "warnings": proposal["warnings"] + ambiguity_warnings, "errors": [],
                "unresolved_required_fields": unresolved_fields,
                "unresolved_party_names": unresolved_parties if proposal["fact_mode"] == "source_extracted" else [],
                "used_compact_fallback": used_compact,
            })
            counts["generated"] += 1
        except Exception as exc:
            counts["qwen_failed"] += 1
            _write_json(record_dir / VALIDATION_FILE, {
                "schema_version": "agent1-input-validation-v1", "template_id": runtime["template_id"],
                "status": "qwen_failed", "runtime_document_kind": kind,
                "eligible_for_alpaca_export": False,
                "errors": [f"{type(exc).__name__}: {exc}"], "warnings": [],
            })

    summary: dict[str, int | str] = {**counts, "status": "demo_unreviewed", "model": model_tag}
    _write_json(runtime_dir / SUMMARY_FILE, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=Path("artifacts/runtime-demo"))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--case-date", default="2026-08-14")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--overwrite-proposals", action="store_true")
    args = parser.parse_args()
    if args.num_ctx < 1 or args.num_predict < 1 or args.timeout_seconds < 1:
        raise SystemExit("context, prediction, and timeout values must be positive")
    summary = generate_agent1_inputs(
        args.runtime_dir, model_tag=args.model, num_ctx=args.num_ctx,
        num_predict=args.num_predict, timeout_seconds=args.timeout_seconds,
        case_date=args.case_date, base_url=args.base_url,
        overwrite_proposals=args.overwrite_proposals,
        progress=True,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
