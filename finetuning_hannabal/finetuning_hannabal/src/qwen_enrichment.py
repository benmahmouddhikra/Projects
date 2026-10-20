"""Compact, contract-only Qwen enrichment for canonical template records.

Canonical records remain the evidence-bearing source of truth.  This module only
passes section/clause text and existing placeholder metadata to Qwen, and merges
schema-valid labels back into a separate canonical artifact set.
"""

from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator

from normalize import sha256_text, stable_id
from ollama_extract import OllamaConfig, installed_models
from validate import schema_errors, validate_canonical


CONTRACT_KINDS = {"contract_template", "completed_contract"}
PACKET_SCHEMA = "qwen-contract-label-packet-v1.schema.json"
RESPONSE_SCHEMA = "qwen-contract-label-response-v1.schema.json"
PROMPT_NAME = "qwen-contract-packet-labeling-v1.txt"


def _feature_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _canonical_paths(canonical_dir: Path) -> list[Path]:
    if canonical_dir.is_file():
        return [canonical_dir]
    return sorted(canonical_dir.rglob("canonical.json"), key=lambda path: path.as_posix().casefold())


def _artifact_key(canonical_path: Path, canonical_dir: Path) -> str:
    if canonical_dir.is_file():
        return canonical_path.parent.name
    return canonical_path.parent.relative_to(canonical_dir).as_posix()


def _contract_record(record: dict[str, Any]) -> bool:
    return record.get("document", {}).get("document_kind") in CONTRACT_KINDS


def _packet_variables(record: dict[str, Any], clause_ids: set[str]) -> list[dict[str, Any]]:
    referenced = {
        variable_id
        for section in record["sections"]
        for clause in section["clauses"]
        if clause["clause_id"] in clause_ids
        for variable_id in clause["variable_ids"]
    }
    return [
        {
            "variable_id": variable["variable_id"],
            "canonical_token": variable["canonical_token"],
            "value_type": variable["value_type"],
            "requirement_status": variable["requirement_status"],
        }
        for variable in record["variables"]
        if variable["variable_id"] in referenced
    ]


def _make_packet(record: dict[str, Any], section: dict[str, Any], clauses: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    clause_ids = {clause["clause_id"] for clause in clauses}
    packet_id = stable_id("pkt", record["template_id"], section["section_id"], scope, *sorted(clause_ids))
    return {
        "schema_version": "qwen-contract-label-packet-v1",
        "packet_id": packet_id,
        "template_id": record["template_id"],
        "document_kind": record["document"]["document_kind"],
        "scope": scope,
        "section": {
            "section_id": section["section_id"],
            "heading": section["heading"],
            "kind": section["kind"],
            "requirement_status": section["requirement_status"],
        },
        "clauses": [
            {
                "clause_id": clause["clause_id"],
                "heading": clause["heading"],
                "text": clause["text"],
                "requirement_status": clause["requirement_status"],
                "variable_ids": clause["variable_ids"],
            }
            for clause in clauses
        ],
        "variables": _packet_variables(record, clause_ids),
    }


def build_packets(record: dict[str, Any], *, max_chars: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build model packets without accepting non-contract documents."""
    if not _contract_record(record):
        return [], [{"code": "non_contract_document_kind", "message": "Record is outside contract-only enrichment scope."}]
    packets: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for section in record["sections"]:
        if not section["clauses"]:
            failures.append({"code": "section_without_clauses", "message": f"{section['section_id']} cannot form a packet."})
            continue
        section_packet = _make_packet(record, section, section["clauses"], "section")
        if len(json.dumps(section_packet, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
            packets.append(section_packet)
            continue
        for clause in section["clauses"]:
            packet = _make_packet(record, section, [clause], "clause")
            packets.append(packet)
            if len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"))) > max_chars:
                failures.append({
                    "code": "clause_packet_exceeds_max_chars",
                    "message": f"{packet['packet_id']} exceeds max_chars and is retained untruncated for evidence fidelity.",
                })
    return packets, failures


def build_packet_artifacts(canonical_dir: Path, output_dir: Path, *, max_chars: int) -> dict[str, int]:
    """Write contract-only packets. Existing packet IDs make reruns deterministic."""
    packets_dir = output_dir / "packets"
    records = 0
    skipped = 0
    packets_written = 0
    failures: list[dict[str, Any]] = []
    packet_index: list[dict[str, str]] = []
    seen_templates: set[str] = set()
    for canonical_path in _canonical_paths(canonical_dir):
        record = _read_json(canonical_path)
        artifact_key = _artifact_key(canonical_path, canonical_dir)
        if not _contract_record(record):
            skipped += 1
            failures.append({"artifact_key": artifact_key, "code": "non_contract_document_kind"})
            continue
        records += 1
        if record["template_id"] in seen_templates:
            failures.append({"artifact_key": artifact_key, "code": "duplicate_template_id"})
            continue
        seen_templates.add(record["template_id"])
        packets, packet_failures = build_packets(record, max_chars=max_chars)
        for failure in packet_failures:
            failures.append({"artifact_key": artifact_key, "template_id": record["template_id"], **failure})
        for packet in packets:
            errors = schema_errors(packet, PACKET_SCHEMA)
            if errors:
                raise ValueError("Packet schema validation failed: " + "; ".join(errors))
            _write_json(packets_dir / f"{packet['packet_id']}.json", packet)
            packet_index.append({"packet_id": packet["packet_id"], "template_id": packet["template_id"], "artifact_key": artifact_key})
            packets_written += 1
    _write_json(output_dir / "packet-index.json", packet_index)
    _write_json(output_dir / "packet-failures.json", failures)
    summary = {"contract_records": records, "non_contract_skipped": skipped, "packets": packets_written, "failures": len(failures)}
    _write_json(output_dir / "packet-summary.json", summary)
    return summary


def validate_response_for_packet(response: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    errors = schema_errors(response, RESPONSE_SCHEMA)
    if errors:
        return errors
    for field in ("packet_id", "template_id"):
        if response[field] != packet[field]:
            errors.append(f"semantic:{field}:does not match packet")
    if response["section_id"] != packet["section"]["section_id"]:
        errors.append("semantic:section_id:does not match packet")

    def exact_ids(field: str, actual: list[str], expected: list[str]) -> None:
        if len(actual) != len(set(actual)):
            errors.append(f"semantic:{field}:IDs must be unique")
        if set(actual) != set(expected):
            errors.append(f"semantic:{field}:must label exactly the packet IDs")

    exact_ids("clause_labels", [label["clause_id"] for label in response["clause_labels"]], [clause["clause_id"] for clause in packet["clauses"]])
    exact_ids("variable_labels", [label["variable_id"] for label in response["variable_labels"]], [variable["variable_id"] for variable in packet["variables"]])
    return errors


def _packet_paths(packets_dir: Path) -> list[Path]:
    index_path = packets_dir / "packet-index.json"
    if not index_path.is_file():
        return []
    index = _read_json(index_path)
    if not isinstance(index, list):
        raise ValueError("packet-index.json must be an array")
    paths = [packets_dir / "packets" / f"{entry['packet_id']}.json" for entry in index]
    if len(paths) != len(set(paths)):
        raise ValueError("packet-index.json contains duplicate packet IDs")
    return paths


def _call_ollama(packet: dict[str, Any], *, schema: dict[str, Any], prompt: str, config: OllamaConfig) -> dict[str, Any]:
    body = {
        "model": config.model_tag,
        "prompt": prompt + "\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True),
        "stream": False,
        "format": schema,
        "options": {"temperature": 0, "num_ctx": config.num_ctx},
    }
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/api/generate",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as http_response:
            envelope = json.load(http_response)
        return json.loads(envelope["response"])
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"Qwen packet request failed: {type(exc).__name__}: {exc}") from exc


def run_packet_labeling(packets_dir: Path, *, model_tag: str, num_ctx: int, retry_failures: bool = False) -> dict[str, int]:
    """Call Qwen sequentially; only schema- and packet-valid responses are retained."""
    output_dir = packets_dir
    response_dir = output_dir / "responses"
    failure_dir = output_dir / "failures"
    prompt_path = _feature_root() / "prompts" / PROMPT_NAME
    prompt = prompt_path.read_text(encoding="utf-8")
    schema = _read_json(_feature_root() / "schemas" / RESPONSE_SCHEMA)
    validator = Draft202012Validator(schema)
    available = installed_models()
    config = OllamaConfig(model_tag=model_tag, num_ctx=num_ctx)
    processed = succeeded = failed = resumed = 0
    for packet_path in _packet_paths(packets_dir):
        packet = _read_json(packet_path)
        response_path = response_dir / f"{packet['packet_id']}.json"
        failure_path = failure_dir / f"{packet['packet_id']}.json"
        if response_path.is_file():
            try:
                if not validate_response_for_packet(_read_json(response_path), packet):
                    resumed += 1
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        if failure_path.is_file() and not retry_failures:
            resumed += 1
            continue
        processed += 1
        try:
            if model_tag not in available:
                raise RuntimeError(f"Configured model {model_tag!r} is not installed/reachable. Installed models: {available or 'none'}.")
            response = _call_ollama(packet, schema=schema, prompt=prompt, config=config)
            errors = [error.message for error in validator.iter_errors(response)] + validate_response_for_packet(response, packet)
            if errors:
                raise ValueError("; ".join(sorted(set(errors))))
            _write_json(response_path, response)
            failure_path.unlink(missing_ok=True)
            succeeded += 1
        except Exception as exc:
            _write_json(failure_path, {"packet_id": packet["packet_id"], "template_id": packet["template_id"], "error": f"{type(exc).__name__}: {exc}"})
            failed += 1
    _write_json(output_dir / "run-config.json", {
        "schema_version": "qwen-contract-packet-run-v1", "model_tag": model_tag, "num_ctx": num_ctx,
        "prompt_sha256": sha256_text(prompt), "response_schema": RESPONSE_SCHEMA,
    })
    summary = {"processed": processed, "succeeded": succeeded, "failed": failed, "resumed": resumed}
    _write_json(output_dir / "run-summary.json", summary)
    return summary


def _label_value(labels: list[tuple[str, Any]], target: str) -> Any:
    values = {value for source, value in labels if source == target}
    if len(values) != 1:
        raise ValueError(f"Conflicting labels for {target}")
    return values.pop()


def merge_contract_labels(record: dict[str, Any], packets: list[dict[str, Any]], responses: list[dict[str, Any]], *, model_tag: str | None, prompt_sha256: str | None) -> dict[str, Any]:
    """Atomically merge a complete response set into a copy of canonical data."""
    if not _contract_record(record):
        raise ValueError("Non-contract records are outside enrichment scope")
    packet_by_id = {packet["packet_id"]: packet for packet in packets}
    if len(packet_by_id) != len(packets):
        raise ValueError("Duplicate packet IDs")
    if {response["packet_id"] for response in responses} != set(packet_by_id):
        raise ValueError("Missing or unexpected packet response")
    section_labels: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    clause_labels: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    variable_labels: dict[str, list[tuple[str, tuple[str, str]]]] = defaultdict(list)
    for response in responses:
        packet = packet_by_id[response["packet_id"]]
        errors = validate_response_for_packet(response, packet)
        if errors:
            raise ValueError("Invalid response: " + "; ".join(errors))
        section_labels[response["section_id"]].extend([
            ("kind", response["section_kind"]), ("requirement_status", response["section_requirement_status"]),
        ])
        for label in response["clause_labels"]:
            clause_labels[label["clause_id"]].append(("requirement_status", label["requirement_status"]))
        for label in response["variable_labels"]:
            variable_labels[label["variable_id"]].extend([
                ("value_type", (label["value_type"], label["requirement_status"])),
            ])
    result = copy.deepcopy(record)
    for section in result["sections"]:
        if section["section_id"] in section_labels:
            labels = section_labels[section["section_id"]]
            section["kind"] = _label_value(labels, "kind")
            section["requirement_status"] = _label_value(labels, "requirement_status")
        for clause in section["clauses"]:
            if clause["clause_id"] in clause_labels:
                clause["requirement_status"] = _label_value(clause_labels[clause["clause_id"]], "requirement_status")
    for variable in result["variables"]:
        if variable["variable_id"] in variable_labels:
            value_type, requirement_status = _label_value(variable_labels[variable["variable_id"]], "value_type")
            variable["value_type"] = value_type
            variable["requirement_status"] = requirement_status
    action_id = stable_id("action", result["template_id"], "qwen_packet_enrichment", *sorted(packet_by_id))
    result["audit_history"].append({
        "action_id": action_id, "action": "qwen_packet_enrichment", "actor_type": "model",
        "actor": model_tag or "qwen", "details": f"Applied validated labels from {len(packets)} compact packet(s); source text and evidence were unchanged.",
    })
    result["versions"]["model"] = model_tag
    result["versions"]["prompt"] = prompt_sha256
    return result


def merge_artifacts(canonical_dir: Path, packets_dir: Path, output_dir: Path) -> dict[str, int]:
    """Create enriched canonical copies only when every expected packet succeeded."""
    packets = [_read_json(path) for path in _packet_paths(packets_dir)]
    packets_by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        packets_by_template[packet["template_id"]].append(packet)
    config_path = packets_dir / "run-config.json"
    config = _read_json(config_path) if config_path.is_file() else {}
    model_tag, prompt_sha256 = config.get("model_tag"), config.get("prompt_sha256")
    merged = skipped = failed = 0
    index: list[dict[str, Any]] = []
    seen_templates: set[str] = set()
    for canonical_path in _canonical_paths(canonical_dir):
        record = _read_json(canonical_path)
        artifact_key = _artifact_key(canonical_path, canonical_dir)
        report_path = output_dir / "records" / artifact_key / "merge.json"
        if not _contract_record(record):
            skipped += 1
            _write_json(report_path, {"status": "skipped", "reason": "non_contract_document_kind"})
            continue
        if record["template_id"] in seen_templates:
            failed += 1
            _write_json(report_path, {"status": "failed", "reason": "duplicate_template_id"})
            continue
        seen_templates.add(record["template_id"])
        expected = packets_by_template.get(record["template_id"], [])
        responses: list[dict[str, Any]] = []
        failure_reasons: list[str] = []
        for packet in expected:
            response_path = packets_dir / "responses" / f"{packet['packet_id']}.json"
            failure_path = packets_dir / "failures" / f"{packet['packet_id']}.json"
            if failure_path.is_file() or not response_path.is_file():
                failure_reasons.append(f"missing_valid_response:{packet['packet_id']}")
                continue
            try:
                responses.append(_read_json(response_path))
            except (OSError, json.JSONDecodeError):
                failure_reasons.append(f"unreadable_response:{packet['packet_id']}")
        if not expected:
            failure_reasons.append("no_packets")
        if failure_reasons:
            failed += 1
            _write_json(report_path, {"status": "failed", "reasons": failure_reasons})
            index.append({"artifact_key": artifact_key, "template_id": record["template_id"], "status": "failed"})
            continue
        try:
            enriched = merge_contract_labels(record, expected, responses, model_tag=model_tag, prompt_sha256=prompt_sha256)
            extraction = _read_json(canonical_path.parent / "extraction.json")
            errors = validate_canonical(enriched, extraction)
            if errors:
                raise ValueError("; ".join(errors))
            target_dir = output_dir / "records" / artifact_key
            _write_json(target_dir / "canonical.json", enriched)
            _write_json(target_dir / "validation.json", {"canonical_schema_and_semantics_valid": True, "canonical_errors": []})
            _write_json(report_path, {"status": "merged", "packet_count": len(expected)})
            index.append({"artifact_key": artifact_key, "template_id": record["template_id"], "status": "merged"})
            merged += 1
        except Exception as exc:
            failed += 1
            _write_json(report_path, {"status": "failed", "reasons": [f"{type(exc).__name__}: {exc}"]})
            index.append({"artifact_key": artifact_key, "template_id": record["template_id"], "status": "failed"})
    _write_json(output_dir / "index.json", index)
    summary = {"merged": merged, "failed": failed, "non_contract_skipped": skipped}
    _write_json(output_dir / "summary.json", summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packets = commands.add_parser("packets")
    packets.add_argument("--canonical-dir", type=Path, required=True)
    packets.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen-packets"))
    packets.add_argument("--max-chars", type=int, default=6000)
    run = commands.add_parser("run")
    run.add_argument("--packets-dir", type=Path, default=Path("artifacts/qwen-packets"))
    run.add_argument("--model", default="qwen2.5:3b")
    run.add_argument("--num-ctx", type=int, default=4096)
    run.add_argument("--retry-failures", action="store_true")
    merge = commands.add_parser("merge")
    merge.add_argument("--canonical-dir", type=Path, required=True)
    merge.add_argument("--packets-dir", type=Path, default=Path("artifacts/qwen-packets"))
    merge.add_argument("--output-dir", type=Path, default=Path("artifacts/enriched-canonical"))
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "packets":
        if args.max_chars < 1:
            raise SystemExit("--max-chars must be positive")
        summary = build_packet_artifacts(args.canonical_dir, args.output_dir, max_chars=args.max_chars)
    elif args.command == "run":
        summary = run_packet_labeling(args.packets_dir, model_tag=args.model, num_ctx=args.num_ctx, retry_failures=args.retry_failures)
    else:
        summary = merge_artifacts(args.canonical_dir, args.packets_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
