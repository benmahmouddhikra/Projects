from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

from agent2_intake import adapt_agent2_input, contains_forbidden_model_identifier
from alpaca_export import export_alpaca_dataset, validate_alpaca_row
from contract_target_builder import build_contract_target, build_retrieved_evidence, validate_contract_target


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_record(feature_root: Path) -> tuple[str, Path]:
    runtime_dir = feature_root / "artifacts" / "runtime-demo"
    for entry in _read(runtime_dir / "index.json"):
        record_dir = runtime_dir / "records" / entry["artifact_key"]
        if (record_dir / "agent1-input.json").is_file():
            validation = _read(record_dir / "agent1-input-validation.json")
            if validation.get("status") == "demo_unreviewed" and not validation.get("errors"):
                return entry["artifact_key"], record_dir
    raise AssertionError("The checked-in demo corpus has no valid Agent 1 input")


def _adapt(feature_root: Path):
    artifact_key, record_dir = _valid_record(feature_root)
    payload = _read(record_dir / "agent1-input.json")
    runtime = _read(record_dir / "runtime.json")
    upstream = _read(record_dir / "agent1-input-validation.json")
    proposal_path = record_dir / "agent1-qwen-proposal.json"
    proposal = _read(proposal_path) if proposal_path.is_file() else None
    facts, report = adapt_agent2_input(
        payload, runtime, upstream,
        artifact_key=artifact_key,
        raw_input_ref="agent1-input.json",
        proposal=proposal,
    )
    assert facts is not None, report
    return artifact_key, record_dir, payload, runtime, facts, report


def test_agent2_intake_removes_operational_ids_and_maps_every_runtime_variable(feature_root):
    _, _, payload, runtime, facts, report = _adapt(feature_root)
    assert payload["metadata"]["session_id"]
    assert payload["metadata"]["user_id"]
    assert not contains_forbidden_model_identifier(facts)
    assert report["forbidden_identifier_paths"] == []
    assert report["status"] == "demo_unreviewed"
    assert report["training_eligible"] is False
    assert [item["variable_id"] for item in facts["variable_bindings"]] == [
        item["variable_id"] for item in runtime["variables"]
    ]
    assert all(
        item["value"] == item["canonical_token"]
        for item in facts["variable_bindings"]
        if item["status"] in {"missing", "ambiguous"}
    )


def test_agent2_intake_rejects_conflicting_duplicate_fact_locations(feature_root):
    artifact_key, record_dir = _valid_record(feature_root)
    payload = _read(record_dir / "agent1-input.json")
    runtime = _read(record_dir / "runtime.json")
    upstream = _read(record_dir / "agent1-input-validation.json")
    if not runtime["variables"]:
        # The corpus begins with completed contracts; select a blank record with variables.
        runtime_dir = feature_root / "artifacts" / "runtime-demo"
        for entry in _read(runtime_dir / "index.json"):
            candidate = runtime_dir / "records" / entry["artifact_key"]
            if (candidate / "agent1-input.json").is_file() and _read(candidate / "runtime.json")["variables"]:
                artifact_key, record_dir = entry["artifact_key"], candidate
                payload = _read(candidate / "agent1-input.json")
                runtime = _read(candidate / "runtime.json")
                upstream = _read(candidate / "agent1-input-validation.json")
                break
    variable_id = runtime["variables"][0]["variable_id"]
    payload = copy.deepcopy(payload)
    payload["informations_principales"][variable_id] = "VALUE_A"
    payload["autres_informations"][variable_id] = "VALUE_B"
    facts, report = adapt_agent2_input(
        payload, runtime, upstream,
        artifact_key=artifact_key,
        raw_input_ref="agent1-input.json",
        proposal=None,
    )
    assert facts is None
    assert any("conflicting values" in error for error in report["errors"])


def test_target_is_complete_source_order_with_only_deterministic_substitution(feature_root):
    artifact_key, _, _, runtime, facts, _ = _adapt(feature_root)
    canonical = _read(feature_root / "artifacts" / "enriched-canonical" / "records" / artifact_key / "canonical.json")
    extraction = _read(feature_root / "artifacts" / "full" / "records" / artifact_key / "extraction.json")
    target, report = build_contract_target(facts, runtime, canonical, extraction)
    assert target is not None, report
    assert report["valid_for_demo_export"] is True
    assert report["training_eligible"] is False
    assert report["checks"]["unsupported_legal_content"]["status"] == "passed"
    assert len(report["passages"]) == len(extraction["blocks"])

    invented = target + "\nINVENTED LEGAL CLAUSE"
    invented_report = validate_contract_target(invented, facts, runtime, canonical, extraction)
    assert invented_report["valid_for_demo_export"] is False
    assert any("differs from deterministic source reconstruction" in error for error in invented_report["errors"])


def test_retrieval_package_is_exact_runtime_evidence(feature_root):
    _, _, _, runtime, _, _ = _adapt(feature_root)
    evidence, errors = build_retrieved_evidence(runtime, demo=True)
    assert errors == []
    assert evidence is not None
    assert evidence["template"] == runtime
    assert evidence["clauses"] == []
    assert evidence["retrieval_status"] == "demo_runtime_evidence"


def test_one_record_demo_export_is_strict_resumable_and_production_refuses_unreviewed(feature_root, tmp_path):
    artifact_key, source_record = _valid_record(feature_root)
    runtime = _read(source_record / "runtime.json")
    runtime_dir = tmp_path / "runtime-demo"
    copied_record = runtime_dir / "records" / artifact_key
    shutil.copytree(source_record, copied_record)
    (runtime_dir / "index.json").write_text(json.dumps([{
        "artifact_key": artifact_key,
        "runtime": f"records/{artifact_key}/runtime.json",
        "runtime_status": runtime["runtime_status"],
        "template_id": runtime["template_id"],
    }]), encoding="utf-8")

    canonical_dir = tmp_path / "canonical"
    extraction_dir = tmp_path / "extraction"
    (canonical_dir / artifact_key).mkdir(parents=True)
    (extraction_dir / artifact_key).mkdir(parents=True)
    shutil.copy2(
        feature_root / "artifacts" / "enriched-canonical" / "records" / artifact_key / "canonical.json",
        canonical_dir / artifact_key / "canonical.json",
    )
    shutil.copy2(
        feature_root / "artifacts" / "full" / "records" / artifact_key / "extraction.json",
        extraction_dir / artifact_key / "extraction.json",
    )

    output = tmp_path / "alpaca-demo"
    first = export_alpaca_dataset(
        runtime_dir, output, canonical_dir=canonical_dir,
        extraction_dir=extraction_dir, mode="demo", expected_records=1,
    )
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = export_alpaca_dataset(
        runtime_dir, output, canonical_dir=canonical_dir,
        extraction_dir=extraction_dir, mode="demo", expected_records=1,
    )
    assert first == second
    assert first_bytes == {path.name: path.read_bytes() for path in output.iterdir()}
    assert first["exported_rows"] == 1
    row = json.loads((output / "alpaca.jsonl").read_text(encoding="utf-8"))
    assert set(row) == {"instruction", "input", "output"}
    assert validate_alpaca_row(row) == []
    assert "session_id" not in json.dumps(row)
    assert "user_id" not in json.dumps(row)
    manifest = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest["status"] == "demo_unreviewed"
    assert manifest["training_eligible"] is False

    production = export_alpaca_dataset(
        runtime_dir, tmp_path / "alpaca-production", canonical_dir=canonical_dir,
        extraction_dir=extraction_dir, mode="production", expected_records=1,
    )
    assert production["exported_rows"] == 0
    assert production["rejected_records"] == 1

