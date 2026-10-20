from __future__ import annotations

import copy
import json

from agent1_demo_inputs import (
    INPUT_FILE,
    PROPOSAL_FILE,
    SUMMARY_FILE,
    VALIDATION_FILE,
    agent1_schema_errors,
    assemble_agent1_payload,
    generate_agent1_inputs,
    normalize_proposal_for_runtime,
    proposal_schema_errors,
    validate_proposal_for_runtime,
)


def _runtime(*, completed: bool = False) -> dict:
    return {
        "schema_version": "contract-template-runtime-v1",
        "runtime_status": "demo_unreviewed",
        "template_id": "tpl_0123456789abcdef0123",
        "canonical_ref": {
            "schema_version": "contract-template-v1",
            "template_id": "tpl_0123456789abcdef0123",
            "source_ids": ["src_0123456789abcdef0123"],
        },
        "document": {
            "document_kind": "completed_contract" if completed else "contract_template",
            "template_state": "completed" if completed else "blank",
            "title": "Demo agreement",
            "contract_family": "unknown",
            "contract_type": None,
            "languages": ["en"],
        },
        "jurisdictions": [],
        "parties": [],
        "variables": [] if completed else [
            {
                "variable_id": "employer_name", "canonical_token": "[EMPLOYER_NAME]",
                "value_type": "unknown", "party_id": None,
                "requirement_status": "unspecified", "allowed_choices": [],
            },
            {
                "variable_id": "start_date", "canonical_token": "[START_DATE]",
                "value_type": "unknown", "party_id": None,
                "requirement_status": "unspecified", "allowed_choices": [],
            },
        ],
        "sections": [
            {
                "section_id": "sec_0123456789abcdef0123", "order": 1,
                "kind": "parties", "heading": "Parties",
                "requirement_status": "unspecified", "condition": None,
                "clauses": [
                    {
                        "clause_id": "clause_0123456789abcdef0123", "order": 1,
                        "clause_type": "unclassified", "heading": None,
                        "text": "Demo Company and Demo Person",
                        "variable_ids": [] if completed else ["employer_name", "start_date"],
                        "legal_references": [], "requirement_status": "unspecified", "condition": None,
                    }
                ],
            }
        ],
        "unparsed_content": [
            {
                "content_id": "unparsed_0123456789abcdef0123",
                "text": "Address: Source Address", "reason": "no_boundary",
            }
        ],
    }


def _proposal(*, completed: bool = False) -> dict:
    return {
        "schema_version": "qwen-agent1-demo-proposal-v1",
        "template_id": "tpl_0123456789abcdef0123",
        "fact_mode": "source_extracted" if completed else "synthetic",
        "document": {
            "type_contrat": "demo agreement",
            "sous_type": "demo subtype",
            "pays": "Demo Country",
            "juridiction": "Demo Jurisdiction",
            "langue_contrat": "en",
            "evidence_clause_ids": ["clause_0123456789abcdef0123"],
            "evidence_content_ids": [],
        },
        "parties": [
            {
                "party_key": "party_employer",
                "name": "Source Company" if completed else None,
                "type": "entreprise",
                "role": "employer",
                "evidence_clause_ids": ["clause_0123456789abcdef0123"],
                "evidence_content_ids": [],
            }
        ],
        "variable_bindings": [] if completed else [
            {
                "variable_id": "employer_name", "value_type": "text",
                "destination": "party_name", "party_key": "party_employer",
                "status": "provided", "evidence_clause_ids": ["clause_0123456789abcdef0123"],
                "evidence_content_ids": [],
            },
            {
                "variable_id": "start_date", "value_type": "date",
                "destination": "informations_principales", "party_key": None,
                "status": "provided", "evidence_clause_ids": ["clause_0123456789abcdef0123"],
                "evidence_content_ids": [],
            },
        ],
        "additional_facts": [
            {
                "key": "address", "destination": "party_contact",
                "party_key": "party_employer", "value": "Source Address",
                "evidence_clause_ids": [],
                "evidence_content_ids": ["unparsed_0123456789abcdef0123"],
            }
        ] if completed else [],
        "warnings": [],
    }


def test_blank_proposal_maps_every_variable_and_builds_fictional_payload():
    runtime = _runtime()
    proposal = _proposal()
    assert validate_proposal_for_runtime(proposal, runtime) == []
    payload, errors = assemble_agent1_payload(proposal, runtime, case_date="2026-08-14")
    assert errors == []
    assert payload is not None
    assert agent1_schema_errors(payload) == []
    assert payload["parties"][0]["nom"] == "DEMO_COMPANY_001"
    assert payload["informations_principales"]["start_date"] == "2026-08-14"
    assert payload["metadata"]["date_generation"] == "2026-08-14T00:00:00Z"


def test_completed_contract_preserves_source_extracted_facts_and_stays_unreviewed(tmp_path):
    runtime = _runtime(completed=True)
    proposal = _proposal(completed=True)
    records = tmp_path / "records" / "one"
    records.mkdir(parents=True)
    (records / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
    (tmp_path / "index.json").write_text(json.dumps([
        {"runtime": "records/one/runtime.json", "template_id": runtime["template_id"]}
    ]), encoding="utf-8")

    def fake_qwen(*_args, **_kwargs):
        return proposal

    summary = generate_agent1_inputs(
        tmp_path, model_tag="qwen-test", num_ctx=32768, num_predict=8192,
        timeout_seconds=1, case_date="2026-08-14", qwen_call=fake_qwen,
    )
    assert summary["generated"] == 1
    assert summary["completed_contracts"] == 1
    payload = json.loads((records / INPUT_FILE).read_text(encoding="utf-8"))
    validation = json.loads((records / VALIDATION_FILE).read_text(encoding="utf-8"))
    assert payload["parties"][0]["nom"] == "Source Company"
    assert payload["parties"][0]["informations_contact"]["address"] == "Source Address"
    assert validation["privacy_review_required"] is True
    assert validation["eligible_for_alpaca_export"] is False
    assert (records / PROPOSAL_FILE).is_file()
    assert (tmp_path / SUMMARY_FILE).is_file()


def test_proposal_rejects_missing_variables_unknown_evidence_and_injection():
    runtime = _runtime()
    proposal = _proposal()
    missing = copy.deepcopy(proposal)
    missing["variable_bindings"].pop()
    assert any("must map every runtime variable" in item for item in validate_proposal_for_runtime(missing, runtime))

    unknown_evidence = copy.deepcopy(proposal)
    unknown_evidence["document"]["evidence_clause_ids"] = ["clause_ffffffffffffffffffff"]
    assert any("unknown clause" in item for item in validate_proposal_for_runtime(unknown_evidence, runtime))

    injected = copy.deepcopy(proposal)
    injected["invented_clause"] = "unsupported legal text"
    assert proposal_schema_errors(injected)

    invented_variable = copy.deepcopy(proposal)
    invented_variable["variable_bindings"][0]["variable_id"] = "invented_variable"
    errors = validate_proposal_for_runtime(invented_variable, runtime)
    assert any("unknown runtime variable" in item for item in errors)


def test_completed_fact_requires_source_evidence_and_blank_names_are_not_generated_by_qwen():
    completed_runtime = _runtime(completed=True)
    completed = _proposal(completed=True)
    completed["additional_facts"][0]["evidence_content_ids"] = []
    assert any("requires evidence" in item for item in validate_proposal_for_runtime(completed, completed_runtime))

    blank = _proposal()
    blank["parties"][0]["name"] = "Not allowed"
    assert validate_proposal_for_runtime(blank, _runtime())


def test_blank_qwen_extras_are_removed_with_an_explicit_warning():
    runtime = _runtime()
    raw = _proposal()
    raw["parties"][0]["name"] = "Generated Name"
    raw["additional_facts"] = [
        {
            "key": "unsupported", "destination": "autres_informations", "party_key": None,
            "value": "Generated", "evidence_clause_ids": [], "evidence_content_ids": [],
        }
    ]
    normalized = normalize_proposal_for_runtime(raw, runtime)
    assert normalized["parties"][0]["name"] is None
    assert normalized["additional_facts"] == []
    assert any("dropped 1" in warning for warning in normalized["warnings"])
    assert any("removed Qwen-generated" in warning for warning in normalized["warnings"])
    assert validate_proposal_for_runtime(normalized, runtime) == []


def test_completed_qwen_invented_variables_and_duplicate_keys_are_normalized():
    runtime = _runtime(completed=True)
    raw = _proposal(completed=True)
    raw["parties"][0]["name"] = None
    raw["variable_bindings"] = [
        {
            "variable_id": "invented_variable", "value_type": "text",
            "destination": "informations_principales", "party_key": "party_employer",
            "status": "missing", "evidence_clause_ids": [], "evidence_content_ids": [],
        }
    ]
    raw["additional_facts"][0]["party_key"] = "party_employer"
    raw["additional_facts"][0]["destination"] = "informations_principales"
    duplicate = copy.deepcopy(raw["additional_facts"][0])
    raw["additional_facts"].append(duplicate)
    normalized = normalize_proposal_for_runtime(raw, runtime)
    assert normalized["variable_bindings"] == []
    assert normalized["additional_facts"][1]["party_key"] is None
    assert normalized["additional_facts"][1]["key"].endswith("_002")
    assert validate_proposal_for_runtime(normalized, runtime) == []


def test_duplicate_party_keys_and_unsupported_single_language_are_normalized():
    runtime = _runtime()
    raw = _proposal()
    duplicate = copy.deepcopy(raw["parties"][0])
    duplicate["role"] = "client"
    raw["parties"].append(duplicate)
    raw["document"]["langue_contrat"] = "English"
    normalized = normalize_proposal_for_runtime(raw, runtime)
    assert normalized["parties"][1]["party_key"].endswith("_002")
    assert normalized["document"]["langue_contrat"] == "en"
    assert any("renamed duplicate party_key" in warning for warning in normalized["warnings"])
    assert any("replaced unsupported language" in warning for warning in normalized["warnings"])
    assert validate_proposal_for_runtime(normalized, runtime) == []


def test_unresolved_required_document_value_is_explicit_ambiguity():
    proposal = _proposal()
    proposal["document"]["juridiction"] = None
    payload, errors = assemble_agent1_payload(proposal, _runtime(), case_date="2026-08-14")
    assert errors == []
    assert payload is not None
    assert payload["metadata"]["juridiction"] == "AMBIGUITY"
