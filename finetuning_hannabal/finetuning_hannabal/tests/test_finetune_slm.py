from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import finetune_slm
from finetune_slm import (
    FineTuneConfig,
    FineTuneError,
    TokenizedExample,
    ValidatedExample,
    build_ollama_commands,
    execute,
    split_examples,
    tokenize_and_filter,
    tokenize_example,
    validate_cuda_environment,
    validate_dataset_directory,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = "<pad>"
    padding_side = "right"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = ""
        for message in messages:
            text += f"<{message['role']}>\n{message['content']}\n"
        if add_generation_prompt:
            text += "<assistant>\n"
        if not tokenize:
            return text
        # Stable, deterministic word-level IDs are sufficient for boundary tests.
        return [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) for word in text.split()]


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_and_manifest(index: int, *, source_id: str | None = None, output: str = "TARGET"):
    source_id = source_id or f"src_{index}"
    template_id = f"tpl_{index}"
    clause_id = f"clause_{index}"
    model_input = {
        "agent2_generation_facts": {
            "document": {"language": "en"},
            "parties": [],
            "variable_bindings": [],
        },
        "retrieval": {
            "template": {
                "schema_version": "contract-template-runtime-v1",
                "template_id": template_id,
                "runtime_status": "demo_unreviewed",
                "canonical_ref": {"schema_version": "contract-template-v1", "source_ids": [source_id], "template_id": template_id},
                "document": {
                    "contract_family": "unknown",
                    "contract_type": None,
                    "document_kind": "contract_template",
                    "languages": ["en"],
                    "template_state": "blank",
                    "title": "Test",
                },
                "jurisdictions": [],
                "parties": [],
                "sections": [{
                    "section_id": f"sec_{index}", "order": 1, "heading": "Terms",
                    "kind": "unclassified", "requirement_status": "unspecified", "condition": None,
                    "clauses": [{
                        "clause_id": clause_id, "order": 1, "heading": "Terms", "text": "Evidence",
                        "clause_type": "unclassified", "requirement_status": "unspecified",
                        "condition": None, "legal_references": [], "variable_ids": [],
                    }],
                }],
                "unparsed_content": [],
                "variables": [],
            },
            "clauses": [],
        },
    }
    row = {
        "instruction": "Generate only from supplied evidence.",
        "input": _canonical(model_input),
        "output": output,
    }
    row_hash = hashlib.sha256(_canonical(row).encode()).hexdigest()
    manifest = {
        "schema_version": "alpaca-export-manifest-v1",
        "row_id": f"alpaca_{index}",
        "row_index": index,
        "status": "demo_unreviewed",
        "training_eligible": False,
        "artifact_key": source_id,
        "template_id": template_id,
        "source_ids": [source_id],
        "clause_ids": [clause_id],
        "jurisdictions": [],
        "languages": ["en"],
        "paths": {"alpaca_dataset": "alpaca-demo/alpaca.jsonl"},
        "hashes": {"row_sha256": row_hash},
        "review": {"structure": "pending", "legal": "pending", "privacy": "pending", "sanitization": "not_required"},
    }
    return row, manifest


def _write_dataset(path: Path, count: int = 6) -> None:
    path.mkdir(parents=True)
    pairs = [_row_and_manifest(index) for index in range(count)]
    (path / "alpaca.jsonl").write_text("".join(_canonical(row) + "\n" for row, _ in pairs), encoding="utf-8")
    (path / "manifest.jsonl").write_text("".join(_canonical(manifest) + "\n" for _, manifest in pairs), encoding="utf-8")
    (path / "summary.json").write_text(json.dumps({
        "schema_version": "alpaca-export-summary-v1",
        "mode": "demo",
        "status": "demo_unreviewed",
        "training_eligible": False,
        "exported_rows": count,
        "eligible_examples": 0,
        "minimum_volume_met": False,
        "audit_errors": [],
    }), encoding="utf-8")


def _validated(index: int, *, source_id: str | None = None, output: str = "TARGET") -> ValidatedExample:
    row, manifest = _row_and_manifest(index, source_id=source_id, output=output)
    return ValidatedExample(
        row=row,
        manifest=manifest,
        parsed_input=json.loads(row["input"]),
        document_kind="contract_template",
        contract_family="unknown",
    )


def test_demo_requires_explicit_override_and_checks_hashes(tmp_path):
    dataset = tmp_path / "alpaca-demo"
    _write_dataset(dataset)
    with pytest.raises(FineTuneError, match="allow-unreviewed-demo"):
        validate_dataset_directory(dataset)

    examples, summary, hashes = validate_dataset_directory(dataset, allow_unreviewed_demo=True)
    assert len(examples) == 6
    assert summary["mode"] == "demo"
    assert set(hashes) == {"alpaca.jsonl", "manifest.jsonl", "summary.json"}

    manifests = (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    broken = json.loads(manifests[0])
    broken["hashes"]["row_sha256"] = "0" * 64
    manifests[0] = _canonical(broken)
    (dataset / "manifest.jsonl").write_text("\n".join(manifests) + "\n", encoding="utf-8")
    with pytest.raises(FineTuneError, match="row_sha256"):
        validate_dataset_directory(dataset, allow_unreviewed_demo=True)


def test_forbidden_operational_identifier_is_not_bypassed(tmp_path):
    dataset = tmp_path / "alpaca-demo"
    _write_dataset(dataset)
    rows = (dataset / "alpaca.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(rows[0])
    parsed = json.loads(row["input"])
    parsed["agent2_generation_facts"]["session_id"] = "forbidden"
    row["input"] = _canonical(parsed)
    rows[0] = _canonical(row)
    (dataset / "alpaca.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(FineTuneError, match="session_id"):
        validate_dataset_directory(dataset, allow_unreviewed_demo=True)


def test_production_requires_200_eligible_rows(tmp_path):
    dataset = tmp_path / "alpaca-production"
    _write_dataset(dataset)
    manifests = []
    for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        manifest = json.loads(line)
        manifest["status"] = "eligible"
        manifest["training_eligible"] = True
        manifests.append(manifest)
    (dataset / "manifest.jsonl").write_text("".join(_canonical(item) + "\n" for item in manifests), encoding="utf-8")
    (dataset / "summary.json").write_text(json.dumps({
        "schema_version": "alpaca-export-summary-v1",
        "mode": "production",
        "status": "eligible",
        "training_eligible": True,
        "exported_rows": len(manifests),
        "eligible_examples": len(manifests),
        "minimum_volume_met": False,
        "audit_errors": [],
    }), encoding="utf-8")
    with pytest.raises(FineTuneError, match="at least 200"):
        validate_dataset_directory(dataset)


def test_completion_mask_and_overlength_exclusion_never_truncate():
    tokenizer = FakeTokenizer()
    short = _validated(0, output="short target")
    long = _validated(1, output=" ".join(["target"] * 200))
    tokenized = tokenize_example(short, tokenizer)
    assert tokenized.labels[:tokenized.prompt_tokens] == [-100] * tokenized.prompt_tokens
    assert all(label != -100 for label in tokenized.labels[tokenized.prompt_tokens:])

    usable, excluded = tokenize_and_filter([short, long], tokenizer, max_sequence_length=80)
    assert [item.source.manifest["row_id"] for item in usable] == ["alpaca_0"]
    assert excluded[0]["row_id"] == "alpaca_1"
    assert excluded[0]["reason"].endswith("no text was truncated")


def test_grouped_split_is_deterministic_and_has_no_source_leakage():
    tokenizer = FakeTokenizer()
    examples = [_validated(index, source_id=f"src_{index // 2}") for index in range(12)]
    tokenized = [tokenize_example(example, tokenizer) for example in examples]
    first = split_examples(tokenized, seed=42)
    second = split_examples(tokenized, seed=42)
    assert {name: [item.source.manifest["row_id"] for item in values] for name, values in first.items()} == {
        name: [item.source.manifest["row_id"] for item in values] for name, values in second.items()
    }
    owners = {}
    assert all(first[name] for name in ("train", "validation", "test"))
    for split_name, values in first.items():
        for item in values:
            for source_id in item.source.manifest["source_ids"]:
                assert source_id not in owners or owners[source_id] == split_name
                owners[source_id] = split_name


def test_execute_with_mocked_trainer_writes_non_production_artifacts(tmp_path, monkeypatch):
    dataset = tmp_path / "alpaca-demo"
    output = tmp_path / "finetuned"
    _write_dataset(dataset, count=12)
    calls = []

    def fake_training(config, tokenizer, splits):
        calls.append((config, splits))
        (config.output_dir / "adapter").mkdir()
        return {"train": {"loss": 1.0}, "legal_quality_kpi_claims": []}

    monkeypatch.setattr(finetune_slm, "run_training", fake_training)
    result = execute(FineTuneConfig(
        dataset_dir=dataset,
        output_dir=output,
        allow_unreviewed_demo=True,
        max_sequence_length=1000,
    ), tokenizer=FakeTokenizer())
    assert calls
    assert result["dataset"]["rows_audited"] == 12
    assert result["dataset"]["production_ready"] is False
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["status"] == "demo_training_complete"
    assert (output / "splits" / "train.jsonl").is_file()


def test_cuda_guard_refuses_cpu_and_insufficient_vram():
    class NoCuda:
        @staticmethod
        def is_available():
            return False

    class TorchNoCuda:
        cuda = NoCuda()

    with pytest.raises(FineTuneError, match="CUDA-enabled"):
        validate_cuda_environment(TorchNoCuda(), minimum_vram_gb=16)

    class Props:
        name = "Small GPU"
        total_memory = 4 * 1024 ** 3

    class SmallCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(index):
            return Props()

        @staticmethod
        def is_bf16_supported():
            return False

    class TorchSmall:
        cuda = SmallCuda()

    with pytest.raises(FineTuneError, match="at least 16 GB"):
        validate_cuda_environment(TorchSmall(), minimum_vram_gb=16)


def test_ollama_commands_use_merged_gguf_and_verify_model(tmp_path, monkeypatch):
    llama_cpp = tmp_path / "llama.cpp"
    converter = llama_cpp / "convert_hf_to_gguf.py"
    quantizer = llama_cpp / "build" / "bin" / "Release" / ("llama-quantize.exe" if finetune_slm.os.name == "nt" else "llama-quantize")
    converter.parent.mkdir(parents=True)
    converter.write_text("", encoding="utf-8")
    quantizer.parent.mkdir(parents=True)
    quantizer.write_text("", encoding="utf-8")
    monkeypatch.setattr(finetune_slm.shutil, "which", lambda name: "C:/Tools/ollama.exe")
    commands, modelfile, quantized = build_ollama_commands(
        llama_cpp_dir=llama_cpp,
        merged_dir=tmp_path / "merged-hf",
        export_dir=tmp_path / "ollama",
        ollama_model="hannibal-demo",
    )
    assert commands[0][2] == str(tmp_path / "merged-hf")
    assert commands[1][-1] == "Q4_K_M"
    assert commands[2][1:3] == ["create", "hannibal-demo"]
    assert commands[3][1:] == ["show", "hannibal-demo"]
    assert modelfile.name == "Modelfile"
    assert quantized.name.endswith("q4_K_M.gguf")


def test_checked_in_demo_audits_all_65_rows_without_modification(feature_root, tmp_path):
    dataset = feature_root / "artifacts" / "alpaca-demo"
    before = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in ("alpaca.jsonl", "manifest.jsonl", "summary.json")}
    result = execute(FineTuneConfig(
        dataset_dir=dataset,
        output_dir=tmp_path / "preflight",
        allow_unreviewed_demo=True,
        preflight_only=True,
        max_sequence_length=1_000_000,
    ), tokenizer=FakeTokenizer())
    after = {name: hashlib.sha256((dataset / name).read_bytes()).hexdigest() for name in before}
    assert result["dataset"]["rows_audited"] == 65
    assert result["dataset"]["rows_usable"] == 65
    assert before == after
