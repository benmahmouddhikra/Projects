"""Fine-tune Hannibal Agent 2 with Qwen2.5 3B using guarded QLoRA SFT.

The command consumes an existing Alpaca export directory.  It never changes the
source dataset and deliberately keeps model/training imports lazy so dataset
audits and unit tests can run without a CUDA fine-tuning environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from jsonschema import Draft202012Validator

from alpaca_export import validate_alpaca_row
from validate import load_schema


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_SEQUENCE_LENGTH = 8192
DEFAULT_MINIMUM_VRAM_GB = 16.0
MINIMUM_PRODUCTION_EXAMPLES = 200
MANIFEST_SCHEMA = "alpaca-export-manifest-v1.schema.json"
REQUIRED_DATASET_FILES = ("alpaca.jsonl", "manifest.jsonl", "summary.json")
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

AMBIGUITIES = (
    {
        "issue": "AMBIGUITY: Base model",
        "source_a": "Project sources recommend Llama 3.1 8B or Mistral 7B.",
        "source_b": "Neither source selects Qwen.",
        "decision_required": "Resolved for this demo: use Qwen/Qwen2.5-3B-Instruct.",
    },
    {
        "issue": "AMBIGUITY: Maximum sequence length",
        "source_a": "The source training profile specifies 4096 tokens.",
        "source_b": "The exported full-contract examples can exceed that length.",
        "decision_required": "Resolved for this demo: use 8192 tokens and exclude, rather than truncate, larger rows.",
    },
    {
        "issue": "AMBIGUITY: Qwen learning rate",
        "source_a": "The sources specify 2e-5 for Llama and 5e-6 for Mistral.",
        "source_b": "No Qwen-specific learning rate is supplied.",
        "decision_required": "Resolved for this demo: default to 1e-4.",
    },
    {
        "issue": "AMBIGUITY: Adapter routing",
        "source_a": "The architecture defines specialized adapter families.",
        "source_b": "The current demo export classifies every record as contract_family=unknown.",
        "decision_required": "Resolve record families before registering a production adapter; this output remains mixed and demo-only.",
    },
    {
        "issue": "AMBIGUITY: Dataset eligibility",
        "source_a": "Production training requires reviewed records and at least 200 eligible examples.",
        "source_b": "The current demo has 65 unreviewed, ineligible rows.",
        "decision_required": "Obtain at least 200 reviewed eligible records for production; only --allow-unreviewed-demo permits this non-production run.",
    },
)


class FineTuneError(RuntimeError):
    """Raised for a guarded, user-actionable fine-tuning failure."""


@dataclass(frozen=True)
class FineTuneConfig:
    dataset_dir: Path
    output_dir: Path
    model_name_or_path: str = DEFAULT_MODEL
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    epochs: float = 2.0
    learning_rate: float = 1e-4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = -1
    seed: int = 42
    minimum_vram_gb: float = DEFAULT_MINIMUM_VRAM_GB
    allow_unreviewed_demo: bool = False
    preflight_only: bool = False
    resume_from_checkpoint: Path | None = None
    export_ollama: bool = False
    llama_cpp_dir: Path | None = None
    ollama_model: str = "hannibal-qwen2.5-3b-demo"


@dataclass
class ValidatedExample:
    row: dict[str, Any]
    manifest: dict[str, Any]
    parsed_input: dict[str, Any]
    document_kind: str
    contract_family: str


@dataclass
class TokenizedExample:
    source: ValidatedExample
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return len(self.input_ids)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FineTuneError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FineTuneError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise FineTuneError(f"Cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FineTuneError(f"Blank JSONL line in {path} at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FineTuneError(f"Malformed JSONL in {path} at line {line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise FineTuneError(f"Expected a JSON object in {path} at line {line_number}")
        rows.append(value)
    return rows


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _manifest_schema_errors(manifest: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(load_schema(MANIFEST_SCHEMA))
    errors = sorted(validator.iter_errors(manifest), key=lambda error: [str(item) for item in error.absolute_path])
    return [
        f"manifest:{'/'.join(str(item) for item in error.absolute_path) or '$'}:{error.message}"
        for error in errors
    ]


def _row_evidence_identity_errors(
    parsed_input: dict[str, Any], manifest: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    retrieval = parsed_input.get("retrieval", {})
    template = retrieval.get("template", {}) if isinstance(retrieval, dict) else {}
    if template.get("template_id") != manifest.get("template_id"):
        errors.append("identity:Alpaca template_id does not match manifest")
    source_ids = template.get("canonical_ref", {}).get("source_ids")
    if source_ids != manifest.get("source_ids"):
        errors.append("identity:Alpaca source_ids do not match manifest")
    expected_clause_ids = sorted({
        str(clause.get("clause_id"))
        for section in template.get("sections", [])
        if isinstance(section, dict)
        for clause in section.get("clauses", [])
        if isinstance(clause, dict) and clause.get("clause_id")
    })
    if expected_clause_ids != sorted(manifest.get("clause_ids", [])):
        errors.append("identity:Alpaca clause_ids do not match manifest")
    languages = template.get("document", {}).get("languages", [])
    if languages != manifest.get("languages"):
        errors.append("identity:Alpaca languages do not match manifest")
    jurisdictions = [item.get("country_code") for item in template.get("jurisdictions", []) if isinstance(item, dict)]
    if jurisdictions != manifest.get("jurisdictions"):
        errors.append("identity:Alpaca jurisdictions do not match manifest")
    return errors


def validate_dataset_directory(
    dataset_dir: Path, *, allow_unreviewed_demo: bool = False
) -> tuple[list[ValidatedExample], dict[str, Any], dict[str, str]]:
    """Load and strictly audit one existing Alpaca export directory."""
    dataset_dir = dataset_dir.resolve()
    missing = [name for name in REQUIRED_DATASET_FILES if not (dataset_dir / name).is_file()]
    if missing:
        raise FineTuneError("Dataset directory is missing required files: " + ", ".join(missing))

    rows = _read_jsonl(dataset_dir / "alpaca.jsonl")
    manifests = _read_jsonl(dataset_dir / "manifest.jsonl")
    summary = _read_json(dataset_dir / "summary.json")
    errors: list[str] = []

    if len(rows) != len(manifests):
        errors.append(f"dataset:row/manifest count mismatch ({len(rows)} != {len(manifests)})")
    if summary.get("exported_rows") != len(rows):
        errors.append("dataset:summary exported_rows does not match alpaca.jsonl")
    if summary.get("audit_errors"):
        errors.append("dataset:summary contains upstream audit errors")

    mode = summary.get("mode")
    if mode not in {"demo", "production"}:
        errors.append("dataset:summary mode must be demo or production")
    if mode == "demo" and not allow_unreviewed_demo:
        errors.append("eligibility:demo rows require --allow-unreviewed-demo")
    if mode == "production":
        if len(rows) < MINIMUM_PRODUCTION_EXAMPLES:
            errors.append(f"eligibility:production requires at least {MINIMUM_PRODUCTION_EXAMPLES} rows")
        if not summary.get("training_eligible") or not summary.get("minimum_volume_met"):
            errors.append("eligibility:production summary is not training eligible or does not meet minimum volume")

    manifest_validator_errors: list[str] = []
    row_ids: set[str] = set()
    row_hashes: set[str] = set()
    validated: list[ValidatedExample] = []
    for index, pair in enumerate(zip(rows, manifests)):
        row, manifest = pair
        row_errors = validate_alpaca_row(row)
        if set(row) != {"instruction", "input", "output"}:
            row_errors.append("alpaca:row must contain exactly instruction, input, and output")
        for field in ("instruction", "input", "output"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                row_errors.append(f"alpaca:{field} must be a non-empty string")

        manifest_errors = _manifest_schema_errors(manifest)
        manifest_validator_errors.extend(manifest_errors)
        if manifest.get("row_index") != index:
            row_errors.append("identity:manifest row_index does not match JSONL position")
        row_id = manifest.get("row_id")
        if row_id in row_ids:
            row_errors.append("identity:duplicate manifest row_id")
        if isinstance(row_id, str):
            row_ids.add(row_id)
        row_hash = _sha256_text(_canonical_json(row))
        if manifest.get("hashes", {}).get("row_sha256") != row_hash:
            row_errors.append("integrity:manifest row_sha256 does not match Alpaca row")
        if row_hash in row_hashes:
            row_errors.append("integrity:duplicate Alpaca row hash")
        row_hashes.add(row_hash)

        try:
            parsed_input = json.loads(row.get("input", ""))
        except (json.JSONDecodeError, TypeError):
            parsed_input = {}
        if isinstance(parsed_input, dict):
            row_errors.extend(_row_evidence_identity_errors(parsed_input, manifest))
        else:
            parsed_input = {}

        if mode == "demo":
            if manifest.get("status") != "demo_unreviewed" or manifest.get("training_eligible") is not False:
                row_errors.append("eligibility:demo manifest must remain unreviewed and ineligible")
        elif mode == "production":
            if manifest.get("status") != "eligible" or manifest.get("training_eligible") is not True:
                row_errors.append("eligibility:every production manifest must be eligible")

        if row_errors:
            errors.extend(f"row[{index}] {error}" for error in sorted(set(row_errors)))
            continue
        document = parsed_input.get("retrieval", {}).get("template", {}).get("document", {})
        validated.append(ValidatedExample(
            row=row,
            manifest=manifest,
            parsed_input=parsed_input,
            document_kind=str(document.get("document_kind", "unknown")),
            contract_family=str(document.get("contract_family", "unknown")),
        ))

    if manifest_validator_errors:
        errors.extend(f"manifest {error}" for error in manifest_validator_errors)
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = f"\n- ... {len(errors) - 30} more error(s)" if len(errors) > 30 else ""
        raise FineTuneError(f"Dataset validation failed:\n{preview}{suffix}")

    hashes = {name: _sha256_file(dataset_dir / name) for name in REQUIRED_DATASET_FILES}
    return validated, summary, hashes


def build_conversation(row: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Keep Alpaca fields distinct while mapping them to Qwen chat turns."""
    prompt = [{
        "role": "user",
        "content": f"{row['instruction']}\n\nInput:\n{row['input']}",
    }]
    completion = [{"role": "assistant", "content": row["output"]}]
    return prompt, completion


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, list) and (not value or isinstance(value[0], int)):
        return [int(item) for item in value]
    if isinstance(value, dict) and isinstance(value.get("input_ids"), list):
        return [int(item) for item in value["input_ids"]]
    raise FineTuneError("Tokenizer chat template did not return a flat input-id list")


def tokenize_example(example: ValidatedExample, tokenizer: Any) -> TokenizedExample:
    prompt, completion = build_conversation(example.row)
    prompt_ids = _token_ids(tokenizer.apply_chat_template(
        prompt, tokenize=True, add_generation_prompt=True,
    ))
    full_ids = _token_ids(tokenizer.apply_chat_template(
        prompt + completion, tokenize=True, add_generation_prompt=False,
    ))
    if not prompt_ids or len(full_ids) <= len(prompt_ids):
        raise FineTuneError(f"Row {example.manifest['row_id']} has no supervised completion tokens")
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise FineTuneError(
            f"Row {example.manifest['row_id']} chat template is not prefix-stable; completion masking would be unsafe"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return TokenizedExample(
        source=example,
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        prompt_tokens=len(prompt_ids),
        completion_tokens=len(full_ids) - len(prompt_ids),
    )


def tokenize_and_filter(
    examples: Sequence[ValidatedExample], tokenizer: Any, *, max_sequence_length: int
) -> tuple[list[TokenizedExample], list[dict[str, Any]]]:
    usable: list[TokenizedExample] = []
    excluded: list[dict[str, Any]] = []
    for example in examples:
        tokenized = tokenize_example(example, tokenizer)
        if tokenized.total_tokens > max_sequence_length:
            excluded.append({
                "row_id": example.manifest["row_id"],
                "row_index": example.manifest["row_index"],
                "template_id": example.manifest["template_id"],
                "source_ids": example.manifest["source_ids"],
                "prompt_tokens": tokenized.prompt_tokens,
                "completion_tokens": tokenized.completion_tokens,
                "total_tokens": tokenized.total_tokens,
                "maximum_tokens": max_sequence_length,
                "reason": "formatted example exceeds maximum; no text was truncated",
            })
        else:
            usable.append(tokenized)
    return usable, excluded


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _group_components(examples: Sequence[TokenizedExample]) -> list[list[TokenizedExample]]:
    groups = _DisjointSet(len(examples))
    seen: dict[str, int] = {}
    for index, example in enumerate(examples):
        manifest = example.source.manifest
        keys = [f"template:{manifest['template_id']}"] + [f"source:{item}" for item in manifest["source_ids"]]
        for key in keys:
            if key in seen:
                groups.union(index, seen[key])
            else:
                seen[key] = index
    components: dict[int, list[TokenizedExample]] = defaultdict(list)
    for index, example in enumerate(examples):
        components[groups.find(index)].append(example)
    return list(components.values())


def _stratum(component: Sequence[TokenizedExample]) -> str:
    kinds = sorted({item.source.document_kind for item in component})
    languages = sorted({language for item in component for language in item.source.manifest.get("languages", [])})
    return f"{'|'.join(kinds) or 'unknown'}::{','.join(languages) or 'unknown'}"


def split_examples(
    examples: Sequence[TokenizedExample], *, seed: int = 42
) -> dict[str, list[TokenizedExample]]:
    """Deterministically group by source/template and approximately stratify."""
    if len(examples) < len(SPLIT_NAMES):
        raise FineTuneError("At least three usable examples are required for non-empty train/validation/test splits")
    components = _group_components(examples)
    if len(components) < len(SPLIT_NAMES):
        raise FineTuneError("Source/template grouping leaves fewer than three independent groups")

    rng = random.Random(seed)
    buckets: dict[str, list[list[TokenizedExample]]] = defaultdict(list)
    for component in components:
        buckets[_stratum(component)].append(component)
    for components_in_bucket in buckets.values():
        rng.shuffle(components_in_bucket)

    result: dict[str, list[TokenizedExample]] = {name: [] for name in SPLIT_NAMES}
    total = len(examples)
    global_targets = {name: total * SPLIT_RATIOS[name] for name in SPLIT_NAMES}
    for stratum in sorted(buckets):
        bucket = buckets[stratum]
        bucket_size = sum(len(component) for component in bucket)
        stratum_counts = Counter()
        for component in bucket:
            def score(split_name: str) -> tuple[float, float, int]:
                stratum_deficit = bucket_size * SPLIT_RATIOS[split_name] - stratum_counts[split_name]
                global_deficit = global_targets[split_name] - len(result[split_name])
                return (stratum_deficit, global_deficit, -SPLIT_NAMES.index(split_name))
            selected = max(SPLIT_NAMES, key=score)
            result[selected].extend(component)
            stratum_counts[selected] += len(component)

    empty = [name for name in SPLIT_NAMES if not result[name]]
    for empty_name in empty:
        donors = sorted(
            (name for name in SPLIT_NAMES if len(_group_components(result[name])) > 1),
            key=lambda name: len(result[name]), reverse=True,
        )
        if not donors:
            raise FineTuneError("Unable to create non-empty grouped train/validation/test splits")
        donor = donors[0]
        donor_components = sorted(_group_components(result[donor]), key=lambda group: (len(group), group[0].source.manifest["row_id"]))
        moved = donor_components[0]
        moved_ids = {id(item) for item in moved}
        result[donor] = [item for item in result[donor] if id(item) not in moved_ids]
        result[empty_name].extend(moved)

    for split_name in SPLIT_NAMES:
        result[split_name].sort(key=lambda item: item.source.manifest["row_id"])
    return result


def _split_identity_report(splits: dict[str, list[TokenizedExample]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    template_owners: dict[str, str] = {}
    source_owners: dict[str, str] = {}
    for name, examples in splits.items():
        languages = Counter(language for item in examples for language in item.source.manifest.get("languages", []))
        kinds = Counter(item.source.document_kind for item in examples)
        report[name] = {
            "rows": len(examples),
            "languages": dict(sorted(languages.items())),
            "document_kinds": dict(sorted(kinds.items())),
            "minimum_tokens": min(item.total_tokens for item in examples),
            "maximum_tokens": max(item.total_tokens for item in examples),
        }
        for item in examples:
            template_id = item.source.manifest["template_id"]
            if template_id in template_owners and template_owners[template_id] != name:
                raise FineTuneError(f"Template leakage detected for {template_id}")
            template_owners[template_id] = name
            for source_id in item.source.manifest["source_ids"]:
                if source_id in source_owners and source_owners[source_id] != name:
                    raise FineTuneError(f"Source leakage detected for {source_id}")
                source_owners[source_id] = name
    return report


def _resolved_config(config: FineTuneConfig, source_hashes: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    values = asdict(config)
    for key in ("dataset_dir", "output_dir", "resume_from_checkpoint", "llama_cpp_dir"):
        values[key] = str(values[key]) if values[key] is not None else None
    values.update({
        "method": "qlora",
        "quantization": {"bits": 4, "type": "nf4", "double_quantization": True, "compute_dtype": "auto_bf16_else_fp16"},
        "optimizer": "paged_adamw_8bit",
        "scheduler": "linear",
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "packing": False,
        "completion_only_loss": True,
        "lora_target_modules": list(LORA_TARGET_MODULES),
        "split_ratios": SPLIT_RATIOS,
        "source_hashes": source_hashes,
        "dataset_mode": summary.get("mode"),
        "production_ready": summary.get("mode") == "production",
        "ambiguities": list(AMBIGUITIES),
    })
    return values


def _ensure_output_is_safe(config: FineTuneConfig, source_hashes: dict[str, str]) -> None:
    output_dir = config.output_dir.resolve()
    if output_dir == config.dataset_dir.resolve():
        raise FineTuneError("Output directory must not be the source dataset directory")
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    if config.resume_from_checkpoint is None:
        raise FineTuneError(f"Output directory is not empty: {output_dir}; use a new directory or --resume-from-checkpoint")
    existing_path = output_dir / "run-config.json"
    if not existing_path.is_file():
        raise FineTuneError("Resume output is missing run-config.json")
    existing = _read_json(existing_path)
    if existing.get("source_hashes") != source_hashes:
        raise FineTuneError("Resume dataset hashes do not match the existing run")


def write_prepared_artifacts(
    config: FineTuneConfig,
    splits: dict[str, list[TokenizedExample]],
    excluded: list[dict[str, Any]],
    summary: dict[str, Any],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    split_report = _split_identity_report(splits)
    for split_name, examples in splits.items():
        _atomic_write_jsonl(output_dir / "splits" / f"{split_name}.jsonl", (item.source.row for item in examples))
        _atomic_write_jsonl(output_dir / "splits" / f"{split_name}-manifest.jsonl", (item.source.manifest for item in examples))
    _atomic_write_jsonl(output_dir / "excluded.jsonl", excluded)
    report = {
        "schema_version": "hannibal-finetune-dataset-report-v1",
        "status": "demo_non_production" if summary.get("mode") == "demo" else "eligible",
        "production_ready": summary.get("mode") == "production",
        "rows_audited": int(summary.get("exported_rows", 0)),
        "rows_usable": sum(len(items) for items in splits.values()),
        "rows_excluded_for_length": len(excluded),
        "maximum_sequence_length": config.max_sequence_length,
        "no_truncation": True,
        "splits": split_report,
        "source_hashes": source_hashes,
        "legal_quality_kpi_claims": [],
        "warning": "Training loss and perplexity are diagnostics, not evidence of legal correctness.",
    }
    _atomic_write_json(output_dir / "dataset-report.json", report)
    _atomic_write_json(output_dir / "run-config.json", _resolved_config(config, source_hashes, summary))
    return report


def _require_tokenizer(model_name_or_path: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FineTuneError("Transformers is required; install requirements-finetune.txt") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def validate_cuda_environment(torch_module: Any, *, minimum_vram_gb: float) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise FineTuneError("QLoRA training requires a CUDA-enabled PyTorch installation and NVIDIA GPU")
    properties = torch_module.cuda.get_device_properties(0)
    total_gb = float(properties.total_memory) / (1024 ** 3)
    if total_gb < minimum_vram_gb:
        raise FineTuneError(f"QLoRA profile requires at least {minimum_vram_gb:g} GB VRAM; detected {total_gb:.1f} GB")
    bf16 = bool(torch_module.cuda.is_bf16_supported())
    return {"device": properties.name, "vram_gb": round(total_gb, 2), "compute_dtype": "bfloat16" if bf16 else "float16"}


def _to_dataset(examples: Sequence[TokenizedExample]) -> Any:
    from datasets import Dataset
    return Dataset.from_list([{
        "input_ids": item.input_ids,
        "attention_mask": item.attention_mask,
        "labels": item.labels,
    } for item in examples])


def _data_collator(tokenizer: Any, torch_module: Any):
    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        input_ids, attention_masks, labels = [], [], []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * padding)
            attention_masks.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch_module.tensor(input_ids, dtype=torch_module.long),
            "attention_mask": torch_module.tensor(attention_masks, dtype=torch_module.long),
            "labels": torch_module.tensor(labels, dtype=torch_module.long),
        }
    return collate


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
    loss = result.get("eval_loss")
    if isinstance(loss, (int, float)):
        result["perplexity"] = math.exp(loss) if loss < 50 else float("inf")
    return result


def run_training(config: FineTuneConfig, tokenizer: Any, splits: dict[str, list[TokenizedExample]]) -> dict[str, Any]:
    try:
        import bitsandbytes  # noqa: F401
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise FineTuneError("The complete QLoRA stack is required; install requirements-finetune.txt") from exc

    hardware = validate_cuda_environment(torch, minimum_vram_gb=config.minimum_vram_gb)
    compute_dtype = torch.bfloat16 if hardware["compute_dtype"] == "bfloat16" else torch.float16
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=quantization_config,
        device_map={"": 0},
        dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(LORA_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    ))
    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if unexpected:
        raise FineTuneError("Unexpected non-LoRA trainable parameters: " + ", ".join(unexpected[:10]))

    trainer_output = config.output_dir / "checkpoints"
    args = SFTConfig(
        output_dir=str(trainer_output),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=hardware["compute_dtype"] == "bfloat16",
        fp16=hardware["compute_dtype"] == "float16",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="none",
        logging_steps=1,
        packing=False,
        max_length=None,
        completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=_to_dataset(splits["train"]),
        eval_dataset=_to_dataset(splits["validation"]),
        processing_class=tokenizer,
        data_collator=_data_collator(tokenizer, torch),
    )
    train_result = trainer.train(
        resume_from_checkpoint=str(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
    )
    adapter_dir = config.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    metrics = {
        "hardware": hardware,
        "train": _json_safe_metrics(train_result.metrics),
        "validation": _json_safe_metrics(trainer.evaluate(eval_dataset=_to_dataset(splits["validation"]))),
        "test": _json_safe_metrics(trainer.evaluate(eval_dataset=_to_dataset(splits["test"]))),
        "legal_quality_kpi_claims": [],
        "warning": "Loss and perplexity do not establish legal correctness.",
    }
    _atomic_write_json(config.output_dir / "metrics.json", metrics)
    return metrics


def _find_first(paths: Sequence[Path], description: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FineTuneError(f"Cannot find {description}; checked: " + ", ".join(str(path) for path in paths))


def build_ollama_commands(
    *, llama_cpp_dir: Path, merged_dir: Path, export_dir: Path, ollama_model: str
) -> tuple[list[list[str]], Path, Path]:
    llama_cpp_dir = llama_cpp_dir.resolve()
    converter = _find_first((llama_cpp_dir / "convert_hf_to_gguf.py",), "llama.cpp HF-to-GGUF converter")
    executable = "llama-quantize.exe" if os.name == "nt" else "llama-quantize"
    quantizer = _find_first((
        llama_cpp_dir / "build" / "bin" / "Release" / executable,
        llama_cpp_dir / "build" / "bin" / executable,
        llama_cpp_dir / executable,
    ), "llama.cpp quantizer")
    ollama = shutil.which("ollama")
    if not ollama:
        raise FineTuneError("ollama executable is not available on PATH")
    export_dir.mkdir(parents=True, exist_ok=True)
    f16_path = export_dir / "hannibal-qwen2.5-3b-f16.gguf"
    quantized_path = export_dir / "hannibal-qwen2.5-3b-q4_K_M.gguf"
    modelfile = export_dir / "Modelfile"
    commands = [
        [sys.executable, str(converter), str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"],
        [str(quantizer), str(f16_path), str(quantized_path), "Q4_K_M"],
        [ollama, "create", ollama_model, "-f", str(modelfile)],
        [ollama, "show", ollama_model],
    ]
    return commands, modelfile, quantized_path


def export_ollama_model(config: FineTuneConfig, tokenizer: Any) -> dict[str, Any]:
    if config.llama_cpp_dir is None:
        raise FineTuneError("--export-ollama requires --llama-cpp-dir")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise FineTuneError("PEFT and Transformers are required for Ollama export") from exc

    adapter_dir = config.output_dir / "adapter"
    if not adapter_dir.is_dir():
        raise FineTuneError(f"Adapter directory does not exist: {adapter_dir}")
    merged_dir = config.output_dir / "merged-hf"
    base = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        dtype=torch.float16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(merged_dir)

    commands, modelfile, quantized_path = build_ollama_commands(
        llama_cpp_dir=config.llama_cpp_dir,
        merged_dir=merged_dir,
        export_dir=config.output_dir / "ollama",
        ollama_model=config.ollama_model,
    )
    modelfile.write_text(f"FROM {json.dumps(str(quantized_path.resolve()))}\nPARAMETER num_ctx {config.max_sequence_length}\n", encoding="utf-8", newline="\n")
    for command in commands:
        subprocess.run(command, check=True)
    report = {
        "model": config.ollama_model,
        "quantization": "Q4_K_M",
        "gguf": str(quantized_path),
        "modelfile": str(modelfile),
        "verified_with_ollama_show": True,
    }
    _atomic_write_json(config.output_dir / "ollama" / "export-report.json", report)
    return report


def execute(config: FineTuneConfig, *, tokenizer: Any | None = None) -> dict[str, Any]:
    config = FineTuneConfig(**{**asdict(config),
        "dataset_dir": config.dataset_dir.resolve(),
        "output_dir": config.output_dir.resolve(),
        "resume_from_checkpoint": config.resume_from_checkpoint.resolve() if config.resume_from_checkpoint else None,
        "llama_cpp_dir": config.llama_cpp_dir.resolve() if config.llama_cpp_dir else None,
    })
    examples, summary, source_hashes = validate_dataset_directory(
        config.dataset_dir, allow_unreviewed_demo=config.allow_unreviewed_demo,
    )
    _ensure_output_is_safe(config, source_hashes)
    tokenizer = tokenizer or _require_tokenizer(config.model_name_or_path)
    usable, excluded = tokenize_and_filter(
        examples, tokenizer, max_sequence_length=config.max_sequence_length,
    )
    splits = split_examples(usable, seed=config.seed)
    dataset_report = write_prepared_artifacts(config, splits, excluded, summary, source_hashes)
    result: dict[str, Any] = {"dataset": dataset_report, "preflight_only": config.preflight_only}
    if config.preflight_only:
        _atomic_write_json(config.output_dir / "status.json", {"status": "preflight_complete", "production_ready": False})
        return result
    result["metrics"] = run_training(config, tokenizer, splits)
    if config.export_ollama:
        result["ollama"] = export_ollama_model(config, tokenizer)
    status = "demo_training_complete" if summary.get("mode") == "demo" else "training_complete"
    _atomic_write_json(config.output_dir / "status.json", {
        "status": status,
        "production_ready": summary.get("mode") == "production",
        "ollama_exported": config.export_ollama,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory containing alpaca.jsonl, manifest.jsonl, and summary.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-vram-gb", type=float, default=DEFAULT_MINIMUM_VRAM_GB)
    parser.add_argument("--allow-unreviewed-demo", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--export-ollama", action="store_true")
    parser.add_argument("--llama-cpp-dir", type=Path)
    parser.add_argument("--ollama-model", default="hannibal-qwen2.5-3b-demo")
    return parser


def config_from_args(args: argparse.Namespace) -> FineTuneConfig:
    output_dir = args.output_dir or args.dataset_dir.resolve().parent / "finetuned" / "hannibal-demo-qwen2.5-3b"
    positive = {
        "max sequence length": args.max_sequence_length,
        "epochs": args.epochs,
        "learning rate": args.learning_rate,
        "LoRA rank": args.lora_rank,
        "LoRA alpha": args.lora_alpha,
        "train batch size": args.per_device_train_batch_size,
        "evaluation batch size": args.per_device_eval_batch_size,
        "gradient accumulation steps": args.gradient_accumulation_steps,
        "minimum VRAM": args.minimum_vram_gb,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise FineTuneError("These arguments must be positive: " + ", ".join(invalid))
    if not 0 <= args.lora_dropout < 1:
        raise FineTuneError("LoRA dropout must be in [0, 1)")
    if args.max_steps == 0 or args.max_steps < -1:
        raise FineTuneError("--max-steps must be -1 or a positive integer")
    if args.preflight_only and args.export_ollama:
        raise FineTuneError("--preflight-only cannot be combined with --export-ollama")
    if args.export_ollama and args.llama_cpp_dir is None:
        raise FineTuneError("--export-ollama requires --llama-cpp-dir")
    return FineTuneConfig(
        dataset_dir=args.dataset_dir,
        output_dir=output_dir,
        model_name_or_path=args.model_name_or_path,
        max_sequence_length=args.max_sequence_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        seed=args.seed,
        minimum_vram_gb=args.minimum_vram_gb,
        allow_unreviewed_demo=args.allow_unreviewed_demo,
        preflight_only=args.preflight_only,
        resume_from_checkpoint=args.resume_from_checkpoint,
        export_ollama=args.export_ollama,
        llama_cpp_dir=args.llama_cpp_dir,
        ollama_model=args.ollama_model,
    )


def main() -> int:
    try:
        config = config_from_args(build_parser().parse_args())
        result = execute(config)
    except (FineTuneError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
