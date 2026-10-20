# Contract Template Dataset — Deterministic Pilot and Full-Corpus Demo

This repository implements the source-grounded canonical contract-template pipeline and a resumable, demo-only Agent 2 conversion through strict Alpaca rows. It inventories the corpus without modifying source documents, preserves provider-neutral canonical evidence, validates compact runtime records, sanitizes Agent 1 inputs for Agent 2, reconstructs contract targets from ordered source extraction, and retains audit provenance.

It does not change the Agent 1 schema, fine-tune adapters, move legal logic into Agent 1 or Agent 3, create embeddings, publish to a knowledge-base vendor, or integrate production RAG/runtime inference. Demo Alpaca rows remain unreviewed and are not training-eligible.

## Implemented layout

- `schemas/`: strict Draft 2020-12 schemas for manifest, extraction, coordinate-only model labels, canonical templates, and review records.
- `config/taxonomy.json`: controlled document, template, family, requirement, and value-type vocabularies plus conservative deterministic routing hints.
- `config/pilot-manifest.json`: the exact 36 paths and a selection reason for each; runtime selection is never random.
- `prompts/`: coordinate-only Qwen instructions that prohibit generated source text.
- `src/inventory.py`: complete read-only inventory, SHA-256 hashing, count checks, and exact-duplicate grouping.
- `src/extract.py`: page-based PDF extraction, OOXML DOCX paragraph/table/header/footer extraction, hidden read-only Microsoft Word DOC extraction, and a safe-mode temporary LibreOffice fallback boundary.
- `src/normalize.py`: Unicode NFC and line-ending normalization, stable normalization mappings, deterministic IDs, language hints, and placeholder normalization.
- `src/ollama_extract.py`: sequential Ollama structured output with temperature `0`, configurable model/context, strict schema validation, grounded coordinate checks, and retry only for malformed structured output.
- `src/assemble.py`: deterministic/source-coordinate assembly, meaningful unparsed-content preservation, and guarded completed-contract reclassification.
- `src/validate.py`: JSON Schema, evidence, hash, coordinate, placeholder, cross-reference, ordering, eligibility, and provider-neutrality checks.
- `src/pilot.py`: end-to-end inventory/pilot execution, review routing, artifact generation, and before/after source-hash verification.
- `src/full_corpus.py`: resumable processing of every source, exact-duplicate grouping, demo-candidate filtering, optional review routing, and before/after source-hash verification.
- `review_app.py` and `src/review_workspace.py`: local-only, server-rendered review dashboard with risk-based coverage, immutable artifact provenance, atomic working-copy revisions, strict corrections, and evidence-bound sanitization.
- `tests/`: automated coverage for the required deterministic and safety boundaries.

## Runtime requirements

Python 3.12 was used for the pilot. Required Python packages are listed in `requirements.txt`.

Legacy `.doc` extraction uses Microsoft Word when available. Word is launched in a separate hidden process with alerts disabled, macro automation security forced to disabled, external link updates disabled, documents opened read-only and excluded from recent files, and save operations disabled. On non-Windows systems, the pipeline exposes a LibreOffice `--headless --safe-mode` conversion boundary that writes only to a temporary directory. LibreOffice was not installed on the pilot machine.

No OCR implementation is included. Encrypted, unreadable, empty, malformed, and likely-OCR records are preserved and routed to review rather than guessed.

## Run

From the repository root:

```powershell
python src\inventory.py --templates-dir templates --output artifacts\manifest.jsonl --strict-counts
python -m pytest -q
python src\pilot.py
python review_app.py
```

The dashboard binds only to `127.0.0.1:5000`. It asks for a reviewer name and creates resumable state under `review-workspace/records/<review_id>/`; it never writes to generated `artifacts/` or legal sources under `templates/`.

Microsoft Word automation may need permission to run outside a restricted execution sandbox. It remains hidden and read-only.

To request another installed Ollama tag or context length:

```powershell
python src\pilot.py --model qwen3.5:4b-q4_K_M --num-ctx 4096
```

The configured tag is not downloaded automatically. The code checks `/api/tags`; an unavailable model never causes fabricated labels or pilot results.

### Full-corpus demo run

The full-corpus runner processes every supported file under `templates/`. For the requested no-review demo using the currently installed Ollama tag:

```powershell
python src\full_corpus.py --model qwen2.5:3b --num-ctx 4096 --skip-review
```

The run is sequential and resumable. Re-running the same command reuses compatible completed content groups. Use `--force` only when every group must be regenerated. Use `--no-model` for deterministic extraction and filename routing without Ollama; this is less suitable for classifying opaque filenames.

Legacy `.doc` extraction runs in an isolated worker with a 60-second default timeout. If Word automation hangs on one file, that source is recorded as unreadable and the corpus continues. Change the boundary with `--doc-timeout 120`. After interrupting a stuck older run, restart without `--force`; completed groups are resumed.

The runner writes under `artifacts/full/`:

- `manifest.jsonl`: every physical source path;
- `records/<artifact_key>/`: extraction, optional model labels, canonical output, validation, and resumable state for each unique content group;
- `demo-candidates.jsonl`: successfully extracted and valid `contract_template` or `completed_contract` groups, including records whose template state is `blank`, `completed`, `mixed`, or `unknown`;
- `excluded.jsonl`: failed, unreadable, invalid, unknown, and non-contract groups;
- `index.json` and `run-summary.json`: complete routing and aggregate counts.

Exact duplicates are processed once while all physical paths remain recorded. `--skip-review` does not mark anything approved: candidate canonical records retain `training_eligibility: review_required`, and the candidate index identifies them as unreviewed demo candidates. This command does not produce Alpaca rows.

## Artifact contract

`artifacts/manifest.jsonl` contains one validated record for every physical source file, including all duplicate paths. `artifacts/pilot/index.json` maps the 36 fixed selections to processed content groups. Each group directory contains:

- `extraction.json`: raw/normalized text, stable blocks, raw-to-normalized mapping, method, status, counts, hashes, and warnings;
- `canonical.json`: a source-grounded proposal when extraction succeeds;
- `labeling-response.json`: only when the configured model actually returned a schema-valid, coordinate-grounded response;
- `validation.json`: extraction and canonical validation results.

`artifacts/review-queue/` holds one review record per unique pilot content group. Canonical records contain no embeddings, vector-provider identifiers, or publication configuration.

## Pilot result — 2026-08-11

- Full inventory: `1,276` files — `955` PDF, `297` DOC, `24` DOCX — across all 15 source folders.
- Exact duplicates: one group, retaining both `CSP_Agreement_Final (1).pdf` and `CSP_Agreement_Final.pdf` paths.
- Pilot: `36` physical files, `35` unique content groups.
- Extraction: `34` successful groups and `1` encrypted group.
- Canonical proposals: `34`; all `34` passed strict schema and semantic validation.
- Review queue: `35` records.
- Human-reviewed, legally approved, or training-eligible: `0`.
- Qwen-assisted records: `0`. The requested `qwen3.5:4b-q4_K_M` candidate was not installed. Ollama exposed only the previously evaluated `qwen2.5:3b` baseline, which this pilot intentionally did not substitute.
- Source integrity: all `1,276` source SHA-256 hashes were identical before and after the run.
- Tests: `12 passed`.

The initial deterministic classifications are routing proposals, not legal or human-reviewed conclusions. Contract-like files without detected square-bracket placeholders start as `completed_contract` and require confirmation, privacy review, and sanitization. Non-contract kinds remain truthful and excluded from contract-template training while still appearing in the review queue.

## Reclassification safeguard

A completed record can be directly reclassified to `contract_template` only through `reclassify_completed_record` after all of the following are recorded:

- sanitization completed;
- privacy review approved;
- structure review approved;
- legal review approved;
- a named human actor, review-record ID, and reason.

The initial classification, source identity, sanitization history, reclassification history, and audit action remain attached. No source file is modified.

## Remaining ambiguity

```text
AMBIGUITY: Non-contract corpus eligibility (AMBIGUITY-023).
Source A: Every collected file must be inventoried, reviewed, and available for a future knowledge-base/training destination.
Source B: This feature is limited to contract templates, while the corpus includes policies, forms, letters, court filings, resolutions, guidance, and other non-contract records.
Decision required: Define separate destinations, a separately reviewed transformation, or audit-only retention for non-contract records.
```

This ambiguity does not block inventory, extraction, validation, truthful classification, or review routing. It blocks only a claim about the eventual non-contract publication/training destination.

```text
AMBIGUITY: No-review demo candidates versus production fine-tuning eligibility.
Source A: The approved feature plan requires human structure and legal review, plus privacy review and sanitization for completed or mixed records, before training eligibility.
Source B: The demo run is explicitly requested without human review and retains contract templates, completed contracts, and mixed records as future Alpaca candidates.
Decision required: Before any actual fine-tuning export, decide whether review remains mandatory or formally replace the existing eligibility rule and define the privacy/sanitization treatment for completed and mixed records.
```

## Compact Qwen enrichment and runtime records

The canonical record remains the evidence-bearing audit record. Contract-only Qwen
enrichment is a separate, resumable post-processing flow; it never sends the
canonical JSON to the model and never overwrites canonical source artifacts.

```powershell
python src\qwen_enrichment.py packets --canonical-dir artifacts\full\records --output-dir artifacts\qwen-packets
python src\qwen_enrichment.py run --packets-dir artifacts\qwen-packets --model qwen2.5:3b --num-ctx 4096
python src\qwen_enrichment.py merge --canonical-dir artifacts\full\records --packets-dir artifacts\qwen-packets --output-dir artifacts\enriched-canonical
python src\runtime_templates.py --canonical-dir artifacts\enriched-canonical\records --output-dir artifacts\runtime
```

Only `contract_template` and `completed_contract` records produce Qwen packets.
The runtime step accepts only blank `contract_template` records that have already
become `training_eligibility: eligible`; completed contracts must first pass the
existing sanitization, privacy, legal, structure, and reclassification safeguards.
`contract-template-runtime-v1` is a compact RAG/training-source record, not an
Alpaca training row.

For an explicitly unreviewed demonstration only, including completed contracts:

```powershell
python src\runtime_templates.py --canonical-dir artifacts\enriched-canonical\records --output-dir artifacts\runtime-demo --demo-unreviewed
```

Each record from this command has `runtime_status: demo_unreviewed`. It must not
be represented as training-eligible or used for actual fine-tuning until it
passes the existing review and sanitization safeguards.

## Agent 1 demo inputs

Generate one raw Agent 1 demo input beside every runtime-demo record with the
locally installed Qwen model and its 32K context:

```powershell
python src\agent1_demo_inputs.py --runtime-dir artifacts\runtime-demo --model qwen2.5:3b --num-ctx 32768 --num-predict 8192 --case-date 2026-08-14
```

The command is sequential and resumable. Each record directory retains
`agent1-qwen-response.json`, the normalized `agent1-qwen-proposal.json`,
`agent1-input.json` when assembly succeeds, and
`agent1-input-validation.json`. Blank templates use deterministic fictional
facts. Completed contracts use only source-extracted facts cited to runtime
clause or unparsed-content IDs. Unresolved required fields are rendered as the
literal `AMBIGUITY` and recorded in validation warnings.

Every generated input remains `demo_unreviewed` and
`eligible_for_alpaca_export: false`. Completed-contract inputs additionally set
`privacy_review_required: true`. The raw Agent 1 payload still contains its
required synthetic `session_id` and `user_id`; the Agent 2 intake adapter removes
them from every model-facing and Alpaca artifact while retaining raw inputs as
non-model audit files.

## Agent 2 grounded demo and Alpaca export

Run the deterministic pipeline after the resumable Agent 1 generator has
written any available inputs:

```powershell
python src/alpaca_export.py --runtime-dir artifacts/runtime-demo --output-dir artifacts/alpaca-demo --canonical-dir artifacts/enriched-canonical/records --extraction-dir artifacts/full/records --mode demo --expected-records 90
```

The pipeline audits all 90 indexed runtime records. A record without a valid
Agent 1 input is written to `rejected.jsonl`; rerunning later processes it
without requiring a force flag. Valid records receive:

- `agent2-generation-facts.json` with `session_id` and `user_id` removed;
- `agent2-generation-validation.json` with mappings, hashes, and normalization audit;
- `retrieved-evidence.json` containing the exact runtime documentary evidence;
- `contract-target.txt` reconstructed from complete extraction order;
- `contract-target-validation.json` with passage-level grounding and validation results.

Dataset artifacts are written to `artifacts/alpaca-demo/` as `alpaca.jsonl`,
`manifest.jsonl`, `rejected.jsonl`, and `summary.json`. Demo Stage 5 has no human
review step by explicit project decision, but it never marks a row eligible:
all manifest rows are `demo_unreviewed` with `training_eligible: false`.
Production mode requires approved source eligibility and a separate target
review bound to the exact target hash.

## Qwen2.5 3B QLoRA fine-tuning

`src/finetune_slm.py` consumes a terminal-supplied Alpaca export and trains only
an Agent 2 adapter. Agent 1 remains the information collector, the retrieved
documentary material remains the only source of legal content, and Agent 3 is
not involved in training.

Install a CUDA-enabled PyTorch build suitable for the training host, then use
the isolated training dependencies:

```powershell
python -m pip install -r requirements-finetune.txt
```

The default QLoRA profile requires a CUDA GPU with at least 16 GB VRAM. The
current workstation's CPU-only PyTorch build and 4 GB GPU can run neither this
profile nor the 8192-token training batches. It can still inspect output files
and use the final quantized model through Ollama.

Run an exact-token preflight without loading model weights:

```powershell
python src/finetune_slm.py artifacts/alpaca-demo --output-dir artifacts/finetuned/hannibal-demo-preflight --allow-unreviewed-demo --preflight-only
```

Run the explicitly non-production demo on the external training GPU:

```powershell
python src/finetune_slm.py artifacts/alpaca-demo --output-dir artifacts/finetuned/hannibal-demo-qwen2.5-3b --allow-unreviewed-demo
```

The script verifies row and manifest hashes, source/template/clause identities,
eligibility metadata, and forbidden identifiers before model loading. It uses
Qwen's chat template, supervises only assistant completion tokens, groups
source/template-related records into the same deterministic 80/10/10 split,
and never truncates. Rows above the explicit 8192-token demo limit are written
unchanged to `excluded.jsonl`. Training and evaluation loss are diagnostics;
they do not establish legal correctness or satisfy the project KPIs.

The current 65-row `alpaca-demo` corpus is unreviewed, below the source-defined
minimum of 200 eligible examples, and has unknown adapter families. The
`--allow-unreviewed-demo` flag is therefore mandatory and every output remains
demo-only. Production exports cannot use that flag to bypass schema, integrity,
review, source-grounding, or minimum-volume checks.

To merge the trained adapter, convert it with an existing llama.cpp checkout,
quantize it as `Q4_K_M`, and register it locally in Ollama:

```powershell
python src/finetune_slm.py artifacts/alpaca-demo --output-dir artifacts/finetuned/hannibal-demo-qwen2.5-3b --allow-unreviewed-demo --export-ollama --llama-cpp-dir C:\path\to\llama.cpp --ollama-model hannibal-qwen2.5-3b-demo
```

Ollama packaging requires `convert_hf_to_gguf.py`, a built
`llama-quantize`/`llama-quantize.exe`, and `ollama` on `PATH`. The script merges
against the same Hugging Face base used for QLoRA before GGUF conversion; it
does not attach the QLoRA Safetensors adapter directly to the installed Ollama
base.
