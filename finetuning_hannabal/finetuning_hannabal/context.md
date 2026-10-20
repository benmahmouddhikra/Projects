# Hannibal V2 Project Context

## 1. Read this first

This file is the compact source-grounded context for the Hannibal V2 fine-tuning work.

Current sources of truth:

1. `Hannibal V2.pdf` — specification sections 1–6.
2. `Hannibal R&D.pdf` — specification sections 7–14.
3. `contrat_example.md` — examples of templates, contracts and Q&A, risk records, and clause records.

Approved feature-level decisions are recorded in `features/contract-template-dataset/PLAN.md`. The fixed Agent 1 interface supplied for that feature is recorded in `features/contract-template-dataset/agent_1_json_output.md`. These additions define feature and interface behavior only; they do not supply law or legal content.

No external facts have been used to resolve gaps. Every unresolved gap or conflict is labeled `AMBIGUITY`.

## 2. Product definition

Hannibal is E-Tafakna's AI Legal Contract Reasoning Engine for generating professional contracts adapted to multiple MENA jurisdictions.

It is:

- A multi-agent system.
- A RAG-based legal retrieval system.
- A consumer of several proprietary documentary databases.
- A legal reasoning and drafting engine.
- A validation and risk-analysis engine.
- A document-generation pipeline.

It is not:

- A chatbot.
- A simple GPT contract generator.
- A model that learns or invents law through fine-tuning.

Core doctrine: legal rules come exclusively from the documentary indexes. The AI reasons, organizes, drafts, reformulates, and explains.

## 3. Runtime architecture

### Agent 1 — Information Collector

Collects contract type, country, language, parties, amounts, duration, and specific information. It outputs structured JSON to Agent 2 and never drafts.

### Agent 2 — Hannibal Core

Analyzes the request, chooses indexes, retrieves content through RAG, selects clauses, constructs the contract, checks coherence, and analyzes risks.

All legal reasoning intelligence is concentrated here.

### Agent 3 — Document Formatter

Handles layout, numbering, table of contents, signatures, and export to DOCX, PDF, and HTML. It has no legal logic.

### Pipeline

```text
User
-> information collection
-> contract analysis
-> index choice
-> RAG documentary research
-> clause selection
-> generation
-> validation
-> risk analysis
-> final document
```

## 4. Why fine-tune

Current problems identified by the R&D source:

- Clauses can be vague or generic.
- Structure can be inconsistent.
- Essential clauses can be omitted.
- Style can sound repetitive, long, or “LLM-like.”
- MENA cultural adaptation is limited.
- Dates and amounts can be formatted incorrectly.

Fine-tuning objectives:

- Improve structure.
- Reduce format hallucinations.
- Improve drafting quality.
- Reduce clause omissions.
- Adapt MENA style.
- Reduce lawyer review time.

Fine-tuning does not teach laws. Laws remain in RAG.

## 5. Training strategy snapshot

### Data types to collect

- Corrected final contracts.
- Prompt → contract pairs.
- Lawyer-drafted clauses.
- Structure/blueprint examples.
- Erroneous contracts paired with corrections.

### Standard record

```json
{
  "instruction": "Task requested from the model",
  "input": "User and contract context",
  "output": "Corrected target result"
}
```

### Dataset volume

- Minimum: 200.
- Recommended: 500–1,000.
- Optimal: 2,000+.
- Quality takes priority over quantity.

### Planned transformations

- A: intent → blueprint, 200 examples.
- B: blueprint → clauses, 300 examples.
- C: clause → drafted version, 500 examples.
- D: contract → omissions/inconsistencies, 200 examples.

### Model and adaptation candidates

- Llama 3.1 8B or Mistral 7B.
- GPT-4o-mini through Azure fine-tuning as an alternative.
- LoRA recommended.
- QLoRA as an alternative.
- Full fine-tuning discouraged.

### Adapter families

- Employment: CDI, CDD, freelance, employment.
- Commercial: services, NDA, MSA, partnership.
- SaaS: SaaS, API access, cloud services.
- NDA: confidentiality and NDA.
- Phase 2: Investment, Real Estate, Procurement.

## 6. Example data vocabulary

### Blueprint

A contract structure record describing identity, jurisdiction/language compatibility, retrieval preferences, mandatory and conditional sections, section purposes, variables, required elements, adaptation hints, behavior flags, and a vector.

The supplied employment blueprint is Arabic by default, supports UAE, permits Middle East and international structures, has ten populated sections, uses a similarity threshold of `0.65`, and requests at most three similar results.

### Contract/Q&A record

A Tunisian contract context plus fifteen repeated question/answer groups and a vector. Two examples are supplied: personal services and a simple exchange without balancing payment.

### Risk record

A record describing a risky clause through detection patterns, risk level, legal basis, recommendation, impact, probability, keywords, embedding text, and vector.

### Legal-clause record

An identifier, clause text, and vector.

### Placeholder

The sources show bracketed placeholders such as `[Nom]`, `[montant]`, `[CONTRAT COMPLET]`, and `[À COMPLÉTER]`. They require automatic format/replacement checks but do not define one canonical placeholder grammar.

### Generated contract template

A reviewed, sanitized, source-grounded JSON record derived from a collected source document for contract-template retrieval. It replaces the flattened production blueprint format demonstrated in `contrat_example.md` without modifying the source document.

### Canonical placeholder

A rendered missing-value token in `[UPPER_SNAKE_CASE]` form whose internal variable identifier uses `lower_snake_case`. This grammar was approved for the contract-template feature on 2026-08-11.

### Agent 1 demo input

A raw Agent 1 payload paired with one `demo_unreviewed` runtime contract for pipeline demonstration. Blank templates receive visibly fictional values. Completed contracts may receive facts extracted from their existing text, retain source-evidence identifiers, and require privacy review. The artifact is neither model-facing input nor an Alpaca row and is not training-eligible.

### Qwen Agent 1 proposal

An unreviewed, schema-constrained intermediate record that maps runtime contract evidence to the unchanged Agent 1 interface. It may classify and extract facts but cannot add legal clauses, laws, citations, or training approval. Deterministic code validates and assembles the final Agent 1 demo input.

## 7. Source-defined targets

| Target | Value |
|---|---:|
| Required sections | 100% in the objective table; greater than 99% in the KPI table |
| Malformed placeholders | 0 |
| Correctly replaced variables | Greater than 99% |
| Missing clauses | Less than 2% |
| Internal contradictions | Less than 1% |
| Legal hallucinations | 0% |
| Drafting/legal style | Greater than 85% |
| MENA adaptation | Greater than 80% |
| Accepted without major change | Greater than 90% |
| Lawyer correction time | Less than 10 minutes |
| Generation time | Less than 15 seconds |

Validation includes 50 test contracts and an A/B production exposure of 10% of users.

## 8. Source-defined implementation values

```text
Learning rate: 2e-5 Llama / 5e-6 Mistral
Epochs: 2
Batch size: 4 on 24 GB / 8 on 40 GB
LoRA rank: 16
LoRA alpha: 32
QLoRA quantization: 4-bit
Maximum sequence length: 4096
Optimizer: paged_adamw_8bit
```

GPU choices:

- Minimal: NVIDIA A10G, 24 GB, ~€1/hour.
- Recommended: NVIDIA A100, 40 GB, ~€3/hour.
- Optimal: NVIDIA H100, 80 GB, ~€6/hour.

Cloud choices listed: Azure NCas-v4, RunPod, Vast.ai, Google Colab Pro+.

Libraries listed: `unsloth`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`, and `wandb`.

## 10. Decision rule

Until an ambiguity is resolved:

- Preserve all source statements.
- Do not invent a legal rule, product feature, schema, service, threshold, or schedule.
- Continue only with work that does not depend on the missing decision.
- Keep the invariant that all law comes from RAG.
