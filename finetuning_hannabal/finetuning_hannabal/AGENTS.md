# Hannibal V2 Agent Instructions

## Scope

This repository concerns the fine-tuning of Hannibal V2, E-Tafakna's AI Legal Contract Reasoning Engine.

The only project sources of truth currently available are:

- `Hannibal V2.pdf`, sections 1–6: project purpose, three-agent architecture, operating principles, pipeline, and fine-tuning boundary.
- `Hannibal R&D.pdf`, sections 7–14: problems, objectives, datasets, fine-tuning strategy, infrastructure, roadmap, and KPIs.
- `contrat_example.md`: examples of contract blueprints, contract-and-question records, risk-clause records, and legal-clause records.

Do not add legal, product, architectural, training, or operational claims from general knowledge or external sources. If the supplied sources do not decide something, write `AMBIGUITY` and preserve the competing possibilities.

## Non-negotiable system rules

1. Hannibal is a Legal Contract Reasoning Engine, not a chatbot and not a simple GPT contract generator.
2. The runtime architecture has three defined agents:
   - Agent 1 — Information Collector.
   - Agent 2 — Hannibal Core / Legal Reasoning Engine.
   - Agent 3 — Document Formatter.
3. Agent 1 collects information and never drafts a contract.
4. Agent 2 owns request analysis, index selection, RAG retrieval, clause selection, contract construction, coherence validation, and risk analysis.
5. Agent 3 owns layout and export to DOCX, PDF, and HTML. It contains no legal logic.
6. The model must never invent law. Legal content comes exclusively from the documentary indexes used by RAG.
7. Fine-tuning improves reasoning, organization, drafting, reformulation, legal style, structure, clause choice, and global coherence. It does not replace the legal indexes or teach the model the law.
8. Preserve jurisdiction, language, source tracking, and auditability when the source records provide them.
9. Never silently repair a contradiction in the specification. Record it as `AMBIGUITY`.

## Required processing flow

Keep changes aligned with this source-defined flow:

1. Collect user information.
2. Analyze the requested contract.
3. Choose the relevant indexes.
4. Retrieve documentary evidence through RAG.
5. Select clauses.
6. Generate the contract.
7. Validate it.
8. Analyze risks.
9. Produce the final formatted document.

Do not move legal reasoning into Agent 1 or Agent 3.

## Data rules

### Fine-tuning records

The source specifies the standard Alpaca shape:

```json
{
  "instruction": "What the model must do",
  "input": "Contract facts and user context",
  "output": "The corrected target contract or target reasoning result"
}
```

Keep those three fields distinct. The sources do not define an alternative production training schema.

Prioritize data quality over volume. The source levels are:

- Minimum: 200 examples, acceptable quality.
- Recommended: 500–1,000 examples, good quality.
- Optimal: 2,000 or more examples, excellent quality.

The planned dataset families are:

- A — Intent → Blueprint.
- B — Blueprint → Clauses.
- C — Clause → Drafted legal version.
- D — Contract → Omissions/inconsistencies analysis.

### Documentary examples

Treat `contrat_example.md` as an example of several record families, not as one ready-to-train JSON document:

- Contract blueprint/template metadata with mandatory and conditional sections, per-section variable points, required elements, adaptation hints, retrieval preferences, and a vector.
- Tunisian contract context with repeated question, answer, answer-offset, answer-presence fields, and a vector.
- Risk-clause metadata with detection patterns, severity, legal basis, recommendation, impact, probability, keywords, embedding text, and a vector.
- Legal-clause text with an identifier and vector.

Do not fabricate missing vector values, delimiters, schemas, questions, answers, laws, or clauses. Do not treat the displayed `...` vector fragments as complete embeddings.

### Contract quality

At minimum, validations must cover the source-defined concerns:

- Required sections.
- Missing clauses.
- Malformed or unreplaced placeholders.
- Internal contradictions.
- Legal hallucinations.
- Legal drafting style.
- MENA adaptation.
- Review time and generation time.

The exact KPI conflicts are recorded in `context.md`; do not choose one conflicting threshold without an explicit decision.

## Fine-tuning constraints

- Preferred method: LoRA.
- Alternative: QLoRA.
- Full fine-tuning is discouraged.
- Recommended base-model candidates: Llama 3.1 8B or Mistral 7B.
- Alternative: GPT-4o-mini through Azure fine-tuning.
- Agent 2 must be able to load specialized adapters dynamically.
- Adapter names, covered contract types, and priorities must remain aligned with `architecture.md`.

Do not silently select the unresolved base model, cloud service, GPU option, or LoRA/QLoRA variant.

## Ambiguity protocol

Use this form close to the affected text or code:

```text
AMBIGUITY: <unresolved issue>
Source A: <what one source states>
Source B: <what the other source states or what is missing>
Decision required: <the exact decision that must be made>
```

An ambiguity is not permission to invent a default. Safe validation that enforces an undisputed invariant—especially “law only from RAG”—may proceed while the ambiguity remains documented.

## Completion checklist

Before declaring work complete:

- Trace every material behavior to one of the three source files.
- Confirm Agent 1 does not draft and Agent 3 contains no legal logic.
- Confirm Agent 2 retrieves legal material before using it.
- Confirm fine-tuning has not become a source of law.
- Validate the Alpaca `instruction` / `input` / `output` separation.
- Check mandatory sections, clause omissions, placeholders, and contradictions.
- Preserve source and jurisdiction metadata where available.
- Report every unresolved source conflict as `AMBIGUITY`.
- Do not claim a KPI has passed without using its source-defined measurement method.

## Known critical ambiguities

- `AMBIGUITY`: The architecture defines only three agents, while Dataset D says its examples are generated by “Agent 4.”
- `AMBIGUITY`: Required-section completeness is stated as both 100% and greater than 99%.
- `AMBIGUITY`: Phase 4 is assigned to weeks 7–8, while the global calendar places integration and A/B deployment in weeks 9–12.
- `AMBIGUITY`: The exact Agent 1 → Agent 2 and Agent 2 → Agent 3 payload schemas are not specified.
- `AMBIGUITY`: The source recommends both Llama 3.1 8B and Mistral 7B without selecting one.

See `context.md` for the complete ambiguity register.
