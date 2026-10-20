# Guide de navigation — Projets_Interns

> Généré le 2026-10-20. Ce guide résume le contenu de chaque projet du dossier `C:\Users\user\Desktop\Projets_Interns`, comment le lancer, et ce qu'il faut savoir.
> Les noms de variables d'environnement sont listés, **jamais leurs valeurs**.

---

## Vue d'ensemble

| Projet | Sujet | Stack principale | LLM | Lancement | Git |
|---|---|---|---|---|---|
| [`dpo_legalchat`](#1-dpo_legalchat) | Chatbot RAG droit tunisien + boucle de feedback → DPO/GRPO | FastAPI + React/Vite + SQLite | `Qwen/Qwen2.5-1.5B-Instruct` (transformers) | API :8000 / UI :5173 | `mayartahri3/dpo_legalchat` |
| [`E-Tafakna-Legal-RAG-DPO`](#2-e-tafakna-legal-rag-dpo-hannibal) | « Hannibal » : chatbot RAG juridique + corrections expertes → DPO sur Kaggle | FastAPI + Next.js + FAISS | `qwen3:4b` (Ollama), DPO sur `Qwen3-4B-Thinking-2507` | API :8010 / UI :3000 | `HassenProgg/E-Tafakna-Legal-RAG-DPO` |
| [`Elysa`](#3-elysa-elyssa) | « Elyssa » : assistant RAG droit du travail + guide de style pour LoRA | Flask + FAISS + llama-server | `Qwen2.5-7B-Instruct` (par défaut) | UI :5000 / LLM :8081 | `louaybennoamen/Elysa` |
| [`finetuning_hannabal`](#4-finetuning_hannabal-hannibal-v2) | Hannibal V2 : pipeline de données contrats + préparation fine-tuning QLoRA (Agent 2) | Python CLI + Flask (revue) + Ollama | `Qwen/Qwen2.5-3B-Instruct` (QLoRA SFT) | Scripts CLI / revue :5000 | ❌ pas un dépôt git |
| `Hannibal_V2_Technical_Report.pdf` | Rapport technique du projet 4 | — | — | — | — |

**Points communs :** les 4 projets sont des briques E-Tafakna (droit tunisien / MENA), tous en Python. Les projets 1, 2 et 3 sont des chatbots RAG avec FAISS ; les projets 1 et 2 ajoutent une boucle feedback → DPO.

⚠️ **Conflit de port :** `Elysa` (app.py) et `finetuning_hannabal` (review_app.py) utilisent tous les deux le port **5000** — ne pas les lancer en même temps.

---

## 1. `dpo_legalchat`

📁 `dpo_legalchat\` — 🔗 https://github.com/mayartahri3/dpo_legalchat.git (branche `main`)

### Objectif
Chatbot RAG sur le droit tunisien qui cite les articles de loi et s'améliore avec les retours utilisateurs :
- Question → recherche FAISS des articles → réponse d'un LLM local avec citations.
- Avocats et juristes internes proposent des corrections, validées par des experts.
- Les corrections validées alimentent un fine-tuning **DPO / GRPO** (LoRA). PPO n'est pas utilisé.
- Chaque version est évaluée sur un jeu de test figé avant d'être promue.
- Les citoyens ne corrigent pas : ils choisissent entre deux styles de réponse.

### Stack
- **Backend :** FastAPI, uvicorn, pydantic v2, sentence-transformers, faiss-cpu, transformers, torch, trl, peft, datasets, accelerate
- **Modèles :** LLM `Qwen/Qwen2.5-1.5B-Instruct`, embeddings `sentence-transformers/all-MiniLM-L6-v2`
- **BDD :** SQLite (`backend/feedback.db`)
- **Frontend :** React 19, Vite 8, axios
- **Entraînement :** Google Colab (GPU), webhook n8n en option

### Où trouver quoi
```
backend/
  api.py              ← POINT D'ENTRÉE FastAPI (toutes les routes /api/*)
  minirag.py          EmbeddingIndex (FAISS) + LocalLLM
  feedback_store.py   SQLite : feedback, corrections, statuts
  validation.py       Score des corrections → envoi aux experts
  dashboard.py        Métriques SQL
  dataset_builder.py  Paires d'entraînement + split train/test
  dpo_train.py, grpo_train.py, reward_model.py, train_common.py
  ppo_train.py        (inutilisé)
  eval_metrics.py, model_eval.py           Évaluation / comparaison de versions
  export_for_colab.py, colab_train.py, import_colab_model.py   Workflow Colab
  articles/           43 JSON d'articles de loi
  datasets/           dataset_v1/v2/v4 (.jsonl + .meta.json)
  models/             Adaptateurs LoRA v1..v3 + registry.json
  reward_models/      registry.json
frontend/src/
  App.jsx             Routage (chat, articles, review, dashboard, training)
  api.js              Client HTTP (VITE_API_URL, sinon http://localhost:8000)
  components/         ChatView, Sidebar, Articles, ReviewQueue, Dashboard, Training, FeedbackBlock
```

**Routes API principales :** `/api/chat`, `/api/feedback`, `/api/correction`, `/api/style-preference`, `/api/review/pending|decide`, `/api/dashboard`, `/api/articles`, `/api/train*`, `/api/models/compare|history|import`, `/api/promote/{version}`, `/api/health`.

### Lancer
```powershell
# Backend (port 8000)
cd dpo_legalchat\backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000

# Frontend (port 5173)
cd dpo_legalchat\frontend
npm install
npm run dev
```
**Entraînement Colab :** `python eval_metrics.py --make-split` → `python export_for_colab.py` → sur Colab `colab_train.py --export-zip feedback_export.zip --method all --epochs 3` → en local `import_colab_model.py --export-dir ...`. Promotion d'une version via la page « Entraînement » de l'UI, puis redémarrer uvicorn.

**Variables d'env** (`backend/.env`) : `ARTICLES_DIR`, `DB_PATH`, `EMBED_CACHE_DIR`, `EMBED_MODEL`, `LLM_MODEL`, `FORCE_BASE_MODEL`, `TOP_K`, `MAX_NEW_TOKENS`, `AUTO_TRAIN_MIN_EXAMPLES`, `AUTO_TRAIN_EPOCHS`, `AUTO_TRAIN_VIA_N8N`, `N8N_WEBHOOK_URL`, `N8N_WEBHOOK_TOKEN`, `PUBLIC_API_URL`, `EXTERNAL_API_TOKEN` — Frontend : `VITE_API_URL`.

### ⚠️ À savoir
- Des fichiers exclus par `.gitignore` sont **quand même suivis par git** : `backend/.env`, `frontend/.env`, `backend/feedback.db`, `models/` (.safetensors/.bin), `node_modules/` (~1087 fichiers)… Des secrets ont peut-être été poussés sur GitHub.
- Dossiers imbriqués suspects : `backend/models/models/v1`, `backend/models/reward_model/reward_models/v1`.
- `@vitejs/plugin-react` est importé dans `vite.config.js` mais absent de `package.json` → `npm install` sur un clone propre peut échouer.

---

## 2. `E-Tafakna-Legal-RAG-DPO` (Hannibal)

📁 `E-Tafakna-Legal-RAG-DPO\` — 🔗 https://github.com/HassenProgg/E-Tafakna-Legal-RAG-DPO.git (branche `main`)

### Objectif
« Hannibal — E-Tafakna » : chatbot de questions juridiques (RAG).
- Recherche dans un corpus local (FAISS + embeddings `BAAI/bge-m3`), génération avec Ollama `qwen3:4b`.
- Les utilisateurs notent les réponses et proposent des corrections ; un expert valide / rejette / reformule → paires de préférence DPO.
- **Entraînement DPO uniquement sur Kaggle** (GPU T4, QLoRA NF4, LoRA rang 8) sur `Qwen/Qwen3-4B-Thinking-2507`.
- L'adaptateur est réimporté, packagé en GGUF Q4_K_M, enregistré comme modèle Ollama `hannibal-dpo-*`, évalué contre la base, puis activé après approbation de l'expert.

### Stack
- **Backend :** FastAPI, uvicorn, ollama, transformers, sentence-transformers, faiss-cpu, torch
- **Frontend :** Next.js, React, TypeScript
- **Stockage :** index FAISS + fichiers JSON/JSONL (pas de BDD)

### Où trouver quoi
```
backend/api.py         ← POINT D'ENTRÉE FastAPI : /ask, /feedback, /corrections, /health, ~30 routes /admin/*
src/
  retrieve.py, generator.py, intent.py, load_data.py, build_index.py   Pipeline RAG
  feedback/            models.py, store.py (avis et corrections)
  validation/          rag_checker.py, confidence.py, aggregator.py, expert.py
  dpo/                 dataset, kaggle_bridge, kaggle_train, checkpoint, package_ollama,
                       deployment, model_registry, comparison, evaluation, scheduler…
  evaluation/          evaluate.py, evaluate_generation.py, generate_questions.py
  test_*.py            Tests dans chaque module
frontend/
  app/page.tsx         Chat
  app/admin/page.tsx   Admin (Suivi / Corrections / Entraînement / Versions)
  lib/api.ts           Client API
  next.config.ts       Proxy /api/backend/* → backend
data/raw/              25 documents juridiques JSON ; data/questions.json
indexes/               legal_index.faiss + metadata.json
notebooks/kaggle_dpo_training.ipynb   Notebook de référence Kaggle
scripts/               build_kaggle_notebook.py, prepare_kaggle_run.py
```

### Lancer (PowerShell, depuis la racine du projet)
```powershell
# Installation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r backend/requirements.txt
Copy-Item frontend/.env.local.example frontend/.env.local
ollama pull qwen3:4b
cd frontend; npm.cmd ci

# Backend (port 8010) — Ollama doit tourner
$env:E_TAFAKNA_LOCAL_ADMIN_MODE="1"
.\.venv\Scripts\python.exe -m uvicorn backend.api:app --host 127.0.0.1 --port 8010

# Frontend (autre terminal) → http://127.0.0.1:3000  (admin : /admin)
cd frontend; npm.cmd run dev -- --hostname 127.0.0.1
```
- Avant un export DPO : `python -m src.dpo.checkpoint --refresh`
- Tests : `python -m unittest src.dpo.test_pipeline src.dpo.test_baseline_benchmark` ; frontend : `node node_modules/typescript/bin/tsc --noEmit`

**Variables d'env :** backend `E_TAFAKNA_LOCAL_ADMIN_MODE`, `E_TAFAKNA_LOCAL_FILES_ONLY`, `E_TAFAKNA_OLLAMA_MODEL` — frontend `BACKEND_API_URL` (obligatoire), `NEXT_PUBLIC_API_URL` (facultative).

### ⚠️ À savoir
- Le mode admin local **n'a pas d'authentification** → rester sur 127.0.0.1.
- Beaucoup de données sont hors git (`data/feedback/`, `data/dpo/`, `models/`, `*.gguf`, `.env*`…) → absentes d'un clone neuf.
- Seulement 8 paires d'entraînement + 2 de validation : le circuit marche, mais aucun gain de qualité prouvé (minimum 4 paires uniques).
- Pas d'entraînement local ; la fusion CPU demande 16 Go RAM + 26 Go disque.
- Dépendances npm en version `latest` (non figées).

---

## 3. `Elysa` (Elyssa)

📁 `Elysa\` — 🔗 https://github.com/louaybennoamen/Elysa.git (branche `main`)

### Objectif
Deux choses dans le même dépôt :
- **README :** petits modèles affinés en droit tunisien (QLoRA + Unsloth sur Kaggle, ~1 660 conversations synthétiques arabe/derja/français), publiés sur Hugging Face sous `Hedi-Bk/...` : Qwen3 8B (LoRA 2 et 3 epochs, GGUF, versions « CORRECTION ») et Gemma 4B (LoRA + GGUF, 2 epochs).
- **Code :** assistant juridique RAG local « Elyssa » (Flask) sur le droit du travail.
- **Principe (`docs/COMPLIANCE.md`) :** les LoRA n'apprennent que le style et le raisonnement, avec des placeholders ; les vraies références légales viennent **uniquement du RAG**.
- **Style (`docs/Elyssa_Style_Guide_v2.md`) :** 5 sections par réponse (Résumé, Analyse, Références, Recommandations, Conclusion) ; placeholders `[REFERENCE] [ARTICLE] [LOI] [JURISPRUDENCE] [SOURCE] [DATE]` ; réponse dans la langue de l'utilisateur.

### Stack
Flask, requests, FAISS (`IndexFlatIP`), embeddings `paraphrase-multilingual-MiniLM-L12-v2`, LLM servi par un `llama-server` compatible OpenAI (par défaut `Qwen/Qwen2.5-7B-Instruct`, temperature 0.05).

### Où trouver quoi
```
app.py                 ← POINT D'ENTRÉE Flask : GET "/" et POST "/ask", prompt système, appel LLM
train.py               Vérification avant entraînement (--adapter style|reasoning) — ne lance PAS l'entraînement
rag/retriever.py       Classe LawRetriever : découpe des articles, index, search(top_k=5)
data/laws/             code_travail.txt, travailleurs_etrangers.txt (textes de loi bruts)
index/                 laws.index, texts.txt, sources.txt (index FAISS persistant)
dataset/               train.jsonl (9), val.jsonl (3), golden_benchmark.jsonl (3) + validate_dataset.py
docs/                  COMPLIANCE.md, Elyssa_Style_Guide_v2.md
```

### Lancer
```powershell
cd Elysa
pip install -r requirements.txt
# 1) Démarrer d'abord un llama-server (modèle GGUF) sur 127.0.0.1:8081
python app.py                        # → http://127.0.0.1:5000
python rag/retriever.py              # tester le RAG seul (reconstruit l'index s'il manque)
python dataset/validate_dataset.py   # valider les datasets
python train.py --adapter style      # validation à blanc
```
**Variables d'env :** `ELYSSA_LLM_URL`, `ELYSSA_MODEL_NAME`. Pas de README d'installation : les commandes ci-dessus sont déduites du code.

### ⚠️ À savoir
- Le **README et le code ne parlent pas des mêmes modèles** (Qwen3 8B / Gemma 4B d'un côté, Qwen2.5-7B de l'autre). Bug signalé sur le Qwen3 « CORRECTION » : boucle « I am a Legal Assistant ».
- L'index ne contient que des articles de `code_travail.txt` (23 entrées) : la regex ne semble pas reconnaître les en-têtes type `Art. 258 (nouveau) – ...` (non vérifié en exécutant).
- Datasets volontairement minuscules (jeux de départ). Pas de tests, pas de `.gitignore`.

---

## 4. `finetuning_hannabal` (Hannibal V2)

📁 Le vrai projet est **un niveau plus bas** : `finetuning_hannabal\finetuning_hannabal\` (noté `ROOT` ci-dessous). ❌ Ce n'est pas un dépôt git (pas de sauvegarde distante !).

### Objectif
Pipeline de données et préparation du fine-tuning de **Hannibal V2**, le « AI Legal Contract Reasoning Engine » d'E-Tafakna (génération de contrats pour les pays MENA).
- Corpus de contrats PDF/DOC/DOCX → inventaire (SHA-256) → extraction → normalisation → JSON canonique validé → enrichissement Qwen (Ollama) → enregistrements runtime → entrées démo Agent 1 → lignes Alpaca → entraînement QLoRA.
- Architecture à 3 agents : **Agent 1** collecte, **Agent 2** raisonne (celui qu'on fine-tune), **Agent 3** met en forme.
- Règle centrale : le modèle n'invente jamais le droit — le contenu juridique vient du RAG.

### Stack
- Modèle de base : `Qwen/Qwen2.5-3B-Instruct` (`src\finetune_slm.py`, ligne 30)
- QLoRA SFT : rank 16, alpha 32, dropout 0.05, lr 1e-4, 2 epochs, split 80/10/10 (pas de DPO)
- `requirements-finetune.txt` : transformers, trl, peft, accelerate, datasets, bitsandbytes
- `requirements.txt` : jsonschema, Flask, PyPDF2, pywin32, pytest (+ reportlab pour le PDF)
- Ollama `qwen2.5:3b` (enrichissement), llama.cpp (export GGUF Q4_K_M)

### Où trouver quoi (depuis `ROOT`)
```
README.md                      ← GUIDE PRINCIPAL avec toutes les commandes
AGENTS.md                      Règles non négociables + protocole AMBIGUITY
architecture.md, context.md, implementation_roadmap.md
Hannibal V2.pdf, Hannibal R&D.pdf, contrat_example.md   Sources de vérité (specs)
src/
  inventory.py, extract.py, normalize.py, ollama_extract.py, assemble.py, validate.py   Étapes du pipeline
  pilot.py                     Pilote sur 36 fichiers
  full_corpus.py               Corpus complet
  qwen_enrichment.py, runtime_templates.py
  agent1_demo_inputs.py, agent2_intake.py, contract_target_builder.py, alpaca_export.py
  finetune_slm.py              ← Entraînement QLoRA + export Ollama
  review_workspace.py          Logique de revue humaine
review_app.py + review_ui/     Tableau de bord de revue Flask (127.0.0.1:5000)
review-workspace/records/      ~40 états de revue sauvegardés
config/                        taxonomy.json, pilot-manifest.json
schemas/                       17 schémas JSON
prompts/                       4 prompts Qwen
reports/                       Hannibal_V2_Technical_Report.md + generate_report_pdf.py
tests/                         11 fichiers pytest
artifacts/                     VIDE (sorties / checkpoints absents)
```

### Lancer (depuis `ROOT`)
```powershell
python src\inventory.py --templates-dir templates --output artifacts\manifest.jsonl --strict-counts
python -m pytest -q
python src\pilot.py
python src\full_corpus.py --model qwen2.5:3b --num-ctx 4096 --skip-review
python src\qwen_enrichment.py packets|run|merge ...
python src\runtime_templates.py ... --demo-unreviewed
python src\agent1_demo_inputs.py --runtime-dir artifacts\runtime-demo --model qwen2.5:3b --num-ctx 32768
python src\alpaca_export.py --runtime-dir artifacts/runtime-demo --output-dir artifacts/alpaca-demo ... --mode demo
pip install -r requirements-finetune.txt
python src\finetune_slm.py artifacts/alpaca-demo --output-dir artifacts/finetuned/hannibal-demo-qwen2.5-3b --allow-unreviewed-demo [--preflight-only]
python review_app.py                    # revue humaine
python reports\generate_report_pdf.py   # régénérer le PDF
```
Aucune variable d'environnement : tout passe par la ligne de commande.

### ⚠️ À savoir
- Les dossiers `templates\` (le corpus de contrats) et `features\` sont **absents** ; `artifacts\` est vide.
- QLoRA demande un GPU CUDA ≥ 16 Go de VRAM — le poste actuel (PyTorch CPU, GPU 4 Go) ne suffit pas.
- Le modèle de base diffère des specs (Llama 3.1 8B / Mistral 7B recommandés, marqué AMBIGUITY).
- **Pas de git** → penser à initialiser un dépôt ou faire une sauvegarde.

### 📄 `Hannibal_V2_Technical_Report.pdf` (à la racine)
« Hannibal V2 — Contract Data Pipeline and Fine-Tuning Readiness Report » (E-Tafakna, 17 août 2026), version PDF de `ROOT\reports\Hannibal_V2_Technical_Report.md`.
Chiffres : 1 276 fichiers sources, 660 candidats contrats, 90 enregistrements runtime, 65 lignes Alpaca (36 EN, 26 AR, 3 FR), 51 tests réussis. Les données restent `demo_unreviewed` et **non éligibles à l'entraînement** (minimum 200 exemples). Le rapport décrit l'état de préparation, pas des performances validées.

---

## Accès rapide

| Je veux… | Aller à |
|---|---|
| Voir l'API du chatbot dpo_legalchat | `dpo_legalchat\backend\api.py` |
| Voir l'API Hannibal RAG | `E-Tafakna-Legal-RAG-DPO\backend\api.py` |
| Voir le notebook DPO Kaggle | `E-Tafakna-Legal-RAG-DPO\notebooks\kaggle_dpo_training.ipynb` |
| Voir le pipeline DPO complet (Hannibal) | `E-Tafakna-Legal-RAG-DPO\src\dpo\` |
| Voir le workflow Colab DPO/GRPO | `dpo_legalchat\backend\colab_train.py` |
| Voir l'app Elyssa + son prompt système | `Elysa\app.py` |
| Voir les règles de style / conformité Elyssa | `Elysa\docs\` |
| Voir le script de fine-tuning Hannibal V2 | `finetuning_hannabal\finetuning_hannabal\src\finetune_slm.py` |
| Comprendre l'architecture Hannibal V2 | `finetuning_hannabal\finetuning_hannabal\architecture.md` + `README.md` |
| Lire le rapport technique | `Hannibal_V2_Technical_Report.pdf` |
