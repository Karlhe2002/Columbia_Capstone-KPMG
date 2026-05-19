# Introduction.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a healthcare RAG (Retrieval-Augmented Generation) system that analyzes New York State Medicaid policy documents and provides citation-grounded answers to compliance queries. The system combines document parsing, semantic chunking, a Neo4j graph database, **hybrid dense + sparse retrieval fused via Reciprocal Rank Fusion (RRF)**, cross-encoder reranking, and LLM-based response generation. An LLM-driven query understanding layer extracts themes, keywords, dates, and document filters to steer retrieval.

**Tech Stack:** Python 3.9+, Neo4j 5.17+, BGE-M3 embeddings, BAAI/bge-reranker-base, OpenAI / Gemini / Ollama LLMs, Streamlit

## Commands

### Setup & Installation

```bash
# Initial setup
python -m venv .venv  # Windows
python3 -m venv .venv # macOS/Linux
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
pip install -e .

# Start Neo4j database (required for most operations)
cd docker
#(Windows)
Copy-Item .env.example .env
#(macOS/Linux)：
cp .env.example .env

# Open docker desktop
docker compose up -d
cd ..

# Test database connection
python scripts/test_neo4j.py
```

### Core Pipeline Commands

Run these in sequence for the full pipeline:

```bash
# 1. Parse documents (PDF/DOCX → JSON)
python scripts/ingestion_parse.py --config configs/ingest_parse.yaml

# 2. Chunk documents (choose one strategy)
python scripts/do_asterisk_chunking.py      # Delimiter-based (recommended)
python scripts/do_fix_size_chunking.py      # Fixed-size chunks
python scripts/do_semantic_chunking.py      # Embedding-based chunking

# 3. Ingest chunks into Neo4j graph
python scripts/ingest_graph.py --chunk_dir data/chunks/section_semantic_chunking_result

# 4. Run web interface
streamlit run frontend/app.py
```

### Testing & Development

```bash
# Test LLM query
python scripts/llm_query.py

# Evaluate retrieval accuracy
python scripts/evaluate.py --tested_result results.json --ground_truth gt.json --output eval.json

# Reset Neo4j graph (keep container)
python scripts/reset_graph.py

# Reset Neo4j completely (delete all data and container)
cd docker
docker compose down -v
```

### Database Access

- Neo4j Browser: http://localhost:7474
- Credentials: Load from `docker/.env`
- Streamlit App: http://localhost:8501

## Architecture

### Data Flow

```
Raw Documents (data/raw/)
    ↓
[doc_parsing] → Structured JSON (data/processed/)
    ↓
[chunking] → JSONL chunks (data/chunks/)
    ↓
[embedding + graph_builder] → Neo4j Graph Database
    ↓
[User Query]
    ↓
[LLMFilterExtractor: themes, keywords, dates, doc_class filters]
    ↓
        ┌─────────────────────┐
        │  Dense Vector       │ (BGE-M3 → chunk_vec)
        │  Sparse Lexical     │ (Cypher token / CONTAINS scoring)
        └──────────┬──────────┘
                   ↓
        [RRF Fusion (k = 60)]
                   ↓
        [Cross-Encoder Rerank (alpha = 0.3, z-score blend)]
                   ↓
        [LLM Generation] → Citation-grounded Answer + Follow-ups
```

### Key Components

**Document Processing:**
- `src/healthcare_rag_llm/doc_parsing/doc_parsing.py` - PDF/DOCX parser with OCR support
- Entry point: `scripts/ingestion_parse.py`

**Chunking Strategies:**
- `src/healthcare_rag_llm/chunking/fix_size_chunking.py` - Fixed character chunks
- `src/healthcare_rag_llm/chunking/pattern_chunking.py` - Delimiter-based (asterisk separators)
- `src/healthcare_rag_llm/chunking/semantic_chunking.py` - Embedding-aware boundary detection
- `src/healthcare_rag_llm/chunking/section_chunking.py` - Section-header aware splitting
- `src/healthcare_rag_llm/chunking/section_semantic_chunking.py` - Section + semantic hybrid (used by the bulk rebuild pipeline `scripts/rebuild_db.py` and `scripts/batch_parse_chunk.py`, defaults: `max_chunk_chars=1200`, `similarity_threshold=0.35`)

**Embedding System:**
- Model: BAAI/bge-m3 (1024-dim dense vectors; sparse vectors supported by the model but not stored in Neo4j)
- Class: `HealthcareEmbedding` in `src/healthcare_rag_llm/embedding/HealthcareEmbedding.py`
- Auto-detects GPU support via torch

**Graph Database:**
- Schema: `Authority → Document → Page → Chunk` (relationships: `ISSUED`, `CONTAINS`, `HAS_CHUNK` / `HAS_TABLE` / `HAS_OCR`)
- Vector index: `chunk_vec` on `Chunk.denseEmbedding` (1024-dim, cosine similarity)
- No sparse / full-text index — sparse retrieval runs as Cypher token matching against `Chunk.text` and document metadata
- Connection: `Neo4jConnector` in `src/healthcare_rag_llm/graph_builder/neo4j_loader.py`

**Query Understanding (Theme-Aware):**
- `src/healthcare_rag_llm/filters/llm_filter_extractor.py` (`LLMFilterExtractor`) — uses the LLM to extract:
  - `themes` (actions, actors, organizations, domains, objects, intent, temporal_cues)
  - `search_themes`, `semantic_keywords`, `retrieval_query`
  - Date bounds and `temporal_focus`
- Filter metadata loaded via `src/healthcare_rag_llm/filters/load_metadata.py`
- Outputs feed both retrieval branches and the reranker (no explicit "theme-aware" toggle — it's always on)

**Retrieval & Reranking:**
- Dense vector search: `src/healthcare_rag_llm/graph_builder/queries.py::query_chunks()` — embeds the raw question with BGE-M3 and queries `chunk_vec`; optional keyword boost via `_keyword_signal()`
- Sparse lexical search: `query_chunks_sparse()` in the same file — weighted Cypher `CONTAINS` over `text` (×1.0), `title` (×2.0), `doc_type` (×2.25), `authority` (×1.75); up to 16 terms from query + extractor hints, stopwords filtered
- Fusion: Reciprocal Rank Fusion (`ResponseGenerator._fuse_ranked_hits_rrf`, default `rrf_k=60`) — each hit gets `retrieval_sources`, `dense_score`, `sparse_score`, `rrf_score`
- Reranking: `src/healthcare_rag_llm/reranking/reranker.py` (`apply_rerank_to_chunks`) — BAAI/bge-reranker-base cross-encoder; when invoked from `ResponseGenerator` it z-score normalizes and blends `alpha * z(dense) + (1-alpha) * z(rerank)` with **alpha=0.3**
- Defaults in `answer_question()`: `top_k=8`, `rerank_top_k=30`

**LLM Interface:**
- Runtime providers in `LLMClient` (`src/healthcare_rag_llm/llm/llm_client.py`): `openai`, `gemini`, `ollama` (Anthropic is configured in YAML but not implemented in the client)
- Default model: `gpt-5.4-mini-2026-03-17` (see `configs/api_config.yaml` → `default_provider: openai_official`)
- Config loader: `src/healthcare_rag_llm/utils/api_config.py` (`APIConfigManager.get_default_config()`)

**Response Generation:**
- Orchestrator: `ResponseGenerator` in `src/healthcare_rag_llm/llm/response_generator.py`
  - `answer_question()` — Q&A flow with hybrid retrieval + follow-up question generation
  - `answer_compare_definitions()` — runs two filtered retrievals (`doc_classes=['policy']` vs `['provider_manual']`) and returns structured JSON `compare_sections`
- Alternative orchestrator with parsed evidence: `src/healthcare_rag_llm/llm/response_gen_json.py`
- System prompts:
  - `configs/system_prompt.txt` — Q&A citation-focused prompt
  - `configs/system_prompt_compare.txt` — Compare-definitions prompt
- Output: Answer with exact quotes and `[doc_id:page — date]` style citations

### Neo4j Graph Schema

```cypher
# Nodes
(Authority {name, abbr})
(Document {doc_id, title, url, doc_type, effective_date, category, doc_class})
(Page {uid, doc_id, page_no})
(Chunk {chunk_id, text, type, pages, denseEmbedding, doc_class})
# Chunk.type ∈ {"text", "table", "ocr"}

# Relationships
(Authority)-[:ISSUED]->(Document)
(Document)-[:CONTAINS]->(Page)
(Page)-[:HAS_CHUNK]->(Chunk)
(Page)-[:HAS_TABLE]->(Chunk)
(Page)-[:HAS_OCR]->(Chunk)

# Constraints
UNIQUE: authority_name, doc_id, page_uid, chunk_id

# Indexes
VECTOR INDEX chunk_vec: Chunk.denseEmbedding (1024 dim, cosine)
# Sparse retrieval uses Cypher token / CONTAINS scoring — no full-text or sparse vector index
```

## Configuration Files

**`configs/ingest_parse.yaml`** - Paths for raw/processed/chunked data
**`configs/api_config.yaml`** - LLM providers (`openai_official`, `gemini`, `anthropic`), models, tokens. `default_provider: openai_official`, `default_model: gpt-5.4-mini-2026-03-17`. **Note:** `anthropic` is listed in YAML but not yet implemented in `LLMClient`; `ollama` works in code but is not in this YAML (used directly in batch tests).
**`configs/system_prompt.txt`** - Q&A system prompt with citation rules
**`configs/system_prompt_compare.txt`** - Compare-definitions system prompt
**`docker/.env`** - Neo4j credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

Use `docker/.env.example` as template for `.env` file.

## Data Handling

- **NEVER commit data files**: `data/raw/` and `data/processed/` are git-ignored
- Raw documents: Place in `data/raw/Childrens Evolution of Care/State/Medicaid Updates/`
- Parsed outputs: Auto-generated in `data/processed/`
- Chunks: Auto-generated in `data/chunks/{method}_result/`
- Share data via Google Drive/SharePoint, not git

## Git Workflow

- Branch naming: `feature/<description>` or `bugfix/<description>`
- Always sync with main before pushing: `git merge main`
- Create Pull Requests into `main` branch (require 1+ reviewer)
- Delete branches after merge
- Current state: `main` now contains the dense + sparse retrieval pipeline (formerly `feature/dense-sparse-retrieval`) merged in alongside the theme-aware query understanding and Compare Definitions UI

## System Prompt & Compliance Focus

The system enforces **NYS Medicaid policy compliance** with strict citation requirements:

1. Answer ONLY using provided context chunks
2. Quote exact lines with citations: `[<doc_id:page> — <date>]`
3. Prefer recent guidance when conflicted
4. Preserve exact dates, codes, dollar figures
5. Keep decisions actionable and concise
6. Flag missing evidence explicitly

This ensures responses are grounded, traceable, and suitable for regulated healthcare environments.

## Code Conventions

- **JSONL format** for chunks: One JSON object per line with `chunk_id`, `text`, `pages`, `metadata`
- **Chunk IDs**: `{filename}_{index}`
- **Page numbers**: 1-indexed
- **Vector embeddings**: 1024-dimensional float arrays
- **Error handling**: Log and skip malformed documents (don't halt pipeline)

## Platform-Specific Notes

**Windows:**
- Use backslash `\` in config file paths (e.g., `data\raw\...`)
- Document conversion: Requires MS Word COM interface (pywin32) or LibreOffice
- OCR: Install Tesseract separately

**macOS/Linux:**
- Use forward slash `/` in paths
- Document conversion: Requires LibreOffice (`brew install libreoffice` or `apt-get`)
- OCR: Install via package manager (`brew install tesseract` or `apt-get`)

## Recent Development

**Now on `main`:**
- **Dense + sparse hybrid retrieval** with Reciprocal Rank Fusion (`query_chunks_sparse`, `_fuse_ranked_hits_rrf`, default `rrf_k=60`)
- **Theme-aware query understanding** via `LLMFilterExtractor` (themes, search_themes, semantic_keywords, retrieval_query, date bounds) feeding both retrieval branches and the reranker
- **Compare Definitions** Streamlit page (`frontend/pages/2_Compare.py`) with dual policy / provider-manual retrieval and structured JSON comparison output (`answer_compare_definitions`)
- **Cross-encoder reranking** kept on top of fusion (BAAI/bge-reranker-base, z-score blend, alpha=0.3)
- **Bulk DB rebuild pipeline** via `scripts/rebuild_db.py` and `scripts/batch_parse_chunk.py` using `section_semantic_chunking` (`max_chunk_chars=1200`, `similarity_threshold=0.35`)
- **API configuration system** (`configs/api_config.yaml` + `utils/api_config.py`) consolidated on `gpt-5.4-mini-2026-03-17` as default
- **Evaluation tooling** under `scripts/evaluate/` (chunk → ingest → test → optional LLM-as-judge) plus comparison batch testers in `src/healthcare_rag_llm/testing/`

**Comparison test result sets (`data/test_results/`):**
- `comparison_dense&sparse_*_60_query.json` — current hybrid pipeline (dense + sparse + RRF)
- `comparison_baseline_*_60q.json` — baseline (dense + rerank only, no sparse / RRF) imported from the prior `feature/llm_extract` branch
- `comparison_theme_aware_*.json` — runs that exercise theme-aware filter extraction, imported from the prior `feature/theme-aware-hybrid-retrieval` branch
- `exp_*_{fix_size|semantic|asterisk}_k5_{noRerank|rerank_a0.5}_*.json` — chunking + rerank ablations via `RAGBatchTester`

**Known gaps / follow-ups:**
- `anthropic` provider listed in `configs/api_config.yaml` but not implemented in `LLMClient`
- `ollama` works in `LLMClient` but is not registered in `api_config.yaml`
- Sparse retrieval is Cypher lexical scoring, not BM25 or stored sparse vectors — upgrading to a real full-text / BM25 index (or persisting BGE-M3 sparse vectors) is a natural next step

## Point-to-Point (P2P) Experiment Branches

> **TL;DR — the canonical, best-performing pipeline lives on `main`.** The `*-point-to-point` branches are exploratory variants of the **Compare Definitions** flow only, kept around for evaluation studies. They are not deployment targets.

### What "point-to-point" means here

Point-to-point is **not** a different retrieval engine or a separate evaluation harness. It is a reworked **Compare Definitions answering pattern** that turns the single-shot policy-vs-provider-manual answer into a **two-stage, row-aligned** LLM workflow:

1. **Retrieve** policy chunks and provider-manual chunks separately (same dual `doc_classes` filter as main).
2. **Row Planner LLM** (`generate_compare_row_plan` + `_normalize_compare_row_plan`) — decomposes the question into 2–3 operational sub-questions, then for **each row picks exactly one provider-manual chunk and one policy chunk that answer the same sub-question**. A deterministic normalizer dedupes and enforces the 1-to-1 pairing.
3. **Writer LLM** (`write_compare_from_row_plan`) — uses the approved plan plus the row generator's system prompt (`configs/system_prompt_compare_row_generator.txt` + a writer-mode `system_prompt_compare.txt`) to emit `aligned_pairs` (side-by-side rows), `similarities`, `differences`, and `caveats`.

Key files (only present on the p2p branches): `generate_compare_row_plan` / `write_compare_from_row_plan` / `_normalize_compare_row_plan` in `src/healthcare_rag_llm/llm/response_generator.py`, the new planner prompt `configs/system_prompt_compare_row_generator.txt`, and the row-aligned UI in `frontend/pages/2_Compare.py` (`format_compare_tables`, `format_compare_debug`, Stop/Cancel controls).

### How it differs from `main`

| Aspect | `main` (canonical) | `*-point-to-point` (experimental) |
|---|---|---|
| Compare LLM calls | **1** single-shot | **2** — planner + writer |
| Output schema | `policy_definition` / `provider_manual_definition` blocks | `aligned_pairs` row table + `caveats` (legacy keys left empty) |
| Compare prompt | `configs/system_prompt_compare.txt` (full compare assistant) | Same file rewritten as **writer** + new **planner** prompt |
| Compare UI | Two definition columns | Provider-Manual ↔ Policy table, expandable planner/writer debug, Stop button |
| Q&A flow | Hybrid retrieval, dense+sparse+RRF, theme-aware filters | Same as main on `dense-sparse-point-to-point`; dense-only on the other two |
| Test runners | `generate_test_result_comparison_streamlit.py` (current) | Removed / renamed (e.g. `gernerate_test_result_compare.py` on `theme-aware-point-to-point`) |

### Variants

All three p2p branches share the same planner/writer framework. They differ only in the **retrieval + query-understanding stack** plugged underneath:

- **`dense-sparse-point-to-point`** — current hybrid (dense + sparse + RRF + rerank + theme-aware filter extractor) wired into the p2p planner. Closest to what `main` does today, just with the row-aligned compare output.
- **`theme-aware-point-to-point`** — dense-only retrieval + theme-aware filter extractor; planner consumes `search_themes` only.
- **`llm-extract-point-to-point`** — dense-only baseline without `themes` / `search_themes`; the planner receives no theme hints.

Per-variant test artifacts live in `data/test_results/comparison_*_{dense_sparse,theme_aware,llm_extract}_p2p.json` (all three sets have been copied onto `dense-sparse-point-to-point` so they can be diffed side by side).

### Why we still keep them

- Side-by-side **compare-UX A/B**: row-aligned answers vs the monolithic per-source definitions on `main`.
- **Retrieval ablation under a fixed answering pattern**: dense+sparse vs dense+themes vs dense+llm-extract, with everything else held equal.
- Engineering experiments — async generation cancel, planner/writer debug transparency, chunk-edge text sanitization — that may later be cherry-picked into `main`.

### Bottom line

Production work, the latest documentation, the maintained test runners, and the most up-to-date retrieval defaults all live on **`main`**. Treat the `*-point-to-point` branches as **archived experiments**; consult them when you need their compare artifacts or want to revisit the row-aligned UX, but do not deploy from them.
