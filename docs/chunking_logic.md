# Chunking Logic Summary

The project implements **3 chunking strategies**, located in `src/healthcare_rag_llm/chunking/`.

---

## 1. Fixed-Size Chunking

**Source**: `src/healthcare_rag_llm/chunking/fix_size_chunking.py`
**Script**: `scripts/do_fix_size_chunking.py`

Splits text by a fixed character count with optional overlap.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chunk_chars` | 5000 | Maximum characters per chunk |
| `overlap` | 0 | Number of overlapping characters between consecutive chunks |
| `glob_pattern` | `*.json` | Input file matching pattern |

**Process**:
1. Load processed JSON files containing `full_text` and `pages` metadata
2. Merge OCR fallback text into corresponding pages
3. Split `full_text` into chunks of `max_chunk_chars`, moving the cursor back by `overlap`
4. Extract table chunks and OCR image chunks separately

**Characteristics**: Simplest and fastest strategy. Suitable for generic documents as a baseline.

---

## 2. Pattern-Based / Asterisk Chunking

**Source**: `src/healthcare_rag_llm/chunking/pattern_chunking.py`
**Script**: `scripts/do_asterisk_chunking.py`

Splits documents at positions matching a regex separator pattern.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `separator_char` | `*` | Character used as separator |
| `min_repeats` | 10 | Minimum consecutive repeats required to trigger a split |
| `max_chunk_chars` | 1200 | Maximum chunk size for oversized segments (secondary split) |

**Process**:
1. Load JSON files and merge OCR fallback
2. Build regex pattern: `{separator_char}{min_repeats,}` (no line-start anchor)
3. Find all separator runs using `regex.finditer()`
4. Extract segments between separator runs as chunks
5. If a segment exceeds `max_chunk_chars`, split it by size (no overlap)
6. Build page-span mapping to associate chunks with page numbers
7. Add table and OCR chunks, sorted by page order (within same page: text=0, table=1, ocr=2)

**Characteristics**: Structure-aware. Separator runs are removed from output. Best suited for documents with clear delimiters (e.g., Medicaid Updates).

---

## 3. Semantic Chunking

**Source**: `src/healthcare_rag_llm/chunking/semantic_chunking.py`
**Script**: `scripts/do_semantic_chunking.py`

Intelligently merges text units based on cosine similarity of sentence-transformer embeddings.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `unit` | `sentence` | Base segmentation unit (`sentence` or `paragraph`) |
| `similarity_threshold` | 0.80 | Cosine similarity threshold for merging |
| `hysteresis` | 0.02 | Hysteresis band to reduce jitter near the threshold |
| `max_chunk_chars` | 5000 | Hard limit on chunk character count |

**Process**:
1. Load JSON files and merge OCR fallback
2. Tokenize `full_text` into units (sentences via NLTK PunktSentenceTokenizer; paragraphs by splitting on `\n\n`)
3. Generate embeddings for all units using SentenceTransformer
4. Hysteresis-based merging algorithm:
   - Compute cosine similarity between the current chunk's average embedding and the next unit
   - Similarity > `threshold + hysteresis/2` → **merge**
   - Similarity < `threshold - hysteresis/2` → **split**
   - In between → **maintain current state** (reduces oscillation)
5. Force split if merging would exceed `max_chunk_chars`
6. Add table and OCR chunks

**Characteristics**: Highest quality but slowest. Best suited for specialized content such as healthcare documents.

---

## 4. Section-Based Chunking

**Source**: `src/healthcare_rag_llm/chunking/section_chunking.py`
**Script**: `scripts/do_section_chunking.py`

Splits a policy-guidelines TXT file at subsection boundaries identified from its Table of Contents.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `txt_path` | — | Path to the input TXT file |
| `output_dir` | — | Output directory for JSONL chunks |
| `doc_id` | `<stem>.pdf` | Document identifier for JSONL records |
| `category` | `pharmacy` | Category tag for each chunk |
| `max_chunk_chars` | 0 | Maximum chunk size; 0 = no limit (each section is one chunk) |

**Process**:
1. Read the TXT file and parse the Table of Contents to extract ordered section/subsection titles
2. Locate the body text (after ToC pages)
3. Clean page headers (`Policy Guidelines / NYRx`) and footers (`2025-3 October 2025 <page>`)
4. Split body text at each ToC title boundary — each subsection becomes one chunk
5. Estimate page numbers from footer markers in the original text
6. If `max_chunk_chars` > 0, further split oversized sections by character count

**Characteristics**: Structure-aware via ToC parsing. Best suited for policy manuals with a clear Table of Contents (e.g., Pharmacy Policy Guidelines).

---

## 5. Section + Semantic Chunking (Hybrid)

**Source**: `src/healthcare_rag_llm/chunking/section_semantic_chunking.py`
**Script**: `scripts/do_section_semantic_chunking.py`

Two-level hybrid: first splits at ToC section boundaries (hard splits), then applies semantic chunking within each section to further split large sections.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `txt_path` | — | Path to the input TXT file |
| `output_dir` | — | Output directory for JSONL chunks |
| `doc_id` | `<stem>.pdf` | Document identifier for JSONL records |
| `category` | `pharmacy` | Category tag for each chunk |
| `max_chunk_chars` | 1200 | Maximum chunk size |
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `unit` | `sentence` | Base segmentation unit (`sentence` or `paragraph`) |
| `similarity_threshold` | 0.35 | Cosine similarity threshold for merging |
| `hysteresis` | 0.02 | Hysteresis band around threshold |

**Process**:
1. **Level 1 (Section split)**: Parse ToC, clean page markers, split at subsection boundaries — same as Section-Based Chunking
2. **Level 2 (Semantic split)**: For each section:
   - If section ≤ `max_chunk_chars` → keep as a single chunk (no further splitting)
   - If section > `max_chunk_chars` → tokenize into sentences, generate embeddings, apply hysteresis-based merging to produce semantically coherent sub-chunks

**Characteristics**: Combines structural awareness (never merges across sections) with semantic coherence (splits large sections at natural topic boundaries). Produces ~214 chunks vs 63 from pure section chunking. Default `unit=sentence` with `threshold=0.35` because PDF-extracted text uses single `\n` line wrapping (no `\n\n` paragraph separators).

---

## Shared Features Across All Strategies

### Chunk Types

Each strategy produces 3 types of chunks:

| chunk_type | Description |
|------------|-------------|
| `text` | Main document text content |
| `table` | Tables extracted from metadata (converted to RFC 4180 CSV format) |
| `ocr_image` | OCR fallback content (includes bounding box information) |

### Unified OCR Merge Logic

All strategies apply the same OCR merge before chunking:
- If a page's text is < 50 characters and OCR blocks exist → **replace** with OCR text
- If a page has normal text and OCR blocks exist → **append** as `[OCR Supplement]`

### Output Format

All strategies output JSONL files. Each record follows this schema:

```json
{
  "doc_id": "<file_name>",
  "chunk_id": "<file_name>::0007",
  "char_start": 0,
  "char_end": 5000,
  "pages": [1, 2],
  "text": "<chunk content>",
  "chunk_type": "text | table | ocr_image",
  "category": "medicaid_update"        // only in pattern-based and semantic chunking
}
```

---

## Usage in Production

In `scripts/rebuild_db.py`, different document types use different strategies:

| Document Type | Strategy | Key Parameters |
|---------------|----------|----------------|
| Medicaid Updates | Pattern-Based (asterisk) | `separator_char="*"`, `min_repeats=10`, `max_chunk_chars=1200` |
| Children Waiver | Semantic | `threshold=0.55`, `unit="paragraph"`, `max_chunk_chars=1200` |
| Pharmacy Policy Guidelines | Section-Based | `category="pharmacy"` |

### Commands

```bash
# Fixed-size chunking
python scripts/do_fix_size_chunking.py --max-chars 1200 --overlap 150

# Pattern-based chunking
python scripts/do_asterisk_chunking.py --sep "*" --min-repeats 10 --max-chars 1200

# Semantic chunking
python scripts/do_semantic_chunking.py --unit paragraph --threshold 0.55 --max-chars 1200

# Section-based chunking (Pharmacy)
python scripts/do_section_chunking.py --input data/processed/pharmacy/Pharmacy_Policy_Guidelines.txt --output data/chunks/section_chunking_result

# Section + Semantic hybrid chunking (Pharmacy)
python scripts/do_section_semantic_chunking.py --input data/processed/pharmacy/Pharmacy_Policy_Guidelines.txt --output data/chunks/section_semantic_chunking_result

# Full pipeline (orchestrates all steps)
python scripts/rebuild_db.py
```

---

## Strategy Comparison

| Strategy | Split Basis | Speed | Quality | Use Case |
|----------|-------------|-------|---------|----------|
| **Fixed-Size** | Character count | Fastest | Basic | Generic documents, baseline |
| **Pattern-Based** | Regex separators | Fast | Structure-aware | Documents with clear delimiters |
| **Semantic** | Embedding similarity | Slow | Highest | Specialized healthcare content |
| **Section-Based** | ToC title boundaries | Fast | Structure-aware | Policy manuals with clear ToC |
| **Section+Semantic** | ToC + embedding similarity | Medium | Structure + semantic | Large policy manuals needing finer splits |

---

## Chunking Quality Evaluation

Chunking quality is assessed through a **two-layer evaluation system** that measures both retrieval accuracy and downstream answer quality.

### Layer 1: Traditional Retrieval Evaluation

**Source**: `src/healthcare_rag_llm/evaluate/evaluate.py`

Performs subset matching against ground truth to measure whether the correct documents and pages were retrieved.

| Metric | Calculation | What It Measures |
|--------|-------------|------------------|
| **Document-level accuracy** | retrieved docs ⊇ ground truth docs | Whether the correct documents were retrieved |
| **Page-level accuracy** | retrieved pages ⊇ ground truth pages | Whether the correct pages were retrieved |

Fast and deterministic. Directly reflects the retrieval quality of chunking + embedding.

### Layer 2: LLM-as-Judge Evaluation

**Source**: `src/healthcare_rag_llm/evaluate/llm_evaluate.py`

Uses an LLM as a judge to score generated answers (0–1 scale). Currently uses **3 core metrics** (reduced from an initial 5; Citation Quality and Completeness were removed as redundant):

| Metric | Weight (w/ GT) | Weight (w/o GT) | What It Measures |
|--------|----------------|------------------|------------------|
| **Faithfulness** | 30% | 60% | Are all claims in the answer supported by the retrieved chunks? |
| **Answer Relevance** | 30% | 40% | Does the answer directly and completely address the query? |
| **Correctness** | 40% | N/A | Does the answer semantically match the ground truth? (requires GT) |

An **Overall Score** is computed as the weighted average of the above metrics.

### Evaluation Pipeline

**Source**: `scripts/evaluate/evaluate_pipeline.py`

The pipeline automatically runs batch experiments across different chunking and retrieval configurations:

```
Chunking (multiple configs) → Neo4j ingestion → Retrieval + Generation → Dual-layer evaluation → CSV summary
```

**Chunking configurations tested** (historically; not all active at once):
```python
ChunkingConfig("semantic",  {"threshold": 0.80, "max_chars": 2000})
ChunkingConfig("fix_size",  {"max_chars": 1200, "overlap": 150})
ChunkingConfig("fix_size",  {"max_chars": 2000, "overlap": 250})
ChunkingConfig("asterisk",  {})
```

**Retrieval configurations tested** (historically; not all active at once):
```python
RetrievalConfig(top_k=5, rerank=False, alpha=0.0)   # Dense search only
RetrievalConfig(top_k=5, rerank=True,  alpha=0.3)   # Dense 70%, Rerank 30%
RetrievalConfig(top_k=5, rerank=True,  alpha=0.5)   # Dense 50%, Rerank 50%
RetrievalConfig(top_k=5, rerank=True,  alpha=0.7)   # Dense 30%, Rerank 70%
```

> Note: The pipeline runs different subsets of these configs per batch. Check `evaluate_pipeline.py` `main()` for the currently active configurations.

Results are aggregated into `data/evaluation_results/batch_evaluation_results.csv` for cross-experiment comparison.

### Ground Truth

**File**: `data/testing_queries/testing_query.json`

Contains test queries, their correct document/page references, and reference answers used by both evaluation layers.

### Interpreting Results

| Retrieval Accuracy | LLM Score | Interpretation |
|--------------------|-----------|----------------|
| High | High | Optimal configuration |
| High | Low | Good chunks, but generation has issues |
| Low | High | Poor chunks, LLM is compensating |
| Low | Low | Chunking strategy needs adjustment |

### Experiment Results (18 Complete Experiments)

Results are stored in `data/evaluation_results/`, `data/llm_eval_results/`, and `data/test_results/`.

| Exp | Chunking | Rerank | Alpha | Doc Acc | Page Acc | Faithfulness | Relevance | Overall LLM |
|-----|----------|--------|-------|---------|----------|--------------|-----------|-------------|
| 001 | fix_size | No | - | 0.455 | 0.455 | 1.000 | 1.000 | 0.232 |
| 001 | fix_size | Yes | 0.5 | 0.636 | 0.545 | 0.964 | 0.874 | 0.889 |
| 001 | semantic | No | - | 0.455 | 0.364 | 1.000 | 0.752 | 0.883 |
| 001 | semantic | Yes | 0.5 | 0.545 | 0.364 | 0.989 | 0.775 | 0.904 |
| 002 | fix_size | Yes | 0.5 | 0.545 | 0.545 | 1.000 | 0.854 | 0.858 |
| 002 | semantic | Yes | 0.5 | 0.455 | 0.364 | 0.982 | 0.810 | 0.913 |
| 003 | asterisk | Yes | 0.5 | 0.545 | 0.455 | 0.959 | 0.741 | 0.790 |
| 003 | fix_size | Yes | 0.5 | 0.727 | 0.545 | 0.984 | 0.860 | 0.934 |
| 003 | semantic | No | - | 0.455 | 0.364 | 0.991 | 0.775 | 0.880 |
| 004 | fix_size | Yes | 0.5 | 0.636 | 0.545 | 1.000 | 0.775 | 0.910 |
| 004 | semantic | Yes | 0.5 | 0.455 | 0.364 | 0.975 | 0.830 | 0.869 |
| 005 | asterisk | Yes | 0.5 | 0.636 | 0.545 | 0.995 | 0.855 | 0.939 |
| 005 | fix_size | No | - | 0.364 | 0.364 | 0.998 | 0.727 | 0.795 |
| 006 | fix_size | Yes | 0.5 | 0.727 | 0.545 | 1.000 | 0.838 | 0.843 |
| 007 | fix_size | No | - | 0.636 | 0.545 | 0.980 | 0.782 | 0.837 |
| 008 | fix_size | Yes | 0.5 | 0.636 | 0.545 | 0.995 | 0.823 | 0.867 |
| 009 | asterisk | No | - | 0.545 | 0.455 | 1.000 | 0.784 | 0.871 |
| 010 | asterisk | Yes | 0.5 | 0.636 | 0.545 | 0.980 | 0.826 | 0.859 |

### Key Findings

**By chunking method (averages)**:

| Method | Avg Doc Acc | Avg Overall LLM | Notes |
|--------|-------------|-----------------|-------|
| Semantic | 0.473 | 0.890 | Highest LLM quality, lowest retrieval accuracy |
| Asterisk | 0.591 | 0.865 | Balanced performance |
| Fix-size | 0.596 | 0.796 | Best retrieval accuracy, variable LLM scores |

**Impact of reranking**:

| Setting | Avg Doc Acc | Avg Overall LLM |
|---------|-------------|-----------------|
| With rerank | 0.598 | 0.881 |
| Without rerank | 0.485 | 0.750 |

Reranking consistently improves both retrieval accuracy (+23%) and answer quality (+13%).

**Top configurations**:
1. **exp_005 asterisk + rerank a0.5** → Overall LLM 0.939 (highest)
2. **exp_003 fix_size + rerank a0.5** → Doc Acc 0.727 + Overall 0.934

**Bottleneck**: Faithfulness is consistently near 1.0 (not a concern). **Answer Relevance (~0.816 avg)** is the primary area for improvement.