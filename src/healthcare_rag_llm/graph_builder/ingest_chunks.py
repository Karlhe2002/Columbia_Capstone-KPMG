# src/healthcare_rag_llm/graph_builder/ingest_chunks.py
import json
from datetime import datetime, date
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, List, Optional

from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import get_embedding_singleton

def _parse_effective_date(value: Optional[str]) -> Optional[date]:
    """
    Convert various date string formats into a datetime.date object so Neo4j
    receives a proper DATE value.
    """
    if not value:
        return None

    raw = value.strip()
    if not raw:
        return None

    # Try ISO first (covers most of our rows).
    try:
        return datetime.fromisoformat(raw).date()
    except ValueError:
        pass

    # Fall back to a handful of common US formats.
    candidates = [
        "%m/%d/%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    return None


def _write_batch(session, batch_rows: List[Dict[str, Any]]):
    """
    Batch write a set of records into Neo4j.

    Structure:
      (:Authority)-[:ISSUED]->(:Document {category, doc_class})
        └─[:CONTAINS]->(:Page)
             ├─[:HAS_CHUNK]->(:Chunk {type:'text'})
             ├─[:HAS_TABLE]->(:Chunk {type:'table'})
             └─[:HAS_OCR]->(:Chunk {type:'ocr'})
    """
    session.run("""
    UNWIND $batch AS row

    // --- Authority + Document ---
    MERGE (a:Authority {name: row.authority})
      ON CREATE SET a.abbr = row.authority_abbr

    MERGE (d:Document {doc_id: row.doc_id})
      ON CREATE SET d.title = row.title,
                    d.url = row.url,
                    d.doc_type = row.doc_type,
                    d.effective_date = row.effective_date,
                    d.category = row.category,
                    d.doc_class = row.doc_class
      // Keep category up to date even if the Document already exists
      SET d.category = coalesce(row.category, d.category),
          d.doc_class = coalesce(row.doc_class, d.doc_class)

    MERGE (a)-[:ISSUED]->(d)

    // --- Pages ---
    WITH row, d
    UNWIND row.pages AS pno
      MERGE (p:Page {uid: row.doc_id + ':' + toString(pno)})
        ON CREATE SET p.doc_id = row.doc_id, p.page_no = pno
      MERGE (d)-[:CONTAINS]->(p)

      MERGE (c:Chunk {chunk_id: row.chunk_id})
        ON CREATE SET c.text = row.text,
                      c.type = row.chunk_type,
                      c.pages = row.pages,
                      c.denseEmbedding = row.denseEmbedding,
                      c.doc_class = row.doc_class

                
    // --- NEW (add source labels) ---
                
      // Source-specific labels for separate vector indexes
      //FOREACH (_ IN CASE WHEN row.doc_class = 'policy' THEN [1] ELSE [] END |
      //  SET c:PolicyChunk
      //)
      //FOREACH (_ IN CASE WHEN row.doc_class = 'provider_manual' THEN [1] ELSE [] END |
      //  SET c:ManualChunk
      //)




      // --- Relationship by chunk type ---
      FOREACH (_ IN CASE WHEN row.chunk_type = 'text' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_CHUNK]->(c)
      )
      FOREACH (_ IN CASE WHEN row.chunk_type = 'table' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_TABLE]->(c)
      )
      FOREACH (_ IN CASE WHEN row.chunk_type = 'ocr' THEN [1] ELSE [] END |
        MERGE (p)-[:HAS_OCR]->(c)
      )
    """, {"batch": batch_rows})
# Add doc_class info into the previous function

def ingest_chunks(
    jsonl_path: str,
    doc_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    batch_size: int = 30,
    embed_batch_size: int = 8
):
    """
    Ingest a single JSONL chunk file into Neo4j.

    Graph: Authority -> Document (category property) -> Page -> Chunk
    - Links Authority, Document, Page, Chunk
    - Unifies all chunk types (text/table/ocr)
    - Every chunk gets a denseEmbedding (for vector search)
    - Batch embedding (embed_batch_size texts at once) for ~5x speedup
    - Batch Neo4j write for stability/performance
    """
    embedder = get_embedding_singleton()
    connector = Neo4jConnector()

    chunk_file = Path(jsonl_path)
    if not chunk_file.exists():
        raise FileNotFoundError(f"❌ File not found: {jsonl_path}")

    # Read all records
    records = []
    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            doc_id_raw = (record.get("doc_id") or "").strip()
            if doc_id_raw:
                records.append(record)

    if not records:
        print(f"[SKIP] {chunk_file.name} (no records)")
        return

    # Batch encode embeddings (embed_batch_size at a time, M1 friendly)
    all_texts = [(r.get("text") or "").strip() for r in records]
    all_vecs = []
    for start in range(0, len(all_texts), embed_batch_size):
        batch_texts = all_texts[start:start + embed_batch_size]
        enc = embedder.encode(
            batch_texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        for vec in enc["dense_vecs"]:
            all_vecs.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))

    # Write to Neo4j
    batch: List[Dict[str, Any]] = []
    chunk_count = 0

    with connector.driver.session() as session:
        for i, record in enumerate(tqdm(records, desc=f"Ingesting {chunk_file.name}")):
            doc_id = (record.get("doc_id") or "").strip().lower()
            chunk_id = record.get("chunk_id")
            pages = record.get("pages", []) or [1]
            text = all_texts[i]

            if isinstance(chunk_id, str) and "::table" in chunk_id:
                chunk_type = "table"
            elif isinstance(chunk_id, str) and "::ocr" in chunk_id:
                chunk_type = "ocr"
            else:
                chunk_type = "text"

            category = (record.get("category") or "").strip()
            if not category:
                dm = (doc_metadata or {}).get(doc_id, {})
                category = (dm.get("category") or "unknown").strip() or "unknown"

            meta = (doc_metadata or {}).get(doc_id, {})
            batch.append({
                "doc_id": doc_id,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "doc_type": meta.get("doc_type", "PDF"),
                "effective_date": _parse_effective_date(
                    record.get("effective_date") or meta.get("effective_date")
                ),
                "authority": meta.get("authority", "Unknown"),
                "authority_abbr": meta.get("authority_abbr", ""),
                "category": category,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "text": text,
                "denseEmbedding": all_vecs[i],
                "pages": pages,
                "doc_class": (meta.get("doc_class") or "").strip(),
            })
            chunk_count += 1

            if len(batch) >= batch_size:
                _write_batch(session, batch)
                batch = []

        if batch:
            _write_batch(session, batch)

    connector.close()
    print(f"✅ Ingested {chunk_count} chunks (Document.category set) from {chunk_file.name}")
