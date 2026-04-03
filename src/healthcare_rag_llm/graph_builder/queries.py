from datetime import datetime, date
import re
from typing import Optional, Union, Iterable, List, Dict, Any

import pandas as pd  # kept because it's used in the __main__ block

from healthcare_rag_llm.graph_builder.neo4j_loader import Neo4jConnector
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding

SPARSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "should", "the", "to",
    "today", "what", "when", "where", "which", "who", "why", "with",
}


def _normalize_keywords(keywords, max_items: int = 12) -> List[str]:
    if not keywords:
        return []
    out = []
    seen = set()
    for raw in keywords:
        cleaned = " ".join(str(raw or "").strip().lower().split())
        if len(cleaned) < 2 or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
        if len(out) >= max_items:
            break
    return out


def _keyword_signal(result: Dict[str, Any], keywords: List[str]) -> Dict[str, float]:
    text = " ".join(str(result.get("text", "")).lower().split())
    title = " ".join(str(result.get("title", "")).lower().split())
    doc_type = " ".join(str(result.get("doc_type", "")).lower().split())
    authority = " ".join(str(result.get("authority", "")).lower().split())

    keyword_hits = 0
    keyword_score = 0.0

    for keyword in keywords:
        if " " in keyword:
            text_hit = keyword in text
            title_hit = keyword in title
            doc_type_hit = keyword in doc_type
            authority_hit = keyword in authority
        else:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            text_hit = bool(re.search(pattern, text))
            title_hit = bool(re.search(pattern, title))
            doc_type_hit = bool(re.search(pattern, doc_type))
            authority_hit = bool(re.search(pattern, authority))

        if not any((text_hit, title_hit, doc_type_hit, authority_hit)):
            continue

        keyword_hits += 1
        if text_hit:
            keyword_score += 0.08
        if title_hit:
            keyword_score += 0.12
        if doc_type_hit:
            keyword_score += 0.15
        if authority_hit:
            keyword_score += 0.10

    return {
        "keyword_hits": float(keyword_hits),
        "keyword_score": float(keyword_score),
    }


def _build_sparse_terms(
    query_text: str,
    keywords=None,
    max_items: int = 16,
) -> List[str]:
    """
    Build a lexical term list for sparse retrieval from both the raw query and
    any LLM-derived keyword/theme hints.
    """
    terms = _normalize_keywords(keywords, max_items=max_items)
    raw_query = " ".join(str(query_text or "").strip().lower().split())
    if not raw_query:
        return terms

    for token in re.findall(r"[a-z0-9][a-z0-9/_-]*", raw_query):
        if token in SPARSE_STOPWORDS or len(token) < 3:
            continue
        terms.append(token)
        if len(terms) >= max_items:
            break

    deduped = []
    seen = set()
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
        if len(deduped) >= max_items:
            break
    return deduped


def _chunk_relationship_types(include_table: bool, include_ocr: bool) -> List[str]:
    rel_types = ["HAS_CHUNK"]
    if include_table:
        rel_types.append("HAS_TABLE")
    if include_ocr:
        rel_types.append("HAS_OCR")
    return rel_types


def _normalize_date_filter(
    value: Optional[Union[str, datetime, date]]
) -> Optional[date]:
    """
    Normalize a date-like value into a date object (or None).

    Accepted inputs:
      - None          -> None
      - date          -> same date (unchanged)
      - datetime      -> datetime.date()
      - str           -> parsed as ISO format or 'YYYY-MM-DD'
    """
    if value is None:
        return None

    # Already a date (but not a datetime subclass)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value

    # Datetime -> date
    if isinstance(value, datetime):
        return value.date()

    # String formats
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        # First try Python's ISO parser (covers many variants).
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            pass

        # Fallback to a strict YYYY-MM-DD format.
        try:
            return datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid date format: {raw}")

    # Any other type is treated as "no usable date"
    return None


def query_chunks(
    query_embedding,
    top_k: int = 5,
    search_k: Optional[int] = None,
    include_table: bool = True,
    include_ocr: bool = True,
    authority_names: Optional[Iterable[str]] = None,
    doc_titles: Optional[Iterable[str]] = None,
    doc_types: Optional[Iterable[str]] = None,
    doc_classes: Optional[Iterable[str]] = None,
    min_effective_date: Optional[Union[str, datetime, date]] = None,
    max_effective_date: Optional[Union[str, datetime, date]] = None,
    keywords=None,
) -> List[Dict[str, Any]]:
    """
    Perform a vector search over Chunk.denseEmbedding via the 'chunk_vec' index,
    then traverse to Page, Document, and Authority.

    Returns a list of result dicts, each with:
      - chunk_id, text, chunk_type, doc_id, title, url, doc_type,
        effective_date, authority, pages, score
    """
    connector = Neo4jConnector()
    try:
        with connector.driver.session() as session:
            authority_names_list = list(authority_names) if authority_names is not None else None
            doc_titles_list = list(doc_titles) if doc_titles is not None else None
            doc_types_list = list(doc_types) if doc_types is not None else None
            doc_classes_list = list(doc_classes) if doc_classes is not None else None
            has_doc_class_filter = bool(doc_classes_list)
            chunk_rel_types = _chunk_relationship_types(include_table, include_ocr)

            # Determine which chunk types to consider
            if include_table and include_ocr:
                type_filter = "WHERE c.type IN ['text', 'table', 'ocr']"
            elif include_table:
                type_filter = "WHERE c.type IN ['text', 'table']"
            elif include_ocr:
                type_filter = "WHERE c.type IN ['text', 'ocr']"
            else:
                type_filter = "WHERE c.type = 'text'"

            # Build filter conditions for documents/authorities
            # Use ($param IS NULL OR condition) pattern so None values don't filter
            doc_filter_conditions = []
            doc_filter_conditions.append("($authority_names IS NULL OR a.name IN $authority_names)")
            doc_filter_conditions.append("($doc_titles IS NULL OR d.title IN $doc_titles)")
            doc_filter_conditions.append("($doc_types IS NULL OR d.doc_type IN $doc_types)")
            doc_filter_conditions.append("($doc_classes IS NULL OR d.doc_class IN $doc_classes)")
            doc_filter_conditions.append("($min_effective_date IS NULL OR d.effective_date >= $min_effective_date)")
            doc_filter_conditions.append("($max_effective_date IS NULL OR d.effective_date <= $max_effective_date)")
            
            doc_filter_where = "WHERE " + " AND ".join(doc_filter_conditions)

            if search_k is None:
                if has_doc_class_filter:
                    # For source-constrained retrieval, a moderate candidate pool is enough.
                    search_k = max(top_k * 5, 100)
                else:
                    # Preserve original QA behavior: rank globally first, then filter.
                    search_k = 999999

            if has_doc_class_filter:
                # Filter-first retrieval for source-specific comparison use cases:
                # restrict by doc_class/doc filters first, then score within filtered chunks.
                type_filter_condition = type_filter.replace("WHERE ", "", 1)
                doc_filter_condition = doc_filter_where.replace("WHERE ", "", 1)
                cypher = f"""
                MATCH (a:Authority)-[:ISSUED]->(d:Document)-[:CONTAINS]->(p:Page)-[rel]->(c:Chunk)
                WHERE type(rel) IN $chunk_rel_types AND {type_filter_condition} AND {doc_filter_condition}
                WITH DISTINCT c, d, a
                WITH c, d, a,
                     reduce(dot = 0.0, i IN range(0, size(c.denseEmbedding) - 1) |
                         dot + (c.denseEmbedding[i] * $query_embedding[i])) AS dot,
                     sqrt(reduce(norm_c = 0.0, x IN c.denseEmbedding | norm_c + (x * x))) AS norm_c,
                     sqrt(reduce(norm_q = 0.0, x IN $query_embedding | norm_q + (x * x))) AS norm_q
                WITH c, d, a,
                     CASE
                         WHEN norm_c = 0.0 OR norm_q = 0.0 THEN 0.0
                         ELSE dot / (norm_c * norm_q)
                     END AS score
                ORDER BY score DESC
                LIMIT $search_k
                RETURN
                    c.chunk_id        AS chunk_id,
                    c.text            AS text,
                    c.type            AS chunk_type,
                    d.doc_id          AS doc_id,
                    d.title           AS title,
                    d.url             AS url,
                    d.doc_type        AS doc_type,
                    d.category        AS category,
                    d.doc_class       AS doc_class,
                    d.effective_date  AS effective_date,
                    a.name            AS authority,
                    c.pages           AS pages,
                    score
                ORDER BY score DESC
                LIMIT $top_k
                """
            else:
                # Rank ALL chunks first, then filter, then return top k.
                # Query all chunks to ensure we rank all relevant chunks.
                # Then filter by document/authority criteria, then return top k.
                cypher = f"""
                // Step 1: Rank ALL chunks by vector similarity (query large number to get all relevant chunks)
                CALL db.index.vector.queryNodes('chunk_vec', $search_k, $query_embedding)
                YIELD node, score
                // Step 2: Match chunks and apply type filter
                MATCH (c:Chunk {{chunk_id: node.chunk_id}})
                {type_filter}
                // Step 3: Traverse to documents and authorities to get metadata for filtering
                MATCH (p:Page)-[rel]->(c)
                WHERE type(rel) IN $chunk_rel_types
                MATCH (p)<-[:CONTAINS]-(d:Document)<-[:ISSUED]-(a:Authority)
                // Step 4: Filter by document/authority criteria (when filters are None, conditions evaluate to True)
                {doc_filter_where}
                // Step 5: Return top k from filtered and ranked results (highest scores first)
                RETURN DISTINCT
                    c.chunk_id        AS chunk_id,
                    c.text            AS text,
                    c.type            AS chunk_type,
                    d.doc_id          AS doc_id,
                    d.title           AS title,
                    d.url             AS url,
                    d.doc_type        AS doc_type,
                    d.category        AS category,
                    d.doc_class       AS doc_class,
                    d.effective_date  AS effective_date,
                    a.name            AS authority,
                    c.pages           AS pages,
                    score
                ORDER BY score DESC
                LIMIT $top_k
                """

            # Always include all filter parameters (even if None) so Cypher NULL checks work
            params = {
                "query_embedding": query_embedding,
                "search_k": search_k,
                "top_k": top_k,
                "authority_names": authority_names_list,
                "doc_titles": doc_titles_list,
                "doc_types": doc_types_list,
                "doc_classes": doc_classes_list,
                "chunk_rel_types": chunk_rel_types,
                "min_effective_date": _normalize_date_filter(min_effective_date),
                "max_effective_date": _normalize_date_filter(max_effective_date),
            }

            result = session.run(cypher, params)
            data = result.data()
    finally:
        connector.close()

    normalized_keywords = _normalize_keywords(keywords)
    if normalized_keywords:
        for row in data:
            vector_score = float(row.get("score", 0.0) or 0.0)
            signal = _keyword_signal(row, normalized_keywords)
            row["vector_score"] = vector_score
            row["keyword_hits"] = int(signal["keyword_hits"])
            row["keyword_score"] = signal["keyword_score"]
            row["score"] = vector_score + signal["keyword_score"]
        data.sort(
            key=lambda row: (
                float(row.get("score", 0.0) or 0.0),
                int(row.get("keyword_hits", 0) or 0),
            ),
            reverse=True,
        )

    return data[:top_k]


def query_chunks_sparse(
    query_text: str,
    top_k: int = 5,
    search_k: Optional[int] = None,
    include_table: bool = True,
    include_ocr: bool = True,
    authority_names: Optional[Iterable[str]] = None,
    doc_titles: Optional[Iterable[str]] = None,
    doc_types: Optional[Iterable[str]] = None,
    doc_classes: Optional[Iterable[str]] = None,
    min_effective_date: Optional[Union[str, datetime, date]] = None,
    max_effective_date: Optional[Union[str, datetime, date]] = None,
    keywords=None,
) -> List[Dict[str, Any]]:
    """
    Perform an independent sparse retrieval pass using lexical term matches over
    chunk text and related document metadata.

    This is intentionally separate from vector retrieval so the caller can later
    fuse dense and sparse results in a controlled way.
    """
    sparse_terms = _build_sparse_terms(query_text, keywords=keywords)
    if not sparse_terms:
        return []

    connector = Neo4jConnector()
    try:
        with connector.driver.session() as session:
            authority_names_list = list(authority_names) if authority_names is not None else None
            doc_titles_list = list(doc_titles) if doc_titles is not None else None
            doc_types_list = list(doc_types) if doc_types is not None else None
            doc_classes_list = list(doc_classes) if doc_classes is not None else None
            chunk_rel_types = _chunk_relationship_types(include_table, include_ocr)

            if include_table and include_ocr:
                type_filter = "WHERE c.type IN ['text', 'table', 'ocr']"
            elif include_table:
                type_filter = "WHERE c.type IN ['text', 'table']"
            elif include_ocr:
                type_filter = "WHERE c.type IN ['text', 'ocr']"
            else:
                type_filter = "WHERE c.type = 'text'"

            doc_filter_conditions = []
            doc_filter_conditions.append("($authority_names IS NULL OR a.name IN $authority_names)")
            doc_filter_conditions.append("($doc_titles IS NULL OR d.title IN $doc_titles)")
            doc_filter_conditions.append("($doc_types IS NULL OR d.doc_type IN $doc_types)")
            doc_filter_conditions.append("($doc_classes IS NULL OR d.doc_class IN $doc_classes)")
            doc_filter_conditions.append("($min_effective_date IS NULL OR d.effective_date >= $min_effective_date)")
            doc_filter_conditions.append("($max_effective_date IS NULL OR d.effective_date <= $max_effective_date)")

            type_filter_condition = type_filter.replace("WHERE ", "", 1)
            doc_filter_condition = " AND ".join(doc_filter_conditions)
            if search_k is None:
                search_k = max(top_k * 10, 100)

            cypher = f"""
            MATCH (a:Authority)-[:ISSUED]->(d:Document)-[:CONTAINS]->(p:Page)-[rel]->(c:Chunk)
            WHERE type(rel) IN $chunk_rel_types AND {type_filter_condition} AND {doc_filter_condition}
            WITH DISTINCT c, d, a,
                 toLower(coalesce(c.text, '')) AS text_lc,
                 toLower(coalesce(d.title, '')) AS title_lc,
                 toLower(coalesce(d.doc_type, '')) AS doc_type_lc,
                 toLower(coalesce(a.name, '')) AS authority_lc
            WITH c, d, a, text_lc, title_lc, doc_type_lc, authority_lc,
                 [term IN $terms
                    WHERE text_lc CONTAINS term
                       OR title_lc CONTAINS term
                       OR doc_type_lc CONTAINS term
                       OR authority_lc CONTAINS term] AS matched_terms
            WHERE size(matched_terms) > 0
            WITH c, d, a, matched_terms,
                 reduce(score = 0.0, term IN matched_terms |
                    score
                    + CASE WHEN text_lc CONTAINS term THEN 1.0 ELSE 0.0 END
                    + CASE WHEN title_lc CONTAINS term THEN 2.0 ELSE 0.0 END
                    + CASE WHEN doc_type_lc CONTAINS term THEN 2.25 ELSE 0.0 END
                    + CASE WHEN authority_lc CONTAINS term THEN 1.75 ELSE 0.0 END
                 ) AS sparse_score
            RETURN
                c.chunk_id        AS chunk_id,
                c.text            AS text,
                c.type            AS chunk_type,
                d.doc_id          AS doc_id,
                d.title           AS title,
                d.url             AS url,
                d.doc_type        AS doc_type,
                d.category        AS category,
                d.doc_class       AS doc_class,
                d.effective_date  AS effective_date,
                a.name            AS authority,
                c.pages           AS pages,
                sparse_score      AS score,
                sparse_score      AS sparse_score,
                size(matched_terms) AS keyword_hits,
                matched_terms     AS matched_terms
            ORDER BY score DESC, keyword_hits DESC
            LIMIT $search_k
            """

            params = {
                "terms": sparse_terms,
                "search_k": search_k,
                "authority_names": authority_names_list,
                "doc_titles": doc_titles_list,
                "doc_types": doc_types_list,
                "doc_classes": doc_classes_list,
                "chunk_rel_types": chunk_rel_types,
                "min_effective_date": _normalize_date_filter(min_effective_date),
                "max_effective_date": _normalize_date_filter(max_effective_date),
            }

            result = session.run(cypher, params)
            data = result.data()
    finally:
        connector.close()

    return data[:top_k]


def check_match_page_level(
    gt_doc_ids: Optional[Iterable[str]],
    gt_page_nos: Iterable[Iterable[int]],
    results: Optional[List[Dict[str, Any]]],
    only_highest_score: bool = False,
) -> Optional[bool]:
    """
    Check if retrieval results match the ground truth at (doc_id, page) level.

    Inputs:
      - gt_doc_ids: list of ground-truth document IDs
      - gt_page_nos: list of lists of ground-truth page numbers
      - results: list of result dicts from query_chunks
      - only_highest_score: if True, only use the highest-scoring result(s)

    Output:
      - True / False if ground truth provided
      - None if no results or no ground truth doc IDs
    """
    if not results:
        return None

    if only_highest_score:
        max_score = max(r["score"] for r in results)
        results = [r for r in results if r["score"] == max_score]

    if not gt_doc_ids:
        return None

    # Build a mapping: doc_id -> set of retrieved pages
    doc_page_map: Dict[str, set] = {}
    for r in results:
        doc_id = r.get("doc_id")
        if not doc_id:
            continue

        # Try to interpret pages from different shapes ("page" or "pages")
        pages_field = r.get("pages")
        page_single = r.get("page")

        pages: set = set()
        if isinstance(pages_field, (list, tuple, set)):
            pages.update(pages_field)
        elif isinstance(page_single, (int, float)) and page_single is not None:
            pages.add(int(page_single))

        if not pages:
            continue

        if doc_id not in doc_page_map:
            doc_page_map[doc_id] = set()
        doc_page_map[doc_id].update(pages)

    # Compare each ground-truth doc and its pages to retrieved ones
    for idx, gt_doc_id in enumerate(gt_doc_ids):
        if gt_doc_id not in doc_page_map:
            return False

        expected_pages = set(gt_page_nos[idx])
        if not expected_pages.issubset(doc_page_map[gt_doc_id]):
            return False

    return True


def check_match_doc_level(
    gt_doc_ids: Optional[Iterable[str]],
    results: List[Dict[str, Any]],
    only_highest_score: bool = False,
) -> Optional[bool]:
    """
    Check if retrieval results contain all ground-truth document IDs.

    Inputs:
      - gt_doc_ids: iterable of ground-truth document IDs
      - results: list of result dicts from query_chunks
      - only_highest_score: if True, restrict to highest-scoring result(s)

    Output:
      - True / False if ground truth provided
      - None if gt_doc_ids is empty/None
    """
    if not gt_doc_ids:
        return None

    if not results:
        return False

    filtered_results = results
    if only_highest_score:
        max_score = max(r["score"] for r in results)
        filtered_results = [r for r in results if r["score"] == max_score]

    retrieved_doc_ids = {r.get("doc_id") for r in filtered_results if r.get("doc_id")}

    for gt_doc_id in gt_doc_ids:
        if gt_doc_id not in retrieved_doc_ids:
            return False

    return True
