# src/healthcare_rag_llm/llm/response_generator.py

import json
import re
from datetime import date, datetime
from typing import Dict, List, Optional
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks, query_chunks_sparse
from healthcare_rag_llm.embedding.HealthcareEmbedding import get_embedding_singleton
from healthcare_rag_llm.reranking.reranker import apply_rerank_to_chunks
from healthcare_rag_llm.llm.chat_history import ChatHistory
from healthcare_rag_llm.utils.prompt_config import load_system_prompt


class ResponseGenerator:
    """
    End-to-end RAG (Retrieval-Augmented Generation) pipeline.
    Responsible for:
      1. Retrieving relevant context documents
      2. Building the final LLM prompt
      3. Generating the answer using the LLM
    """

    def __init__(self, llm_client: LLMClient,
                 system_prompt: Optional[str] = None,
                 use_reranker: bool = True,
                 filter_extractor=None,
                 chat_history: ChatHistory = None):
        # Load system prompt from config if not provided
        if system_prompt is None:
            system_prompt = load_system_prompt()
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.embedder = None
        self.use_reranker = use_reranker
        self.filter_extractor = filter_extractor
        self.chat_history = chat_history or ChatHistory()
        try:
            self.compare_system_prompt = load_system_prompt("configs/system_prompt_compare.txt")
        except FileNotFoundError:
            self.compare_system_prompt = self.system_prompt

    def _get_embedder(self):
        if self.embedder is None:
            self.embedder = get_embedding_singleton()
        return self.embedder

    @staticmethod
    def _get_doc_title(chunk: Dict) -> str:
        return (
            chunk.get("title")
            or chunk.get("doc_title")
            or chunk.get("doc_id")
            or "Unknown Document"
        )

    @staticmethod
    def _compact_text(text: str, max_chars: int = 500) -> str:
        cleaned = " ".join((text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[:max_chars].rstrip() + " ..."

    @staticmethod
    def _dedupe_terms(values: List[str], max_items: int = 12) -> List[str]:
        out = []
        seen = set()
        for value in values or []:
            cleaned = " ".join(str(value or "").strip().split())
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(cleaned)
            if len(out) >= max_items:
                break
        return out

    @classmethod
    def _build_keyword_hints(cls, filters: Dict) -> List[str]:
        return cls._dedupe_terms(
            list(filters.get("keywords") or [])
            + list(filters.get("search_themes") or [])
            + list(filters.get("semantic_keywords") or [])
        )

    @classmethod
    def _build_rerank_query(cls, base_query: str, filters: Dict) -> str:
        keyword_hints = cls._build_keyword_hints(filters)
        if not keyword_hints:
            return base_query
        return f"{base_query}\nKey themes: {', '.join(keyword_hints[:8])}"

    @staticmethod
    def _chunk_fusion_key(chunk: Dict) -> str:
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if chunk_id:
            return chunk_id
        doc_id = str(chunk.get("doc_id") or "").strip()
        pages = str(chunk.get("pages") or "").strip()
        return f"{doc_id}::{pages}"

    @classmethod
    def _fuse_ranked_hits_rrf(
        cls,
        dense_hits: List[Dict],
        sparse_hits: List[Dict],
        *,
        rrf_k: int = 60,
        max_results: Optional[int] = None,
    ) -> List[Dict]:
        fused: Dict[str, Dict] = {}

        for source_name, hits in (("dense", dense_hits or []), ("sparse", sparse_hits or [])):
            for rank, hit in enumerate(hits, start=1):
                key = cls._chunk_fusion_key(hit)
                if key not in fused:
                    base = dict(hit)
                    base["retrieval_sources"] = []
                    base["rrf_score"] = 0.0
                    base.setdefault("dense_score", None)
                    base.setdefault("sparse_score", None)
                    fused[key] = base

                entry = fused[key]
                entry["rrf_score"] += 1.0 / (rrf_k + rank)
                if source_name not in entry["retrieval_sources"]:
                    entry["retrieval_sources"].append(source_name)
                if source_name == "dense":
                    entry["dense_score"] = hit.get("score")
                    entry.setdefault("vector_score", hit.get("vector_score", hit.get("score")))
                else:
                    entry["sparse_score"] = hit.get("score")
                    entry["matched_terms"] = hit.get("matched_terms", entry.get("matched_terms"))

        fused_hits = list(fused.values())
        fused_hits.sort(
            key=lambda hit: (
                float(hit.get("rrf_score", 0.0) or 0.0),
                float(hit.get("dense_score", 0.0) or 0.0),
                float(hit.get("sparse_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        for hit in fused_hits:
            hit["score"] = float(hit.get("rrf_score", 0.0) or 0.0)
        if max_results is not None:
            return fused_hits[:max_results]
        return fused_hits

    @classmethod
    def _log_retrieval_stage(cls, label: str, hits: List[Dict], limit: int = 5) -> None:
        pass

    @classmethod
    def _log_retrieval_breakdown(
        cls,
        *,
        query: str,
        retrieval_query: str,
        keyword_hints: List[str],
        dense_hits: List[Dict],
        sparse_hits: List[Dict],
        fused_hits: List[Dict],
        final_hits: List[Dict],
        label: str = "qa",
    ) -> None:
        pass

    @staticmethod
    def _parse_effective_date(value) -> Optional[date]:
        if not value:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        if isinstance(value, datetime):
            return value.date()
        raw = str(value).strip()
        if not raw or raw.upper() == "N/A":
            return None
        for parser in (
            lambda s: datetime.fromisoformat(s).date(),
            lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
            lambda s: datetime.strptime(s, "%B %Y").date(),
            lambda s: datetime.strptime(s, "%b %Y").date(),
            lambda s: datetime.strptime(s, "%B %d, %Y").date(),
            lambda s: datetime.strptime(s, "%b %d, %Y").date(),
        ):
            try:
                return parser(raw)
            except ValueError:
                continue
        return None

    @classmethod
    def _latest_chunk(cls, chunks: List[Dict]) -> Optional[Dict]:
        dated_chunks = []
        for chunk in chunks or []:
            parsed = cls._parse_effective_date(chunk.get("effective_date"))
            if parsed is not None:
                dated_chunks.append((parsed, chunk))
        if not dated_chunks:
            return None
        dated_chunks.sort(key=lambda item: item[0], reverse=True)
        return dated_chunks[0][1]

    @classmethod
    def _build_question_recency_brief(cls, chunks: List[Dict]) -> str:
        newest_chunk = cls._latest_chunk(chunks)
        if not newest_chunk:
            return (
                "Recency briefing:\n"
                "- No usable effective dates were available in the final retrieved chunks.\n"
                "- Do not guess recency. If recency matters, say it could not be determined from the provided metadata."
            )

        doc_title = cls._get_doc_title(newest_chunk)
        effective_date = newest_chunk.get("effective_date") or "N/A"
        pages = newest_chunk.get("pages", "N/A")
        return (
            "Recency briefing:\n"
            f"- Newest relevant retrieved source: {doc_title}\n"
            f"- Effective date: {effective_date}\n"
            f"- Pages: {pages}\n"
            "- Put the most weight on this source when answering the user's current-guidance question unless the user explicitly asks about a historical period.\n"
            "- If older sources differ, explain that they are older and treat them as supporting or superseded context."
        )

    @classmethod
    def _build_compare_recency_brief(cls, policy_chunks: List[Dict], provider_manual_chunks: List[Dict]) -> str:
        policy_latest = cls._latest_chunk(policy_chunks)
        provider_latest = cls._latest_chunk(provider_manual_chunks)

        def _side_lines(label: str, chunk: Optional[Dict]) -> List[str]:
            if not chunk:
                return [
                    f"- Newest {label} source: unavailable",
                    f"- Newest {label} effective date: unavailable",
                ]
            return [
                f"- Newest {label} source: {cls._get_doc_title(chunk)}",
                f"- Newest {label} effective date: {chunk.get('effective_date') or 'N/A'}",
                f"- Newest {label} pages: {chunk.get('pages', 'N/A')}",
            ]

        lines = ["Recency briefing:"]
        lines.extend(_side_lines("policy", policy_latest))
        lines.extend(_side_lines("provider manual", provider_latest))

        policy_date = cls._parse_effective_date(policy_latest.get("effective_date")) if policy_latest else None
        provider_date = cls._parse_effective_date(provider_latest.get("effective_date")) if provider_latest else None

        if policy_date and provider_date:
            if policy_date > provider_date:
                lines.append("- More up-to-date source type overall: policy")
                lines.append("- Put the most weight on the newest policy source when describing current guidance, while still explaining any meaningful provider-manual differences.")
            elif provider_date > policy_date:
                lines.append("- More up-to-date source type overall: provider manual")
                lines.append("- Put the most weight on the newest provider manual source when describing current guidance, while still explaining any meaningful policy differences.")
            else:
                lines.append("- More up-to-date source type overall: tied on effective date")
                lines.append("- Treat both source types as equally current on recency grounds and resolve the answer using grounded content differences.")
        else:
            lines.append("- More up-to-date source type overall: cannot be determined from available effective dates")
            lines.append("- Do not guess overall recency; say so explicitly if it matters to the answer.")

        return "\n".join(lines)

    @classmethod
    def _format_chunks(cls, chunks: List[Dict], compact_text: bool = False) -> str:
        if not chunks:
            return "(No relevant chunks found.)"
        return "\n\n".join(
            [
                f"[Document Title: {cls._get_doc_title(chunk)}] -[Chunk ID: {chunk['chunk_id']}]"
                f"-[Effective Date: {chunk.get('effective_date') or 'N/A'}]"
                f"-[pages: {chunk['pages']}] - [Chunk Content: "
                f"{cls._compact_text(chunk.get('text', '')) if compact_text else chunk.get('text', '')}]"
                for chunk in chunks
            ]
        )

    @staticmethod
    def _clean_compare_concept(value: str) -> str:
        value = " ".join((value or "").split()).strip()
        value = re.sub(r"^[\"'`“”‘’]+|[\"'`“”‘’]+$", "", value).strip()
        value = re.sub(r"^[\s:;,\-]+|[\s:;,\-?.!]+$", "", value).strip()
        return value

    @classmethod
    def _resolve_compare_concept(cls, question: str, concept: Optional[str] = None) -> str:
        raw = (concept or question or "").strip()
        if not raw:
            return ""

        # Prefer explicitly quoted target concepts, e.g. Define 'medical necessity'...
        quoted_match = re.search(r"[\"'`“”‘’]([^\"'`“”‘’]{2,160})[\"'`“”‘’]", raw)
        if quoted_match:
            quoted_value = cls._clean_compare_concept(quoted_match.group(1))
            if quoted_value:
                return quoted_value

        candidate = " ".join(raw.split()).strip()
        candidate = re.sub(
            r"^(define|explain|describe|compare|contrast|summarize|tell me about)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"^(what is|what's|how is|how do you define)\s+",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\s+(in|across|between|for)\s+policy\s+(vs\.?|versus|and)\s+provider\s+manual\.?$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = re.sub(
            r"\s+in\s+(policy|provider manual)\s+documents?\.?$",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = cls._clean_compare_concept(candidate)
        return candidate or cls._clean_compare_concept(raw)

    def answer_question(self, question: str, top_k: int = 8, rerank_top_k: int = 30, history: Optional[List[Dict]] = None) -> Dict:
        """
        Full question-answering pipeline.

        Steps:
          1. Retrieve relevant chunks (RAG retrieval)
          2. Construct context + prompt
          3. Generate final response using the LLM
        """
        self.chat_history.add("user", question)

        # 0. Smart filter
        filters = self.filter_extractor.extract(question) if self.filter_extractor else {}
        retrieval_query = (filters.get("retrieval_query") or question).strip()
        normalized_query = (filters.get("normalized_query") or question).strip()
        rerank_query = self._build_rerank_query(normalized_query, filters)
        keyword_hints = self._build_keyword_hints(filters)

        # 1. Encode query as vector
        # Keep retrieval stable while we iterate on query understanding.
        query_vec = self._get_embedder().encode([question])["dense_vecs"][0].tolist()
        
        # 2. Retrieve more chunks initially (for reranking)
        initial_k = rerank_top_k if self.use_reranker else top_k
        dense_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            authority_names=filters.get("authority_names"),
            doc_titles=filters.get("doc_titles"),
            doc_types=filters.get("doc_types"),
            min_effective_date=filters.get("min_effective_date"),
            max_effective_date=filters.get("max_effective_date"),
            keywords=keyword_hints,
        )
        sparse_chunks = query_chunks_sparse(
            retrieval_query,
            top_k=initial_k,
            authority_names=filters.get("authority_names"),
            doc_titles=filters.get("doc_titles"),
            doc_types=filters.get("doc_types"),
            min_effective_date=filters.get("min_effective_date"),
            max_effective_date=filters.get("max_effective_date"),
            keywords=keyword_hints,
        )
        retrieved_chunks = self._fuse_ranked_hits_rrf(
            dense_chunks,
            sparse_chunks,
            max_results=max(initial_k * 2, initial_k),
        )

        # 3. Apply reranking if enabled
        if self.use_reranker and retrieved_chunks:
            retrieved_chunks = apply_rerank_to_chunks(
                query=rerank_query,
                chunks=retrieved_chunks,
                combine_with_dense=True,  
                alpha=0.3,  
                text_key="text",
                dense_score_key="score"
            )
        #4 Take top k chunks
        final_chunks = retrieved_chunks[:top_k]
        self._log_retrieval_breakdown(
            query=question,
            retrieval_query=retrieval_query,
            keyword_hints=keyword_hints,
            dense_hits=dense_chunks,
            sparse_hits=sparse_chunks,
            fused_hits=retrieved_chunks,
            final_hits=final_chunks,
            label="qa",
        )
        
        # 3. Context
        context = self._format_chunks(final_chunks)
        recency_brief = self._build_question_recency_brief(final_chunks)
        
        # 6. Generate response 
        user_msg = f"""
Question:
{question}

Context Chunks (authoritative; cite only these):
{context}

Chunks format:
[Document Title: <doc_title>] -[Chunk ID: <chunk_id>]-[Effective Date: <effective_date or N/A>]-[pages: <pages>] - [Chunk Content: <chunk_content>]
Use the provided Effective Date metadata to determine which grounded source is the most recent.

{recency_brief}

Output sections (exactly):
- Answer
- Evidence (quoted)
- Caveats (if any)
Each bullet must have a citation like [doc_title or doc_title:page — Mon DD, YYYY].
Answer requirements:
- Start with a direct answer to the user's question.
- If dates are available, use clear wording such as "According to the newest relevant source dated ..." near the beginning of the answer.
- State what the newest source says before discussing older supporting material.
- Put the most effort on the newest-source briefing above when deciding how to frame the answer.
- If multiple sources conflict, explain which source is newer and give it more weight unless the user explicitly asks about a historical period or date-bounded question.
- If dates are missing or unclear, say that explicitly instead of guessing.
- Keep the answer grounded in the provided chunks only.
Internal decision process:
- Determine whether the user is asking about current guidance or a historical period.
- Identify the most relevant chunks for the question.
- Use Effective Date metadata to find the newest relevant source.
- Answer from the newest relevant source first, then add older grounded context only if it helps.
If Caveats has no meaningful content, output exactly:
Caveats
None
Formatting requirements (strict):
1) Use section headers on their own lines exactly as:
   Answer: <answer text>
   Evidence: <evidence text>
   Caveats: <caveats text>
2) Leave one blank line between sections.
3) Do NOT put section headers inline with sentence content (forbidden: "... April 2023. Evidence:").
4) In Answer section, include only answer text (no "Evidence" or "Caveats" words as section labels).
""".strip()

        # llm_response = self.llm_client.chat(
        #     user_prompt=user_msg,
        #     system_prompt=self.system_prompt
        # )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.chat_history.get_messages())
        if messages and messages[-1]['role'] == "user":
            messages[-1]["content"] = user_msg
        else:
            messages.append({"role": "user", "content": user_msg})
        
        llm_response = self.llm_client.chat(messages=messages)
        self.chat_history.add("assistant", llm_response)

        followup_questions = self._generate_followup_questions(
            question=question,
            answer=llm_response,
            retrieved_chunks=final_chunks,
        )

        return {
            "question": question,
            "answer": llm_response,
            "retrieved_docs": final_chunks,
            "followup_questions": followup_questions,
            "query_understanding": {
                "retrieval_query": retrieval_query,
                "normalized_query": normalized_query,
                "search_themes": keyword_hints,
                "filters": filters,
            },
        }

    def _generate_followup_questions(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict],
        mode: str = "qa",
    ) -> List[str]:
        """Generate follow-up questions that are answerable from already retrieved chunks."""
        try:
            if not retrieved_chunks:
                return []

            allowed_chunk_ids = {str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks if chunk.get("chunk_id")}
            context = self._format_chunks(retrieved_chunks, compact_text=True)

            if mode == "compare":
                prompt = (
                    "You are generating suggested follow-up questions for a NYS Medicaid comparison assistant.\n"
                    "The user just compared a concept across policy vs provider manual sources.\n"
                    "Goal: produce questions that help the user dig deeper into the comparison, "
                    "answerable using ONLY the retrieved chunks provided below.\n"
                    "Hard rules:\n"
                    "1) Suggest exactly 3 concise follow-up questions.\n"
                    "2) Every suggested question must be directly answerable from the retrieved chunks.\n"
                    "3) Each question should focus on ONE of these directions:\n"
                    "   - Drill into a specific difference or similarity found in the comparison\n"
                    "   - Compare a related concept across the same two source types (policy vs provider manual)\n"
                    "   - Clarify which source is more current or authoritative for a specific point\n"
                    "4) For each question, include at least one supporting chunk_id from the retrieved chunks.\n"
                    "5) Do not use outside knowledge.\n"
                    "6) Return ONLY valid JSON (no markdown, no commentary) as an array with this schema:\n"
                    '[{"question":"...","supports":["chunk_id_1","chunk_id_2"]}]\n\n'
                    f"Original question:\n{question}\n\n"
                    f"Current answer (abbreviated):\n{answer[:500]}\n\n"
                    f"Retrieved chunks:\n{context}\n"
                )
            else:
                prompt = (
                    "You are generating suggested follow-up questions for a NYS Medicaid Q&A assistant.\n"
                    "Goal: produce questions that can be answered using ONLY the retrieved chunks provided below.\n"
                    "Hard rules:\n"
                    "1) Suggest exactly 3 concise follow-up questions.\n"
                    "2) Every suggested question must be directly answerable from the retrieved chunks.\n"
                    "3) For each question, include at least one supporting chunk_id from the retrieved chunks.\n"
                    "4) Do not use outside knowledge.\n"
                    "5) Return ONLY valid JSON (no markdown, no commentary) as an array with this schema:\n"
                    '[{"question":"...","supports":["chunk_id_1","chunk_id_2"]}]\n\n'
                    f"Original question:\n{question}\n\n"
                    f"Current answer (abbreviated):\n{answer[:500]}\n\n"
                    f"Retrieved chunks:\n{context}\n"
                )
            raw = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            cleaned = re.sub(r'^```(?:json)?\s*', '', (raw or "").strip())
            cleaned = re.sub(r'\s*```$', '', cleaned)
            parsed = json.loads(cleaned)

            if not isinstance(parsed, list):
                return []

            questions: List[str] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                supports = item.get("supports", [])
                if not q or not isinstance(supports, list):
                    continue
                support_ids = [str(s).strip() for s in supports if str(s).strip()]
                if not support_ids:
                    continue
                if any(sid not in allowed_chunk_ids for sid in support_ids):
                    continue
                if q not in questions:
                    questions.append(q)
                if len(questions) == 3:
                    break

            return questions
        except Exception as e:
            print(f"[followup] Failed to generate follow-up questions: {e}")
            return []

    def answer_compare_definitions(
        self,
        question: str,
        concept: Optional[str] = None,
        top_k_per_source: int = 4,
        rerank_top_k: int = 30,
    ) -> Dict:
        """
        Compare a concept definition across policy vs provider manual documents.

        Retrieval strategy:
          1) Retrieve by doc_classes=['policy']
          2) Retrieve by doc_classes=['provider_manual']
          3) Build dual-source context and ask LLM for side-by-side comparison
        """
        self.chat_history.add("user", question)
        filters = self.filter_extractor.extract(question) if self.filter_extractor else {}

        resolved_concept = self._resolve_compare_concept(question, concept)
        default_query = resolved_concept or question
        retrieval_query = (filters.get("retrieval_query") or default_query).strip()
        normalized_query = (filters.get("normalized_query") or question).strip()
        rerank_query = self._build_rerank_query(normalized_query, filters)
        keyword_hints = self._build_keyword_hints(filters)
        query_vec = self._get_embedder().encode([default_query])["dense_vecs"][0].tolist()
        initial_k = rerank_top_k if self.use_reranker else top_k_per_source

        base_filters = {
            "authority_names": filters.get("authority_names"),
            "doc_titles": filters.get("doc_titles"),
            "doc_types": filters.get("doc_types"),
            "min_effective_date": filters.get("min_effective_date"),
            "max_effective_date": filters.get("max_effective_date"),
            "keywords": keyword_hints,
        }

        source_search_k = max(initial_k * 5, initial_k)
        policy_filters = {**base_filters, "doc_classes": ["policy"]}
        provider_manual_filters = {**base_filters, "doc_classes": ["provider_manual"]}

        policy_dense_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            search_k=source_search_k,
            **policy_filters,
        )
        policy_sparse_chunks = query_chunks_sparse(
            retrieval_query,
            top_k=initial_k,
            search_k=source_search_k,
            **policy_filters,
        )
        provider_manual_dense_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            search_k=source_search_k,
            **provider_manual_filters,
        )
        provider_manual_sparse_chunks = query_chunks_sparse(
            retrieval_query,
            top_k=initial_k,
            search_k=source_search_k,
            **provider_manual_filters,
        )
        policy_chunks = self._fuse_ranked_hits_rrf(
            policy_dense_chunks,
            policy_sparse_chunks,
            max_results=max(initial_k * 2, initial_k),
        )
        provider_manual_chunks = self._fuse_ranked_hits_rrf(
            provider_manual_dense_chunks,
            provider_manual_sparse_chunks,
            max_results=max(initial_k * 2, initial_k),
        )

        if self.use_reranker and policy_chunks:
            policy_chunks = apply_rerank_to_chunks(
                query=rerank_query,
                chunks=policy_chunks,
                combine_with_dense=True,
                alpha=0.3,
                text_key="text",
                dense_score_key="score",
            )
        if self.use_reranker and provider_manual_chunks:
            provider_manual_chunks = apply_rerank_to_chunks(
                query=rerank_query,
                chunks=provider_manual_chunks,
                combine_with_dense=True,
                alpha=0.3,
                text_key="text",
                dense_score_key="score",
            )

        policy_final = policy_chunks[:top_k_per_source]
        provider_manual_final = provider_manual_chunks[:top_k_per_source]
        self._log_retrieval_breakdown(
            query=question,
            retrieval_query=retrieval_query,
            keyword_hints=keyword_hints,
            dense_hits=policy_dense_chunks,
            sparse_hits=policy_sparse_chunks,
            fused_hits=policy_chunks,
            final_hits=policy_final,
            label="compare_policy",
        )
        self._log_retrieval_breakdown(
            query=question,
            retrieval_query=retrieval_query,
            keyword_hints=keyword_hints,
            dense_hits=provider_manual_dense_chunks,
            sparse_hits=provider_manual_sparse_chunks,
            fused_hits=provider_manual_chunks,
            final_hits=provider_manual_final,
            label="compare_provider_manual",
        )

        policy_context = self._format_chunks(policy_final, compact_text=True)
        provider_context = self._format_chunks(provider_manual_final, compact_text=True)
        recency_brief = self._build_compare_recency_brief(policy_final, provider_manual_final)
        target_concept = resolved_concept or "the requested concept"

        user_msg = f"""
Question:
{question}

Target concept:
{target_concept}

Policy Context Chunks (authoritative; cite only these):
{policy_context}

Provider Manual Context Chunks (authoritative; cite only these):
{provider_context}

Chunks format:
[Document Title: <doc_title>] -[Chunk ID: <chunk_id>]-[Effective Date: <effective_date or N/A>]-[pages: <pages>] - [Chunk Content: <chunk_content>]
Use the provided Effective Date metadata to determine which grounded source is more recent.

{recency_brief}

Output as a valid JSON object with exactly these keys:
{{
  "headline_summary": "1-2 sentence summary comparing both sources and clearly surfacing which source is more recent when supported by dates",
  "policy_definition": "Policy definition with inline citations [doc_title:page - date]",
  "provider_manual_definition": "Provider manual definition with inline citations",
  "similarities": ["similarity point 1 with citation", "similarity point 2"],
  "differences": ["difference point 1 with citation", "difference point 2"],
  "caveats": "Any caveats, or null if none"
}}
Do NOT wrap the JSON in markdown code fences. Return ONLY the JSON object.
Each field must have inline citations like [doc_title or doc_title:page — Mon DD, YYYY].
Comparison requirements:
- In "headline_summary", answer the user's question directly and make the recency conclusion explicit.
- Use clear wording such as "According to the newest policy source dated ..." or "According to the newest provider manual source dated ...".
- Explicitly state the newest relevant policy date and the newest relevant provider manual date when available.
- Clearly say which source type is more up to date overall for this issue.
- Put the most effort on the recency briefing above when deciding which source should guide the user now.
- If the provider manual is more recent, say so in the headline summary and explain that it should receive more weight by default.
- If policy is more recent, say so in the headline summary and explain that it should receive more weight by default.
- If dates are missing or unclear, say that explicitly instead of guessing.
- If the user explicitly asks about a historical period, honor that instead of defaulting to the newest source.
- If either side has insufficient evidence, state that explicitly in the relevant field.
Internal decision process:
- Identify the newest relevant policy chunk by Effective Date.
- Identify the newest relevant provider manual chunk by Effective Date.
- Compare those newest sources first.
- In "policy_definition", lead with the newest policy evidence.
- In "provider_manual_definition", lead with the newest provider manual evidence.
- Use older evidence only as background or to explain differences.
""".strip()

        messages = []
        if self.compare_system_prompt:
            messages.append({"role": "system", "content": self.compare_system_prompt})
        messages.extend(self.chat_history.get_messages())
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = user_msg
        else:
            messages.append({"role": "user", "content": user_msg})

        llm_response = self.llm_client.chat(messages=messages)
        self.chat_history.add("assistant", llm_response)
        compare_sections = self._parse_compare_response(llm_response)

        all_chunks = policy_final + provider_manual_final
        followup_questions = self._generate_followup_questions(
            question=question,
            answer=llm_response,
            retrieved_chunks=all_chunks,
            mode="compare",
        )

        return {
            "question": question,
            "concept": resolved_concept or concept,
            "answer": llm_response,
            "compare_sections": compare_sections,
            "followup_questions": followup_questions,
            "query_understanding": {
                "retrieval_query": retrieval_query,
                "normalized_query": normalized_query,
                "search_themes": keyword_hints,
                "filters": filters,
            },
            "retrieved_docs": {
                "policy": policy_final,
                "provider_manual": provider_manual_final,
            },
        }

    @staticmethod
    def _parse_compare_response(llm_response: str) -> Dict:
        """Parse structured JSON from compare definitions LLM response."""
        cleaned = re.sub(r'^```(?:json)?\s*', '', llm_response.strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "headline_summary": llm_response,
                "policy_definition": "",
                "provider_manual_definition": "",
                "similarities": [],
                "differences": [],
                "caveats": None,
                "_parse_failed": True,
            }
        for key in ("similarities", "differences"):
            val = parsed.get(key, [])
            if isinstance(val, str):
                parsed[key] = [val] if val else []
        return parsed
