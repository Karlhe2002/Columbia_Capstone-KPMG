# src/healthcare_rag_llm/llm/response_generator.py

import json
import re
from typing import Dict, List, Optional
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks
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

    @classmethod
    def _format_chunks(cls, chunks: List[Dict], compact_text: bool = False) -> str:
        if not chunks:
            return "(No relevant chunks found.)"
        return "\n\n".join(
            [
                # "chunk_id" is an internal identifier; we include it for traceability
                # but label it explicitly so the LLM does not treat it as a cite key.
                f"[Document Title: {cls._get_doc_title(chunk)}]"
                f"-[pages: {chunk['pages']}] - [Chunk Content: "
                f"{cls._compact_text(chunk.get('text', '')) if compact_text else chunk.get('text', '')}]"
                f" - [Internal Chunk ID (do not cite): {chunk.get('chunk_id', '')}]"
                for chunk in chunks
            ]
        )

    def answer_question(self, question: str, top_k: int = 5, rerank_top_k: int = 20, history: Optional[List[Dict]] = None) -> Dict:
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

        # 1. Encode query as vector
        query_vec = self._get_embedder().encode([question])["dense_vecs"][0].tolist()
        
        # 2. Retrieve more chunks initially (for reranking)
        initial_k = rerank_top_k if self.use_reranker else top_k
        # retrieved_chunks = query_chunks(query_vec, top_k=initial_k)
        retrieved_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            authority_names=filters.get("authority_names"),
            doc_titles=filters.get("doc_titles"),
            doc_types=filters.get("doc_types"),
            min_effective_date=filters.get("min_effective_date"),
            max_effective_date=filters.get("max_effective_date"),
            keywords=filters.get("keywords")
        )

        # 3. Apply reranking if enabled
        if self.use_reranker and retrieved_chunks:
            retrieved_chunks = apply_rerank_to_chunks(
                query=question,
                chunks=retrieved_chunks,
                combine_with_dense=True,  
                alpha=0.3,  
                text_key="text",
                dense_score_key="score"
            )
        #4 Take top k chunks
        final_chunks = retrieved_chunks[:top_k]
        
        # 3. Context
        context = self._format_chunks(final_chunks)
        
        # 6. Generate response 
        user_msg = f"""
Question:
{question}

Context Chunks (authoritative; cite only these):
{context}

Chunks format:
[Document Title: <doc_title>] -[pages: <pages>] - [Chunk Content: <chunk_content>] - [Internal Chunk ID (do not cite): <chunk_id>]

Output sections (exactly):
- Answer
- Evidence (quoted)
- Caveats (if any)
Each bullet must have a citation like [doc_title or doc_title:page — Mon DD, YYYY].
Do NOT cite chunk IDs. Citations must use Document Title (doc_title) + page (if available).
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
        }

    def _generate_followup_questions(
        self,
        question: str,
        answer: str,
        retrieved_chunks: List[Dict],
    ) -> List[str]:
        """Generate follow-up questions that are answerable from already retrieved chunks."""
        try:
            if not retrieved_chunks:
                return []

            allowed_chunk_ids = {str(chunk.get("chunk_id", "")) for chunk in retrieved_chunks if chunk.get("chunk_id")}
            context = self._format_chunks(retrieved_chunks, compact_text=True)

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
        top_k_per_source: int = 5,
        rerank_top_k: int = 20,
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

        retrieval_query = concept.strip() if concept and concept.strip() else question
        query_vec = self._get_embedder().encode([retrieval_query])["dense_vecs"][0].tolist()
        initial_k = rerank_top_k if self.use_reranker else top_k_per_source

        base_filters = {
            "authority_names": filters.get("authority_names"),
            "doc_titles": filters.get("doc_titles"),
            "doc_types": filters.get("doc_types"),
            "min_effective_date": filters.get("min_effective_date"),
            "max_effective_date": filters.get("max_effective_date"),
            "keywords": filters.get("keywords"),
        }

        source_search_k = max(initial_k * 5, initial_k)
        policy_filters = {**base_filters, "doc_classes": ["policy"]}
        provider_manual_filters = {**base_filters, "doc_classes": ["provider_manual"]}

        policy_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            search_k=source_search_k,
            **policy_filters,
        )
        provider_manual_chunks = query_chunks(
            query_vec,
            top_k=initial_k,
            search_k=source_search_k,
            **provider_manual_filters,
        )

        if self.use_reranker and policy_chunks:
            policy_chunks = apply_rerank_to_chunks(
                query=question,
                chunks=policy_chunks,
                combine_with_dense=True,
                alpha=0.3,
                text_key="text",
                dense_score_key="score",
            )
        if self.use_reranker and provider_manual_chunks:
            provider_manual_chunks = apply_rerank_to_chunks(
                query=question,
                chunks=provider_manual_chunks,
                combine_with_dense=True,
                alpha=0.3,
                text_key="text",
                dense_score_key="score",
            )

        policy_final = policy_chunks[:top_k_per_source]
        provider_manual_final = provider_manual_chunks[:top_k_per_source]

        policy_context = self._format_chunks(policy_final, compact_text=True)
        provider_context = self._format_chunks(provider_manual_final, compact_text=True)
        target_concept = concept if concept and concept.strip() else "the requested concept"

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
[Document Title: <doc_title>] -[Chunk ID: <chunk_id>]-[pages: <pages>] - [Chunk Content: <chunk_content>]

Output as a valid JSON object with exactly these keys:
{{
  "headline_summary": "1-2 sentence summary comparing both sources",
  "policy_definition": "Policy definition with inline citations [doc_title:page - date]",
  "provider_manual_definition": "Provider manual definition with inline citations",
  "similarities": ["similarity point 1 with citation", "similarity point 2"],
  "differences": ["difference point 1 with citation", "difference point 2"],
  "caveats": "Any caveats, or null if none"
}}
Do NOT wrap the JSON in markdown code fences. Return ONLY the JSON object.
Each field must have inline citations like [doc_title or doc_title:page — Mon DD, YYYY].
If either side has insufficient evidence, state that explicitly in the relevant field.
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

        return {
            "question": question,
            "concept": concept,
            "answer": llm_response,
            "compare_sections": compare_sections,
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