# src/healthcare_rag_llm/llm/response_generator.py

from typing import Dict, List, Optional
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.graph_builder.queries import query_chunks
from healthcare_rag_llm.embedding.HealthcareEmbedding import HealthcareEmbedding
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
        self.embedder = HealthcareEmbedding()
        self.use_reranker = use_reranker
        self.filter_extractor = filter_extractor
        self.chat_history = chat_history or ChatHistory()
        try:
            self.compare_system_prompt = load_system_prompt("configs/system_prompt_compare.txt")
        except FileNotFoundError:
            self.compare_system_prompt = self.system_prompt

    @staticmethod
    def _format_chunks(chunks: List[Dict]) -> str:
        if not chunks:
            return "(No relevant chunks found.)"
        return "\n\n".join(
            [
                f"[Document ID: {chunk['doc_id']}] -[Chunk ID: {chunk['chunk_id']}]"
                f"-[pages: {chunk['pages']}] - [Chunk Content: {chunk['text']}]"
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
        query_vec = self.embedder.encode([question])["dense_vecs"][0].tolist()
        
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
        context = "\n\n".join(
            [f"[Document ID: {chunk['doc_id']}] -[Chunk ID: {chunk['chunk_id']}]-[pages: {chunk['pages']}] - [Chunk Content: {chunk['text']}]" 
             for chunk in final_chunks]
        )
        
        # 6. Generate response 
        user_msg = f"""
Question:
{question}

Context Chunks (authoritative; cite only these):
{context}

Chunks format:
[Document ID: <doc_id>] -[Chunk ID: <chunk_id>]-[pages: <pages>] - [Chunk Content: <chunk_content>]

Output sections (exactly):
- Answer
- Evidence (quoted)
- Caveats (if any)
Each bullet must have a citation like [doc or doc:page — Mon DD, YYYY].
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

        return {
            "question": question,
            "answer": llm_response,
            "retrieved_docs": final_chunks,
        }

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
        query_vec = self.embedder.encode([retrieval_query])["dense_vecs"][0].tolist()
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

        policy_context = self._format_chunks(policy_final)
        provider_context = self._format_chunks(provider_manual_final)
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
[Document ID: <doc_id>] -[Chunk ID: <chunk_id>]-[pages: <pages>] - [Chunk Content: <chunk_content>]

Output sections (exactly):
- Policy Definition
- Provider Manual Definition
- Comparison (similarities and differences)
- Evidence (quoted)
- Caveats (if any)
Each bullet must have a citation like [doc or doc:page — Mon DD, YYYY].
If either side has insufficient evidence, state that explicitly.
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

        return {
            "question": question,
            "concept": concept,
            "answer": llm_response,
            "retrieved_docs": {
                "policy": policy_final,
                "provider_manual": provider_manual_final,
            },
        }