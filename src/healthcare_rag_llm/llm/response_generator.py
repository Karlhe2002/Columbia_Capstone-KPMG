# src/healthcare_rag_llm/llm/response_generator.py

import json
import re
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from healthcare_rag_llm.llm.llm_client import GenerationCancelled, LLMClient
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
            self.compare_writer_system_prompt = load_system_prompt("configs/system_prompt_compare.txt")
        except FileNotFoundError:
            self.compare_writer_system_prompt = self.system_prompt
        try:
            self.compare_row_generator_system_prompt = load_system_prompt(
                "configs/system_prompt_compare_row_generator.txt"
            )
        except FileNotFoundError:
            self.compare_row_generator_system_prompt = self.compare_writer_system_prompt
        # Backward-compatible alias for older compare code paths that still reference this name.
        self.compare_system_prompt = self.compare_writer_system_prompt

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
    def _trim_incomplete_trailing_text(text: str) -> str:
        # If a chunk ends mid-sentence, trim back to the last complete sentence so the model
        # does not quote visibly broken fragments.
        raw = str(text or "")
        if not raw.strip():
            return ""
        stripped = raw.rstrip()
        if stripped[-1] in ".!?)]}\"'":
            return stripped

        sentence_matches = list(re.finditer(r"[.!?](?=\s|$)", stripped))
        if not sentence_matches:
            return stripped

        cut_idx = sentence_matches[-1].end()
        candidate = stripped[:cut_idx].rstrip()
        if len(candidate) >= max(int(len(stripped) * 0.6), 80):
            return candidate
        return stripped

    @staticmethod
    def _terminal_token_is_suspiciously_truncated(text: str) -> bool:
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return False
        match = re.search(r"\b([A-Za-z][A-Za-z\-]{4,})(?=[.!?]['\")\]]*\s*$)", cleaned)
        if not match:
            return False

        token = match.group(1).lower().rstrip("-")
        allowed_complete_terms = {
            "today", "cpt", "hcpcs", "ndc", "cin", "npi", "ffs", "mmc", "ltc", "cbic",
            "scc", "opra", "mevs", "nyrx", "medicare", "medicaid",
        }
        if token in allowed_complete_terms:
            return False

        suspicious_terminal_stems = (
            "administra",
            "administrat",
            "authorizat",
            "beneficiar",
            "categor",
            "clarificat",
            "compariso",
            "coordina",
            "definitio",
            "differenc",
            "dispens",
            "guidelin",
            "identificat",
            "informatio",
            "organizat",
            "pharmac",
            "practicion",
            "prescript",
            "procedur",
            "referenc",
            "similarit",
            "transact",
            "transitio",
            "utiliz",
            "vaccin",
        )
        return token.endswith(suspicious_terminal_stems)

    @staticmethod
    def _sanitize_compare_text_field(text: str) -> str:
        # Clean up compare output text so obviously broken evidence fragments do not reach the UI.
        # This is intentionally pattern-based rather than tied to one phrase, so it can catch
        # many kinds of chunk-edge damage such as chopped quotes, dangling connectors, or
        # incomplete tail words.
        cleaned = " ".join(str(text or "").split()).strip()
        if not cleaned:
            return ""

        cleaned = (
            cleaned
            .replace("â€”", "—")
            .replace("Ã¢â‚¬â€", "—")
            .replace("â€œ", '"')
            .replace("â€", '"')
            .replace("â€˜", "'")
            .replace("â€™", "'")
        )

        cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)

        def _strip_incomplete_quoted_tails(value: str) -> str:
            # If a quoted fragment ends with a suspicious chopped tail, keep the sentence but
            # remove the broken quote wrapper so it reads like a paraphrase instead of a fragment.
            pattern = r"'([^'\n]{8,}?)'(?=\s*(?:\[\d+\])?[.,;:]?\s*$)"

            def _replace(match):
                inner = match.group(1).strip()
                if not inner:
                    return ""
                if re.search(r"\b(?:and|or|to|for|with|of|in|on|by|the|this|that|these|those)$", inner, re.I):
                    return inner
                if re.search(r"\b[A-Z]$", inner):
                    return inner
                if re.search(r"[A-Za-z]{1,4}$", inner) and not re.search(r"[.!?]$", inner):
                    return inner
                return match.group(0)

            return re.sub(pattern, _replace, value)

        def _trim_broken_tail(value: str) -> str:
            # Trim only clearly broken endings such as a stray capital letter or a dangling
            # connector. Do not strip ordinary short valid ending words like "card" or acronyms
            # such as "NYRx", because that can damage otherwise-correct text.
            if not value:
                return value

            broken_tail_patterns = [
                r"\s+[A-Z](?=\s*(?:\[\d+\])?[.,;:]?\s*$)",  # e.g. "Managed C"
                r"\b(?:and|or|to|for|with|of|in|on|by|the|this|that|these|those)(?=\s*(?:\[\d+\])?[.,;:]?\s*$)",
            ]

            trimmed = value
            for pattern in broken_tail_patterns:
                candidate = re.sub(pattern, "", trimmed).rstrip()
                if candidate != trimmed and len(candidate) >= max(30, int(len(trimmed) * 0.6)):
                    trimmed = candidate
                    break
            if ResponseGenerator._terminal_token_is_suspiciously_truncated(trimmed):
                candidate = re.sub(r"\s+[^\s]+(?=[.!?]['\")\]]*\s*$)", "", trimmed).rstrip(" ,;:")
                if len(candidate) >= max(30, int(len(trimmed) * 0.6)):
                    trimmed = candidate
            return trimmed

        def _trim_to_last_safe_boundary(value: str) -> str:
            # If the field still looks damaged, fall back to the last clean sentence/phrase boundary.
            if not value:
                return value
            if (
                re.search(r"[.!?](?:\s*(?:\[\d+\])?[\"')\]]*)?\s*$", value)
                and not ResponseGenerator._terminal_token_is_suspiciously_truncated(value)
            ):
                return value

            boundary_matches = list(re.finditer(r"[.!?;:](?=\s|$)", value))
            if boundary_matches:
                candidate = value[:boundary_matches[-1].end()].rstrip()
                if len(candidate) >= max(30, int(len(value) * 0.6)):
                    return candidate

            comma_matches = list(re.finditer(r",(?=\s)", value))
            if comma_matches:
                candidate = value[:comma_matches[-1].start()].rstrip()
                if len(candidate) >= max(30, int(len(value) * 0.7)):
                    return candidate
            return value

        cleaned = _strip_incomplete_quoted_tails(cleaned)
        cleaned = _trim_broken_tail(cleaned)
        cleaned = _trim_to_last_safe_boundary(cleaned)

        # Balance dangling quotes/parentheses created by model paraphrasing.
        if cleaned.count('"') % 2 == 1:
            cleaned = cleaned.replace('"', "")
        if cleaned.count("'") % 2 == 1 and "'" not in re.sub(r"[A-Za-z]+'[A-Za-z]+", "", cleaned):
            cleaned = cleaned.replace("'", "")
        if cleaned.count("(") > cleaned.count(")"):
            cleaned = cleaned + (")" * (cleaned.count("(") - cleaned.count(")")))
        elif cleaned.count(")") > cleaned.count("("):
            excess = cleaned.count(")") - cleaned.count("(")
            cleaned = cleaned[::-1].replace(")", "", excess)[::-1]

        return cleaned.strip()

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
    def _sort_chunks_newest_first(cls, chunks: List[Dict]) -> List[Dict]:
        # For compare mode, show the freshest evidence first so the model sees the desired ordering directly.
        def _sort_key(chunk: Dict):
            parsed = cls._parse_effective_date(chunk.get("effective_date"))
            return (parsed is not None, parsed or date.min)

        return sorted(chunks or [], key=_sort_key, reverse=True)

    @classmethod
    def _format_chunks(cls, chunks: List[Dict], compact_text: bool = False) -> str:
        if not chunks:
            return "(No relevant chunks found.)"
        return "\n\n".join(
            [
                f"[Document Title: {cls._get_doc_title(chunk)}] -[Chunk ID: {chunk['chunk_id']}]"
                f"-[Effective Date: {chunk.get('effective_date') or 'N/A'}]"
                f"-[pages: {chunk['pages']}] - [Chunk Content: "
                f"{cls._compact_text(cls._trim_incomplete_trailing_text(chunk.get('text', ''))) if compact_text else cls._trim_incomplete_trailing_text(chunk.get('text', ''))}]"
                for chunk in chunks
            ]
        )

    @staticmethod
    def _build_chunk_lookup(chunks: List[Dict]) -> Dict[str, Dict]:
        lookup = {}
        for chunk in chunks or []:
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if chunk_id:
                lookup[chunk_id] = chunk
        return lookup

    @classmethod
    def _filter_chunks_by_ids(cls, chunks: List[Dict], chunk_ids: List[str]) -> List[Dict]:
        wanted = {str(chunk_id).strip() for chunk_id in chunk_ids if str(chunk_id).strip()}
        if not wanted:
            return []
        return [chunk for chunk in chunks or [] if str(chunk.get("chunk_id", "")).strip() in wanted]

    @classmethod
    def _build_compare_row_generator_prompt(
        cls,
        question: str,
        target_concept: str,
        provider_manual_chunks: List[Dict],
        policy_chunks: List[Dict],
    ) -> str:
        provider_context = cls._format_chunks(provider_manual_chunks, compact_text=False)
        policy_context = cls._format_chunks(policy_chunks, compact_text=False)
        recency_brief = cls._build_compare_recency_brief(policy_chunks, provider_manual_chunks)
        newest_provider = cls._latest_chunk(provider_manual_chunks)
        newest_policy = cls._latest_chunk(policy_chunks)

        return f"""
Question:
{question}

Target concept:
{target_concept}

Deterministic recency facts:
- Newest provider-manual chunk id: {newest_provider.get("chunk_id", "N/A") if newest_provider else "N/A"}
- Newest provider-manual effective date: {newest_provider.get("effective_date", "N/A") if newest_provider else "N/A"}
- Newest policy chunk id: {newest_policy.get("chunk_id", "N/A") if newest_policy else "N/A"}
- Newest policy effective date: {newest_policy.get("effective_date", "N/A") if newest_policy else "N/A"}

{recency_brief}

Provider Manual Context Chunks (anchor side):
{provider_context}

Policy Context Chunks (matching side):
{policy_context}

Chunks format:
[Document Title: <doc_title>] -[Chunk ID: <chunk_id>]-[Effective Date: <effective_date or N/A>]-[pages: <pages>] - [Chunk Content: <chunk_content>]

Task:
Create a point-to-point row plan.

Rules for this request:
- First identify the user's main intent.
- Then break the question into a small number of concrete candidate supporting sub-questions.
- Use the question and retrieved evidence to decide which candidate sub-questions are actually answerable.
- Do not create a row unless the evidence supports it.
- Define each row as one concrete operational sub-question, decision point, requirement, exception, or workflow step.
- Do not substitute a narrower pharmacy subdomain, such as long-term care, vaccines, diabetic supplies, or another special case, for the user's main question unless the question explicitly asks about that subdomain.
- If the evidence does not directly answer the user's main billing pathway or decision point, say that limitation clearly and avoid building rows around narrower examples as if they were the main answer.
- If two chunks share the same broad topic but describe different billing scenarios, workflows, or rules, do not place them in the same row.
- Do not use a broad standalone topic label as the row topic unless the evidence itself is that focused.
- It is acceptable for multiple evidence cues to support the same row if they point to the same sub-question.
- It is acceptable for one broad question area to produce multiple rows if the evidence supports distinct sub-questions.
- Rank those sub-questions by usefulness to the user's main intent.
- Start from provider-manual anchor points.
- Keep only rows that directly help answer the main intent or one of the most useful supporting sub-questions.
- Use the newest directly relevant provider-manual chunk first for the core question.
- Use one primary provider-manual chunk and one primary policy chunk per row.
- Choose the single best provider-manual chunk and the single best policy chunk for that row's sub-question; do not spread one row across multiple evidence topics.
- Keep only same-sub-question rows.
- Row 1 should answer the user's main intent as directly as possible.
- Do not let Row 1 drift into a narrower example or special-case scenario unless that is the clearest direct answer to the user's stated question.
- Later rows should answer different supporting sub-questions.
- For broad questions, prefer the row that best answers the user's main intent, even if one side is only indirect or partial.
- If the provider-manual evidence is narrower or less direct than the policy evidence for the main-intent row, keep the row and state that limitation instead of replacing it with a cleaner but less useful supporting row.
- Before creating a row, ask whether both sides truly answer the same operational sub-question or decision point.
- If you cannot state the shared sub-question in one clear sentence, the row is too broad or mismatched and should be dropped.
- Do not pair rows just because both sides share broad domain language, repeated document terms, overlapping keywords, or the same general subject area.
- Surface-level term overlap is not enough. Both sides must address the same rule, pathway, requirement, exception, or operational scenario.
- For broad questions, usually return 2 to 3 useful rows if the evidence supports them.
- For broad questions, try to return at least 2 useful rows when the evidence supports a core row plus a supporting row.
- One strong row is better than several mismatched rows.
- Do not stop too early if a second directly relevant sub-question would give a fuller answer.
- Prefer one core-answer row plus one or two supporting rows over a single row that leaves the answer too narrow.
- If a possible second or third row is still directly relevant but slightly broader, keep it as partial rather than dropping it.
- Do not keep weak, repetitive, or mixed-topic rows.
- In match_reason, briefly explain what sub-question the row answers and why the two sides belong in the same row.

Return ONLY valid JSON with exactly this schema:
{{
  "headline_summary": "1 to 3 complete sentences that directly answer the question",
  "row_plan": [
    {{
      "row_id": "row_1",
      "topic": "one short label for one concrete sub-question or decision point",
      "provider_manual_chunk_ids": ["exactly_one_primary_chunk_id"],
      "provider_manual_point": "one concise grounded provider-manual point",
      "policy_chunk_ids": ["exactly_one_primary_chunk_id"],
      "policy_match_quality": "strong|partial",
      "policy_point": "one concise grounded policy point",
      "match_reason": "one short same-topic rationale"
    }}
  ]
}}
""".strip()

    @classmethod
    def _parse_compare_row_plan(cls, llm_response: str) -> Dict:
        cleaned = re.sub(r'^```(?:json)?\s*', '', (llm_response or "").strip())
        cleaned = re.sub(r'\s*```$', '', cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {"headline_summary": "", "row_plan": []}

        headline_summary = cls._sanitize_compare_text_field(parsed.get("headline_summary", ""))
        row_plan = parsed.get("row_plan", [])
        if not isinstance(row_plan, list):
            return {"headline_summary": headline_summary, "row_plan": []}

        normalized_rows = []
        for idx, item in enumerate(row_plan, 1):
            if not isinstance(item, dict):
                continue

            provider_ids = item.get("provider_manual_chunk_ids", [])
            policy_ids = item.get("policy_chunk_ids", [])
            if isinstance(provider_ids, str):
                provider_ids = [provider_ids]
            if isinstance(policy_ids, str):
                policy_ids = [policy_ids]

            quality = str(item.get("policy_match_quality", "")).strip().lower()
            if quality not in {"strong", "partial"}:
                quality = "partial"

            normalized_rows.append(
                {
                    "row_id": str(item.get("row_id") or f"row_{idx}").strip(),
                    "topic": str(item.get("topic", "")).strip(),
                    "provider_manual_chunk_ids": [str(v).strip() for v in provider_ids if str(v).strip()],
                    "provider_manual_point": cls._sanitize_compare_text_field(item.get("provider_manual_point", "")),
                    "policy_chunk_ids": [str(v).strip() for v in policy_ids if str(v).strip()],
                    "policy_match_quality": quality,
                    "policy_point": cls._sanitize_compare_text_field(item.get("policy_point", "")),
                    "match_reason": cls._sanitize_compare_text_field(item.get("match_reason", "")),
                }
            )

        return {
            "headline_summary": headline_summary,
            "row_plan": normalized_rows,
        }

    def generate_compare_row_plan(
        self,
        question: str,
        target_concept: str,
        provider_manual_chunks: List[Dict],
        policy_chunks: List[Dict],
        cancel_check=None,
    ) -> Tuple[str, Dict]:
        prompt = self._build_compare_row_generator_prompt(
            question=question,
            target_concept=target_concept,
            provider_manual_chunks=provider_manual_chunks,
            policy_chunks=policy_chunks,
        )
        raw = self.llm_client.chat(
            messages=self._compose_messages(
                user_msg=prompt,
                system_prompt=self.compare_row_generator_system_prompt,
            ),
            temperature=0.1,
            cancel_check=cancel_check,
        )
        return raw, self._parse_compare_row_plan(raw)

    @classmethod
    def _build_compare_fallback_point(cls, chunk: Optional[Dict], max_chars: int = 280) -> str:
        if not isinstance(chunk, dict):
            return ""
        raw = cls._trim_incomplete_trailing_text(chunk.get("text", "") or "")
        raw = " ".join(str(raw or "").split()).strip()
        if not raw:
            return ""

        first_sentence = re.search(r"^(.+?[.!?])(?=\s|$)", raw)
        if first_sentence:
            candidate = first_sentence.group(1).strip()
            if len(candidate) >= 30:
                return cls._sanitize_compare_text_field(candidate)

        candidate = cls._compact_text(raw, max_chars=max_chars).strip()
        if candidate and candidate[-1] not in ".!?":
            candidate = candidate.rstrip(" ,;:") + "."
        return cls._sanitize_compare_text_field(candidate)

    @classmethod
    def _normalize_compare_row_plan(
        cls,
        row_plan: Dict,
        provider_manual_chunks: Optional[List[Dict]] = None,
        policy_chunks: Optional[List[Dict]] = None,
    ) -> Dict:
        rows = row_plan.get("row_plan", []) or []
        headline_summary = cls._sanitize_compare_text_field(row_plan.get("headline_summary", ""))
        provider_rank = {
            str(chunk.get("chunk_id", "")).strip(): idx
            for idx, chunk in enumerate(provider_manual_chunks or [])
            if str(chunk.get("chunk_id", "")).strip()
        }
        provider_lookup = cls._build_chunk_lookup(provider_manual_chunks or [])
        policy_lookup = cls._build_chunk_lookup(policy_chunks or [])

        def _row_provider_rank(row: Dict) -> int:
            ranks = [
                provider_rank.get(str(chunk_id).strip(), 10_000)
                for chunk_id in row.get("provider_manual_chunk_ids", []) or []
                if str(chunk_id).strip()
            ]
            return min(ranks) if ranks else 10_000

        normalized_rows = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_copy = dict(row)
            row_copy["_provider_rank"] = _row_provider_rank(row_copy)
            normalized_rows.append(row_copy)

        normalized_rows.sort(key=lambda row: row.get("_provider_rank", 10_000))
        normalized_plan_rows = []
        used_primary_provider_ids = set()

        for item in normalized_rows:
            if len(normalized_plan_rows) >= 4:
                break
            if not isinstance(item, dict):
                continue
            row = dict(item)
            if not row.get("provider_manual_chunk_ids"):
                continue

            provider_ids = [str(v).strip() for v in row.get("provider_manual_chunk_ids", []) or [] if str(v).strip()]
            policy_ids = [str(v).strip() for v in row.get("policy_chunk_ids", []) or [] if str(v).strip()]
            provider_ids = provider_ids[:1]
            policy_ids = policy_ids[:1]
            if not policy_ids and policy_chunks:
                fallback_policy_chunk = policy_chunks[0]
                fallback_policy_id = str(fallback_policy_chunk.get("chunk_id", "")).strip()
                if fallback_policy_id:
                    policy_ids = [fallback_policy_id]
            if not policy_ids:
                continue

            primary_provider_id = provider_ids[0]
            if primary_provider_id in used_primary_provider_ids:
                continue

            row["provider_manual_chunk_ids"] = provider_ids
            row["policy_chunk_ids"] = policy_ids

            topic = cls._sanitize_compare_text_field(row.get("topic", ""))
            if not topic:
                topic = "Retrieved billing guidance"
            row["topic"] = topic

            if not row.get("provider_manual_point"):
                row["provider_manual_point"] = cls._build_compare_fallback_point(
                    provider_lookup.get(provider_ids[0])
                ) or "Relevant provider-manual guidance was retrieved for this point."
            if not row.get("policy_point"):
                row["policy_point"] = cls._build_compare_fallback_point(
                    policy_lookup.get(policy_ids[0])
                ) or "Relevant policy guidance was retrieved for this point."

            row["policy_match_quality"] = (
                row.get("policy_match_quality")
                if row.get("policy_match_quality") in {"strong", "partial"}
                else "partial"
            )
            row["match_reason"] = cls._sanitize_compare_text_field(row.get("match_reason", ""))
            normalized_plan_rows.append(row)
            used_primary_provider_ids.add(primary_provider_id)

        target_row_count = 3
        if not normalized_plan_rows and provider_manual_chunks and policy_chunks:
            normalized_plan_rows = []
            fallback_limit = min(target_row_count, len(provider_manual_chunks))
            for idx in range(fallback_limit):
                provider_chunk = provider_manual_chunks[idx]
                policy_chunk = policy_chunks[min(idx, len(policy_chunks) - 1)]
                provider_id = str(provider_chunk.get("chunk_id", "")).strip()
                policy_id = str(policy_chunk.get("chunk_id", "")).strip()
                if not provider_id or not policy_id:
                    continue
                normalized_plan_rows.append(
                    {
                        "row_id": f"row_{len(normalized_plan_rows) + 1}",
                        "topic": "Retrieved billing guidance",
                        "provider_manual_chunk_ids": [provider_id],
                        "provider_manual_point": cls._build_compare_fallback_point(provider_chunk)
                        or "Relevant provider-manual guidance was retrieved for this question.",
                        "policy_chunk_ids": [policy_id],
                        "policy_match_quality": "partial",
                        "policy_point": cls._build_compare_fallback_point(policy_chunk)
                        or "Relevant policy guidance was retrieved for this question.",
                        "match_reason": "Kept as an inclusive fallback row from the top retrieved provider-manual and policy chunks.",
                    }
                )

        for idx, row in enumerate(normalized_plan_rows, 1):
            row["row_id"] = f"row_{idx}"

        return {
            "headline_summary": headline_summary,
            "row_plan": normalized_plan_rows,
        }

    @classmethod
    def _build_compare_row_refs(cls, row_plan: Dict, chunk_lookup: Dict[str, Dict]) -> List[Dict]:
        refs = []
        for row in row_plan.get("row_plan", []) or []:
            row_ref = {"provider_manual": [], "policy_update": []}
            for side, ids_key in (
                ("provider_manual", "provider_manual_chunk_ids"),
                ("policy_update", "policy_chunk_ids"),
            ):
                side_refs = []
                for chunk_id in row.get(ids_key, []) or []:
                    chunk = chunk_lookup.get(str(chunk_id).strip())
                    if not chunk:
                        continue
                    side_refs.append(
                        {
                            "chunk_id": str(chunk.get("chunk_id", "")).strip(),
                            "doc_id": str(chunk.get("doc_id", "")).strip(),
                            "title": cls._get_doc_title(chunk),
                            "pages": chunk.get("pages", "N/A"),
                            "effective_date": chunk.get("effective_date") or "N/A",
                        }
                    )
                row_ref[side] = side_refs
            refs.append(row_ref)
        return refs

    @classmethod
    def _build_compare_writer_prompt(
        cls,
        question: str,
        target_concept: str,
        row_plan: Dict,
        provider_manual_chunks: List[Dict],
        policy_chunks: List[Dict],
    ) -> str:
        recency_brief = cls._build_compare_recency_brief(policy_chunks, provider_manual_chunks)
        row_plan_json = json.dumps(row_plan, ensure_ascii=True, indent=2)
        planned_headline = cls._sanitize_compare_text_field(row_plan.get("headline_summary", ""))
        provider_context = cls._format_chunks(provider_manual_chunks, compact_text=False)
        policy_context = cls._format_chunks(policy_chunks, compact_text=False)

        return f"""
Question:
{question}

Target concept:
{target_concept}

{recency_brief}

Planner headline draft:
{planned_headline or "N/A"}

Approved row plan:
{row_plan_json}

Provider Manual Referenced Chunks:
{provider_context}

Policy Referenced Chunks:
{policy_context}

Task:
Write the final comparison from the approved row plan.

Hard rules:
1) Do not invent new rows.
2) Keep the row order from the approved row plan.
3) Keep the provider-manual side as the anchor.
4) If a row is partial, reflect that honestly instead of overstating the match.
5) Keep each row focused on the single operational sub-topic already approved.
6) aligned_pairs should be citation-light because references for the first table are rendered deterministically in the UI.
7) Similarities and differences should still use inline citations when available.
8) Use the planner headline draft as the starting point for "headline_summary". You may improve wording for clarity, but preserve its grounded conclusion.
9) headline_summary must be simple: 1 to 2 short sentences that answer the question directly without summarizing every row.
10) Every sentence must be complete and fully finished.
11) Never end headline_summary, aligned_pairs, similarities, differences, or caveats with a clipped word or incomplete phrase.
12) Each aligned_pairs cell should usually contain 2 to 3 short complete sentences.
13) Sentence 1 should state the core rule.
14) Sentence 2 should add the most important operational detail or limitation.
15) Sentence 3 may add one concise nuance only if it improves the comparison.
16) Prefer multiple short complete sentences over one dense long sentence.
17) Do not make first-table cells so short that they lose the operational detail needed to understand the comparison.
18) When grounded evidence allows it, include one concrete detail such as the billing pathway, member group, code family, field requirement, or exception.
19) Prefer one short complete sentence per similarity or difference bullet.
20) Spell out full words such as "administration", "organization", and "identification". Never shorten them to clipped stems like "administra.".
21) If a field would otherwise end in an incomplete thought, rewrite it as a shorter complete sentence.
22) Before returning JSON, check the final word of every text field. If any field ends with an incomplete word, rewrite that field before returning.
23) Write provider_manual as a direct statement of current guidance from the provider-manual chunk for that row.
24) Write policy_update as the closest matching policy statement for the same row topic, and say plainly when it is broader or less specific.
25) For broad questions, keep headline_summary high-level and avoid summarizing every row.
26) Use bracket citations in similarities and differences, not raw chunk ids or parenthesized file tokens.
27) If the approved row plan is empty, return aligned_pairs as an empty list and explain in caveats that no approved comparison row survived validation, even though chunks may still have been retrieved.

Return ONLY valid JSON with exactly these keys:
{{
  "headline_summary": "1 to 2 short complete sentences that directly answer the user's question",
  "aligned_pairs": [
    {{"provider_manual": "2 to 3 short complete sentences with no inline citations", "policy_update": "2 to 3 short complete sentences with no inline citations"}}
  ],
  "similarities": ["One short complete sentence similarity with inline citation when available"],
  "differences": ["One short complete sentence difference with inline citation when available"],
  "caveats": "Complete sentence caveat, or null if none"
}}
""".strip()

    def write_compare_from_row_plan(
        self,
        question: str,
        target_concept: str,
        row_plan: Dict,
        provider_manual_chunks: List[Dict],
        policy_chunks: List[Dict],
        cancel_check=None,
    ) -> str:
        prompt = self._build_compare_writer_prompt(
            question=question,
            target_concept=target_concept,
            row_plan=row_plan,
            provider_manual_chunks=provider_manual_chunks,
            policy_chunks=policy_chunks,
        )
        return self.llm_client.chat(
            messages=self._compose_messages(
                user_msg=prompt,
                system_prompt=self.compare_writer_system_prompt,
            ),
            cancel_check=cancel_check,
        )

    @classmethod
    def _compare_sections_to_payload(cls, sections: Dict) -> Dict:
        return {
            "headline_summary": sections.get("headline_summary", ""),
            "aligned_pairs": sections.get("aligned_pairs", []) or [],
            "similarities": sections.get("similarities", []) or [],
            "differences": sections.get("differences", []) or [],
            "caveats": sections.get("caveats"),
        }

    @classmethod
    def _build_compare_debug_payload(
        cls,
        planner_raw: str,
        planner_row_plan: Dict,
        normalized_row_plan: Dict,
        writer_raw: str,
        writer_sections: Dict,
    ) -> Dict:
        return {
            "planner_raw": planner_raw,
            "planner_row_plan": planner_row_plan,
            "normalized_row_plan": normalized_row_plan,
            "writer_raw": writer_raw,
            "writer_sections": cls._compare_sections_to_payload(writer_sections or {}),
            "repair_flags_before": [],
            "repair_raw": None,
            "repair_sections": None,
            "repair_flags_after": [],
            "forced_sections": None,
        }

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

    def _compose_messages(self, user_msg: str, system_prompt: Optional[str] = None) -> List[Dict]:
        # Build the model input from already-saved history plus the new user request.
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.chat_history.get_messages())
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = user_msg
        else:
            messages.append({"role": "user", "content": user_msg})
        return messages

    def _finalize_turn(self, question: str, llm_response: str) -> None:
        # This is the "save the new turn" step.
        # We call it only after the answer fully succeeds, so stopped turns never get saved.
        self.chat_history.add("user", question)
        self.chat_history.add("assistant", llm_response)

    def answer_question(
        self,
        question: str,
        top_k: int = 8,
        rerank_top_k: int = 30,
        history: Optional[List[Dict]] = None,
        cancel_check=None,
    ) -> Dict:
        # Full Q&A pipeline.
        # Important rule: keep the current turn temporary until the very end.
        """
        Full question-answering pipeline.

        Steps:
          1. Retrieve relevant chunks (RAG retrieval)
          2. Construct context + prompt
          3. Generate final response using the LLM
        """
        # 0. Smart filter
        if cancel_check and cancel_check():
            raise GenerationCancelled()
        filters = self.filter_extractor.extract(question) if self.filter_extractor else {}
        retrieval_query = (filters.get("retrieval_query") or question).strip()
        normalized_query = (filters.get("normalized_query") or question).strip()
        rerank_query = question

        # 1. Encode query as vector
        # Keep retrieval stable while we iterate on query understanding.
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
            keywords=filters.get("keywords"),
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
        if cancel_check and cancel_check():
            raise GenerationCancelled()
        
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
        messages = self._compose_messages(user_msg=user_msg, system_prompt=self.system_prompt)
        llm_response = self.llm_client.chat(messages=messages, cancel_check=cancel_check)
        if cancel_check and cancel_check():
            raise GenerationCancelled()

        followup_questions = self._generate_followup_questions(
            question=question,
            answer=llm_response,
            retrieved_chunks=final_chunks,
        )
        if cancel_check and cancel_check():
            raise GenerationCancelled()
        self._finalize_turn(question, llm_response)

        return {
            "question": question,
            "answer": llm_response,
            "retrieved_docs": final_chunks,
            "followup_questions": followup_questions,
            "query_understanding": {
                "retrieval_query": retrieval_query,
                "normalized_query": normalized_query,
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
        cancel_check=None,
    ) -> Dict:
        # Full compare pipeline.
        # Same rule as Q&A: keep the current turn temporary until the very end.
        """
        Compare a concept definition across policy vs provider manual documents.

        Retrieval strategy:
          1) Retrieve by doc_classes=['policy']
          2) Retrieve by doc_classes=['provider_manual']
          3) Build dual-source context and ask LLM for side-by-side comparison
        """
        if cancel_check and cancel_check():
            raise GenerationCancelled()
        filters = self.filter_extractor.extract(question) if self.filter_extractor else {}

        resolved_concept = self._resolve_compare_concept(question, concept)
        default_query = resolved_concept or question
        retrieval_query = (filters.get("retrieval_query") or default_query).strip()
        normalized_query = (filters.get("normalized_query") or question).strip()
        rerank_query = question
        query_vec = self._get_embedder().encode([default_query])["dense_vecs"][0].tolist()
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
        if cancel_check and cancel_check():
            raise GenerationCancelled()

        policy_ordered = self._sort_chunks_newest_first(policy_final)
        provider_manual_ordered = self._sort_chunks_newest_first(provider_manual_final)
        # For compare mode, keep chunk text intact so the model does not see chopped evidence mid-sentence.
        target_concept = resolved_concept or "the requested concept"

        planner_raw, planner_row_plan = self.generate_compare_row_plan(
            question=question,
            target_concept=target_concept,
            provider_manual_chunks=provider_manual_ordered,
            policy_chunks=policy_ordered,
            cancel_check=cancel_check,
        )
        row_plan = self._normalize_compare_row_plan(
            planner_row_plan,
            provider_manual_chunks=provider_manual_ordered,
            policy_chunks=policy_ordered,
        )
        if cancel_check and cancel_check():
            raise GenerationCancelled()

        row_provider_ids = []
        row_policy_ids = []
        for row in row_plan.get("row_plan", []) or []:
            row_provider_ids.extend(row.get("provider_manual_chunk_ids", []) or [])
            row_policy_ids.extend(row.get("policy_chunk_ids", []) or [])

        writer_provider_chunks = self._filter_chunks_by_ids(provider_manual_ordered, row_provider_ids)
        writer_policy_chunks = self._filter_chunks_by_ids(policy_ordered, row_policy_ids)
        writer_raw = self.write_compare_from_row_plan(
            question=question,
            target_concept=target_concept,
            row_plan=row_plan,
            provider_manual_chunks=writer_provider_chunks,
            policy_chunks=writer_policy_chunks,
            cancel_check=cancel_check,
        )
        if cancel_check and cancel_check():
            raise GenerationCancelled()

        writer_sections = self._parse_compare_response(writer_raw)
        llm_response = writer_raw
        compare_sections = writer_sections
        if cancel_check and cancel_check():
            raise GenerationCancelled()

        compare_debug = self._build_compare_debug_payload(
            planner_raw=planner_raw,
            planner_row_plan=planner_row_plan,
            normalized_row_plan=row_plan,
            writer_raw=writer_raw,
            writer_sections=writer_sections,
        )

        chunk_lookup = self._build_chunk_lookup(provider_manual_ordered + policy_ordered)
        compare_row_refs = self._build_compare_row_refs(row_plan, chunk_lookup)

        all_chunks = policy_final + provider_manual_final
        followup_questions = self._generate_followup_questions(
            question=question,
            answer=llm_response,
            retrieved_chunks=all_chunks,
            mode="compare",
        )
        if cancel_check and cancel_check():
            raise GenerationCancelled()
        self._finalize_turn(question, llm_response)

        return {
            "question": question,
            "concept": resolved_concept or concept,
            "answer": llm_response,
            "compare_sections": compare_sections,
            "compare_row_plan": row_plan,
            "compare_row_refs": compare_row_refs,
            "compare_debug": compare_debug,
            "followup_questions": followup_questions,
            "query_understanding": {
                "retrieval_query": retrieval_query,
                "normalized_query": normalized_query,
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
                "aligned_pairs": [],
                "similarities": [],
                "differences": [],
                "caveats": None,
                "_parse_failed": True,
            }
        aligned_pairs = parsed.get("aligned_pairs", [])
        if isinstance(aligned_pairs, dict):
            parsed["aligned_pairs"] = [aligned_pairs]
        elif isinstance(aligned_pairs, list):
            normalized_pairs = []
            for item in aligned_pairs:
                if not isinstance(item, dict):
                    continue
                normalized_pairs.append(
                    {
                        "provider_manual": ResponseGenerator._sanitize_compare_text_field(item.get("provider_manual", "")),
                        "policy_update": ResponseGenerator._sanitize_compare_text_field(item.get("policy_update", "")),
                    }
                )
            parsed["aligned_pairs"] = normalized_pairs
        else:
            parsed["aligned_pairs"] = []

        parsed["headline_summary"] = ResponseGenerator._sanitize_compare_text_field(
            parsed.get("headline_summary", "")
        )
        caveat_value = parsed.get("caveats")
        if caveat_value is not None:
            parsed["caveats"] = ResponseGenerator._sanitize_compare_text_field(caveat_value)

        for key in ("policy_definition", "provider_manual_definition", "similarities", "differences"):
            val = parsed.get(key, [])
            if isinstance(val, str):
                parsed[key] = [val] if val else []
            parsed[key] = [
                ResponseGenerator._sanitize_compare_text_field(item)
                for item in parsed.get(key, [])
                if ResponseGenerator._sanitize_compare_text_field(item)
            ]

        if not parsed.get("aligned_pairs"):
            provider_items = parsed.get("provider_manual_definition", [])
            policy_items = parsed.get("policy_definition", [])
            max_len = max(len(provider_items), len(policy_items), 0)
            parsed["aligned_pairs"] = [
                {
                    "provider_manual": provider_items[i] if i < len(provider_items) else "",
                    "policy_update": policy_items[i] if i < len(policy_items) else "",
                }
                for i in range(max_len)
            ]
        return parsed
