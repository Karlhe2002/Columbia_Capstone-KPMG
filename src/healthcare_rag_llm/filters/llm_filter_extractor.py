# src/healthcare_rag_llm/filters/llm_filter_extractor.py

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Dict, List, Optional

from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import APIConfigManager, get_default_model_for_provider

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

THEME_SLOTS = (
    "actions",
    "actors",
    "organizations",
    "domains",
    "objects",
    "intent",
    "temporal_cues",
)

QUERY_UNDERSTANDING_SYSTEM_PROMPT = (
    "You are a query-understanding component for a healthcare policy RAG system.\n"
    "Your job is to transform a user's natural-language question into a compact retrieval plan.\n"
    "\n"
    "Return ONLY a JSON object with exactly these fields:\n"
    "- \"normalized_query\": concise rewrite optimized for retrieval\n"
    "- \"semantic_keywords\": array of short domain terms or synonyms\n"
    "- \"themes\": object containing arrays for actions, actors, organizations, domains, objects, intent, and temporal_cues\n"
    "- \"search_themes\": array of short retrieval-friendly keywords or phrases derived from the question themes\n"
    "- \"doc_types\": array of allowed document types from the provided candidate list\n"
    "- \"min_publish_date\": YYYY-MM-DD or null\n"
    "- \"max_publish_date\": YYYY-MM-DD or null\n"
    "- \"temporal_focus\": one of \"current\", \"historical\", \"unspecified\"\n"
    "\n"
    "Rules:\n"
    "- Preserve the user's meaning; do not answer the question.\n"
    "- Expand acronyms and shorthand when useful for retrieval.\n"
    "- Prefer specific healthcare-policy terms over conversational phrasing.\n"
    "- Keep doc_types and date bounds as structured fields; do not hide them inside themes.\n"
    "- Themes should capture the main concepts in the question, using short phrases.\n"
    "- Use temporal_cues only for raw time wording like 'today', 'current', or 'during covid'; keep date bounds in the date fields.\n"
    "- search_themes should be useful for later keyword search and may include abbreviations, expansions, and short noun phrases.\n"
    "- Only return doc types from the provided allowed list.\n"
    "- Do not invent exact document titles or authorities unless they are present in the question or hints.\n"
    "- If no useful rewrite is needed, set normalized_query to the original question.\n"
    "- If no semantic keywords are helpful, return an empty array.\n"
    "- If no themes are helpful, return empty arrays for each theme slot and an empty search_themes array.\n"
    "- If no date bounds are implied, set both dates to null.\n"
    "- Use \"current\" when the user asks about latest/current/in effect guidance.\n"
    "- Use \"historical\" when the user clearly asks about a past period or changes over time.\n"
    "- Otherwise use \"unspecified\".\n"
)

QUERY_UNDERSTANDING_USER_PROMPT = (
    "User question:\n"
    "{question}\n"
    "\n"
    "Rule-based hints already extracted:\n"
    "{rule_hints}\n"
    "\n"
    "Allowed doc types:\n"
    "{allowed_doc_types}\n"
    "\n"
    "Return ONLY JSON with this structure:\n"
    "{{\n"
    "  \"normalized_query\": \"...\",\n"
    "  \"semantic_keywords\": [\"...\"],\n"
    "  \"themes\": {{\n"
    "    \"actions\": [\"...\"],\n"
    "    \"actors\": [\"...\"],\n"
    "    \"organizations\": [\"...\"],\n"
    "    \"domains\": [\"...\"],\n"
    "    \"objects\": [\"...\"],\n"
    "    \"intent\": [\"...\"],\n"
    "    \"temporal_cues\": [\"...\"]\n"
    "  }},\n"
    "  \"search_themes\": [\"...\"],\n"
    "  \"doc_types\": [\"...\"],\n"
    "  \"min_publish_date\": \"YYYY-MM-DD\" or null,\n"
    "  \"max_publish_date\": \"YYYY-MM-DD\" or null,\n"
    "  \"temporal_focus\": \"current\" | \"historical\" | \"unspecified\"\n"
    "}}\n"
)

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        cleaned = str(item).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _parse_date_expr(query: str) -> Dict[str, str]:
    q = query.lower()
    out: Dict[str, str] = {}

    m_between = re.search(
        r"between\s+(?P<m1>[a-z]+)?\s*(?P<y1>\d{4})\s+and\s+(?P<m2>[a-z]+)?\s*(?P<y2>\d{4})",
        q,
    )
    if m_between:
        m1, y1 = m_between.group("m1"), m_between.group("y1")
        m2, y2 = m_between.group("m2"), m_between.group("y2")
        mm1 = MONTHS.get(m1[:3], 1) if m1 else 1
        mm2 = MONTHS.get(m2[:3], 12) if m2 else 12
        out["min_effective_date"] = date(int(y1), mm1, 1).isoformat()
        out["max_effective_date"] = date(int(y2), mm2, 28).isoformat()
        return out

    m_after = re.search(r"(after|since|post)\s+([a-z]+)?\s*(\d{4})", q)
    if m_after:
        month = m_after.group(2)
        year = int(m_after.group(3))
        mm = MONTHS.get(month[:3], 1) if month else 1
        out["min_effective_date"] = date(year, mm, 1).isoformat()
        return out

    m_before = re.search(r"(before|until|prior|through)\s+([a-z]+)?\s*(\d{4})", q)
    if m_before:
        month = m_before.group(2)
        year = int(m_before.group(3))
        mm = MONTHS.get(month[:3], 12) if month else 12
        out["max_effective_date"] = date(year, mm, 28).isoformat()
        return out

    year_match = re.findall(r"\b(?:19|20)\d{2}\b", q)
    if year_match:
        year = max(int(y) for y in year_match)
        out["min_effective_date"] = f"{year}-01-01"

    return out


class LLMFilterExtractor:
    """
    Extract a lightweight query plan for retrieval.

    The extractor keeps deterministic rules for high-confidence signals and uses
    an LLM only to supplement semantics such as query rewriting, synonym
    expansion, and date interpretation.
    """

    def __init__(
        self,
        authority_map: Dict[str, str],
        acronym_map: Dict[str, str],
        doc_metadata: List[Dict],
        llm_client: Optional[LLMClient] = None,
    ):
        self.authority_map = {k.lower(): v for k, v in (authority_map or {}).items()}
        self.acronym_map = {k.lower(): v for k, v in (acronym_map or {}).items()}

        self.authorities = {
            str(d.get("authority_abbr", "")).lower(): str(d.get("authority_name", ""))
            for d in (doc_metadata or [])
            if d.get("authority_abbr")
        }
        self.doc_titles = [
            str(d.get("doc_title", "")).lower()
            for d in (doc_metadata or [])
            if d.get("doc_title")
        ]
        self.doc_types = sorted({
            str(d.get("doc_type", "")).lower()
            for d in (doc_metadata or [])
            if d.get("doc_type")
        })

        if llm_client is not None:
            self.llm_client = llm_client
        else:
            try:
                api_config_manager = APIConfigManager()
                cfg = api_config_manager.get_default_config()
                self.llm_client = LLMClient(
                    api_key=cfg.api_key,
                    model=get_default_model_for_provider(cfg.provider),
                    provider=cfg.provider,
                    base_url=cfg.base_url,
                )
            except Exception:
                self.llm_client = None

    @staticmethod
    def _extract_json_dict(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            try:
                raw = raw.get("content", "")
            except Exception:
                return {}
        content = raw.strip()
        if "{" in content and "}" in content:
            content = content[content.find("{"): content.rfind("}") + 1]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _normalize_date(value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if not value or not ISO_DATE_RE.match(value):
            return None
        return value

    @staticmethod
    def _normalize_string_list(values: Any, allowed_values: Optional[set[str]] = None, max_items: int = 8) -> List[str]:
        if not isinstance(values, list):
            return []
        out = []
        for value in values:
            cleaned = str(value).strip().lower()
            if not cleaned:
                continue
            if allowed_values is not None and cleaned not in allowed_values:
                continue
            out.append(cleaned)
            if len(out) >= max_items:
                break
        return _dedupe_preserve_order(out)

    @staticmethod
    def _normalize_themes(value: Any, max_items_per_slot: int = 6) -> Dict[str, List[str]]:
        if not isinstance(value, dict):
            return {}

        out: Dict[str, List[str]] = {}
        for slot in THEME_SLOTS:
            normalized = LLMFilterExtractor._normalize_string_list(
                value.get(slot),
                max_items=max_items_per_slot,
            )
            if normalized:
                out[slot] = normalized
        return out

    @staticmethod
    def _normalize_temporal_focus(value: Any) -> str:
        value = str(value or "").strip().lower()
        if value in {"current", "historical", "unspecified"}:
            return value
        return "unspecified"

    def _extract_rule_filters(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        filters: Dict[str, Any] = {}

        authorities = []
        for abbr, full in self.authority_map.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        for abbr, full in self.authorities.items():
            if abbr in q or full.lower() in q:
                authorities.append(full)
        if authorities:
            filters["authority_names"] = _dedupe_preserve_order(authorities)

        keywords = []
        for acronym, full_term in self.acronym_map.items():
            pattern = r"\b" + re.escape(acronym.lower()) + r"\b"
            if re.search(pattern, q):
                keywords.append(full_term)
        if keywords:
            filters["keywords"] = _dedupe_preserve_order(keywords)

        matched_titles = [title for title in self.doc_titles if title and title in q]
        if matched_titles:
            filters["doc_titles"] = _dedupe_preserve_order(matched_titles)

        matched_types = [doc_type for doc_type in self.doc_types if doc_type and doc_type in q]
        if matched_types:
            filters["doc_types"] = _dedupe_preserve_order(matched_types)

        filters.update(_parse_date_expr(q))
        return filters

    def _extract_query_plan_with_llm(self, query: str, rule_filters: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_client is None:
            return {}

        rule_hints = {
            "authority_names": rule_filters.get("authority_names", []),
            "keywords": rule_filters.get("keywords", []),
            "doc_titles": rule_filters.get("doc_titles", []),
            "doc_types": rule_filters.get("doc_types", []),
            "min_effective_date": rule_filters.get("min_effective_date"),
            "max_effective_date": rule_filters.get("max_effective_date"),
        }
        messages = [
            {"role": "system", "content": QUERY_UNDERSTANDING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QUERY_UNDERSTANDING_USER_PROMPT.format(
                    question=query.strip(),
                    rule_hints=json.dumps(rule_hints, ensure_ascii=True),
                    allowed_doc_types=", ".join(self.doc_types) or "(none)",
                ),
            },
        ]

        try:
            raw = self.llm_client.chat(messages=messages)
        except Exception:
            return {}

        parsed = self._extract_json_dict(raw)
        if not parsed:
            return {}

        themes = self._normalize_themes(parsed.get("themes"))
        semantic_keywords = self._normalize_string_list(parsed.get("semantic_keywords"))

        derived_search_themes = []
        for slot in THEME_SLOTS:
            derived_search_themes.extend(themes.get(slot, []))

        search_themes = _dedupe_preserve_order(
            self._normalize_string_list(parsed.get("search_themes"), max_items=12)
            + semantic_keywords
            + derived_search_themes
            + self._normalize_string_list(rule_filters.get("keywords"), max_items=12)
        )

        return {
            "normalized_query": str(parsed.get("normalized_query", "")).strip(),
            "semantic_keywords": semantic_keywords,
            "themes": themes,
            "search_themes": search_themes,
            "doc_types": self._normalize_string_list(parsed.get("doc_types"), allowed_values=set(self.doc_types)),
            "min_effective_date": self._normalize_date(parsed.get("min_publish_date")),
            "max_effective_date": self._normalize_date(parsed.get("max_publish_date")),
            "temporal_focus": self._normalize_temporal_focus(parsed.get("temporal_focus")),
        }

    @staticmethod
    def _build_retrieval_query(original_query: str, filters: Dict[str, Any], llm_plan: Dict[str, Any]) -> str:
        # Keep retrieval_query conservative so query-understanding improvements do
        # not silently distort the embedding query.
        return (llm_plan.get("normalized_query") or original_query).strip()

    def extract(self, query: str) -> Dict:
        """
        Extract a retrieval-oriented query plan.

        Returned fields remain backward compatible with the old filter extractor
        while adding lightweight understanding outputs such as retrieval_query.
        """
        filters: Dict[str, Any] = self._extract_rule_filters(query)
        llm_plan = self._extract_query_plan_with_llm(query, filters)

        # Keep retrieval behavior stable: LLM-inferred doc types are exposed for
        # debugging only, not enforced as filters yet.
        if llm_plan.get("min_effective_date") and not filters.get("min_effective_date"):
            filters["min_effective_date"] = llm_plan["min_effective_date"]
        if llm_plan.get("max_effective_date") and not filters.get("max_effective_date"):
            filters["max_effective_date"] = llm_plan["max_effective_date"]

        filters["normalized_query"] = llm_plan.get("normalized_query") or query.strip()
        filters["semantic_keywords"] = llm_plan.get("semantic_keywords") or []
        filters["themes"] = llm_plan.get("themes") or {}
        filters["search_themes"] = llm_plan.get("search_themes") or []
        filters["suggested_doc_types"] = llm_plan.get("doc_types") or []
        filters["temporal_focus"] = llm_plan.get("temporal_focus") or "unspecified"
        filters["retrieval_query"] = self._build_retrieval_query(query, filters, llm_plan)

        return filters
