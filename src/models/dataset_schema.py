"""Map external datasets onto the internal document shape via FieldMapping + TRANSFORMS."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field


def _identity(v: Any) -> Any:
    return v


def _list_first(v: Any) -> str:
    if isinstance(v, list):
        return str(v[0]) if v else ""
    return str(v) if v else ""


def _list_join(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x)
    return str(v) if v else ""


def _to_float(v: Any) -> float:
    try:
        return round(float(str(v).replace("$", "").replace(",", "").strip()), 2)
    except (ValueError, TypeError):
        return 0.0


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "available", "in stock")


def _unix_to_iso(v: Any) -> str:
    try:
        return datetime.fromtimestamp(float(v), tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return datetime.now(timezone.utc).isoformat()


def _str_truncate_2000(v: Any) -> str:
    return str(v)[:2000] if v else ""


def _to_str(v: Any) -> str:
    return str(v).strip() if v else ""


def _to_int(v: Any) -> int:
    try:
        return int(float(str(v)))
    except (ValueError, TypeError):
        return 0


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "identity": _identity,
    "list_first": _list_first,
    "list_join": _list_join,
    "to_float": _to_float,
    "to_bool": _to_bool,
    "unix_to_iso": _unix_to_iso,
    "str_truncate_2000": _str_truncate_2000,
    "to_str": _to_str,
    "to_int": _to_int,
}


class FieldMapping(BaseModel):
    source_fields: list[str] = Field(default_factory=list)
    transform: str = "identity"
    default: Any = None


_PLACEHOLDER_DESCRIPTIONS = frozenset({
    "", "no overview found.", "no overview found", "no description",
})


class DatasetSchema(BaseModel):
    name: str
    description: str = ""
    id_field: str = "id"
    field_mappings: dict[str, FieldMapping] = Field(default_factory=dict)
    extra_passthrough_fields: list[str] = Field(default_factory=list)
    searchable_fields: list[str] = Field(
        default=["title", "description", "brand", "category"]
    )
    filterable_fields: list[str] = Field(
        default=["category", "brand", "in_stock"]
    )
    sortable_fields: list[str] = Field(default=["indexed_at"])
    embedder_template: str = "{{doc.title}} {{doc.description}}"
    description_fallback_template: str = ""

    # ── Search / Meilisearch (per-domain) ─────────────────────────────────
    query_stop_words_extra: list[str] = Field(
        default_factory=list,
        description="Merged with English baseline stop words for keyword-overlap logic.",
    )
    filter_aliases_llm: dict[str, str] = Field(
        default_factory=dict,
        description="Lowercased LLM filter keys → Meilisearch attribute names.",
    )
    keyword_overlap_fields: list[str] = Field(
        default_factory=lambda: ["title", "description"],
        description="Hit fields scanned for semantic-degradation keyword overlap.",
    )

    # ── Reranker / citations ───────────────────────────────────────────────
    rerank_auxiliary_fields: list[str] = Field(
        default_factory=lambda: ["category", "description"],
        description="source_fields keys appended to cross-encoder text (order matters).",
    )
    rerank_field_labels: dict[str, str] = Field(
        default_factory=dict,
        description="Optional display labels for auxiliary fields in cross-encoder text.",
    )
    citation_tag_fields: list[str] = Field(
        default_factory=list,
        description="Fields treated as comma-separated tokens for citation matching.",
    )
    evaluator_signal_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "semantic_relevance": 0.366,
            "result_coverage": 0.268,
            "ranking_stability": 0.122,
            "freshness_signal": 0.244,
        },
        description=(
            "Pre-rerank evaluator weights for semantic_relevance/result_coverage/"
            "ranking_stability/freshness_signal. Values are normalized at runtime."
        ),
    )

    # ── Intent LLM ─────────────────────────────────────────────────────────
    intent_parse_appendix: str = Field(
        default="",
        description="Appended to the base intent prompt for domain-specific examples.",
    )

    # ── Demo / UI ──────────────────────────────────────────────────────────
    demo_queries: dict[str, str] | None = Field(
        default=None,
        description="If set, overrides CLI demo queries for this schema.",
    )
    ui_product_title: str = Field(
        default="Search",
        description="Short product name for Streamlit / CLI headers.",
    )
    ui_product_subtitle: str = Field(
        default="Find results",
        description="One-line subtitle under the product title.",
    )
    ui_query_placeholder: str = Field(
        default="What are you looking for?",
        description="Placeholder for the search box.",
    )
    ui_image_field: str | None = Field(
        default="poster",
        description="source_fields key for thumbnail; None to hide image column.",
    )
    ui_tag_fields: list[str] = Field(
        default_factory=lambda: ["category"],
        description="Ordered list: first non-empty field supplies facet tags (comma-split).",
    )

    def meilisearch_attributes_to_retrieve(self) -> list[str]:
        """Attributes for Meilisearch ``attributesToRetrieve`` for this index."""
        names: set[str] = {"id"}
        names.update(self.field_mappings.keys())
        names.update(self.extra_passthrough_fields)
        return sorted(names)

    def normalized_evaluator_weights(self) -> dict[str, float]:
        """Return normalized 4-signal evaluator weights for this schema."""
        keys = (
            "semantic_relevance",
            "result_coverage",
            "ranking_stability",
            "freshness_signal",
        )
        raw = {
            k: max(float(self.evaluator_signal_weights.get(k, 0.0)), 0.0)
            for k in keys
        }
        total = sum(raw.values())
        if total <= 0:
            return {
                "semantic_relevance": 0.366,
                "result_coverage": 0.268,
                "ranking_stability": 0.122,
                "freshness_signal": 0.244,
            }
        return {k: raw[k] / total for k in keys}

    def apply(self, raw: dict[str, Any], row_index: int) -> dict[str, Any] | None:
        doc: dict[str, Any] = {}
        for internal_field, mapping in self.field_mappings.items():
            doc[internal_field] = self._extract(raw, mapping)

        if self.id_field and raw.get(self.id_field) is not None:
            doc["id"] = str(raw[self.id_field])
        elif not doc.get("id"):
            doc["id"] = hashlib.md5(
                f"{doc.get('title', '')}_{row_index}".encode()
            ).hexdigest()

        if self._should_skip_short_title(raw, doc):
            return None

        self._apply_description_fallback(doc)

        if not doc.get("indexed_at"):
            offset = int(hashlib.md5(str(row_index).encode()).hexdigest(), 16) % 21600
            doc["indexed_at"] = int(time.time()) - offset
        if not doc.get("indexed_at_iso"):
            doc["indexed_at_iso"] = datetime.fromtimestamp(
                doc["indexed_at"], tz=timezone.utc
            ).isoformat()

        for field in self.extra_passthrough_fields:
            if field in raw and field not in doc:
                doc[field] = raw[field]

        return doc

    def _apply_description_fallback(self, doc: dict[str, Any]) -> None:
        """Fill empty or placeholder descriptions using the fallback template.

        Prevents degenerate embeddings for documents whose description field
        is empty — the embedder would otherwise encode only the title and
        genre, producing a generic vector near the centroid.
        """
        if not self.description_fallback_template:
            return
        desc = str(doc.get("description", "")).strip()
        if desc.lower() in _PLACEHOLDER_DESCRIPTIONS:
            doc["description"] = self.description_fallback_template.format_map(
                {k: v for k, v in doc.items() if isinstance(v, str)}
            )

    def _should_skip_short_title(self, raw: dict[str, Any], doc: dict[str, Any]) -> bool:
        title_mapping = self.field_mappings.get("title")
        if title_mapping:
            for sf in title_mapping.source_fields:
                v = raw.get(sf)
                if v and str(v).strip():
                    return len(str(v).strip()) < 3
            return True
        return len(str(doc.get("title", "")).strip()) < 3

    def _extract(self, raw: dict[str, Any], mapping: FieldMapping) -> Any:
        transform_fn = TRANSFORMS.get(mapping.transform, _identity)
        for field_name in mapping.source_fields:
            value = raw.get(field_name)
            if value is not None and value != "" and value != []:
                return transform_fn(value)
        if mapping.default is not None:
            return (
                transform_fn(mapping.default)
                if mapping.transform != "identity"
                else mapping.default
            )
        return None
