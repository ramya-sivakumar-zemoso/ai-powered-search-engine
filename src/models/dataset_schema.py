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
