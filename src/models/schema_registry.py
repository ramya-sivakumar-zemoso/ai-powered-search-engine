from src.models.dataset_schema import DatasetSchema, FieldMapping

MOVIES_SCHEMA = DatasetSchema(
    name="movies",
    description="TMDB movie dataset — 31,944 records with title, overview, genres",
    id_field="id",
    field_mappings={
        "title": FieldMapping(
            source_fields=["title"],
            transform="to_str",
            default="Untitled",
        ),
        "description": FieldMapping(
            source_fields=["overview"],
            transform="str_truncate_2000",
            default="",
        ),
        "category": FieldMapping(
            source_fields=["genres"],
            transform="list_first",   # first genre becomes the primary category
            default="Uncategorized",
        ),
        "genres_all": FieldMapping(
            source_fields=["genres"],
            transform="list_join",    # all genres as comma string for search
            default="",
        ),
        "brand": FieldMapping(
            source_fields=[],         # no studio/brand field in this dataset
            default="Unknown",
        ),
        "price": FieldMapping(
            source_fields=[],
            transform="to_float",
            default=0.0,
        ),
        "currency": FieldMapping(
            source_fields=[],
            default="USD",
        ),
        "rating": FieldMapping(
            source_fields=[],         # no rating in this dataset
            transform="to_float",
            default=0.0,
        ),
        "in_stock": FieldMapping(
            source_fields=[],
            default=True,
        ),
        "indexed_at": FieldMapping(
            source_fields=["release_date"],
            transform="to_int",       # unix timestamp already
            default=0,
        ),
        "indexed_at_iso": FieldMapping(
            source_fields=["release_date"],
            transform="unix_to_iso",
            default="",
        ),
        "poster": FieldMapping(
            source_fields=["poster"],
            transform="to_str",
            default="",
        ),
    },
    extra_passthrough_fields=["poster"],
    searchable_fields=["title", "description", "genres_all", "category"],
    filterable_fields=["category", "genres_all", "in_stock"],
    sortable_fields=["indexed_at", "rating"],
    embedder_template="{{doc.title}} {{doc.description}} {{doc.genres_all}}",
)


# ── Registry ──────────────────────────────────────────────────────────────────

SCHEMA_REGISTRY: dict[str, DatasetSchema] = {
    "movies": MOVIES_SCHEMA,
}


def get_schema(name: str) -> DatasetSchema:
    """
    Retrieve a schema by name.
    Raises ValueError with a helpful message if the name is unknown.
    """
    if name not in SCHEMA_REGISTRY:
        available = ", ".join(SCHEMA_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset schema '{name}'. "
            f"Available schemas: {available}. "
            f"To add a new one, define a DatasetSchema in schema_registry.py "
            f"and add it to SCHEMA_REGISTRY."
        )
    return SCHEMA_REGISTRY[name]