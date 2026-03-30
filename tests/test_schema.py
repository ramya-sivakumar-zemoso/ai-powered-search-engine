"""Tests for dataset schema transforms, apply logic, and registry."""

import pytest

from src.models.dataset_schema import TRANSFORMS, DatasetSchema, FieldMapping
from src.models.schema_registry import get_schema


# ── Transform Functions ─────────────────────────────────────────────────────


@pytest.mark.parametrize("fn, inp, expected", [
    ("identity",          "hello",       "hello"),
    ("list_first",        ["a", "b"],    "a"),
    ("list_first",        [],            ""),
    ("list_first",        "solo",        "solo"),
    ("list_join",         ["a", "b"],    "a, b"),
    ("list_join",         "solo",        "solo"),
    ("to_float",          "$1,234.56",   1234.56),
    ("to_float",          "bad",         0.0),
    ("to_bool",           True,          True),
    ("to_bool",           "yes",         True),
    ("to_bool",           "no",          False),
    ("to_str",            "  hi  ",      "hi"),
    ("to_str",            None,          ""),
    ("to_int",            "42",          42),
    ("to_int",            "bad",         0),
    ("str_truncate_2000", "x" * 3000,    "x" * 2000),
])
def test_transforms(fn, inp, expected):
    assert TRANSFORMS[fn](inp) == expected


# ── Schema.apply ────────────────────────────────────────────────────────────


def test_schema_apply_normal():
    schema = DatasetSchema(
        name="t",
        id_field="id",
        field_mappings={
            "title": FieldMapping(source_fields=["name"], transform="to_str"),
        },
    )
    doc = schema.apply({"id": "1", "name": "Good Title"}, 0)
    assert doc is not None
    assert doc["id"] == "1"
    assert doc["title"] == "Good Title"


def test_schema_apply_skips_short_title():
    schema = DatasetSchema(
        name="t",
        id_field="id",
        field_mappings={
            "title": FieldMapping(source_fields=["name"], transform="to_str"),
        },
    )
    assert schema.apply({"id": "1", "name": "Hi"}, 0) is None


def test_schema_apply_auto_generates_id():
    schema = DatasetSchema(
        name="t",
        id_field="id",
        field_mappings={
            "title": FieldMapping(source_fields=["name"], transform="to_str"),
        },
    )
    doc = schema.apply({"name": "Test Movie"}, 0)
    assert doc is not None
    assert "id" in doc and len(doc["id"]) > 0


def test_unix_to_iso_invalid_input():
    result = TRANSFORMS["unix_to_iso"]("bad")
    assert "T" in result


# ── Schema Registry ────────────────────────────────────────────────────────


def test_get_schema_known():
    assert get_schema("movies").name == "movies"


def test_get_schema_unknown():
    with pytest.raises(ValueError, match="Unknown dataset schema"):
        get_schema("nonexistent")
