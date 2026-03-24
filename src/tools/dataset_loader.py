"""Load JSON / JSONL / CSV / TSV and normalise rows with a DatasetSchema."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.models.dataset_schema import DatasetSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_and_normalise(
    file_path: str | Path,
    schema: DatasetSchema,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    logger.info("loading_dataset", extra={"file": str(path), "schema": schema.name})
    raw_records = _load_raw(path, limit)
    logger.info("raw_records_loaded", extra={"count": len(raw_records)})

    documents: list[dict[str, Any]] = []
    skipped = 0
    for idx, raw in enumerate(tqdm(raw_records, desc=f"Normalising ({schema.name})")):
        doc = schema.apply(raw, idx)
        if doc is None:
            skipped += 1
            continue
        documents.append(doc)

    logger.info(
        "normalisation_complete",
        extra={
            "total_raw": len(raw_records),
            "normalised": len(documents),
            "skipped": skipped,
            "schema": schema.name,
        },
    )
    return documents


def _load_raw(path: Path, limit: int | None) -> list[dict[str, Any]]:
    ext = path.suffix.lower()
    if ext == ".json":
        return _load_json(path, limit)
    if ext == ".jsonl":
        return _load_jsonl(path, limit)
    if ext in (".csv", ".tsv"):
        return _load_csv(path, limit, sep="\t" if ext == ".tsv" else ",")
    raise ValueError(
        f"Unsupported extension {ext!r}. Use .json, .jsonl, .csv, or .tsv"
    )


def _load_json(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data[:limit] if limit else data


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("jsonl_parse_error", extra={"line": i + 1, "error": str(e)})
                continue
            if limit and len(records) >= limit:
                break
    return records


def _load_csv(path: Path, limit: int | None, sep: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        df = pd.read_csv(path, sep=sep, nrows=limit, low_memory=False)
        df = df.where(df.notna(), None)
        return df.to_dict(orient="records")
    except ImportError:
        records = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for i, row in enumerate(reader):
                records.append(dict(row))
                if limit and i + 1 >= limit:
                    break
        return records
