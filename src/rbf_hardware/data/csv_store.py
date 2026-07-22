"""Validated append-only CSV storage for the streaming inference pipeline."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


class CsvRowError(ValueError):
    """Raised when an append-only CSV contains a malformed or partial row."""


class NumericCsvStore:
    def __init__(self, path: Path, column_count: int) -> None:
        if column_count <= 0:
            raise ValueError("column_count must be positive.")
        self.path = path.expanduser().resolve()
        self.column_count = column_count

    def read_rows(self, start_index: int = 0) -> np.ndarray:
        if start_index < 0:
            raise ValueError("start_index must be non-negative.")
        if not self.path.exists():
            return np.empty((0, self.column_count), dtype=np.float32)

        parsed_rows: list[list[float]] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row_index, row in enumerate(csv.reader(handle)):
                if not row:
                    raise CsvRowError(f"{self.path}: row {row_index + 1} is empty.")
                if len(row) != self.column_count:
                    raise CsvRowError(
                        f"{self.path}: row {row_index + 1} has {len(row)} columns; "
                        f"expected {self.column_count}."
                    )
                if row_index < start_index:
                    continue
                try:
                    values = [float(value) for value in row]
                except ValueError as error:
                    raise CsvRowError(
                        f"{self.path}: row {row_index + 1} contains a non-numeric value."
                    ) from error
                if not np.isfinite(values).all():
                    raise CsvRowError(
                        f"{self.path}: row {row_index + 1} contains NaN or infinity."
                    )
                parsed_rows.append(values)

        if not parsed_rows:
            return np.empty((0, self.column_count), dtype=np.float32)
        return np.asarray(parsed_rows, dtype=np.float32)

    def row_count(self) -> int:
        return int(len(self.read_rows()))

    def append_rows(self, rows: np.ndarray | Sequence[Sequence[float]]) -> int:
        values = np.asarray(rows, dtype=np.float64)
        values = np.atleast_2d(values)
        if values.shape[1] != self.column_count:
            raise ValueError(
                f"Cannot append shape {values.shape} to {self.column_count}-column CSV."
            )
        if not np.isfinite(values).all():
            raise ValueError("Cannot append NaN or infinity to a numeric CSV.")
        if len(values) == 0:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerows(values.tolist())
            handle.flush()
            os.fsync(handle.fileno())
        return int(len(values))


class CsvRecordStore:
    def __init__(self, path: Path, header: Sequence[str]) -> None:
        if not header or len(set(header)) != len(header):
            raise ValueError("CSV record header must contain unique column names.")
        self.path = path.expanduser().resolve()
        self.header = tuple(header)

    def row_count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return 0
        if tuple(rows[0]) != self.header:
            raise CsvRowError(
                f"{self.path}: header {rows[0]} does not match expected {list(self.header)}."
            )
        for row_index, row in enumerate(rows[1:], start=2):
            if len(row) != len(self.header):
                raise CsvRowError(
                    f"{self.path}: row {row_index} has {len(row)} columns; "
                    f"expected {len(self.header)}."
                )
        return len(rows) - 1

    def append_records(self, records: Iterable[Sequence[object]]) -> int:
        prepared = [tuple(record) for record in records]
        for record in prepared:
            if len(record) != len(self.header):
                raise ValueError(
                    f"Record has {len(record)} values; expected {len(self.header)}."
                )
        if not prepared:
            return 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if write_header:
                writer.writerow(self.header)
            writer.writerows(prepared)
            handle.flush()
            os.fsync(handle.fileno())
        return len(prepared)
