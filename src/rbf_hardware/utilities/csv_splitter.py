#!/usr/bin/env python3
"""Split alternating CSV rows into device-input and device-output datasets."""

from __future__ import annotations

import csv
from pathlib import Path


def split_dataset(source: Path, input_path: Path, output_path: Path) -> tuple[int, int, int]:
    input_rows = 0
    output_rows = 0
    column_count: int | None = None

    with (
        source.open("r", encoding="utf-8-sig", newline="") as source_file,
        input_path.open("w", encoding="utf-8-sig", newline="") as input_file,
        output_path.open("w", encoding="utf-8-sig", newline="") as output_file,
    ):
        reader = csv.reader(source_file)
        input_writer = csv.writer(input_file, lineterminator="\n")
        output_writer = csv.writer(output_file, lineterminator="\n")

        for row_number, row in enumerate(reader, start=1):
            if column_count is None:
                column_count = len(row)
            elif len(row) != column_count:
                raise ValueError(
                    f"{source}: row {row_number} has {len(row)} columns; expected {column_count}"
                )

            if row_number % 2 == 1:
                input_writer.writerow(row)
                input_rows += 1
            else:
                output_writer.writerow(row)
                output_rows += 1

    return input_rows, output_rows, column_count or 0

