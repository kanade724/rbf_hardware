#!/usr/bin/env python3
"""Split alternating CSV rows into device-input and device-output datasets."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        prefix = source.stem.removesuffix("_dataset")
        input_path = args.output_dir / f"{prefix}_in_dataset.csv"
        output_path = args.output_dir / f"{prefix}_out_dataset.csv"
        input_rows, output_rows, columns = split_dataset(source, input_path, output_path)
        print(
            f"{source.name}\tinput={input_rows}\toutput={output_rows}\t"
            f"columns={columns}\t{input_path}\t{output_path}"
        )


if __name__ == "__main__":
    main()
