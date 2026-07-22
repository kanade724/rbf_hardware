"""Migration of the 16-sheet Gaussian workbook to one normalized long CSV."""

from __future__ import annotations

import csv
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .xlsx_conversion import (
    MAIN_NS,
    NS,
    cell_value,
    date_style_indexes,
    shared_strings,
    worksheet_paths,
)


CALIBRATION_HEADER = (
    "group_index",
    "differential_level",
    "amplitude",
    "mean",
    "std_dev",
)


def _read_sheet_rows(book: zipfile.ZipFile, sheet_path: str) -> list[list[str]]:
    strings = shared_strings(book)
    date_styles = date_style_indexes(book)
    workbook = ET.parse(book.open("xl/workbook.xml")).getroot()
    workbook_properties = workbook.find("m:workbookPr", NS)
    date_1904 = (
        workbook_properties is not None
        and workbook_properties.attrib.get("date1904") in {"1", "true"}
    )
    rows: list[list[str]] = []
    with book.open(sheet_path) as stream:
        for _event, element in ET.iterparse(stream, events=("end",)):
            if element.tag != f"{{{MAIN_NS}}}row":
                continue
            row = [
                cell_value(cell, strings, date_styles, date_1904)
                for cell in element.findall("m:c", NS)
            ]
            rows.append(row)
            element.clear()
    return rows


def load_differential_levels(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["differential_level"]:
            raise ValueError(
                f"{path} must contain exactly one 'differential_level' column."
            )
        levels = [row["differential_level"] for row in reader]
    if not levels:
        raise ValueError(f"Differential-level CSV is empty: {path}")
    return levels


def migrate_gaussian_workbook(
    workbook_path: Path,
    differential_levels_path: Path,
    destination_path: Path,
    *,
    expected_groups: int = 16,
) -> tuple[int, int]:
    levels = load_differential_levels(differential_levels_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_suffix(destination_path.suffix + ".tmp")

    written_rows = 0
    with zipfile.ZipFile(workbook_path) as book, temporary_path.open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        sheets = worksheet_paths(book)
        expected_names = [f"Group_{index}" for index in range(1, expected_groups + 1)]
        actual_names = [name for name, _path in sheets]
        if actual_names != expected_names:
            raise ValueError(
                f"Expected worksheets {expected_names}; found {actual_names}."
            )

        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CALIBRATION_HEADER)
        for group_index, (_sheet_name, sheet_path) in enumerate(sheets, start=1):
            sheet_rows = _read_sheet_rows(book, sheet_path)
            data_rows = sheet_rows[1:]
            if len(data_rows) != len(levels):
                raise ValueError(
                    f"Group_{group_index} has {len(data_rows)} rows; expected {len(levels)}."
                )
            for level, source_row in zip(levels, data_rows):
                if len(source_row) < 4:
                    raise ValueError(
                        f"Group_{group_index} contains a row with fewer than four columns."
                    )
                writer.writerow(
                    (group_index, level, source_row[1], source_row[2], source_row[3])
                )
                written_rows += 1
        handle.flush()
        os.fsync(handle.fileno())

    temporary_path.replace(destination_path)
    return expected_groups, written_rows
