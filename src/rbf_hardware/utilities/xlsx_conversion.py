#!/usr/bin/env python3
"""Convert every worksheet in one or more XLSX files to UTF-8 CSV."""

from __future__ import annotations

import csv
import datetime as dt
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"m": MAIN_NS, "r": REL_NS, "p": PKG_REL_NS}
DATE_FORMAT_IDS = set(range(14, 23)) | set(range(27, 37)) | set(range(45, 48)) | set(range(50, 59)) | {164}
CELL_REF = re.compile(r"([A-Z]+)(\d+)")
INVALID_FILENAME = re.compile(r"[<>:\"/\\|?*]")


def column_index(cell_ref: str) -> int:
    match = CELL_REF.fullmatch(cell_ref)
    if not match:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - 64
    return value - 1


def shared_strings(book: zipfile.ZipFile) -> list[str]:
    path = "xl/sharedStrings.xml"
    if path not in book.namelist():
        return []
    result: list[str] = []
    with book.open(path) as stream:
        for event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == f"{{{MAIN_NS}}}si":
                result.append("".join(node.text or "" for node in elem.iter(f"{{{MAIN_NS}}}t")))
                elem.clear()
    return result


def date_style_indexes(book: zipfile.ZipFile) -> set[int]:
    path = "xl/styles.xml"
    if path not in book.namelist():
        return set()
    root = ET.parse(book.open(path)).getroot()
    custom_date_ids: set[int] = set()
    for fmt in root.findall("m:numFmts/m:numFmt", NS):
        fmt_id = int(fmt.attrib["numFmtId"])
        code = re.sub(r'"[^"]*"|\\.|\[[^]]*\]', "", fmt.attrib.get("formatCode", ""))
        if re.search(r"[ymdhis]", code, flags=re.I):
            custom_date_ids.add(fmt_id)
    date_ids = DATE_FORMAT_IDS | custom_date_ids
    cell_xfs = root.find("m:cellXfs", NS)
    if cell_xfs is None:
        return set()
    return {
        index
        for index, xf in enumerate(cell_xfs)
        if int(xf.attrib.get("numFmtId", "0")) in date_ids
    }


def worksheet_paths(book: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook = ET.parse(book.open("xl/workbook.xml")).getroot()
    rels = ET.parse(book.open("xl/_rels/workbook.xml.rels")).getroot()
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("p:Relationship", NS)
    }
    sheets: list[tuple[str, str]] = []
    for sheet in workbook.findall("m:sheets/m:sheet", NS):
        target = targets[sheet.attrib[f"{{{REL_NS}}}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        sheets.append((sheet.attrib["name"], target))
    return sheets


def excel_datetime(value: str, date_1904: bool) -> str:
    number = float(value)
    origin = dt.datetime(1904, 1, 1) if date_1904 else dt.datetime(1899, 12, 30)
    converted = origin + dt.timedelta(days=number)
    if converted.time() == dt.time():
        return converted.date().isoformat()
    if converted.date() == origin.date():
        return converted.time().isoformat(timespec="seconds")
    return converted.isoformat(sep=" ", timespec="seconds")


def cell_value(cell: ET.Element, strings: list[str], date_styles: set[int], date_1904: bool) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        inline = cell.find("m:is", NS)
        return "" if inline is None else "".join(node.text or "" for node in inline.iter(f"{{{MAIN_NS}}}t"))
    value_node = cell.find("m:v", NS)
    value = "" if value_node is None or value_node.text is None else value_node.text
    if cell_type == "s" and value:
        return strings[int(value)]
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    if cell_type == "e":
        return value
    style_index = int(cell.attrib.get("s", "0"))
    if value and style_index in date_styles:
        try:
            return excel_datetime(value, date_1904)
        except ValueError:
            pass
    return value


def convert_sheet(
    book: zipfile.ZipFile,
    sheet_path: str,
    destination: Path,
    strings: list[str],
    date_styles: set[int],
    date_1904: bool,
) -> tuple[int, int]:
    max_columns = 0
    row_count = 0
    with book.open(sheet_path) as stream, destination.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        for event, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag != f"{{{MAIN_NS}}}row":
                continue
            row: list[str] = []
            for cell in elem.findall("m:c", NS):
                index = column_index(cell.attrib["r"])
                if index >= len(row):
                    row.extend([""] * (index + 1 - len(row)))
                row[index] = cell_value(cell, strings, date_styles, date_1904)
            writer.writerow(row)
            row_count += 1
            max_columns = max(max_columns, len(row))
            elem.clear()
    return row_count, max_columns


def convert_workbook(source: Path, output_dir: Path) -> list[tuple[Path, str, int, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[tuple[Path, str, int, int]] = []
    with zipfile.ZipFile(source) as book:
        strings = shared_strings(book)
        date_styles = date_style_indexes(book)
        workbook = ET.parse(book.open("xl/workbook.xml")).getroot()
        workbook_props = workbook.find("m:workbookPr", NS)
        date_1904 = workbook_props is not None and workbook_props.attrib.get("date1904") in {"1", "true"}
        sheets = worksheet_paths(book)
        multiple_sheets = len(sheets) > 1
        for sheet_name, sheet_path in sheets:
            safe_sheet = INVALID_FILENAME.sub("_", sheet_name).strip(" .") or "Sheet"
            filename = f"{source.stem}__{safe_sheet}.csv" if multiple_sheets else f"{source.stem}.csv"
            destination = output_dir / filename
            rows, columns = convert_sheet(book, sheet_path, destination, strings, date_styles, date_1904)
            converted.append((destination, sheet_name, rows, columns))
    return converted

