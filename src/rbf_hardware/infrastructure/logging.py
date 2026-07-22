"""Shared application logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"


def _prepare_windows_readable_utf8_log(path: Path, file_mode: str) -> None:
    """Keep UTF-8 logs auto-detectable by Windows PowerShell 5 Get-Content."""
    if file_mode == "w":
        path.write_text("", encoding="utf-8-sig")
        return
    if not path.exists() or path.stat().st_size == 0:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("rb") as handle:
        if handle.read(3) == UTF8_BOM:
            return
    existing_text = path.read_text(encoding="utf-8")
    path.write_text(existing_text, encoding="utf-8-sig")


def setup_logging(path: Path, level_name: str = "INFO", file_mode: str = "w") -> logging.Logger:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {level_name}")
    if file_mode not in {"a", "w"}:
        raise ValueError("logging.file_mode must be a or w.")
    path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_windows_readable_utf8_log(path, file_mode)

    logger = logging.getLogger("rbf_hardware")
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    # The preparation step already applies truncate semantics for mode='w'.
    # Append mode preserves the UTF-8 BOM needed by Windows PowerShell 5.
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
