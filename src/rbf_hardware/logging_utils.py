from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(path: Path, level_name: str = "INFO", file_mode: str = "w") -> logging.Logger:
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {level_name}")
    if file_mode not in {"a", "w"}:
        raise ValueError("logging.file_mode must be a or w.")
    path.parent.mkdir(parents=True, exist_ok=True)

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
    file_handler = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

