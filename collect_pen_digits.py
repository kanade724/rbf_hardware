"""Launch the Pen Digits drawing collector."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rbf_hardware.configuration.settings import (
    load_config,
    resolve_inference_paths,
)
from rbf_hardware.data.csv_store import NumericCsvStore
from rbf_hardware.infrastructure.logging import setup_logging
from rbf_hardware.ui.pen_digits_collector import PenDigitsCollector


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect append-only 16-value Pen Digits samples from mouse strokes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Hardware project configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the configured raw-sample CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    paths = resolve_inference_paths(config)
    output_path = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else paths.raw_samples_file
    )
    logger = setup_logging(
        paths.log_file,
        level_name=str(config["logging"]["level"]),
        file_mode="a",
    )
    logger.info("[采集] Pen Digits绘图采集器已启动，共享文件=%s", output_path)
    sample_store = NumericCsvStore(
        output_path,
        column_count=int(config["inference"]["input_features"]),
    )
    root = tk.Tk()
    PenDigitsCollector(root, sample_store, logger)
    root.mainloop()


if __name__ == "__main__":
    main()
