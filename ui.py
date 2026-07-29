"""Launch the unified Pen Digits collection and hardware-inference GUI."""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from dataclasses import replace
from pathlib import Path
from tkinter import messagebox


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rbf_hardware.configuration.settings import load_config, resolve_inference_paths
from rbf_hardware.data.csv_store import NumericCsvStore
from rbf_hardware.infrastructure.logging import setup_logging
from rbf_hardware.inference.pipeline import StreamingInferencePipeline
from rbf_hardware.ui.unified_application import UnifiedPenDigitsApplication
from rbf_hardware.ui.windowing import enable_windows_high_dpi


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open the unified Pen Digits scientific GUI for drawing, collection, "
            "manual hardware inference, and result presentation."
        )
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
        help="Override the configured raw-sample CSV path used by the GUI pipeline.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("empirical", "mean"),
        default=None,
        help="Override empirical hardware-response sampling.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    configured_paths = resolve_inference_paths(config)
    output_path = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else configured_paths.raw_samples_file
    )
    paths = replace(
        configured_paths,
        raw_samples_file=output_path,
    )
    logger = setup_logging(
        paths.log_file,
        level_name=str(config["logging"]["level"]),
        file_mode="a",
    )
    sampling_mode = arguments.sampling_mode or str(config["inference"]["sampling_mode"])
    logger.info(
        "[GUI] Pen Digits统一科研实验台已启动，原始数据=%s，采样模式=%s",
        output_path,
        sampling_mode,
    )
    sample_store = NumericCsvStore(
        output_path,
        column_count=int(config["inference"]["input_features"]),
    )
    enable_windows_high_dpi()
    root = tk.Tk()
    try:
        pipeline = StreamingInferencePipeline.build(
            config,
            paths,
            logger,
            sampling_mode=sampling_mode,
        )
    except Exception as error:
        logger.exception("[GUI] 初始化推理流水线失败")
        root.withdraw()
        messagebox.showerror("Initialization Failed", str(error))
        root.destroy()
        raise SystemExit(1) from error
    UnifiedPenDigitsApplication(
        root,
        sample_store=sample_store,
        pipeline=pipeline,
        logger=logger,
        sampling_mode=sampling_mode,
        monitor_interval_ms=max(
            200,
            round(float(config["inference"]["poll_interval_seconds"]) * 1000),
        ),
    )
    root.mainloop()
    logger.info("[GUI] Pen Digits统一科研实验台已关闭")


if __name__ == "__main__":
    main()
