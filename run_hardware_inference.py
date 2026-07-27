"""Run the single-process append-only simulated-hardware inference pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rbf_hardware.configuration.settings import (
    load_config,
    resolve_inference_paths,
)
from rbf_hardware.infrastructure.logging import setup_logging
from rbf_hardware.inference.pipeline import StreamingInferencePipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor Pen Digits rows, quantize them, simulate 16x16 hardware "
            "responses, and predict with checkpoints/weights.pt."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Hardware project configuration file.",
    )
    parser.add_argument("--once", action="store_true", help="Process available rows and exit.")
    parser.add_argument(
        "--sampling-mode",
        choices=("empirical", "mean"),
        default=None,
        help="Override empirical cycle sampling; mean is useful for deterministic verification.",
    )
    parser.add_argument("--raw-input", type=Path, default=None)
    parser.add_argument("--differential-output", type=Path, default=None)
    parser.add_argument("--hardware-output", type=Path, default=None)
    parser.add_argument("--experiment-output-dir", type=Path, default=None)
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--report-output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--response-bank", type=Path, default=None)
    return parser.parse_args()


def quit_requested() -> bool:
    if os.name != "nt":
        return False
    import msvcrt

    while msvcrt.kbhit():
        if msvcrt.getwch().lower() == "q":
            return True
    return False


def resolved_override(path: Path | None, fallback: Path) -> Path:
    return path.expanduser().resolve() if path is not None else fallback


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    configured_paths = resolve_inference_paths(config)
    paths = replace(
        configured_paths,
        raw_samples_file=resolved_override(arguments.raw_input, configured_paths.raw_samples_file),
        differential_features_file=resolved_override(
            arguments.differential_output, configured_paths.differential_features_file
        ),
        hardware_features_file=resolved_override(
            arguments.hardware_output, configured_paths.hardware_features_file
        ),
        experiment_output_dir=resolved_override(
            arguments.experiment_output_dir, configured_paths.experiment_output_dir
        ),
        predictions_file=resolved_override(
            arguments.predictions_output, configured_paths.predictions_file
        ),
        report_file=resolved_override(arguments.report_output, configured_paths.report_file),
        checkpoint_file=resolved_override(arguments.checkpoint, configured_paths.checkpoint_file),
        empirical_response_file=resolved_override(
            arguments.response_bank, configured_paths.empirical_response_file
        ),
    )
    logger = setup_logging(
        paths.log_file,
        level_name=str(config["logging"]["level"]),
        file_mode="a",
    )
    logger.info("[推理] 追加式Pen Digits流水线已启动")
    logger.info("[同步] 原始样本文件：%s", paths.raw_samples_file)
    logger.info("[同步] 差分特征文件：%s", paths.differential_features_file)
    logger.info("[同步] 模拟硬件特征文件：%s", paths.hardware_features_file)
    logger.info("[实验] 独立聚合表目录：%s", paths.experiment_output_dir)
    logger.info("[同步] 推理报告文件：%s", paths.report_file)
    logger.info("[推理] checkpoint：%s", paths.checkpoint_file)
    logger.info("[推理] 400循环实测硬件响应库：%s", paths.empirical_response_file)
    logger.info(
        "[推理] 自适应随机噪声：小信号±%.1f%%，大信号±%.1f%%",
        float(config["inference"]["empirical_noise_maximum_rate"]) * 100,
        float(config["inference"]["empirical_noise_minimum_rate"]) * 100,
    )

    pipeline = StreamingInferencePipeline.build(
        config,
        paths,
        logger,
        sampling_mode=arguments.sampling_mode,
    )
    if arguments.once:
        try:
            progress = pipeline.process_once()
        except PermissionError as error:
            locked_file = error.filename or "运行时CSV"
            logger.error(
                "[同步] 文件正被WPS、Excel或其他程序占用：%s；请关闭该CSV后重试",
                locked_file,
            )
            raise SystemExit(
                f"文件被占用：{locked_file}。请关闭WPS、Excel或CSV预览器后重试。"
            ) from None
        print(
            "Processed "
            f"normalized={progress.normalized_rows}, "
            f"simulated={progress.simulated_rows}, "
            f"predicted={progress.predicted_rows}"
        )
        return

    inference = config["inference"]
    print("正在监听绘图 CSV；按 q 或 Ctrl+C 结束。", flush=True)
    try:
        pipeline.run_forever(
            poll_interval_seconds=float(inference["poll_interval_seconds"]),
            debounce_seconds=float(inference["debounce_seconds"]),
            stop_requested=quit_requested,
        )
    except KeyboardInterrupt:
        pass
    logger.info("[推理] Pen Digits流水线已停止")


if __name__ == "__main__":
    main()
