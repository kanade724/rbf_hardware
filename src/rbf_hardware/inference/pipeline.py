"""Single-process, append-only Pen Digits hardware inference orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..configuration.settings import InferencePaths
from ..data.csv_store import CsvRecordStore, CsvRowError, NumericCsvStore
from ..modeling.predictor import JointGaussianPredictor
from .hardware_simulation import GaussianCalibrationBank
from .preprocessing import DifferentialLevelQuantizer


PREDICTION_HEADER = ("sample_index", "predicted_digit")
REPORT_HEADER = (
    "timestamp_utc",
    "sample_index",
    "predicted_digit",
    "top_score",
    "second_score",
    "score_margin",
    "raw_samples_file",
    "differential_features_file",
    "hardware_features_file",
    "checkpoint_file",
)


@dataclass(frozen=True)
class PipelineProgress:
    normalized_rows: int = 0
    simulated_rows: int = 0
    predicted_rows: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.normalized_rows or self.simulated_rows or self.predicted_rows)


class StreamingInferencePipeline:
    def __init__(
        self,
        *,
        paths: InferencePaths,
        input_features: int,
        basis_per_dimension: int,
        quantizer: DifferentialLevelQuantizer,
        calibration_bank: GaussianCalibrationBank,
        predictor: JointGaussianPredictor,
        sampling_mode: str,
        random_seed: int,
        logger: logging.Logger,
    ) -> None:
        self.paths = paths
        self.input_features = input_features
        self.basis_per_dimension = basis_per_dimension
        self.quantizer = quantizer
        self.calibration_bank = calibration_bank
        self.predictor = predictor
        self.sampling_mode = sampling_mode
        self.random_generator = np.random.default_rng(random_seed)
        self.logger = logger
        self.raw_store = NumericCsvStore(paths.raw_samples_file, input_features)
        self.differential_store = NumericCsvStore(
            paths.differential_features_file, input_features
        )
        self.hardware_store = NumericCsvStore(
            paths.hardware_features_file, input_features * basis_per_dimension
        )
        self.prediction_store = CsvRecordStore(paths.predictions_file, PREDICTION_HEADER)
        self.report_store = CsvRecordStore(paths.report_file, REPORT_HEADER)

        checkpoint_features = int(predictor.metadata.get("hardware_input_features", -1))
        expected_hardware_features = input_features * basis_per_dimension
        if checkpoint_features != expected_hardware_features:
            raise ValueError(
                f"Checkpoint expects {checkpoint_features} hardware features; "
                f"pipeline produces {expected_hardware_features}."
            )
        if calibration_bank.group_count != basis_per_dimension:
            raise ValueError(
                f"Calibration has {calibration_bank.group_count} groups; "
                f"expected {basis_per_dimension}."
            )

    @classmethod
    def build(
        cls,
        config: dict,
        paths: InferencePaths,
        logger: logging.Logger,
        *,
        sampling_mode: str | None = None,
    ) -> "StreamingInferencePipeline":
        inference = config["inference"]
        input_features = int(inference["input_features"])
        basis_per_dimension = int(inference["basis_per_dimension"])
        quantizer = DifferentialLevelQuantizer.load(
            paths.differential_levels_file,
            input_features=input_features,
            expected_levels=int(inference["quantization_levels"]),
            raw_minimum=float(inference["raw_minimum"]),
            raw_maximum=float(inference["raw_maximum"]),
        )
        calibration_bank = GaussianCalibrationBank.load(
            paths.gaussian_calibration_file,
            expected_groups=basis_per_dimension,
            expected_levels=quantizer.levels,
        )
        predictor = JointGaussianPredictor.load(paths.checkpoint_file)
        return cls(
            paths=paths,
            input_features=input_features,
            basis_per_dimension=basis_per_dimension,
            quantizer=quantizer,
            calibration_bank=calibration_bank,
            predictor=predictor,
            sampling_mode=sampling_mode or str(inference["sampling_mode"]),
            random_seed=int(inference["random_seed"]),
            logger=logger,
        )

    def _validate_stage_counts(self) -> tuple[int, int, int, int, int]:
        raw_rows = self.raw_store.row_count()
        differential_rows = self.differential_store.row_count()
        hardware_rows = self.hardware_store.row_count()
        prediction_rows = self.prediction_store.row_count()
        report_rows = self.report_store.row_count()
        if differential_rows > raw_rows:
            raise ValueError(
                "Differential CSV contains more rows than the raw sample CSV; "
                "remove or repair the inconsistent downstream rows."
            )
        if hardware_rows > differential_rows:
            raise ValueError(
                "Hardware CSV contains more rows than the differential CSV; "
                "remove or repair the inconsistent downstream rows."
            )
        if prediction_rows > hardware_rows or report_rows > hardware_rows:
            raise ValueError(
                "Prediction or report CSV contains more rows than the hardware CSV."
            )
        return raw_rows, differential_rows, hardware_rows, prediction_rows, report_rows

    def process_once(self) -> PipelineProgress:
        raw_rows, differential_rows, hardware_rows, prediction_rows, report_rows = (
            self._validate_stage_counts()
        )

        new_raw_rows = self.raw_store.read_rows(differential_rows)
        normalized_count = 0
        if len(new_raw_rows):
            differential_values = self.quantizer.transform(new_raw_rows)
            normalized_count = self.differential_store.append_rows(differential_values)
            start_row = differential_rows + 1
            differential_rows += normalized_count
            self.logger.info(
                "[同步] 已归一化并量化原始数据，行=%d..%d，共享文件=%s",
                start_row,
                differential_rows,
                self.paths.differential_features_file,
            )

        new_differential_rows = self.differential_store.read_rows(hardware_rows)
        simulated_count = 0
        if len(new_differential_rows):
            hardware_values = self.calibration_bank.simulate(
                new_differential_rows,
                sampling_mode=self.sampling_mode,
                random_generator=self.random_generator,
            )
            simulated_count = self.hardware_store.append_rows(hardware_values)
            start_row = hardware_rows + 1
            hardware_rows += simulated_count
            self.logger.info(
                "[同步] 已生成16x16模拟硬件数据，行=%d..%d，共享文件=%s，采样模式=%s",
                start_row,
                hardware_rows,
                self.paths.hardware_features_file,
                self.sampling_mode,
            )

        output_start_index = min(prediction_rows, report_rows)
        new_hardware_rows = self.hardware_store.read_rows(output_start_index)
        predicted_count = 0
        if len(new_hardware_rows):
            scores = self.predictor.scores(new_hardware_rows)
            class_indices = scores.argmax(axis=1)
            predictions = self.predictor.classes[class_indices]
            sorted_scores = np.sort(scores, axis=1)
            top_scores = sorted_scores[:, -1]
            second_scores = sorted_scores[:, -2]
            timestamp = datetime.now(timezone.utc).isoformat()
            first_sample_index = output_start_index + 1
            prediction_records = []
            report_records = []
            for offset, prediction in enumerate(predictions):
                sample_index = first_sample_index + offset
                predicted_digit = int(prediction)
                prediction_records.append((sample_index, predicted_digit))
                report_records.append(
                    (
                        timestamp,
                        sample_index,
                        predicted_digit,
                        float(top_scores[offset]),
                        float(second_scores[offset]),
                        float(top_scores[offset] - second_scores[offset]),
                        str(self.paths.raw_samples_file),
                        str(self.paths.differential_features_file),
                        str(self.paths.hardware_features_file),
                        str(self.paths.checkpoint_file),
                    )
                )
            prediction_offset = prediction_rows - output_start_index
            report_offset = report_rows - output_start_index
            self.prediction_store.append_records(prediction_records[prediction_offset:])
            self.report_store.append_records(report_records[report_offset:])
            predicted_count = max(
                len(prediction_records) - prediction_offset,
                len(report_records) - report_offset,
            )
            final_output_row = output_start_index + len(prediction_records)
            self.logger.info(
                "[结果] 已完成推理，行=%d..%d，识别数字=%s，报告=%s",
                first_sample_index,
                final_output_row,
                [int(value) for value in predictions],
                self.paths.report_file,
            )

        return PipelineProgress(
            normalized_rows=normalized_count,
            simulated_rows=simulated_count,
            predicted_rows=predicted_count,
        )

    def run_forever(
        self,
        *,
        poll_interval_seconds: float,
        debounce_seconds: float,
        stop_requested: Callable[[], bool],
    ) -> None:
        observed_signature = self._file_signature(self.paths.raw_samples_file)
        stable_since = time.monotonic() - debounce_seconds
        while not stop_requested():
            current_signature = self._file_signature(self.paths.raw_samples_file)
            if current_signature != observed_signature:
                observed_signature = current_signature
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= debounce_seconds:
                try:
                    self.process_once()
                except CsvRowError as error:
                    self.logger.warning(
                        "[同步] 共享CSV暂时存在未完成行，将稍后重试：%s", error
                    )
                    stable_since = time.monotonic()
            time.sleep(poll_interval_seconds)

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int] | None:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        return metadata.st_mtime_ns, metadata.st_size
