from __future__ import annotations

import csv
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rbf_hardware.configuration.settings import InferencePaths
from rbf_hardware.data.csv_store import CsvRecordStore, NumericCsvStore
from rbf_hardware.inference.hardware_simulation import GaussianCalibrationBank
from rbf_hardware.inference.pipeline import (
    PREDICTION_HEADER,
    REPORT_HEADER,
    StreamingInferencePipeline,
)
from rbf_hardware.inference.preprocessing import DifferentialLevelQuantizer


class DifferentialLevelQuantizerTests(unittest.TestCase):
    def test_normalizes_and_prefers_upper_level_on_midpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            levels_path = Path(temporary_directory) / "levels.csv"
            levels_path.write_text(
                "differential_level\n0\n0.5\n1\n",
                encoding="utf-8",
            )
            quantizer = DifferentialLevelQuantizer.load(
                levels_path,
                input_features=3,
                expected_levels=3,
                raw_minimum=0.0,
                raw_maximum=100.0,
            )
            actual = quantizer.transform(np.asarray([[0.0, 25.0, 100.0]]))
            np.testing.assert_array_equal(actual, np.asarray([[0.0, 0.5, 1.0]]))


class GaussianCalibrationBankTests(unittest.TestCase):
    def test_mean_simulation_uses_dimension_major_group_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "calibration.csv"
            with calibration_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    ("group_index", "differential_level", "amplitude", "mean", "std_dev")
                )
                for group_index in (1, 2):
                    for level_index, level in enumerate((0.0, 0.5, 1.0)):
                        writer.writerow(
                            (group_index, level, 1.0, group_index * 10 + level_index, 0.0)
                        )
            bank = GaussianCalibrationBank.load(
                calibration_path,
                expected_groups=2,
                expected_levels=np.asarray((0.0, 0.5, 1.0)),
            )
            actual = bank.simulate(
                np.asarray([[0.0, 1.0], [0.5, 0.0]]),
                sampling_mode="mean",
                random_generator=np.random.default_rng(42),
            )
            expected = np.asarray(
                [[10.0, 20.0, 12.0, 22.0], [11.0, 21.0, 10.0, 20.0]],
                dtype=np.float32,
            )
            np.testing.assert_array_equal(actual, expected)

    def test_accepts_float32_round_trip_of_csv_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            calibration_path = Path(temporary_directory) / "calibration.csv"
            levels = np.asarray((0.0, 0.058823533, 1.0), dtype=np.float64)
            with calibration_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(
                    ("group_index", "differential_level", "amplitude", "mean", "std_dev")
                )
                for level_index, level in enumerate(levels):
                    writer.writerow((1, level, 1.0, level_index, 0.0))
            bank = GaussianCalibrationBank.load(
                calibration_path,
                expected_groups=1,
                expected_levels=levels,
            )
            actual = bank.simulate(
                np.asarray([[np.float32(levels[1])]], dtype=np.float32),
                sampling_mode="mean",
                random_generator=np.random.default_rng(42),
            )
            np.testing.assert_array_equal(actual, np.asarray([[1.0]], dtype=np.float32))


class NumericCsvStoreTests(unittest.TestCase):
    def test_append_and_resume_by_row_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = NumericCsvStore(Path(temporary_directory) / "rows.csv", 2)
            store.append_rows(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
            self.assertEqual(store.row_count(), 2)
            np.testing.assert_array_equal(
                store.read_rows(start_index=1),
                np.asarray([[3.0, 4.0]], dtype=np.float32),
            )


class _FakePredictor:
    metadata = {"hardware_input_features": 1}
    classes = np.asarray((0, 1), dtype=np.int64)

    @staticmethod
    def scores(features: np.ndarray) -> np.ndarray:
        return np.column_stack((1.0 - features[:, 0], features[:, 0]))


class _CapturingPredictor:
    metadata = {"hardware_input_features": 2}
    classes = np.asarray((0, 1), dtype=np.int64)

    def __init__(self) -> None:
        self.received_features: np.ndarray | None = None

    def scores(self, features: np.ndarray) -> np.ndarray:
        self.received_features = features.copy()
        return np.column_stack((features[:, 0], features[:, 1]))


class StreamingInferencePipelineTests(unittest.TestCase):
    def test_predictor_consumes_simulated_hardware_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = InferencePaths(
                project_root=root,
                workspace_root=root,
                raw_samples_file=root / "raw.csv",
                differential_features_file=root / "differential.csv",
                hardware_features_file=root / "hardware.csv",
                predictions_file=root / "predictions.csv",
                report_file=root / "report.csv",
                differential_levels_file=root / "levels.csv",
                gaussian_calibration_file=root / "calibration.csv",
                checkpoint_file=root / "weights.pt",
                log_file=root / "app.log",
            )
            levels = np.asarray((0.0, 1.0), dtype=np.float64)
            quantizer = DifferentialLevelQuantizer(
                levels=levels,
                input_features=1,
                raw_minimum=0.0,
                raw_maximum=100.0,
            )
            calibration = GaussianCalibrationBank(
                levels=levels,
                amplitudes=np.ones((2, 2)),
                means=np.asarray(((0.0, 0.25), (0.0, 0.75))),
                standard_deviations=np.zeros((2, 2)),
            )
            predictor = _CapturingPredictor()
            NumericCsvStore(paths.raw_samples_file, 1).append_rows([[100.0]])
            pipeline = StreamingInferencePipeline(
                paths=paths,
                input_features=1,
                basis_per_dimension=2,
                quantizer=quantizer,
                calibration_bank=calibration,
                predictor=predictor,
                sampling_mode="mean",
                random_seed=42,
                logger=logging.getLogger("test_hardware_data_flow"),
            )
            progress = pipeline.process_once()
            self.assertEqual(progress.normalized_rows, 1)
            self.assertEqual(progress.simulated_rows, 1)
            self.assertEqual(progress.predicted_rows, 1)
            np.testing.assert_array_equal(
                predictor.received_features,
                np.asarray([[0.25, 0.75]], dtype=np.float32),
            )

    def test_recovers_when_prediction_was_written_before_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = InferencePaths(
                project_root=root,
                workspace_root=root,
                raw_samples_file=root / "raw.csv",
                differential_features_file=root / "differential.csv",
                hardware_features_file=root / "hardware.csv",
                predictions_file=root / "predictions.csv",
                report_file=root / "report.csv",
                differential_levels_file=root / "levels.csv",
                gaussian_calibration_file=root / "calibration.csv",
                checkpoint_file=root / "weights.pt",
                log_file=root / "app.log",
            )
            levels = np.asarray((0.0, 1.0), dtype=np.float64)
            quantizer = DifferentialLevelQuantizer(
                levels=levels,
                input_features=1,
                raw_minimum=0.0,
                raw_maximum=100.0,
            )
            calibration = GaussianCalibrationBank(
                levels=levels,
                amplitudes=np.ones((1, 2)),
                means=np.asarray(((0.0, 1.0),)),
                standard_deviations=np.zeros((1, 2)),
            )
            NumericCsvStore(paths.raw_samples_file, 1).append_rows([[100.0]])
            NumericCsvStore(paths.differential_features_file, 1).append_rows([[1.0]])
            NumericCsvStore(paths.hardware_features_file, 1).append_rows([[1.0]])
            CsvRecordStore(paths.predictions_file, PREDICTION_HEADER).append_records(
                [(1, 1)]
            )
            pipeline = StreamingInferencePipeline(
                paths=paths,
                input_features=1,
                basis_per_dimension=1,
                quantizer=quantizer,
                calibration_bank=calibration,
                predictor=_FakePredictor(),
                sampling_mode="mean",
                random_seed=42,
                logger=logging.getLogger("test_streaming_inference"),
            )
            progress = pipeline.process_once()
            self.assertEqual(progress.predicted_rows, 1)
            self.assertEqual(
                CsvRecordStore(paths.predictions_file, PREDICTION_HEADER).row_count(), 1
            )
            self.assertEqual(CsvRecordStore(paths.report_file, REPORT_HEADER).row_count(), 1)


if __name__ == "__main__":
    unittest.main()
