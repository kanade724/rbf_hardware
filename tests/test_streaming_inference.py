from __future__ import annotations

import csv
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from rbf_hardware.configuration.settings import InferencePaths
from rbf_hardware.data.csv_store import CsvRecordStore, NumericCsvStore
from rbf_hardware.inference.experiment_aggregation import (
    HardwareExperimentAggregator,
    HardwareExperimentRecorder,
)
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


class HardwareExperimentAggregatorTests(unittest.TestCase):
    def test_merges_equal_levels_only_within_one_digit_and_sorts_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "experiment.csv"
            aggregator = HardwareExperimentAggregator(
                output_file,
                input_features=3,
                basis_per_dimension=2,
                differential_levels=np.asarray([0.0, 0.5, 1.0]),
            )
            aggregator.add_sample(
                np.asarray([0.5, 0.0, 0.5], dtype=np.float32),
                np.asarray(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    dtype=np.float32,
                ),
            )

            with output_file.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(
                rows[0],
                [
                    "differential_level_index",
                    "hardware_value_01",
                    "hardware_value_02",
                ],
            )
            np.testing.assert_array_equal(
                np.asarray(rows[1:], dtype=np.float64),
                np.asarray(
                    [
                        [0.0, 3.0, 4.0],
                        [1.0, 6.0, 8.0],
                    ]
                ),
            )

    def test_create_produces_header_only_unique_experiment_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            first = HardwareExperimentAggregator.create(
                output_directory,
                input_features=16,
                basis_per_dimension=16,
                differential_levels=np.linspace(0.0, 1.0, 256),
            )
            second = HardwareExperimentAggregator.create(
                output_directory,
                input_features=16,
                basis_per_dimension=16,
                differential_levels=np.linspace(0.0, 1.0, 256),
            )
            self.assertNotEqual(first.output_file, second.output_file)
            self.assertTrue(first.output_file.is_file())
            with first.output_file.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                self.assertEqual(len(next(csv.reader(handle))), 17)

    def test_recorder_creates_one_independent_table_per_digit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = HardwareExperimentRecorder(
                Path(temporary_directory),
                input_features=2,
                basis_per_dimension=2,
                differential_levels=np.asarray([0.0, 0.5, 1.0]),
            )
            experiments = recorder.record_batch(
                np.asarray([[0.5, 0.0], [0.5, 1.0]], dtype=np.float32),
                np.asarray(
                    [
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                    ],
                    dtype=np.float32,
                ),
            )

            self.assertEqual(len(experiments), 2)
            first_rows = NumericCsvStore(experiments[0].output_file, 3).read_rows(1)
            second_rows = NumericCsvStore(experiments[1].output_file, 3).read_rows(1)
            np.testing.assert_array_equal(
                first_rows,
                np.asarray([[0.0, 3.0, 4.0], [1.0, 1.0, 2.0]]),
            )
            np.testing.assert_array_equal(
                second_rows,
                np.asarray([[1.0, 5.0, 6.0], [2.0, 7.0, 8.0]]),
            )

    def test_writes_zero_and_255_as_integer_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "experiment.csv"
            aggregator = HardwareExperimentAggregator(
                output_file,
                input_features=2,
                basis_per_dimension=1,
                differential_levels=np.linspace(0.0, 1.0, 256),
            )
            aggregator.add_sample(
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([10.0, 20.0], dtype=np.float32),
            )
            with output_file.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual([row[0] for row in rows[1:]], ["0", "255"])


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
    def test_continuous_mode_recovers_from_locked_csv(self) -> None:
        pipeline = object.__new__(StreamingInferencePipeline)
        pipeline.logger = Mock()
        locked_error = PermissionError(
            13,
            "Permission denied",
            "pen_digits_hardware.csv",
        )
        pipeline.process_once = Mock(side_effect=[locked_error, None])

        self.assertFalse(pipeline._process_once_with_recovery())
        self.assertTrue(pipeline._process_once_with_recovery())
        self.assertEqual(pipeline.process_once.call_count, 2)
        pipeline.logger.warning.assert_called_once()
        self.assertIn(
            "稍后自动重试",
            pipeline.logger.warning.call_args.args[0],
        )

    def test_predictor_consumes_simulated_hardware_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = InferencePaths(
                project_root=root,
                workspace_root=root,
                raw_samples_file=root / "raw.csv",
                differential_features_file=root / "differential.csv",
                hardware_features_file=root / "hardware.csv",
                experiment_output_dir=root / "experiments",
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
            experiment_recorder = HardwareExperimentRecorder(
                root / "experiments",
                input_features=1,
                basis_per_dimension=2,
                differential_levels=levels,
            )
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
                experiment_recorder=experiment_recorder,
            )
            progress = pipeline.process_once()
            self.assertEqual(progress.normalized_rows, 1)
            self.assertEqual(progress.simulated_rows, 1)
            self.assertEqual(progress.predicted_rows, 1)
            np.testing.assert_array_equal(
                predictor.received_features,
                np.asarray([[0.25, 0.75]], dtype=np.float32),
            )
            experiment_files = list((root / "experiments").glob("*.csv"))
            self.assertEqual(len(experiment_files), 1)
            with experiment_files[0].open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                experiment_rows = list(csv.reader(handle))
            np.testing.assert_array_equal(
                np.asarray(experiment_rows[1:], dtype=np.float64),
                np.asarray([[1.0, 0.25, 0.75]]),
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
                experiment_output_dir=root / "experiments",
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
