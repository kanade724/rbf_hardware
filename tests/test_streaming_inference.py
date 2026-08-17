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
from rbf_hardware.inference.empirical_response import EmpiricalHardwareResponseBank
from rbf_hardware.inference.experiment_aggregation import (
    HardwareExperimentAggregator,
    HardwareExperimentRecorder,
)
from rbf_hardware.inference.pipeline import (
    PREDICTION_HEADER,
    REPORT_HEADER,
    StreamingInferencePipeline,
)
from rbf_hardware.inference.preprocessing import DifferentialLevelQuantizer
from rbf_hardware.ui.experiment_workflow import ExperimentWorkflowState
from rbf_hardware.ui.pen_digits_collector import (
    CanvasTransform,
    CollectorSettings,
    PenDigitDrawingPad,
)
from rbf_hardware.ui.responsive_layout import (
    ApplicationLayoutMode,
    resolve_application_layout_mode,
)
from rbf_hardware.ui.unified_application import UnifiedPenDigitsApplication


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


class EmpiricalHardwareResponseBankTests(unittest.TestCase):
    @staticmethod
    def _bank() -> EmpiricalHardwareResponseBank:
        samples = np.empty((3, 2, 2), dtype=np.float32)
        for level_index in range(3):
            for cycle_index in range(2):
                base = level_index * 100 + cycle_index * 10
                samples[level_index, cycle_index] = (base + 1, base + 2)
        return EmpiricalHardwareResponseBank(
            levels=np.asarray((0.0, 0.5, 1.0)),
            response_samples=samples,
            group_magnitude_references=np.asarray((250.0, 250.0)),
        )

    def test_empirical_mode_uses_one_shared_cycle_across_dimensions(self) -> None:
        actual = self._bank().simulate(
            np.asarray([[0.0, 1.0], [0.5, 0.0]]),
            sampling_mode="empirical",
            random_generator=np.random.default_rng(42),
            minimum_noise_rate=0.0,
            maximum_noise_rate=0.0,
        ).reshape(2, 2, 2)
        level_bases = np.asarray([[0.0, 200.0], [100.0, 0.0]])
        cycle_offsets = actual[:, :, 0] - level_bases - 1.0
        self.assertTrue(np.isin(cycle_offsets, (0.0, 10.0)).all())
        np.testing.assert_array_equal(cycle_offsets[:, 0], cycle_offsets[:, 1])

    def test_mean_mode_uses_measured_cycle_average(self) -> None:
        actual = self._bank().simulate(
            np.asarray([[0.0, 1.0]]),
            sampling_mode="mean",
            random_generator=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(
            actual,
            np.asarray([[6.0, 7.0, 206.0, 207.0]], dtype=np.float32),
        )

    def test_noise_is_five_percent_for_small_and_one_percent_for_large(self) -> None:
        bank = EmpiricalHardwareResponseBank(
            levels=np.asarray((0.0,)),
            response_samples=np.asarray(
                [[[1.0, 100.0], [1.0, 100.0]]],
                dtype=np.float32,
            ),
            group_magnitude_references=np.asarray((100.0, 100.0)),
        )
        actual = bank.simulate(
            np.zeros((10_000, 1), dtype=np.float32),
            sampling_mode="empirical",
            random_generator=np.random.default_rng(42),
            minimum_noise_rate=0.01,
            maximum_noise_rate=0.05,
        )
        small_relative_noise = np.abs(actual[:, 0] - 1.0)
        large_relative_noise = np.abs(actual[:, 1] - 100.0) / 100.0
        self.assertLessEqual(float(small_relative_noise.max()), 0.05)
        self.assertGreater(float(small_relative_noise.max()), 0.045)
        self.assertLessEqual(float(large_relative_noise.max()), 0.01 + 1.0e-7)
        self.assertGreater(
            float(small_relative_noise.std()),
            float(large_relative_noise.std()) * 3.0,
        )


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


class PenDigitDrawingPadTests(unittest.TestCase):
    def test_canvas_transform_round_trip_preserves_logical_coordinates(self) -> None:
        transform = CanvasTransform(
            scale=1.6,
            offset_x=24.0,
            offset_y=12.0,
        )
        logical_point = (125.0, 375.0)

        display_point = transform.to_display(logical_point)

        self.assertEqual(display_point, (224.0, 612.0))
        np.testing.assert_allclose(
            transform.to_logical(display_point),
            logical_point,
        )

    def test_normalized_features_preserve_pc_pen_digits_coordinate_rule(self) -> None:
        drawing_pad = object.__new__(PenDigitDrawingPad)
        drawing_pad.settings = CollectorSettings(point_count=8)
        drawing_pad.sampled_points = [
            (float(index), float(index))
            for index in range(8)
        ]

        actual = drawing_pad.normalized_features()

        self.assertEqual(actual.shape, (16,))
        np.testing.assert_array_equal(actual[:2], np.asarray((0.0, 100.0)))
        np.testing.assert_array_equal(actual[-2:], np.asarray((100.0, 0.0)))


class UnifiedPenDigitsApplicationTests(unittest.TestCase):
    def test_hardware_output_uses_scientific_notation(self) -> None:
        actual = UnifiedPenDigitsApplication.format_hardware_output(
            [1.0, -0.0025, 123456.0]
        )

        self.assertEqual(
            actual,
            "1.000000e+00, -2.500000e-03, 1.234560e+05",
        )

    def test_experiment_one_saves_without_starting_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application = object.__new__(UnifiedPenDigitsApplication)
            application.inference_busy = False
            application.drawing_pad = Mock()
            application.drawing_pad.is_ready = True
            application.drawing_pad.normalized_features.return_value = np.arange(
                16,
                dtype=np.float32,
            )
            application.sample_store = NumericCsvStore(
                Path(temporary_directory) / "raw.csv",
                16,
            )
            application.saved_count = 0
            application.sample_count_text = Mock()
            application.logger = Mock()
            application.pipeline = Mock()
            application.pipeline_state = Mock()
            application.status_text = Mock()
            application.status_detail = Mock()
            application.inference_button = Mock()
            application._set_stage_state = Mock()
            application._request_inference = Mock()
            application.last_raw_signature = None

            application.save_sample()

            self.assertEqual(application.sample_store.row_count(), 1)
            application.drawing_pad.clear.assert_called_once_with()
            application.pipeline.process_simulation_stage.assert_called_once_with()
            application._request_inference.assert_not_called()
            application.pipeline_state.set.assert_called_once_with("SAMPLE SAVED")
            application._set_stage_state.assert_called_once_with(1)

    def test_experiment_one_tolerates_simulation_stage_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application = object.__new__(UnifiedPenDigitsApplication)
            application.inference_busy = False
            application.drawing_pad = Mock()
            application.drawing_pad.is_ready = True
            application.drawing_pad.normalized_features.return_value = np.arange(
                16,
                dtype=np.float32,
            )
            application.sample_store = NumericCsvStore(
                Path(temporary_directory) / "raw.csv",
                16,
            )
            application.saved_count = 0
            application.sample_count_text = Mock()
            application.logger = Mock()
            application.pipeline = Mock()
            application.pipeline.process_simulation_stage.side_effect = PermissionError(
                13,
                "Permission denied",
                "pen_digits_hardware.csv",
            )
            application.pipeline_state = Mock()
            application.status_text = Mock()
            application.status_detail = Mock()
            application.inference_button = Mock()
            application._set_stage_state = Mock()
            application._request_inference = Mock()
            application.last_raw_signature = None

            application.save_sample()

            self.assertEqual(application.sample_store.row_count(), 1)
            application.pipeline_state.set.assert_called_once_with("SAMPLE SAVED")
            application.inference_button.configure.assert_called_once_with(
                state="normal",
            )

    def test_external_sample_detection_waits_for_experiment_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            application = object.__new__(UnifiedPenDigitsApplication)
            application.inference_busy = False
            application.close_requested = False
            application.sample_store = NumericCsvStore(
                Path(temporary_directory) / "raw.csv",
                1,
            )
            application.sample_store.append_rows([[25.0]])
            application.last_raw_signature = None
            application.saved_count = 0
            application.sample_count_text = Mock()
            application.pipeline_state = Mock()
            application.status_text = Mock()
            application.status_detail = Mock()
            application.logger = Mock()
            application.pipeline = Mock()
            application.pipeline.has_pending_work.return_value = True
            application.inference_button = Mock()
            application.root = Mock()
            application.monitor_interval_ms = 500
            application._monitor_after_id = None
            application._set_stage_state = Mock()
            application._request_inference = Mock()

            application._monitor_new_rows()

            application._request_inference.assert_not_called()
            application.pipeline_state.set.assert_called_once_with(
                "INFERENCE READY",
            )
            application._set_stage_state.assert_called_once_with(1)
            application.root.after.assert_called_once_with(
                500,
                application._monitor_new_rows,
            )

    def test_experiment_two_does_not_run_without_pending_sample(self) -> None:
        application = object.__new__(UnifiedPenDigitsApplication)
        application.inference_busy = False
        application.close_requested = False
        application.pipeline = Mock()
        application.pipeline.has_pending_work.return_value = False
        application.pipeline_state = Mock()
        application.status_text = Mock()
        application.status_detail = Mock()
        application.inference_button = Mock()
        application.logger = Mock()
        application.drawing_pad = Mock()

        application._request_inference()

        application.drawing_pad.set_enabled.assert_not_called()
        application.pipeline_state.set.assert_called_once_with("READY")
        application.inference_button.configure.assert_called_once_with(
            state="disabled",
        )

    def test_pending_status_survives_drawing_pad_resize_callback(self) -> None:
        application = object.__new__(UnifiedPenDigitsApplication)
        application.inference_busy = False
        application.workflow_state = ExperimentWorkflowState.PENDING
        application.save_button = Mock()
        application.status_text = Mock()
        application.status_detail = Mock()

        application._on_drawing_ready(False)

        self.assertIs(
            application.workflow_state,
            ExperimentWorkflowState.PENDING,
        )
        application.status_text.set.assert_not_called()
        application.status_detail.set.assert_not_called()

    def test_pending_after_inference_remains_manual(self) -> None:
        application = object.__new__(UnifiedPenDigitsApplication)
        application.inference_busy = False
        application.workflow_state = ExperimentWorkflowState.COMPLETE
        application.pipeline = Mock()
        application.pipeline.has_pending_work.return_value = True
        application.sample_store = Mock()
        application.sample_store.row_count.return_value = 2
        application.saved_count = 1
        application.sample_count_text = Mock()
        application.pipeline_state = Mock()
        application.status_text = Mock()
        application.status_detail = Mock()
        application.inference_button = Mock()
        application.logger = Mock()
        application._set_stage_state = Mock()
        application._request_inference = Mock()

        has_pending_work = application._refresh_manual_workflow_status(
            preserve_when_no_pending=True,
        )

        self.assertTrue(has_pending_work)
        self.assertIs(
            application.workflow_state,
            ExperimentWorkflowState.PENDING,
        )
        application.pipeline_state.set.assert_called_once_with(
            "INFERENCE READY",
        )
        application._request_inference.assert_not_called()

    def test_close_waits_for_active_inference_worker(self) -> None:
        application = object.__new__(UnifiedPenDigitsApplication)
        application.close_requested = False
        application._root_destroyed = False
        application.inference_busy = True
        application.inference_thread = Mock()
        application.inference_thread.is_alive.return_value = True
        application.workflow_state = ExperimentWorkflowState.PROCESSING
        application.pipeline_state = Mock()
        application.status_text = Mock()
        application.status_detail = Mock()
        application.save_button = Mock()
        application.inference_button = Mock()
        application.drawing_pad = Mock()
        application.logger = Mock()
        application.root = Mock()
        application.root.after.return_value = "close-poll"
        application._layout_refresh_after_id = None
        application._page_extent_after_id = None
        application._result_queue_after_id = None
        application._monitor_after_id = None
        application._close_poll_after_id = None

        application._request_close()

        application.root.destroy.assert_not_called()
        application.inference_thread.is_alive.return_value = False
        application._poll_inference_before_close()

        application.root.destroy.assert_called_once_with()


class ResponsiveLayoutTests(unittest.TestCase):
    def test_wide_window_uses_landscape_layout(self) -> None:
        actual = resolve_application_layout_mode(1220, 930)

        self.assertIs(actual, ApplicationLayoutMode.LANDSCAPE)

    def test_tall_window_uses_portrait_layout(self) -> None:
        actual = resolve_application_layout_mode(800, 1400)

        self.assertIs(actual, ApplicationLayoutMode.PORTRAIT)

    def test_narrow_landscape_window_uses_portrait_layout(self) -> None:
        actual = resolve_application_layout_mode(900, 700)

        self.assertIs(actual, ApplicationLayoutMode.PORTRAIT)

    def test_portrait_layout_uses_hysteresis_before_returning_to_landscape(
        self,
    ) -> None:
        near_breakpoint = resolve_application_layout_mode(
            1000,
            900,
            ApplicationLayoutMode.PORTRAIT,
        )
        clearly_wide = resolve_application_layout_mode(
            1100,
            900,
            ApplicationLayoutMode.PORTRAIT,
        )

        self.assertIs(near_breakpoint, ApplicationLayoutMode.PORTRAIT)
        self.assertIs(clearly_wide, ApplicationLayoutMode.LANDSCAPE)


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

    def test_create_reuses_fixed_experiment_file(self) -> None:
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
            self.assertEqual(first.output_file, second.output_file)
            self.assertEqual(
                first.output_file.name,
                "pen_digits_hardware_experiment.csv",
            )
            self.assertEqual(len(list(output_directory.glob("*.csv"))), 1)
            with second.output_file.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                self.assertEqual(len(next(csv.reader(handle))), 17)

    def test_recorder_overwrites_fixed_table_for_each_digit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            recorder = HardwareExperimentRecorder(
                output_directory,
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
            self.assertEqual(
                experiments[0].output_file,
                experiments[1].output_file,
            )
            self.assertEqual(experiments[1].output_file, recorder.output_file)
            self.assertEqual(len(list(output_directory.glob("*.csv"))), 1)
            final_rows = NumericCsvStore(recorder.output_file, 3).read_rows(1)
            np.testing.assert_array_equal(
                final_rows,
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
    def test_pending_work_detects_any_trailing_pipeline_stage(self) -> None:
        pipeline = object.__new__(StreamingInferencePipeline)
        pipeline.raw_store = Mock()
        pipeline.differential_store = Mock()
        pipeline.hardware_store = Mock()
        pipeline.prediction_store = Mock()
        pipeline.report_store = Mock()
        pipeline.labeled_store = Mock()
        trailing_stage_counts = (
            (2, 1, 1, 1, 1, 1),
            (2, 2, 1, 1, 1, 1),
            (2, 2, 2, 1, 2, 2),
            (2, 2, 2, 2, 1, 2),
            (2, 2, 2, 2, 2, 1),
        )
        for stage_counts in trailing_stage_counts:
            with self.subTest(stage_counts=stage_counts):
                stores = (
                    pipeline.raw_store,
                    pipeline.differential_store,
                    pipeline.hardware_store,
                    pipeline.prediction_store,
                    pipeline.report_store,
                    pipeline.labeled_store,
                )
                for store, row_count in zip(
                    stores,
                    stage_counts,
                    strict=True,
                ):
                    store.row_count.return_value = row_count
                self.assertTrue(pipeline.has_pending_work())

        pipeline.raw_store.row_count.return_value = 2
        pipeline.differential_store.row_count.return_value = 2
        pipeline.hardware_store.row_count.return_value = 2
        pipeline.prediction_store.row_count.return_value = 2
        pipeline.report_store.row_count.return_value = 2
        pipeline.labeled_store.row_count.return_value = 2

        self.assertFalse(pipeline.has_pending_work())

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
                labeled_dataset_file=root / "labeled.csv",
                differential_levels_file=root / "levels.csv",
                empirical_response_file=root / "response_bank.npz",
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
            response_bank = EmpiricalHardwareResponseBank(
                levels=levels,
                response_samples=np.asarray(
                    [
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.25, 0.75], [0.25, 0.75]],
                    ]
                ),
                group_magnitude_references=np.asarray((1.0, 1.0)),
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
                response_bank=response_bank,
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
            self.assertEqual(len(progress.predictions), 1)
            self.assertEqual(progress.predictions[0].sample_index, 1)
            self.assertEqual(progress.predictions[0].predicted_digit, 1)
            self.assertAlmostEqual(progress.predictions[0].score_margin, 0.5)
            self.assertEqual(len(progress.experiment_files), 1)
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
            np.testing.assert_array_equal(
                NumericCsvStore(paths.labeled_dataset_file, 2).read_rows(),
                np.asarray([[100.0, 1.0]], dtype=np.float32),
            )

    def test_simulation_stage_updates_experiment_table_without_predicting(self) -> None:
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
                labeled_dataset_file=root / "labeled.csv",
                differential_levels_file=root / "levels.csv",
                empirical_response_file=root / "response_bank.npz",
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
            response_bank = EmpiricalHardwareResponseBank(
                levels=levels,
                response_samples=np.asarray(
                    [
                        [[0.0, 0.0], [0.0, 0.0]],
                        [[0.25, 0.75], [0.25, 0.75]],
                    ]
                ),
                group_magnitude_references=np.asarray((1.0, 1.0)),
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
                response_bank=response_bank,
                predictor=predictor,
                sampling_mode="mean",
                random_seed=42,
                logger=logging.getLogger("test_hardware_data_flow"),
                experiment_recorder=experiment_recorder,
            )

            simulation_progress = pipeline.process_simulation_stage()

            self.assertEqual(simulation_progress.normalized_rows, 1)
            self.assertEqual(simulation_progress.simulated_rows, 1)
            self.assertEqual(simulation_progress.predicted_rows, 0)
            self.assertEqual(len(simulation_progress.predictions), 0)
            self.assertEqual(len(simulation_progress.experiment_files), 1)
            experiment_files = list((root / "experiments").glob("*.csv"))
            self.assertEqual(len(experiment_files), 1)
            self.assertIsNone(predictor.received_features)
            self.assertEqual(
                CsvRecordStore(paths.predictions_file, PREDICTION_HEADER).row_count(), 0
            )
            self.assertTrue(pipeline.has_pending_prediction_work())
            self.assertFalse(pipeline.has_pending_simulation_work())

            prediction_progress = pipeline.process_prediction_stage()

            self.assertEqual(prediction_progress.normalized_rows, 0)
            self.assertEqual(prediction_progress.simulated_rows, 0)
            self.assertEqual(prediction_progress.predicted_rows, 1)
            self.assertEqual(len(prediction_progress.predictions), 1)
            self.assertEqual(prediction_progress.predictions[0].predicted_digit, 1)
            np.testing.assert_array_equal(
                predictor.received_features,
                np.asarray([[0.25, 0.75]], dtype=np.float32),
            )
            self.assertFalse(pipeline.has_pending_work())

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
                labeled_dataset_file=root / "labeled.csv",
                differential_levels_file=root / "levels.csv",
                empirical_response_file=root / "response_bank.npz",
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
            response_bank = EmpiricalHardwareResponseBank(
                levels=levels,
                response_samples=np.asarray(
                    [[[0.0], [0.0]], [[1.0], [1.0]]]
                ),
                group_magnitude_references=np.asarray((1.0,)),
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
                response_bank=response_bank,
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
            np.testing.assert_array_equal(
                NumericCsvStore(paths.labeled_dataset_file, 2).read_rows(),
                np.asarray([[100.0, 1.0]], dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
