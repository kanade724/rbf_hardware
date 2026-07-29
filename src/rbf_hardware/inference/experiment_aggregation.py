"""Aggregation of one digit into the fixed hardware experiment output."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np


HARDWARE_EXPERIMENT_FILE_NAME = "pen_digits_hardware_experiment.csv"


class HardwareExperimentAggregator:
    """Aggregate one saved digit's differential values and hardware blocks."""

    def __init__(
        self,
        output_file: Path,
        *,
        input_features: int,
        basis_per_dimension: int,
        differential_levels: np.ndarray,
    ) -> None:
        if input_features <= 0:
            raise ValueError("input_features must be positive.")
        if basis_per_dimension <= 0:
            raise ValueError("basis_per_dimension must be positive.")
        self.output_file = output_file.expanduser().resolve()
        self.input_features = input_features
        self.basis_per_dimension = basis_per_dimension
        self.differential_levels = np.asarray(
            differential_levels,
            dtype=np.float64,
        ).reshape(-1)
        if len(self.differential_levels) < 2:
            raise ValueError("differential_levels must contain at least two values.")
        if not np.isfinite(self.differential_levels).all():
            raise ValueError("differential_levels cannot contain NaN or infinity.")
        if np.any(np.diff(self.differential_levels) <= 0):
            raise ValueError("differential_levels must be strictly increasing.")
        self._hardware_sums: dict[int, np.ndarray] = {}
        self._write_snapshot()

    @classmethod
    def create(
        cls,
        output_directory: Path,
        *,
        input_features: int,
        basis_per_dimension: int,
        differential_levels: np.ndarray,
    ) -> "HardwareExperimentAggregator":
        output_file = (
            output_directory.expanduser().resolve()
            / HARDWARE_EXPERIMENT_FILE_NAME
        )
        return cls(
            output_file,
            input_features=input_features,
            basis_per_dimension=basis_per_dimension,
            differential_levels=differential_levels,
        )

    @property
    def header(self) -> tuple[str, ...]:
        hardware_columns = tuple(
            f"hardware_value_{index:02d}"
            for index in range(1, self.basis_per_dimension + 1)
        )
        return ("differential_level_index", *hardware_columns)

    @property
    def aggregated_level_count(self) -> int:
        return len(self._hardware_sums)

    def add_sample(
        self,
        differential_row: np.ndarray,
        hardware_row: np.ndarray,
    ) -> int:
        differential_values = np.asarray(differential_row, dtype=np.float64)
        hardware_values = np.asarray(hardware_row, dtype=np.float64)
        differential_values = np.atleast_2d(differential_values)
        hardware_values = np.atleast_2d(hardware_values)
        expected_hardware_columns = self.input_features * self.basis_per_dimension
        if differential_values.shape[1] != self.input_features:
            raise ValueError(
                "Differential batch has "
                f"{differential_values.shape[1]} columns; expected {self.input_features}."
            )
        if hardware_values.shape[1] != expected_hardware_columns:
            raise ValueError(
                f"Hardware batch has {hardware_values.shape[1]} columns; "
                f"expected {expected_hardware_columns}."
            )
        if len(differential_values) != 1 or len(hardware_values) != 1:
            raise ValueError("One experiment table must contain exactly one saved digit.")
        if not np.isfinite(differential_values).all() or not np.isfinite(
            hardware_values
        ).all():
            raise ValueError("Experiment batches cannot contain NaN or infinity.")

        dimension_blocks = hardware_values.reshape(
            self.input_features,
            self.basis_per_dimension,
        )
        for dimension_index in range(self.input_features):
            differential_level = float(
                np.float32(differential_values[0, dimension_index])
            )
            level_index = self._find_level_index(differential_level)
            hardware_block = dimension_blocks[dimension_index]
            if level_index not in self._hardware_sums:
                self._hardware_sums[level_index] = np.zeros(
                    self.basis_per_dimension,
                    dtype=np.float64,
                )
            self._hardware_sums[level_index] += hardware_block

        self._write_snapshot()
        return self.aggregated_level_count

    def _find_level_index(self, differential_level: float) -> int:
        level_index = int(
            np.abs(self.differential_levels - differential_level).argmin()
        )
        expected_level = float(self.differential_levels[level_index])
        if not np.isclose(
            differential_level,
            expected_level,
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError(
                f"Differential value {differential_level} is not present in "
                "differential_levels.csv."
            )
        return level_index

    def _write_snapshot(self) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.output_file.with_suffix(self.output_file.suffix + ".tmp")
        with temporary_file.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(self.header)
            for level_index in sorted(self._hardware_sums):
                writer.writerow(
                    (
                        level_index,
                        *self._hardware_sums[level_index].tolist(),
                    )
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary_file.replace(self.output_file)


class HardwareExperimentRecorder:
    """Overwrite the fixed aggregate table for every saved digit."""

    def __init__(
        self,
        output_directory: Path,
        *,
        input_features: int,
        basis_per_dimension: int,
        differential_levels: np.ndarray,
    ) -> None:
        self.output_directory = output_directory.expanduser().resolve()
        self.input_features = input_features
        self.basis_per_dimension = basis_per_dimension
        self.differential_levels = np.asarray(
            differential_levels,
            dtype=np.float64,
        )

    @property
    def output_file(self) -> Path:
        return self.output_directory / HARDWARE_EXPERIMENT_FILE_NAME

    def record_batch(
        self,
        differential_rows: np.ndarray,
        hardware_rows: np.ndarray,
    ) -> list[HardwareExperimentAggregator]:
        differential_values = np.asarray(differential_rows, dtype=np.float64)
        hardware_values = np.asarray(hardware_rows, dtype=np.float64)
        differential_values = np.atleast_2d(differential_values)
        hardware_values = np.atleast_2d(hardware_values)
        if len(differential_values) != len(hardware_values):
            raise ValueError(
                "Differential and hardware batches must contain the same number of rows."
            )

        experiments: list[HardwareExperimentAggregator] = []
        for differential_row, hardware_row in zip(
            differential_values,
            hardware_values,
            strict=True,
        ):
            experiment = HardwareExperimentAggregator.create(
                self.output_directory,
                input_features=self.input_features,
                basis_per_dimension=self.basis_per_dimension,
                differential_levels=self.differential_levels,
            )
            experiment.add_sample(differential_row, hardware_row)
            experiments.append(experiment)
        return experiments
