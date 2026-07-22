"""Gaussian-calibrated simulation of the 16-by-16 hardware RBF response bank."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CALIBRATION_COLUMNS = (
    "group_index",
    "differential_level",
    "amplitude",
    "mean",
    "std_dev",
)


@dataclass(frozen=True)
class GaussianCalibrationBank:
    levels: np.ndarray
    amplitudes: np.ndarray
    means: np.ndarray
    standard_deviations: np.ndarray

    @property
    def group_count(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_groups: int,
        expected_levels: np.ndarray,
    ) -> "GaussianCalibrationBank":
        source = path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Gaussian calibration CSV does not exist: {source}")
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CALIBRATION_COLUMNS:
                raise ValueError(
                    f"{source} must use columns {list(CALIBRATION_COLUMNS)}; "
                    f"found {reader.fieldnames}."
                )
            records = list(reader)

        level_count = len(expected_levels)
        expected_rows = expected_groups * level_count
        if len(records) != expected_rows:
            raise ValueError(
                f"{source} contains {len(records)} calibration rows; expected "
                f"{expected_groups} groups x {level_count} levels = {expected_rows}."
            )

        amplitudes = np.empty((expected_groups, level_count), dtype=np.float64)
        means = np.empty_like(amplitudes)
        deviations = np.empty_like(amplitudes)
        seen = np.zeros((expected_groups, level_count), dtype=bool)
        for row_number, record in enumerate(records, start=2):
            try:
                group_index = int(record["group_index"]) - 1
                level = float(record["differential_level"])
                amplitude = float(record["amplitude"])
                mean = float(record["mean"])
                deviation = float(record["std_dev"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{source}: calibration row {row_number} contains an invalid value."
                ) from error
            if not 0 <= group_index < expected_groups:
                raise ValueError(
                    f"{source}: row {row_number} group_index must be 1..{expected_groups}."
                )
            level_index = int(np.searchsorted(expected_levels, level, side="left"))
            if level_index >= level_count or not np.isclose(
                expected_levels[level_index], level, rtol=0.0, atol=1.0e-8
            ):
                raise ValueError(
                    f"{source}: row {row_number} level {level} is not in differential_levels.csv."
                )
            if seen[group_index, level_index]:
                raise ValueError(
                    f"{source}: duplicate group {group_index + 1}, level {level}."
                )
            if not np.isfinite((amplitude, mean, deviation)).all() or deviation < 0:
                raise ValueError(
                    f"{source}: row {row_number} parameters must be finite and std_dev non-negative."
                )
            amplitudes[group_index, level_index] = amplitude
            means[group_index, level_index] = mean
            deviations[group_index, level_index] = deviation
            seen[group_index, level_index] = True

        if not seen.all():
            missing = np.argwhere(~seen)[0]
            raise ValueError(
                f"{source}: missing group {int(missing[0]) + 1}, "
                f"level index {int(missing[1])}."
            )
        return cls(
            levels=np.asarray(expected_levels, dtype=np.float64),
            amplitudes=amplitudes,
            means=means,
            standard_deviations=deviations,
        )

    def simulate(
        self,
        differential_features: np.ndarray,
        *,
        sampling_mode: str,
        random_generator: np.random.Generator,
    ) -> np.ndarray:
        features = np.asarray(differential_features, dtype=np.float64)
        features = np.atleast_2d(features)
        if features.ndim != 2 or not np.isfinite(features).all():
            raise ValueError("Differential features must be a finite two-dimensional matrix.")

        upper_indices = np.searchsorted(self.levels, features, side="left")
        upper_indices = np.clip(upper_indices, 0, len(self.levels) - 1)
        lower_indices = np.maximum(upper_indices - 1, 0)
        lower_distance = np.abs(features - self.levels[lower_indices])
        upper_distance = np.abs(self.levels[upper_indices] - features)
        level_indices = np.where(
            upper_distance <= lower_distance, upper_indices, lower_indices
        )
        selected_levels = self.levels[level_indices]
        if not np.allclose(selected_levels, features, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "Differential features must already be quantized to differential_levels.csv."
            )

        selected_means = self.means[:, level_indices].transpose(1, 2, 0)
        selected_deviations = self.standard_deviations[:, level_indices].transpose(1, 2, 0)
        if sampling_mode == "mean":
            simulated = selected_means
        elif sampling_mode == "gaussian":
            simulated = random_generator.normal(selected_means, selected_deviations)
        else:
            raise ValueError("sampling_mode must be 'gaussian' or 'mean'.")

        # Dimension-major layout: [d1_group1..16, d2_group1..16, ...].
        return simulated.reshape(len(features), -1).astype(np.float32)
