from __future__ import annotations

import unittest

import numpy as np

from rbf_hardware.modeling.joint_gaussian import FullJointGaussianTransformer
from rbf_hardware.training.joint_selection import select_joint_gaussian_parameters


class FullJointGaussianTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = np.random.default_rng(42)
        self.reference = generator.uniform(0.0, 1.0, size=(40, 2)).astype(np.float32)
        first = np.column_stack(
            (
                np.exp(-np.square(self.reference[:, 0] - 0.25) / 0.08),
                np.exp(-np.square(self.reference[:, 0] - 0.75) / 0.08),
            )
        )
        second = np.column_stack(
            (
                np.exp(-np.square(self.reference[:, 1] - 0.25) / 0.08),
                np.exp(-np.square(self.reference[:, 1] - 0.75) / 0.08),
            )
        )
        self.hardware = np.column_stack((first, second)).astype(np.float32)
        self.transformer = FullJointGaussianTransformer(
            dimensions=2,
            basis_per_dimension=2,
            output_features=3,
            joint_sigma=0.3,
            calibration_alpha=0.1,
            factor_lower=1.0e-6,
            factor_upper=1.2,
            epsilon=1.0e-12,
            random_state=42,
            kmeans_max_iter=50,
        ).fit(self.hardware, self.reference)

    def test_hardware_only_transform_is_finite(self) -> None:
        transformed = self.transformer.transform(self.hardware)
        self.assertEqual(transformed.shape, (40, 3))
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue((transformed >= 0).all())

    def test_pc_transform_matches_joint_gaussian_formula(self) -> None:
        actual = self.transformer.ideal_pc_transform(self.reference)
        squared = np.square(
            self.reference[:, None, :] - self.transformer.centers_[None, :, :]
        ).sum(axis=2)
        expected = np.exp(-squared / (2.0 * self.transformer.joint_sigma**2))
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-7)

    def test_state_round_trip_preserves_features(self) -> None:
        restored = FullJointGaussianTransformer.from_state_dict(
            self.transformer.state_dict()
        )
        np.testing.assert_array_equal(
            restored.transform(self.hardware),
            self.transformer.transform(self.hardware),
        )

    def test_joint_search_reports_every_completed_fold_parameter_tuple(self) -> None:
        updates: list[tuple[int, str]] = []
        result = select_joint_gaussian_parameters(
            hardware_features=self.hardware,
            reference_inputs=self.reference,
            labels=np.asarray([0, 1] * 20, dtype=np.int64),
            classes=[0, 1],
            feature_config={
                "dimensions": 2,
                "basis_per_dimension": 2,
                "output_features": 3,
                "factor_clip": {"lower": 1.0e-6},
                "calibration": {"epsilon": 1.0e-12},
                "center_selection": {"max_iter": 20},
                "selection": {
                    "folds": 2,
                    "metric": "macro_f1",
                    "joint_sigma_candidates": [0.3],
                    "calibration_alpha_candidates": [0.1],
                    "factor_upper_candidates": [1.2],
                },
            },
            classifier_config={
                "alpha_selection": {"candidates": [0.01, 0.1]},
            },
            random_state=42,
            progress_update=lambda step, detail: updates.append((step, detail)),
        )

        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(sum(step for step, _detail in updates), 4)
        self.assertTrue(all("折" in detail for _step, detail in updates))
