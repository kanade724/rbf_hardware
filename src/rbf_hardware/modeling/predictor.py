"""Checkpoint-backed predictor for simulated or measured hardware features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .joint_gaussian import FullJointGaussianTransformer


@dataclass(frozen=True)
class JointGaussianPredictor:
    transformer: FullJointGaussianTransformer
    weight: np.ndarray
    bias: np.ndarray
    classes: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, checkpoint_path: Path) -> "JointGaussianPredictor":
        checkpoint = torch.load(
            checkpoint_path.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        if int(checkpoint.get("format_version", 0)) != 2:
            raise ValueError("Expected a format_version=2 joint-Gaussian checkpoint.")
        if checkpoint.get("model_type") != "hardware_full_joint_gaussian_ridge":
            raise ValueError("Checkpoint is not a hardware full-joint-Gaussian model.")
        state = checkpoint["state_dict"]
        weight = state["weight"].detach().cpu().numpy().astype(np.float32, copy=False)
        bias = state["bias"].detach().cpu().numpy().astype(np.float32, copy=False)
        classes_value = checkpoint["classes"]
        if isinstance(classes_value, torch.Tensor):
            classes_value = classes_value.detach().cpu().numpy()
        return cls(
            transformer=FullJointGaussianTransformer.from_state_dict(
                checkpoint["feature_transform"]
            ),
            weight=weight,
            bias=bias,
            classes=np.asarray(classes_value, dtype=np.int64),
            metadata=checkpoint,
        )

    def scores(self, hardware_features: np.ndarray) -> np.ndarray:
        joint_features = self.transformer.transform(hardware_features)
        return joint_features @ self.weight.T + self.bias

    def predict(self, hardware_features: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(hardware_features).argmax(axis=1)]
