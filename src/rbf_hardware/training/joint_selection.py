from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..modeling.joint_gaussian import FullJointGaussianTransformer, benchmark_kmeans
from ..reporting.metrics import evaluate
from .model_selection import _stratified_folds


@dataclass(frozen=True)
class JointGaussianCandidateResult:
    joint_sigma: float
    calibration_alpha: float
    factor_upper: float
    head_alpha: float
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_per_class_recall: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "joint_sigma": self.joint_sigma,
            "calibration_alpha": self.calibration_alpha,
            "factor_upper": self.factor_upper,
            "head_alpha": self.head_alpha,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "mean_per_class_recall": self.mean_per_class_recall,
        }


@dataclass(frozen=True)
class JointGaussianSelectionResult:
    selected_joint_sigma: float
    selected_calibration_alpha: float
    selected_factor_upper: float
    selected_head_alpha: float
    metric: str
    folds: int
    random_state: int
    candidates: list[JointGaussianCandidateResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "strategy": "train_only_stratified_kfold",
            "selected_joint_sigma": self.selected_joint_sigma,
            "selected_calibration_alpha": self.selected_calibration_alpha,
            "selected_factor_upper": self.selected_factor_upper,
            "selected_head_alpha": self.selected_head_alpha,
            "metric": self.metric,
            "folds": self.folds,
            "random_state": self.random_state,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _ridge_predict(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    classes: list[int],
    alpha: float,
) -> np.ndarray:
    class_array = np.asarray(classes, dtype=np.int64)
    class_indices = np.searchsorted(class_array, train_labels)
    targets = np.eye(len(classes), dtype=np.float64)[class_indices]
    design = np.column_stack((np.ones(len(train_features)), train_features)).astype(np.float64)
    validation_design = np.column_stack(
        (np.ones(len(validation_features)), validation_features)
    ).astype(np.float64)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    weights = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ targets,
    )
    return class_array[(validation_design @ weights).argmax(axis=1)]


def select_joint_gaussian_parameters(
    hardware_features: np.ndarray,
    reference_inputs: np.ndarray,
    labels: np.ndarray,
    classes: list[int],
    feature_config: dict[str, Any],
    classifier_config: dict[str, Any],
    random_state: int,
    progress: Callable[[str], None] | None = None,
) -> JointGaussianSelectionResult:
    selection = feature_config["selection"]
    folds = int(selection["folds"])
    metric = str(selection["metric"])
    fold_indices = _stratified_folds(labels, folds, random_state)
    all_indices = np.arange(len(labels), dtype=np.int64)
    dimensions = int(feature_config["dimensions"])
    basis = int(feature_config["basis_per_dimension"])
    outputs = int(feature_config["output_features"])
    lower = float(feature_config["factor_clip"]["lower"])
    epsilon = float(feature_config["calibration"]["epsilon"])
    max_iter = int(feature_config["center_selection"]["max_iter"])
    head_alphas = [float(value) for value in classifier_config["alpha_selection"]["candidates"]]

    keys = [
        (float(sigma), float(calibration), float(upper), head_alpha)
        for sigma in selection["joint_sigma_candidates"]
        for calibration in selection["calibration_alpha_candidates"]
        for upper in selection["factor_upper_candidates"]
        for head_alpha in head_alphas
    ]
    fold_scores: dict[tuple[float, float, float, float], list[tuple[float, float, np.ndarray]]] = {
        key: [] for key in keys
    }

    for fold_number, validation_indices in enumerate(fold_indices, start=1):
        training_mask = np.ones(len(labels), dtype=bool)
        training_mask[validation_indices] = False
        training_indices = all_indices[training_mask]
        centers = benchmark_kmeans(
            reference_inputs[training_indices],
            centers=outputs,
            random_state=random_state,
            max_iter=max_iter,
        )
        if progress is not None:
            progress(
                f"Joint-Gaussian CV fold {fold_number}/{folds}: "
                f"train={len(training_indices)}, validation={len(validation_indices)}, centers fitted"
            )
        for sigma in selection["joint_sigma_candidates"]:
            for calibration in selection["calibration_alpha_candidates"]:
                for upper in selection["factor_upper_candidates"]:
                    transformer = FullJointGaussianTransformer(
                        dimensions=dimensions,
                        basis_per_dimension=basis,
                        output_features=outputs,
                        joint_sigma=float(sigma),
                        calibration_alpha=float(calibration),
                        factor_lower=lower,
                        factor_upper=float(upper),
                        epsilon=epsilon,
                        random_state=random_state,
                        kmeans_max_iter=max_iter,
                    ).fit(
                        hardware_features[training_indices],
                        reference_inputs[training_indices],
                        preset_centers=centers,
                    )
                    fold_train = transformer.transform(hardware_features[training_indices])
                    fold_validation = transformer.transform(hardware_features[validation_indices])
                    for head_alpha in head_alphas:
                        predictions = _ridge_predict(
                            fold_train,
                            labels[training_indices],
                            fold_validation,
                            classes,
                            head_alpha,
                        )
                        evaluation = evaluate(labels[validation_indices], predictions, classes)
                        key = (
                            float(sigma),
                            float(calibration),
                            float(upper),
                            head_alpha,
                        )
                        fold_scores[key].append(
                            (evaluation.accuracy, evaluation.macro_f1, evaluation.recall)
                        )

    candidates: list[JointGaussianCandidateResult] = []
    for sigma, calibration, upper, head_alpha in keys:
        scores = fold_scores[(sigma, calibration, upper, head_alpha)]
        accuracies = np.asarray([item[0] for item in scores], dtype=np.float64)
        macro_f1 = np.asarray([item[1] for item in scores], dtype=np.float64)
        recalls = np.asarray([item[2] for item in scores], dtype=np.float64)
        candidates.append(
            JointGaussianCandidateResult(
                joint_sigma=sigma,
                calibration_alpha=calibration,
                factor_upper=upper,
                head_alpha=head_alpha,
                mean_accuracy=float(accuracies.mean()),
                std_accuracy=float(accuracies.std()),
                mean_macro_f1=float(macro_f1.mean()),
                std_macro_f1=float(macro_f1.std()),
                mean_per_class_recall=recalls.mean(axis=0).tolist(),
            )
        )

    score_name = "mean_macro_f1" if metric == "macro_f1" else "mean_accuracy"
    selected = max(candidates, key=lambda candidate: getattr(candidate, score_name))
    return JointGaussianSelectionResult(
        selected_joint_sigma=selected.joint_sigma,
        selected_calibration_alpha=selected.calibration_alpha,
        selected_factor_upper=selected.factor_upper,
        selected_head_alpha=selected.head_alpha,
        metric=metric,
        folds=folds,
        random_state=random_state,
        candidates=candidates,
    )
