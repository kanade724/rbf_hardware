from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data import HardwareFeatureScaler
from .metrics import evaluate
from .model import RidgeLinearClassifier


@dataclass(frozen=True)
class AlphaCandidateResult:
    alpha: float
    mean_accuracy: float
    std_accuracy: float
    mean_macro_f1: float
    std_macro_f1: float
    mean_per_class_recall: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_macro_f1": self.mean_macro_f1,
            "std_macro_f1": self.std_macro_f1,
            "mean_per_class_recall": self.mean_per_class_recall,
        }


@dataclass(frozen=True)
class AlphaSelectionResult:
    selected_alpha: float
    metric: str
    folds: int
    random_state: int
    candidates: list[AlphaCandidateResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "strategy": "stratified_kfold",
            "selected_alpha": self.selected_alpha,
            "metric": self.metric,
            "folds": self.folds,
            "random_state": self.random_state,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class ClassBiasCandidateResult:
    bias: float
    accuracy: float
    macro_f1: float
    priority_recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "bias": self.bias,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "priority_recall": self.priority_recall,
        }


@dataclass(frozen=True)
class ClassBiasSelectionResult:
    priority_class: int
    selected_bias: float
    minimum_priority_recall: float
    metric: str
    folds: int
    random_state: int
    candidates: list[ClassBiasCandidateResult]

    def bias_vector(self, classes: list[int]) -> np.ndarray:
        result = np.zeros(len(classes), dtype=np.float32)
        result[classes.index(self.priority_class)] = self.selected_bias
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "strategy": "priority_class_bias_cv",
            "priority_class": self.priority_class,
            "selected_bias": self.selected_bias,
            "minimum_priority_recall": self.minimum_priority_recall,
            "metric": self.metric,
            "folds": self.folds,
            "random_state": self.random_state,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
        }


def _stratified_folds(labels: np.ndarray, folds: int, random_state: int) -> list[np.ndarray]:
    counts = np.bincount(labels)
    present_counts = counts[counts > 0]
    if len(present_counts) == 0 or int(present_counts.min()) < folds:
        raise ValueError(
            f"Each class needs at least {folds} samples for stratified cross-validation."
        )
    generator = np.random.default_rng(random_state)
    fold_rows: list[list[int]] = [[] for _ in range(folds)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        for fold_index, partition in enumerate(np.array_split(indices, folds)):
            fold_rows[fold_index].extend(int(index) for index in partition)
    return [np.asarray(sorted(rows), dtype=np.int64) for rows in fold_rows]


def select_ridge_alpha(
    raw_features: np.ndarray,
    labels: np.ndarray,
    classes: list[int],
    classifier_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    random_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AlphaSelectionResult:
    selection_config = classifier_config["alpha_selection"]
    folds = int(selection_config["folds"])
    metric = str(selection_config["metric"])
    fold_indices = _stratified_folds(labels, folds, random_state)
    all_indices = np.arange(len(labels), dtype=np.int64)
    candidates: list[AlphaCandidateResult] = []

    for alpha_value in selection_config["candidates"]:
        alpha = float(alpha_value)
        accuracies: list[float] = []
        macro_f1_scores: list[float] = []
        recalls: list[np.ndarray] = []
        for validation_indices in fold_indices:
            training_mask = np.ones(len(labels), dtype=bool)
            training_mask[validation_indices] = False
            training_indices = all_indices[training_mask]
            scaler = HardwareFeatureScaler(
                scaling=str(preprocessing_config["scaling"]),
                negative_policy=str(preprocessing_config["negative_policy"]),
                epsilon=float(preprocessing_config["epsilon"]),
            ).fit(raw_features[training_indices])
            fold_train = scaler.transform(raw_features[training_indices])
            fold_validation = scaler.transform(raw_features[validation_indices])
            classifier = RidgeLinearClassifier(
                input_features=int(classifier_config["input_features"]),
                classes=classes,
                ridge_alpha=alpha,
                fit_intercept=bool(classifier_config["fit_intercept"]),
                regularize_intercept=bool(classifier_config["regularize_intercept"]),
                device=device,
                dtype=dtype,
            )
            classifier.fit(
                torch.from_numpy(fold_train),
                torch.from_numpy(labels[training_indices]),
            )
            predictions = classifier.predict(torch.from_numpy(fold_validation)).cpu().numpy()
            fold_evaluation = evaluate(labels[validation_indices], predictions, classes)
            accuracies.append(fold_evaluation.accuracy)
            macro_f1_scores.append(fold_evaluation.macro_f1)
            recalls.append(fold_evaluation.recall)
        candidates.append(
            AlphaCandidateResult(
                alpha=alpha,
                mean_accuracy=float(np.mean(accuracies)),
                std_accuracy=float(np.std(accuracies)),
                mean_macro_f1=float(np.mean(macro_f1_scores)),
                std_macro_f1=float(np.std(macro_f1_scores)),
                mean_per_class_recall=np.mean(np.asarray(recalls), axis=0).tolist(),
            )
        )

    score_name = "mean_macro_f1" if metric == "macro_f1" else "mean_accuracy"
    selected = max(candidates, key=lambda candidate: getattr(candidate, score_name))
    return AlphaSelectionResult(
        selected_alpha=selected.alpha,
        metric=metric,
        folds=folds,
        random_state=random_state,
        candidates=candidates,
    )


def select_priority_class_bias(
    raw_features: np.ndarray,
    labels: np.ndarray,
    classes: list[int],
    selected_alpha: float,
    classifier_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    random_state: int,
    device: torch.device,
    dtype: torch.dtype,
) -> ClassBiasSelectionResult:
    calibration_config = classifier_config["score_calibration"]
    folds = int(classifier_config["alpha_selection"]["folds"])
    fold_indices = _stratified_folds(labels, folds, random_state)
    all_indices = np.arange(len(labels), dtype=np.int64)
    out_of_fold_scores = np.empty((len(labels), len(classes)), dtype=np.float32)

    for validation_indices in fold_indices:
        training_mask = np.ones(len(labels), dtype=bool)
        training_mask[validation_indices] = False
        training_indices = all_indices[training_mask]
        scaler = HardwareFeatureScaler(
            scaling=str(preprocessing_config["scaling"]),
            negative_policy=str(preprocessing_config["negative_policy"]),
            epsilon=float(preprocessing_config["epsilon"]),
        ).fit(raw_features[training_indices])
        fold_train = scaler.transform(raw_features[training_indices])
        fold_validation = scaler.transform(raw_features[validation_indices])
        classifier = RidgeLinearClassifier(
            input_features=int(classifier_config["input_features"]),
            classes=classes,
            ridge_alpha=selected_alpha,
            fit_intercept=bool(classifier_config["fit_intercept"]),
            regularize_intercept=bool(classifier_config["regularize_intercept"]),
            device=device,
            dtype=dtype,
        )
        classifier.fit(
            torch.from_numpy(fold_train),
            torch.from_numpy(labels[training_indices]),
        )
        out_of_fold_scores[validation_indices] = (
            classifier.scores(torch.from_numpy(fold_validation)).cpu().numpy()
        )

    priority_class = int(calibration_config["priority_class"])
    priority_index = classes.index(priority_class)
    candidates: list[ClassBiasCandidateResult] = []
    for bias_value in calibration_config["bias_candidates"]:
        bias = float(bias_value)
        calibrated_scores = out_of_fold_scores.copy()
        calibrated_scores[:, priority_index] += bias
        predictions = np.asarray(classes, dtype=np.int64)[calibrated_scores.argmax(axis=1)]
        candidate_evaluation = evaluate(labels, predictions, classes)
        candidates.append(
            ClassBiasCandidateResult(
                bias=bias,
                accuracy=candidate_evaluation.accuracy,
                macro_f1=candidate_evaluation.macro_f1,
                priority_recall=float(candidate_evaluation.recall[priority_index]),
            )
        )

    minimum_recall = float(calibration_config["minimum_priority_recall"])
    feasible = [candidate for candidate in candidates if candidate.priority_recall >= minimum_recall]
    pool = feasible if feasible else candidates
    metric = str(calibration_config["metric"])
    selected = max(
        pool,
        key=lambda candidate: (
            candidate.macro_f1 if metric == "macro_f1" else candidate.accuracy,
            -candidate.bias,
        ),
    )
    return ClassBiasSelectionResult(
        priority_class=priority_class,
        selected_bias=selected.bias,
        minimum_priority_recall=minimum_recall,
        metric=metric,
        folds=folds,
        random_state=random_state,
        candidates=candidates,
    )
