from __future__ import annotations

import csv
import html
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Evaluation:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    confusion_matrix: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray

    def summary(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
        }


def evaluate(labels: np.ndarray, predictions: np.ndarray, classes: list[int]) -> Evaluation:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    class_to_index = {label: index for index, label in enumerate(classes)}
    try:
        true_indices = np.fromiter((class_to_index[int(x)] for x in labels), dtype=np.int64)
        predicted_indices = np.fromiter((class_to_index[int(x)] for x in predictions), dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown class in labels or predictions: {exc.args[0]}") from exc

    count = len(classes)
    flat = np.bincount(true_indices * count + predicted_indices, minlength=count * count)
    matrix = flat.reshape(count, count)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_total = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.int64)
    precision = np.divide(true_positive, predicted_total, out=np.zeros_like(true_positive), where=predicted_total > 0)
    recall = np.divide(true_positive, support, out=np.zeros_like(true_positive), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(precision), where=denominator > 0)
    accuracy = float(true_positive.sum() / len(labels)) if len(labels) else 0.0
    return Evaluation(
        accuracy=accuracy,
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
        confusion_matrix=matrix,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
    )


def confusion_matrix_text(matrix: np.ndarray, classes: list[int]) -> str:
    width = max(5, max(len(str(int(value))) for value in matrix.flat) + 1)
    header = "true\\pred".ljust(10) + "".join(str(label).rjust(width) for label in classes)
    rows = [header]
    for label, row in zip(classes, matrix, strict=True):
        rows.append(str(label).ljust(10) + "".join(str(int(value)).rjust(width) for value in row))
    return "\n".join(rows)


def write_confusion_csv(path: Path, matrix: np.ndarray, classes: list[int]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["true\\pred", *classes])
        for label, row in zip(classes, matrix, strict=True):
            writer.writerow([label, *(int(value) for value in row)])


def write_predictions(path: Path, labels: np.ndarray, predictions: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["label", "prediction", "correct"])
        for label, prediction in zip(labels, predictions, strict=True):
            writer.writerow([int(label), int(prediction), int(label == prediction)])


def write_alpha_search(path: Path, candidates: list[dict[str, object]], classes: list[int]) -> None:
    recall_columns = [f"recall_class_{label}" for label in classes]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "alpha",
                "mean_accuracy",
                "std_accuracy",
                "mean_macro_f1",
                "std_macro_f1",
                *recall_columns,
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate["alpha"],
                    candidate["mean_accuracy"],
                    candidate["std_accuracy"],
                    candidate["mean_macro_f1"],
                    candidate["std_macro_f1"],
                    *candidate["mean_per_class_recall"],
                ]
            )


def write_class_bias_search(path: Path, candidates: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["bias", "accuracy", "macro_f1", "priority_recall"])
        for candidate in candidates:
            writer.writerow(
                [
                    candidate["bias"],
                    candidate["accuracy"],
                    candidate["macro_f1"],
                    candidate["priority_recall"],
                ]
            )


def classification_report_text(evaluation: Evaluation, classes: list[int]) -> str:
    lines = [f"{'class':>10} {'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>10}"]
    for index, label in enumerate(classes):
        lines.append(
            f"{label:>10} {evaluation.precision[index]:>12.4f} {evaluation.recall[index]:>12.4f} "
            f"{evaluation.f1[index]:>12.4f} {int(evaluation.support[index]):>10}"
        )
    lines.extend(
        [
            "",
            f"{'accuracy':>10} {'':>12} {'':>12} {evaluation.accuracy:>12.4f} {int(evaluation.support.sum()):>10}",
            f"{'macro avg':>10} {evaluation.macro_precision:>12.4f} {evaluation.macro_recall:>12.4f} "
            f"{evaluation.macro_f1:>12.4f} {int(evaluation.support.sum()):>10}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_confusion_svg(path: Path, matrix: np.ndarray, classes: list[int]) -> None:
    cell = 58
    left = 105
    top = 85
    right = 24
    bottom = 70
    width = left + cell * len(classes) + right
    height = top + cell * len(classes) + bottom
    maximum = max(int(matrix.max()), 1)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif}.title{font-size:20px;font-weight:700}.axis{font-size:14px;font-weight:600}.tick{font-size:13px}.value{font-size:12px;font-weight:600}</style>',
        f'<text class="title" x="{width / 2:.1f}" y="30" text-anchor="middle">Confusion Matrix</text>',
        f'<text class="axis" x="{left + cell * len(classes) / 2:.1f}" y="55" text-anchor="middle">Predicted label</text>',
        f'<text class="axis" x="22" y="{top + cell * len(classes) / 2:.1f}" text-anchor="middle" transform="rotate(-90 22 {top + cell * len(classes) / 2:.1f})">True label</text>',
    ]
    for index, label in enumerate(classes):
        x = left + index * cell + cell / 2
        y = top + index * cell + cell / 2 + 5
        elements.append(f'<text class="tick" x="{x:.1f}" y="{top - 12}" text-anchor="middle">{html.escape(str(label))}</text>')
        elements.append(f'<text class="tick" x="{left - 16}" y="{y:.1f}" text-anchor="middle">{html.escape(str(label))}</text>')
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            ratio = float(value) / maximum
            red = int(239 - 190 * ratio)
            green = int(246 - 120 * ratio)
            blue = int(255 - 30 * ratio)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            text_fill = "#ffffff" if ratio > 0.55 else "#111827"
            x = left + column_index * cell
            y = top + row_index * cell
            elements.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#ffffff"/>')
            elements.append(
                f'<text class="value" x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                f'text-anchor="middle" fill="{text_fill}">{int(value)}</text>'
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")
