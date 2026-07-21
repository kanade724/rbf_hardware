from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    source: Path
    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    train: DatasetSplit
    test: DatasetSplit
    train_reference_rows: int
    test_reference_rows: int


@dataclass
class HardwareFeatureScaler:
    scaling: str
    negative_policy: str
    epsilon: float
    scale_: float | None = None

    def _handle_negative(self, features: np.ndarray) -> np.ndarray:
        if self.negative_policy == "clamp_zero":
            return np.maximum(features, 0.0)
        return features.copy()

    def fit(self, features: np.ndarray) -> "HardwareFeatureScaler":
        prepared = self._handle_negative(features)
        if self.scaling == "none":
            self.scale_ = 1.0
        elif self.scaling == "train_global_max":
            maximum = float(np.max(prepared))
            if not np.isfinite(maximum) or maximum <= self.epsilon:
                raise ValueError(
                    "Cannot fit train_global_max scaling: training maximum is not greater than epsilon."
                )
            self.scale_ = maximum
        else:
            raise ValueError(f"Unsupported scaling mode: {self.scaling}")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("HardwareFeatureScaler must be fitted before transform().")
        return self._handle_negative(features) / self.scale_

    def state_dict(self) -> dict[str, Any]:
        if self.scale_ is None:
            raise RuntimeError("HardwareFeatureScaler has not been fitted.")
        return {
            "scaling": self.scaling,
            "negative_policy": self.negative_policy,
            "epsilon": self.epsilon,
            "scale": self.scale_,
            "fit_split": "train",
        }


def _load_numeric_csv(path: Path, config: dict[str, Any]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    data_config = config["data"]
    skip_rows = 1 if bool(data_config.get("has_header", False)) else 0
    try:
        values = np.loadtxt(
            path,
            delimiter=str(data_config.get("delimiter", ",")),
            skiprows=skip_rows,
            dtype=np.float64,
            encoding=str(data_config.get("encoding", "utf-8-sig")),
        )
    except ValueError as exc:
        raise ValueError(f"Failed to parse numeric CSV {path}: {exc}") from exc
    values = np.atleast_2d(values)
    if values.size == 0:
        raise ValueError(f"Dataset file is empty: {path}")
    if not np.isfinite(values).all():
        bad_count = int(values.size - np.isfinite(values).sum())
        raise ValueError(f"Dataset contains {bad_count} NaN/Inf values: {path}")
    return values


def _extract_split(name: str, path: Path, config: dict[str, Any]) -> DatasetSplit:
    values = _load_numeric_csv(path, config)
    expected_features = int(config["data"]["input_features"])
    expected_columns = expected_features + 1
    if values.shape[1] != expected_columns:
        raise ValueError(
            f"{name} must contain {expected_columns} columns "
            f"({expected_features} post-RBF features + label), but {path} has {values.shape[1]}. "
            "A 17-column *_in.csv file is before the RBF layer and cannot be sent to the 256-input classifier."
        )

    label_column = int(config["data"].get("label_column", -1))
    labels_raw = values[:, label_column]
    labels = np.rint(labels_raw).astype(np.int64)
    if not np.array_equal(labels_raw, labels.astype(labels_raw.dtype)):
        raise ValueError(f"{name} labels must be integers: {path}")
    allowed = np.asarray(config["data"]["class_labels"], dtype=np.int64)
    unknown = np.setdiff1d(np.unique(labels), allowed)
    if len(unknown):
        raise ValueError(f"{name} contains unknown labels {unknown.tolist()}: {path}")

    features = np.delete(values, label_column, axis=1).astype(np.float32, copy=False)
    return DatasetSplit(name=name, source=path, features=features, labels=labels)


def _validate_reference(
    split: DatasetSplit,
    reference_path: Path,
    config: dict[str, Any],
) -> int:
    reference = _load_numeric_csv(reference_path, config)
    expected_columns = int(config["data"]["reference_input_features"]) + 1
    if reference.shape[1] != expected_columns:
        raise ValueError(
            f"{split.name} reference must have {expected_columns} columns, "
            f"but {reference_path} has {reference.shape[1]}."
        )
    allow_prefix = bool(config["data"].get("allow_reference_prefix", False))
    if len(reference) < len(split.labels):
        raise ValueError(
            f"{split.name} reference has only {len(reference)} rows for {len(split.labels)} hardware rows."
        )
    if len(reference) != len(split.labels) and not allow_prefix:
        raise ValueError(
            f"{split.name} reference has {len(reference)} rows but hardware data has {len(split.labels)}."
        )
    reference_labels_raw = reference[: len(split.labels), -1]
    reference_labels = np.rint(reference_labels_raw).astype(np.int64)
    mismatch = np.flatnonzero(reference_labels != split.labels)
    if len(mismatch):
        first = int(mismatch[0])
        raise ValueError(
            f"{split.name} label mismatch at zero-based row {first}: "
            f"hardware={split.labels[first]}, reference={reference_labels[first]}."
        )
    return int(len(reference))


def load_dataset_bundle(paths: Any, config: dict[str, Any]) -> DatasetBundle:
    train = _extract_split("train", paths.train_file, config)
    test = _extract_split("test", paths.test_file, config)
    if train.features.shape[1] != test.features.shape[1]:
        raise ValueError("Training and test feature counts differ.")

    train_reference_rows = 0
    test_reference_rows = 0
    if bool(config["data"].get("require_reference_label_match", True)):
        train_reference_rows = _validate_reference(train, paths.train_reference_file, config)
        test_reference_rows = _validate_reference(test, paths.test_reference_file, config)

    return DatasetBundle(
        train=train,
        test=test,
        train_reference_rows=train_reference_rows,
        test_reference_rows=test_reference_rows,
    )


def _row_keys(values: np.ndarray) -> set[bytes]:
    contiguous = np.ascontiguousarray(values)
    return {row.tobytes() for row in contiguous}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_dataset_isolation(bundle: DatasetBundle, paths: Any, config: dict[str, Any]) -> dict[str, Any]:
    feature_overlap = len(_row_keys(bundle.train.features) & _row_keys(bundle.test.features))
    train_labeled = np.column_stack((bundle.train.features, bundle.train.labels.astype(np.float32)))
    test_labeled = np.column_stack((bundle.test.features, bundle.test.labels.astype(np.float32)))
    labeled_overlap = len(_row_keys(train_labeled) & _row_keys(test_labeled))
    if bool(config["data"].get("reject_cross_split_duplicates", True)) and feature_overlap:
        raise ValueError(
            f"Data-isolation audit failed: {feature_overlap} exact feature rows occur in both train and test."
        )
    return {
        "train_file": str(paths.train_file),
        "test_file": str(paths.test_file),
        "train_reference_file": str(paths.train_reference_file),
        "test_reference_file": str(paths.test_reference_file),
        "train_rows": int(len(bundle.train.labels)),
        "test_rows": int(len(bundle.test.labels)),
        "cross_split_exact_feature_overlap": feature_overlap,
        "cross_split_exact_labeled_row_overlap": labeled_overlap,
        "reference_labels_verified": True,
        "reject_cross_split_duplicates": bool(
            config["data"].get("reject_cross_split_duplicates", True)
        ),
        "sha256": {
            "train_file": _sha256(paths.train_file),
            "test_file": _sha256(paths.test_file),
            "train_reference_file": _sha256(paths.train_reference_file),
            "test_reference_file": _sha256(paths.test_reference_file),
        },
    }
