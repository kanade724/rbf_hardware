from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


class ConfigError(ValueError):
    """Raised when config.yaml is missing or internally inconsistent."""


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    workspace_root: Path
    train_file: Path
    test_file: Path
    train_reference_file: Path
    test_reference_file: Path
    output_dir: Path
    log_file: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    _validate_config(config)
    return config


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"config.{key} must be a mapping.")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    paths = _require_mapping(config, "paths")
    data = _require_mapping(config, "data")
    preprocessing = _require_mapping(config, "preprocessing")
    classifier = _require_mapping(config, "classifier")
    runtime = _require_mapping(config, "runtime")
    output = _require_mapping(config, "output")
    _require_mapping(config, "logging")
    diagnostics = _require_mapping(config, "diagnostics")
    evaluation = _require_mapping(config, "evaluation")

    required_paths = {
        "workspace_root",
        "train_file",
        "test_file",
        "train_reference_file",
        "test_reference_file",
        "output_dir",
        "log_file",
    }
    missing_paths = sorted(required_paths - paths.keys())
    if missing_paths:
        raise ConfigError(f"Missing paths entries: {', '.join(missing_paths)}")

    input_features = int(data.get("input_features", 0))
    if input_features <= 0:
        raise ConfigError("data.input_features must be a positive integer.")
    if int(classifier.get("input_features", 0)) != input_features:
        raise ConfigError("classifier.input_features must equal data.input_features.")

    class_labels = data.get("class_labels")
    if not isinstance(class_labels, list) or not class_labels:
        raise ConfigError("data.class_labels must be a non-empty list.")
    if len(set(class_labels)) != len(class_labels):
        raise ConfigError("data.class_labels must not contain duplicates.")
    if int(classifier.get("num_classes", 0)) != len(class_labels):
        raise ConfigError("classifier.num_classes must equal len(data.class_labels).")
    if classifier.get("type") != "ridge_regression":
        raise ConfigError("classifier.type must be ridge_regression.")
    if float(classifier.get("ridge_alpha", -1)) < 0:
        raise ConfigError("classifier.ridge_alpha must be non-negative.")
    if not bool(classifier.get("fit_intercept", True)):
        raise ConfigError("This benchmark-compatible implementation requires fit_intercept: true.")
    if classifier.get("prediction") != "argmax":
        raise ConfigError("classifier.prediction must be argmax.")
    alpha_selection = classifier.get("alpha_selection")
    if not isinstance(alpha_selection, dict):
        raise ConfigError("classifier.alpha_selection must be a mapping.")
    if alpha_selection.get("strategy") != "stratified_kfold":
        raise ConfigError("classifier.alpha_selection.strategy must be stratified_kfold.")
    if int(alpha_selection.get("folds", 0)) < 2:
        raise ConfigError("classifier.alpha_selection.folds must be at least 2.")
    if alpha_selection.get("metric") not in {"accuracy", "macro_f1"}:
        raise ConfigError("classifier.alpha_selection.metric must be accuracy or macro_f1.")
    candidates = alpha_selection.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ConfigError("classifier.alpha_selection.candidates must be a non-empty list.")
    if any(float(candidate) < 0 for candidate in candidates):
        raise ConfigError("classifier.alpha_selection.candidates must be non-negative.")
    score_calibration = classifier.get("score_calibration")
    if not isinstance(score_calibration, dict):
        raise ConfigError("classifier.score_calibration must be a mapping.")
    if score_calibration.get("strategy") != "priority_class_bias_cv":
        raise ConfigError("classifier.score_calibration.strategy must be priority_class_bias_cv.")
    priority_class = int(score_calibration.get("priority_class", -1))
    if priority_class not in [int(label) for label in class_labels]:
        raise ConfigError("classifier.score_calibration.priority_class must be in data.class_labels.")
    minimum_recall = float(score_calibration.get("minimum_priority_recall", -1))
    if not 0 <= minimum_recall <= 1:
        raise ConfigError(
            "classifier.score_calibration.minimum_priority_recall must be between 0 and 1."
        )
    if score_calibration.get("metric") not in {"accuracy", "macro_f1"}:
        raise ConfigError("classifier.score_calibration.metric must be accuracy or macro_f1.")
    bias_candidates = score_calibration.get("bias_candidates")
    if not isinstance(bias_candidates, list) or not bias_candidates:
        raise ConfigError(
            "classifier.score_calibration.bias_candidates must be a non-empty list."
        )

    if preprocessing.get("negative_policy") not in {"keep", "clamp_zero"}:
        raise ConfigError("preprocessing.negative_policy must be keep or clamp_zero.")
    if preprocessing.get("scaling") not in {"none", "train_global_max"}:
        raise ConfigError("preprocessing.scaling must be none or train_global_max.")
    if float(preprocessing.get("epsilon", 0)) <= 0:
        raise ConfigError("preprocessing.epsilon must be positive.")

    if runtime.get("device", "auto") not in {"auto", "cpu", "cuda"}:
        raise ConfigError("runtime.device must be auto, cpu, or cuda.")
    if runtime.get("dtype", "float32") not in {"float32", "float64"}:
        raise ConfigError("runtime.dtype must be float32 or float64.")

    required_outputs = {
        "run_prefix",
        "weights_filename",
        "metrics_filename",
        "predictions_filename",
        "confusion_matrix_csv",
        "confusion_matrix_text",
        "confusion_matrix_svg",
        "classification_report_filename",
        "alpha_search_filename",
        "class_bias_search_filename",
        "config_snapshot_filename",
    }
    missing_outputs = sorted(required_outputs - output.keys())
    if missing_outputs:
        raise ConfigError(f"Missing output entries: {', '.join(missing_outputs)}")
    gap_warning = float(diagnostics.get("generalization_gap_warning", -1))
    recall_warning = float(diagnostics.get("class_recall_warning", -1))
    if not 0 <= gap_warning <= 1:
        raise ConfigError("diagnostics.generalization_gap_warning must be between 0 and 1.")
    if not 0 <= recall_warning <= 1:
        raise ConfigError("diagnostics.class_recall_warning must be between 0 and 1.")
    if evaluation.get("test_set_status") not in {
        "pristine_blind_test",
        "observed_during_development",
    }:
        raise ConfigError(
            "evaluation.test_set_status must be pristine_blind_test or observed_during_development."
        )


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    root = project_root()
    path_config = config["paths"]
    configured_workspace = Path(path_config["workspace_root"]).expanduser()
    workspace = (
        configured_workspace.resolve()
        if configured_workspace.is_absolute()
        else (root / configured_workspace).resolve()
    )

    def workspace_path(key: str) -> Path:
        value = Path(path_config[key]).expanduser()
        return value.resolve() if value.is_absolute() else (workspace / value).resolve()

    return ProjectPaths(
        project_root=root,
        workspace_root=workspace,
        train_file=workspace_path("train_file"),
        test_file=workspace_path("test_file"),
        train_reference_file=workspace_path("train_reference_file"),
        test_reference_file=workspace_path("test_reference_file"),
        output_dir=workspace_path("output_dir"),
        log_file=workspace_path("log_file"),
    )


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config["runtime"].get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is cuda, but CUDA is not available.")
    return torch.device(requested)


def resolve_dtype(config: dict[str, Any]) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[config["runtime"]["dtype"]]
