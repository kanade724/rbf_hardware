from __future__ import annotations

import json
import logging
import platform
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..configuration.settings import load_config, resolve_device, resolve_dtype, resolve_paths
from ..data.training_dataset import HardwareFeatureScaler, audit_dataset_isolation, load_dataset_bundle
from ..infrastructure.logging import setup_logging
from ..modeling.ridge_classifier import RidgeLinearClassifier
from ..reporting.metrics import (
    classification_report_text,
    confusion_matrix_text,
    evaluate,
    write_alpha_search,
    write_class_bias_search,
    write_confusion_csv,
    write_confusion_svg,
    write_predictions,
)
from .model_selection import (
    AlphaSelectionResult,
    ClassBiasSelectionResult,
    select_priority_class_bias,
    select_ridge_alpha,
)


@dataclass(frozen=True)
class TrainingResult:
    run_dir: Path
    weights_path: Path
    confusion_matrix_path: Path
    log_path: Path
    test_accuracy: float
    test_macro_f1: float


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _set_reproducibility(config: dict[str, Any]) -> None:
    seed = int(config["runtime"]["random_state"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(config["runtime"].get("deterministic", True))
    torch.use_deterministic_algorithms(deterministic)


def _create_run_dir(output_dir: Path, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_dir / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _negative_fraction(values: np.ndarray) -> float:
    return float(np.count_nonzero(values < 0) / values.size)


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def run_training(config_path: Path) -> TrainingResult:
    resolved_config = config_path.expanduser().resolve()
    config = load_config(resolved_config)
    paths = resolve_paths(config)
    logger = setup_logging(
        paths.log_file,
        level_name=str(config["logging"].get("level", "INFO")),
        file_mode=str(config["logging"].get("file_mode", "w")),
    )
    output_config = config["output"]
    run_dir: Path | None = None
    try:
        logger.info("Starting post-RBF fully connected classifier training")
        logger.info("Configuration: %s", resolved_config)
        logger.info("Workspace root: %s", paths.workspace_root)
        logger.info("Shared log: %s", paths.log_file)
        logger.info("Python %s | PyTorch %s", platform.python_version(), torch.__version__)

        _set_reproducibility(config)
        device = resolve_device(config)
        dtype = resolve_dtype(config)
        logger.info(
            "Runtime: device=%s, dtype=%s, random_state=%s, deterministic=%s",
            device,
            str(dtype).removeprefix("torch."),
            config["runtime"]["random_state"],
            config["runtime"].get("deterministic", True),
        )

        run_dir = _create_run_dir(paths.output_dir, str(output_config["run_prefix"]))
        shutil.copy2(resolved_config, run_dir / str(output_config["config_snapshot_filename"]))
        logger.info("Run output directory: %s", run_dir)

        logger.info("[LOOK] Loading and validating hardware datasets")
        bundle = load_dataset_bundle(paths, config)
        data_audit = audit_dataset_isolation(bundle, paths, config)
        logger.info(
            "Training data: %s | rows=%d, features=%d, range=[%.10g, %.10g]",
            bundle.train.source,
            len(bundle.train.labels),
            bundle.train.features.shape[1],
            float(bundle.train.features.min()),
            float(bundle.train.features.max()),
        )
        logger.info(
            "Test data: %s | rows=%d, features=%d, range=[%.10g, %.10g]",
            bundle.test.source,
            len(bundle.test.labels),
            bundle.test.features.shape[1],
            float(bundle.test.features.min()),
            float(bundle.test.features.max()),
        )
        logger.info(
            "Reference label checks passed: train=%d/%d rows, test=%d/%d prefix rows",
            len(bundle.train.labels),
            bundle.train_reference_rows,
            len(bundle.test.labels),
            bundle.test_reference_rows,
        )
        logger.info(
            "[AUDIT] Split isolation passed: exact feature overlap=%d, exact labeled-row overlap=%d",
            data_audit["cross_split_exact_feature_overlap"],
            data_audit["cross_split_exact_labeled_row_overlap"],
        )
        joint_enabled = bool(config.get("feature_transform", {}).get("enabled", False))
        if joint_enabled:
            logger.info(
                "[AUDIT] Joint model selection consumes train_out plus paired train_reference inputs; "
                "test_out is reserved for final hardware prediction and test_reference is excluded "
                "from hardware feature generation",
            )
        else:
            logger.info(
                "[AUDIT] Model selection, scaler fitting, and classifier fitting consume train_out only; "
                "test_out is used for final prediction/metrics only",
            )
        test_status = str(config["evaluation"]["test_set_status"])
        if test_status == "observed_during_development":
            logger.warning(
                "[LIMIT] test_out was inspected during iterative development: reported metrics are exact for these rows but are not a pristine blind-test estimate; acquire new hardware outputs for a final blind claim",
            )
        if bundle.test_reference_rows > len(bundle.test.labels):
            uncovered = bundle.test_reference_rows - len(bundle.test.labels)
            coverage = len(bundle.test.labels) / bundle.test_reference_rows
            logger.warning(
                "[LOOK] Hardware test coverage is %.2f%%: %d reference inputs have device outputs, %d do not",
                100 * coverage,
                len(bundle.test.labels),
                uncovered,
            )

        if joint_enabled:
            from .joint_pipeline import run_joint_gaussian_training

            artifacts = run_joint_gaussian_training(
                config=config,
                paths=paths,
                bundle=bundle,
                data_audit=data_audit,
                run_dir=run_dir,
                logger=logger,
                device=device,
                dtype=dtype,
                test_status=test_status,
            )
            _close_logger(logger)
            return TrainingResult(
                run_dir=run_dir,
                weights_path=artifacts.weights_path,
                confusion_matrix_path=artifacts.confusion_matrix_path,
                log_path=paths.log_file,
                test_accuracy=artifacts.test_accuracy,
                test_macro_f1=artifacts.test_macro_f1,
            )

        preprocessing = config["preprocessing"]
        classifier_config = config["classifier"]
        classes = [int(label) for label in config["data"]["class_labels"]]
        benchmark_alpha = float(classifier_config["ridge_alpha"])
        alpha_selection_config = classifier_config["alpha_selection"]
        alpha_selection: AlphaSelectionResult | None = None
        selected_alpha = benchmark_alpha
        if bool(alpha_selection_config.get("enabled", False)):
            logger.info(
                "[REASON] The current training split has %d samples for %d correlated features; regularization must be selected inside train only to control variance without test leakage (benchmark alpha=%g)",
                len(bundle.train.labels),
                bundle.train.features.shape[1],
                benchmark_alpha,
            )
            logger.info(
                "[LOOK] Running train-only %d-fold alpha selection; benchmark alpha=%g, metric=%s",
                int(alpha_selection_config["folds"]),
                benchmark_alpha,
                alpha_selection_config["metric"],
            )
            alpha_selection = select_ridge_alpha(
                raw_features=bundle.train.features,
                labels=bundle.train.labels,
                classes=classes,
                classifier_config=classifier_config,
                preprocessing_config=preprocessing,
                random_state=int(config["runtime"]["random_state"]),
                device=device,
                dtype=dtype,
            )
            for candidate in alpha_selection.candidates:
                logger.info(
                    "[LOOK] CV alpha=%g | accuracy=%.4f±%.4f | macro_f1=%.4f±%.4f | recall_8=%.4f",
                    candidate.alpha,
                    candidate.mean_accuracy,
                    candidate.std_accuracy,
                    candidate.mean_macro_f1,
                    candidate.std_macro_f1,
                    candidate.mean_per_class_recall[classes.index(8)] if 8 in classes else float("nan"),
                )
            selected_alpha = alpha_selection.selected_alpha
            selected_candidate = next(
                candidate
                for candidate in alpha_selection.candidates
                if candidate.alpha == selected_alpha
            )
            logger.info(
                "[CHANGE] Selected ridge alpha=%g using training folds only (benchmark baseline=%g)",
                selected_alpha,
                benchmark_alpha,
            )
            logger.info(
                "[REASON] alpha=%g has the highest train-fold mean %s=%.4f; no test labels were used for this choice",
                selected_alpha,
                alpha_selection.metric,
                (
                    selected_candidate.mean_macro_f1
                    if alpha_selection.metric == "macro_f1"
                    else selected_candidate.mean_accuracy
                ),
            )
        else:
            logger.info("[CHANGE] Alpha selection disabled; using configured ridge alpha=%g", selected_alpha)

        score_calibration_config = classifier_config["score_calibration"]
        class_bias_selection: ClassBiasSelectionResult | None = None
        class_bias = np.zeros(len(classes), dtype=np.float32)
        if bool(score_calibration_config.get("enabled", False)):
            logger.info(
                "[REASON] Class 8 is under-predicted and overlaps most strongly with class 0; calibrating only the existing FC bias can improve recall without adding a network layer",
            )
            logger.info(
                "[LOOK] Running train-only priority-class bias calibration: class=%s, minimum_recall=%.2f",
                score_calibration_config["priority_class"],
                float(score_calibration_config["minimum_priority_recall"]),
            )
            class_bias_selection = select_priority_class_bias(
                raw_features=bundle.train.features,
                labels=bundle.train.labels,
                classes=classes,
                selected_alpha=selected_alpha,
                classifier_config=classifier_config,
                preprocessing_config=preprocessing,
                random_state=int(config["runtime"]["random_state"]),
                device=device,
                dtype=dtype,
            )
            for candidate in class_bias_selection.candidates:
                logger.info(
                    "[LOOK] CV class_%d_bias=%g | accuracy=%.4f | macro_f1=%.4f | priority_recall=%.4f",
                    class_bias_selection.priority_class,
                    candidate.bias,
                    candidate.accuracy,
                    candidate.macro_f1,
                    candidate.priority_recall,
                )
            class_bias = class_bias_selection.bias_vector(classes)
            logger.info(
                "[CHANGE] Selected class_%d score bias=%g using out-of-fold training predictions only",
                class_bias_selection.priority_class,
                class_bias_selection.selected_bias,
            )
            selected_bias_candidate = next(
                candidate
                for candidate in class_bias_selection.candidates
                if candidate.bias == class_bias_selection.selected_bias
            )
            logger.info(
                "[REASON] Selected bias meets train-fold class_%d recall %.2f%% and maximizes feasible %s=%.4f",
                class_bias_selection.priority_class,
                100 * selected_bias_candidate.priority_recall,
                class_bias_selection.metric,
                (
                    selected_bias_candidate.macro_f1
                    if class_bias_selection.metric == "macro_f1"
                    else selected_bias_candidate.accuracy
                ),
            )
        else:
            logger.info("[CHANGE] Score calibration disabled")

        scaler = HardwareFeatureScaler(
            scaling=str(preprocessing["scaling"]),
            negative_policy=str(preprocessing["negative_policy"]),
            epsilon=float(preprocessing["epsilon"]),
        ).fit(bundle.train.features)
        train_features = scaler.transform(bundle.train.features)
        test_features = scaler.transform(bundle.test.features)
        logger.info(
            "[CHANGE] Hardware preprocessing fitted on train only: scaling=%s, negative_policy=%s, scale=%.12g",
            scaler.scaling,
            scaler.negative_policy,
            scaler.scale_,
        )
        logger.info(
            "Negative raw values: train=%.2f%%, test=%.2f%%; transformed ranges: train=[%.6g, %.6g], test=[%.6g, %.6g]",
            100 * _negative_fraction(bundle.train.features),
            100 * _negative_fraction(bundle.test.features),
            float(train_features.min()),
            float(train_features.max()),
            float(test_features.min()),
            float(test_features.max()),
        )

        classifier = RidgeLinearClassifier(
            input_features=int(classifier_config["input_features"]),
            classes=classes,
            ridge_alpha=selected_alpha,
            fit_intercept=bool(classifier_config["fit_intercept"]),
            regularize_intercept=bool(classifier_config["regularize_intercept"]),
            device=device,
            dtype=dtype,
        )
        logger.info(
            "[CHANGE] Fitting benchmark-compatible ridge head: 256 -> %d, selected_alpha=%g, intercept_regularized=%s",
            len(classes),
            classifier.ridge_alpha,
            classifier.regularize_intercept,
        )
        classifier.fit(
            torch.from_numpy(train_features),
            torch.from_numpy(bundle.train.labels),
        )
        classifier.add_class_score_bias(torch.from_numpy(class_bias))

        train_predictions = classifier.predict(torch.from_numpy(train_features)).cpu().numpy()
        test_predictions = classifier.predict(torch.from_numpy(test_features)).cpu().numpy()
        train_evaluation = evaluate(bundle.train.labels, train_predictions, classes)
        test_evaluation = evaluate(bundle.test.labels, test_predictions, classes)
        normalized_design = torch.from_numpy(
            np.column_stack(
                [np.ones(len(train_features), dtype=np.float32), train_features]
            )
        ).to(dtype=dtype)
        singular_values = torch.linalg.svdvals(normalized_design)
        design_condition_number = float((singular_values.max() / singular_values.min()).item())
        design_rank = int(torch.linalg.matrix_rank(normalized_design).item())
        max_abs_weight = float(classifier.combined_weights_.detach().abs().max().item())
        logger.info(
            "[LOOK] Design diagnostics: rank=%d/%d, condition_number=%.2f, max_abs_weight=%.4f",
            design_rank,
            int(normalized_design.shape[1]),
            design_condition_number,
            max_abs_weight,
        )
        logger.info(
            "[LOOK] Train accuracy=%.4f, macro_f1=%.4f",
            train_evaluation.accuracy,
            train_evaluation.macro_f1,
        )
        logger.info(
            "[LOOK] Test accuracy=%.4f, macro_f1=%.4f",
            test_evaluation.accuracy,
            test_evaluation.macro_f1,
        )
        logger.info(
            "[RESULT] Train accuracy=%.4f, test accuracy=%.4f, test macro_f1=%.4f, test class_8 recall=%.4f",
            train_evaluation.accuracy,
            test_evaluation.accuracy,
            test_evaluation.macro_f1,
            float(test_evaluation.recall[classes.index(8)]) if 8 in classes else float("nan"),
        )
        diagnostics = config["diagnostics"]
        generalization_gap = train_evaluation.accuracy - test_evaluation.accuracy
        if generalization_gap >= float(diagnostics["generalization_gap_warning"]):
            logger.warning(
                "[LOOK] Generalization gap is %.2f percentage points (train %.2f%% vs test %.2f%%)",
                100 * generalization_gap,
                100 * train_evaluation.accuracy,
                100 * test_evaluation.accuracy,
            )
        recall_threshold = float(diagnostics["class_recall_warning"])
        weak_classes = [
            f"{label}:{test_evaluation.recall[index]:.2%}"
            for index, label in enumerate(classes)
            if test_evaluation.recall[index] < recall_threshold
        ]
        if weak_classes:
            logger.warning(
                "[LOOK] Classes below %.0f%% recall: %s",
                100 * recall_threshold,
                ", ".join(weak_classes),
            )

        matrix_text = confusion_matrix_text(test_evaluation.confusion_matrix, classes)
        logger.info("[LOOK] Test confusion matrix (rows=true, columns=predicted):\n%s", matrix_text)

        metrics = {
            "dataset": config.get("project", {}).get("dataset", "pen_digits"),
            "model_type": "post_rbf_ridge_linear_classifier",
            "train_samples": len(bundle.train.labels),
            "test_samples": len(bundle.test.labels),
            "input_features": int(bundle.train.features.shape[1]),
            "classes": classes,
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "benchmark_ridge_alpha": benchmark_alpha,
            "ridge_alpha": classifier.ridge_alpha,
            "alpha_selection": (
                alpha_selection.as_dict()
                if alpha_selection is not None
                else {"enabled": False, "selected_alpha": selected_alpha}
            ),
            "score_calibration": (
                class_bias_selection.as_dict()
                if class_bias_selection is not None
                else {"enabled": False, "class_bias": class_bias.tolist()}
            ),
            "fit_intercept": classifier.fit_intercept,
            "regularize_intercept": classifier.regularize_intercept,
            "preprocessing": scaler.state_dict(),
            "data_audit": data_audit,
            "evaluation_protocol": {
                "model_selection_uses_test_labels": False,
                "scaler_fit_split": "train",
                "test_set_status": test_status,
                "new_blind_test_required_for_final_claim": bool(
                    config["evaluation"].get(
                        "new_blind_test_required_for_final_claim", True
                    )
                ),
            },
            "diagnostics": {
                "generalization_gap": generalization_gap,
                "design_matrix_rank": design_rank,
                "design_matrix_columns": int(normalized_design.shape[1]),
                "design_condition_number": design_condition_number,
                "max_abs_combined_weight": max_abs_weight,
                "hardware_test_coverage": (
                    len(bundle.test.labels) / bundle.test_reference_rows
                    if bundle.test_reference_rows
                    else 1.0
                ),
                "per_class_recall": {
                    str(label): float(test_evaluation.recall[index])
                    for index, label in enumerate(classes)
                },
            },
            "train": train_evaluation.summary(),
            "test": test_evaluation.summary(),
        }
        state = classifier.cpu_state_dict()
        checkpoint = {
            "format_version": 1,
            "dataset": metrics["dataset"],
            "model_type": metrics["model_type"],
            "input_features": metrics["input_features"],
            "num_classes": len(classes),
            "ridge_alpha": classifier.ridge_alpha,
            "benchmark_ridge_alpha": benchmark_alpha,
            "fit_intercept": classifier.fit_intercept,
            "regularize_intercept": classifier.regularize_intercept,
            "state_dict": {"weight": state["weight"], "bias": state["bias"]},
            "combined_weights": state["combined_weights"],
            "classes": state["classes"],
            "preprocessing": scaler.state_dict(),
            "metrics": metrics,
            "sources": {
                "train_file": str(paths.train_file),
                "test_file": str(paths.test_file),
                "train_reference_file": str(paths.train_reference_file),
                "test_reference_file": str(paths.test_reference_file),
            },
            "config": config,
        }

        weights_path = run_dir / str(output_config["weights_filename"])
        torch.save(checkpoint, weights_path)
        metrics_path = run_dir / str(output_config["metrics_filename"])
        metrics_path.write_text(
            json.dumps(_json_ready(metrics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        predictions_path = run_dir / str(output_config["predictions_filename"])
        write_predictions(predictions_path, bundle.test.labels, test_predictions)
        confusion_csv = run_dir / str(output_config["confusion_matrix_csv"])
        write_confusion_csv(confusion_csv, test_evaluation.confusion_matrix, classes)
        confusion_text = run_dir / str(output_config["confusion_matrix_text"])
        confusion_text.write_text(matrix_text + "\n", encoding="utf-8")
        confusion_svg = run_dir / str(output_config["confusion_matrix_svg"])
        write_confusion_svg(confusion_svg, test_evaluation.confusion_matrix, classes)
        report_path = run_dir / str(output_config["classification_report_filename"])
        report_path.write_text(
            classification_report_text(test_evaluation, classes),
            encoding="utf-8",
        )
        if alpha_selection is not None:
            alpha_search_path = run_dir / str(output_config["alpha_search_filename"])
            write_alpha_search(
                alpha_search_path,
                [candidate.as_dict() for candidate in alpha_selection.candidates],
                classes,
            )
            logger.info("Saved train-only alpha search: %s", alpha_search_path)
        if class_bias_selection is not None:
            class_bias_search_path = run_dir / str(output_config["class_bias_search_filename"])
            write_class_bias_search(
                class_bias_search_path,
                [candidate.as_dict() for candidate in class_bias_selection.candidates],
            )
            logger.info("Saved train-only class-bias search: %s", class_bias_search_path)

        logger.info("Saved CPU-portable weights: %s", weights_path)
        logger.info("Saved confusion matrix CSV: %s", confusion_csv)
        logger.info("Saved confusion matrix SVG: %s", confusion_svg)
        logger.info("Training pipeline completed successfully")
        _close_logger(logger)
        return TrainingResult(
            run_dir=run_dir,
            weights_path=weights_path,
            confusion_matrix_path=confusion_csv,
            log_path=paths.log_file,
            test_accuracy=test_evaluation.accuracy,
            test_macro_f1=test_evaluation.macro_f1,
        )
    except Exception:
        logger.exception("Training pipeline failed")
        if run_dir is not None:
            logger.error("Partial run directory retained for diagnosis: %s", run_dir)
        _close_logger(logger)
        raise
