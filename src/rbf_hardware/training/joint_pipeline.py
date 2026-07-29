from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..modeling.joint_gaussian import FullJointGaussianTransformer, linear_cka
from ..modeling.ridge_classifier import RidgeLinearClassifier
from ..reporting.metrics import (
    classification_report_text,
    confusion_matrix_text,
    evaluate,
    write_confusion_csv,
    write_confusion_svg,
    write_joint_gaussian_search,
    write_predictions,
)
from .joint_selection import JointGaussianSelectionResult, select_joint_gaussian_parameters


@dataclass(frozen=True)
class JointTrainingArtifacts:
    weights_path: Path
    confusion_matrix_path: Path
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


def _checkpoint_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy()).cpu()
    if isinstance(value, dict):
        return {str(key): _checkpoint_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_ready(item) for item in value]
    return value


def _build_transformer(
    feature_config: dict[str, Any],
    random_state: int,
    joint_sigma: float,
    calibration_alpha: float,
    factor_upper: float,
) -> FullJointGaussianTransformer:
    return FullJointGaussianTransformer(
        dimensions=int(feature_config["dimensions"]),
        basis_per_dimension=int(feature_config["basis_per_dimension"]),
        output_features=int(feature_config["output_features"]),
        joint_sigma=joint_sigma,
        calibration_alpha=calibration_alpha,
        factor_lower=float(feature_config["factor_clip"]["lower"]),
        factor_upper=factor_upper,
        epsilon=float(feature_config["calibration"]["epsilon"]),
        random_state=random_state,
        kmeans_max_iter=int(feature_config["center_selection"]["max_iter"]),
    )


def run_joint_gaussian_training(
    *,
    config: dict[str, Any],
    paths: Any,
    bundle: Any,
    data_audit: dict[str, Any],
    run_dir: Path,
    logger: logging.Logger,
    device: torch.device,
    dtype: torch.dtype,
    test_status: str,
) -> JointTrainingArtifacts:
    if bundle.train_reference_features is None or bundle.test_reference_features is None:
        raise ValueError(
            "full_joint_gaussian requires paired train/test reference inputs for training calibration "
            "and the architecture-matched PC comparison."
        )

    feature_config = config["feature_transform"]
    classifier_config = config["classifier"]
    output_config = config["output"]
    classes = [int(label) for label in config["data"]["class_labels"]]
    random_state = int(config["runtime"]["random_state"])
    selection_config = feature_config["selection"]
    selection_result: JointGaussianSelectionResult | None = None

    logger.info(
        "[CHANGE] Enabling full 16-D joint Gaussian: hardware layout=%dx%d, joint outputs=%d",
        int(feature_config["dimensions"]),
        int(feature_config["basis_per_dimension"]),
        int(feature_config["output_features"]),
    )
    logger.info(
        "[REASON] Hardware performs the one-dimensional Gaussian nonlinearities; the digital path "
        "uses train-fitted linear calibration and multiplication only (no inference-time digital exp)",
    )
    logger.info(
        "[AUDIT] train_reference features may fit centers/calibration; test_reference features are "
        "excluded from the hardware path and used only for the separately labeled PC comparison",
    )

    if bool(selection_config.get("enabled", False)):
        transform_count = (
            len(selection_config["joint_sigma_candidates"])
            * len(selection_config["calibration_alpha_candidates"])
            * len(selection_config["factor_upper_candidates"])
        )
        total_candidates = transform_count * len(classifier_config["alpha_selection"]["candidates"])
        logger.info(
            "[LOOK] Running train-only %d-fold joint search: %d transforms, %d total parameter tuples",
            int(selection_config["folds"]),
            transform_count,
            total_candidates,
        )
        with tqdm(
            total=int(selection_config["folds"]) * total_candidates,
            desc="联合参数搜索",
            unit="组合",
            dynamic_ncols=True,
            leave=True,
        ) as search_progress:
            def update_search_progress(step: int, detail: str) -> None:
                search_progress.set_postfix_str(detail, refresh=False)
                search_progress.update(step)

            selection_result = select_joint_gaussian_parameters(
                hardware_features=bundle.train.features,
                reference_inputs=bundle.train_reference_features,
                labels=bundle.train.labels,
                classes=classes,
                feature_config=feature_config,
                classifier_config=classifier_config,
                random_state=random_state,
                progress=lambda message: logger.info("[LOOK] %s", message),
                progress_update=update_search_progress,
            )
        joint_sigma = selection_result.selected_joint_sigma
        calibration_alpha = selection_result.selected_calibration_alpha
        factor_upper = selection_result.selected_factor_upper
        head_alpha = selection_result.selected_head_alpha
        score_name = (
            "mean_macro_f1" if selection_result.metric == "macro_f1" else "mean_accuracy"
        )
        ranked = sorted(
            selection_result.candidates,
            key=lambda item: getattr(item, score_name),
            reverse=True,
        )
        for candidate in ranked[:10]:
            logger.info(
                "[LOOK] CV sigma=%g calibration_alpha=%g upper=%g head_alpha=%g | "
                "accuracy=%.4f+/-%.4f | macro_f1=%.4f+/-%.4f",
                candidate.joint_sigma,
                candidate.calibration_alpha,
                candidate.factor_upper,
                candidate.head_alpha,
                candidate.mean_accuracy,
                candidate.std_accuracy,
                candidate.mean_macro_f1,
                candidate.std_macro_f1,
            )
        selected_candidate = ranked[0]
        logger.info(
            "[CHANGE] Selected joint parameters using training folds only: sigma=%g, "
            "calibration_alpha=%g, factor_upper=%g, head_alpha=%g",
            joint_sigma,
            calibration_alpha,
            factor_upper,
            head_alpha,
        )
        logger.info(
            "[REASON] Selected tuple has the highest train-fold mean %s=%.4f; test labels were not used",
            selection_result.metric,
            getattr(selected_candidate, score_name),
        )
    else:
        joint_sigma = float(feature_config["joint_sigma"])
        calibration_alpha = float(feature_config["calibration"]["alpha"])
        factor_upper = float(feature_config["factor_clip"]["upper"])
        head_alpha = float(classifier_config["ridge_alpha"])
        logger.info("[CHANGE] Joint parameter selection disabled; using fixed configuration")

    final_progress = tqdm(
        total=4,
        desc="最终模型训练",
        unit="阶段",
        dynamic_ncols=True,
        leave=True,
    )
    transformer = _build_transformer(
        feature_config,
        random_state,
        joint_sigma,
        calibration_alpha,
        factor_upper,
    ).fit(bundle.train.features, bundle.train_reference_features)
    train_features = transformer.transform(bundle.train.features)
    test_features = transformer.transform(bundle.test.features)
    final_progress.set_postfix_str("联合特征变换完成", refresh=False)
    final_progress.update(1)
    logger.info(
        "[CHANGE] Joint transformer fitted on train only: sigma=%g, calibration_alpha=%g, "
        "factor_clip=[%g,%g]",
        joint_sigma,
        calibration_alpha,
        transformer.factor_lower,
        transformer.factor_upper,
    )
    logger.info(
        "[LOOK] Joint feature ranges: train=[%.6g, %.6g], test=[%.6g, %.6g]",
        float(train_features.min()),
        float(train_features.max()),
        float(test_features.min()),
        float(test_features.max()),
    )

    classifier = RidgeLinearClassifier(
        input_features=int(feature_config["output_features"]),
        classes=classes,
        ridge_alpha=head_alpha,
        fit_intercept=bool(classifier_config["fit_intercept"]),
        regularize_intercept=bool(classifier_config["regularize_intercept"]),
        device=device,
        dtype=dtype,
    )
    classifier.fit(
        torch.from_numpy(train_features),
        torch.from_numpy(bundle.train.labels),
    )
    train_predictions = classifier.predict(torch.from_numpy(train_features)).cpu().numpy()
    test_predictions = classifier.predict(torch.from_numpy(test_features)).cpu().numpy()
    train_evaluation = evaluate(bundle.train.labels, train_predictions, classes)
    test_evaluation = evaluate(bundle.test.labels, test_predictions, classes)
    final_progress.set_postfix_str("硬件分类器完成", refresh=False)
    final_progress.update(1)

    ideal_train = transformer.ideal_pc_transform(bundle.train_reference_features)
    ideal_test = transformer.ideal_pc_transform(bundle.test_reference_features)
    pc_classifier = RidgeLinearClassifier(
        input_features=int(feature_config["output_features"]),
        classes=classes,
        ridge_alpha=head_alpha,
        fit_intercept=bool(classifier_config["fit_intercept"]),
        regularize_intercept=bool(classifier_config["regularize_intercept"]),
        device=device,
        dtype=dtype,
    )
    pc_classifier.fit(torch.from_numpy(ideal_train), torch.from_numpy(bundle.train.labels))
    pc_train_predictions = pc_classifier.predict(torch.from_numpy(ideal_train)).cpu().numpy()
    pc_test_predictions = pc_classifier.predict(torch.from_numpy(ideal_test)).cpu().numpy()
    pc_train_evaluation = evaluate(bundle.train.labels, pc_train_predictions, classes)
    pc_test_evaluation = evaluate(bundle.test.labels, pc_test_predictions, classes)
    final_progress.set_postfix_str("PC基准完成", refresh=False)
    final_progress.update(1)

    design = np.column_stack((np.ones(len(train_features)), train_features)).astype(np.float64)
    singular_values = np.linalg.svd(design, compute_uv=False)
    design_rank = int(np.linalg.matrix_rank(design))
    design_condition = float(singular_values[0] / max(singular_values[-1], 1.0e-30))
    feature_correlation = float(
        np.corrcoef(train_features.ravel(), ideal_train.ravel())[0, 1]
    )
    feature_rmse = float(np.sqrt(np.mean(np.square(train_features - ideal_train))))
    feature_cka = linear_cka(train_features, ideal_train)
    calibration_r2 = np.asarray(transformer.calibration_r2_, dtype=np.float64)
    generalization_gap = train_evaluation.accuracy - test_evaluation.accuracy
    hardware_pc_gap = pc_test_evaluation.accuracy - test_evaluation.accuracy

    logger.info(
        "[LOOK] Calibration R2: median=%.4f, p05=%.4f, minimum=%.4f",
        float(np.median(calibration_r2)),
        float(np.quantile(calibration_r2, 0.05)),
        float(calibration_r2.min()),
    )
    logger.info(
        "[LOOK] Hardware-vs-PC joint features: flat_correlation=%.4f, linear_CKA=%.4f, RMSE=%.6g",
        feature_correlation,
        feature_cka,
        feature_rmse,
    )
    logger.info(
        "[RESULT] Hardware joint Gaussian: train accuracy=%.4f, test accuracy=%.4f, "
        "test macro_f1=%.4f, class_8 recall=%.4f",
        train_evaluation.accuracy,
        test_evaluation.accuracy,
        test_evaluation.macro_f1,
        float(test_evaluation.recall[classes.index(8)]) if 8 in classes else float("nan"),
    )
    logger.info(
        "[RESULT] Architecture-matched pure-PC Gaussian: test accuracy=%.4f, macro_f1=%.4f; "
        "hardware gap=%.2f percentage points",
        pc_test_evaluation.accuracy,
        pc_test_evaluation.macro_f1,
        100.0 * hardware_pc_gap,
    )

    matrix_text = confusion_matrix_text(test_evaluation.confusion_matrix, classes)
    logger.info("[LOOK] Hardware test confusion matrix (rows=true, columns=predicted):\n%s", matrix_text)
    transform_state = transformer.state_dict()
    selection_metrics = (
        selection_result.as_dict()
        if selection_result is not None
        else {
            "enabled": False,
            "selected_joint_sigma": joint_sigma,
            "selected_calibration_alpha": calibration_alpha,
            "selected_factor_upper": factor_upper,
            "selected_head_alpha": head_alpha,
        }
    )
    metrics = {
        "dataset": config.get("project", {}).get("dataset", "pen_digits"),
        "model_type": "hardware_full_joint_gaussian_ridge",
        "train_samples": int(len(bundle.train.labels)),
        "test_samples": int(len(bundle.test.labels)),
        "hardware_input_features": int(bundle.train.features.shape[1]),
        "joint_output_features": int(train_features.shape[1]),
        "classes": classes,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "ridge_alpha": head_alpha,
        "joint_parameter_selection": selection_metrics,
        "feature_transform": {
            "type": "full_joint_gaussian",
            "hardware_gaussian_source": True,
            "digital_inference_operations": "train-fitted linear calibration plus 16-D product",
            "digital_exp_at_inference": False,
            "uses_reference_at_inference": False,
            "dimensions": transformer.dimensions,
            "basis_per_dimension": transformer.basis_per_dimension,
            "output_features": transformer.output_features,
            "joint_sigma": transformer.joint_sigma,
            "calibration_alpha": transformer.calibration_alpha,
            "factor_clip": [transformer.factor_lower, transformer.factor_upper],
            "calibration_r2": {
                "minimum": float(calibration_r2.min()),
                "p05": float(np.quantile(calibration_r2, 0.05)),
                "median": float(np.median(calibration_r2)),
                "maximum": float(calibration_r2.max()),
            },
            "estimated_digital_macs_per_sample": int(
                transformer.dimensions
                * (transformer.basis_per_dimension + 1)
                * transformer.output_features
            ),
            "estimated_joint_multiplications_per_sample": int(
                (transformer.dimensions - 1) * transformer.output_features
            ),
        },
        "data_audit": data_audit,
        "evaluation_protocol": {
            "model_selection_uses_test_labels": False,
            "transform_fit_split": "train",
            "hardware_inference_uses_reference_inputs": False,
            "pc_comparison_uses_reference_inputs": True,
            "test_set_status": test_status,
            "new_blind_test_required_for_final_claim": bool(
                config["evaluation"].get("new_blind_test_required_for_final_claim", True)
            ),
        },
        "diagnostics": {
            "generalization_gap": generalization_gap,
            "design_matrix_rank": design_rank,
            "design_matrix_columns": int(design.shape[1]),
            "design_condition_number": design_condition,
            "hardware_pc_accuracy_gap": hardware_pc_gap,
            "hardware_pc_feature_flat_correlation": feature_correlation,
            "hardware_pc_feature_linear_cka": feature_cka,
            "hardware_pc_feature_rmse": feature_rmse,
            "per_class_recall": {
                str(label): float(test_evaluation.recall[index])
                for index, label in enumerate(classes)
            },
        },
        "train": train_evaluation.summary(),
        "test": test_evaluation.summary(),
        "pure_pc_architecture_matched": {
            "description": "Direct PC 16-D Gaussian with identical centers, sigma, split, and ridge head",
            "train": pc_train_evaluation.summary(),
            "test": pc_test_evaluation.summary(),
            "per_class_recall": {
                str(label): float(pc_test_evaluation.recall[index])
                for index, label in enumerate(classes)
            },
        },
    }

    state = classifier.cpu_state_dict()
    checkpoint = {
        "format_version": 2,
        "dataset": metrics["dataset"],
        "model_type": metrics["model_type"],
        "hardware_input_features": metrics["hardware_input_features"],
        "joint_output_features": metrics["joint_output_features"],
        "num_classes": len(classes),
        "ridge_alpha": head_alpha,
        "state_dict": {"weight": state["weight"], "bias": state["bias"]},
        "combined_weights": state["combined_weights"],
        "classes": state["classes"],
        "feature_transform": _checkpoint_ready(transform_state),
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
    if selection_result is not None:
        search_path = run_dir / str(output_config["joint_search_filename"])
        write_joint_gaussian_search(
            search_path,
            [candidate.as_dict() for candidate in selection_result.candidates],
            classes,
        )
        logger.info("Saved train-only joint parameter search: %s", search_path)

    final_progress.set_postfix_str("权重与报告已保存", refresh=False)
    final_progress.update(1)
    final_progress.close()
    logger.info("Saved CPU-portable joint weights: %s", weights_path)
    logger.info("Saved hardware confusion matrix CSV: %s", confusion_csv)
    logger.info("Saved hardware confusion matrix SVG: %s", confusion_svg)
    logger.info("Full joint-Gaussian training pipeline completed successfully")
    return JointTrainingArtifacts(
        weights_path=weights_path,
        confusion_matrix_path=confusion_csv,
        test_accuracy=test_evaluation.accuracy,
        test_macro_f1=test_evaluation.macro_f1,
    )
