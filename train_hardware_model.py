from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from rbf_hardware.training.pipeline import run_training


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-dimensional joint Gaussian features from the hardware RBF bank "
            "and train the 256-to-10 ridge classifier."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml (default: rbf-hardware/config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = run_training(arguments.config)
    print(f"Run directory: {result.run_dir}")
    print(f"Weights: {result.weights_path}")
    print(f"Confusion matrix: {result.confusion_matrix_path}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Test macro F1: {result.test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
