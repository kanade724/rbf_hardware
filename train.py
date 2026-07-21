from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rbf_hardware import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 256-to-10 classifier after the hardware RBF feature layer."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="Path to config.yaml (default: rbf-hardware/config.yaml).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_training(args.config)
    print(f"Run directory: {result.run_dir}")
    print(f"Weights: {result.weights_path}")
    print(f"Confusion matrix: {result.confusion_matrix_path}")
    print(f"Test accuracy: {result.test_accuracy:.4f}")
    print(f"Test macro F1: {result.test_macro_f1:.4f}")


if __name__ == "__main__":
    main()

