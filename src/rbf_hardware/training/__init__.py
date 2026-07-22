"""Training orchestration and train-only model selection."""

from .pipeline import TrainingResult, run_training

__all__ = ["TrainingResult", "run_training"]
