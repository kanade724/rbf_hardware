"""Explicit state model for the two-step GUI experiment workflow."""

from __future__ import annotations

from enum import Enum


class ExperimentWorkflowState(str, Enum):
    """Current user-visible state of collection and manual inference."""

    IDLE = "idle"
    DRAWING = "drawing"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"
    CLOSING = "closing"
