"""Responsive layout policy for the unified desktop application."""

from __future__ import annotations

from enum import Enum


class ApplicationLayoutMode(str, Enum):
    """Supported top-level arrangements for the scientific console."""

    LANDSCAPE = "landscape"
    PORTRAIT = "portrait"


PORTRAIT_ENTRY_MAXIMUM_WIDTH = 980
LANDSCAPE_ENTRY_MINIMUM_WIDTH = 1060


def resolve_application_layout_mode(
    viewport_width: int,
    viewport_height: int,
    current_mode: ApplicationLayoutMode | None = None,
) -> ApplicationLayoutMode:
    """Select a stable layout mode for the current window dimensions.

    The different entry and exit widths add hysteresis, preventing repeated
    layout switches while the user drags the window near the breakpoint.
    """

    if viewport_width <= 0 or viewport_height <= 0:
        raise ValueError("Viewport dimensions must be positive.")

    if current_mode is ApplicationLayoutMode.PORTRAIT:
        can_return_to_landscape = (
            viewport_width >= LANDSCAPE_ENTRY_MINIMUM_WIDTH
            and viewport_width >= viewport_height
        )
        return (
            ApplicationLayoutMode.LANDSCAPE
            if can_return_to_landscape
            else ApplicationLayoutMode.PORTRAIT
        )

    should_enter_portrait = (
        viewport_width <= PORTRAIT_ENTRY_MAXIMUM_WIDTH
        or viewport_height > viewport_width
    )
    return (
        ApplicationLayoutMode.PORTRAIT
        if should_enter_portrait
        else ApplicationLayoutMode.LANDSCAPE
    )
