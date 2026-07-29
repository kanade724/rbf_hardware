"""Cross-entry desktop window helpers."""

from __future__ import annotations

import ctypes
import sys


def enable_windows_high_dpi() -> None:
    """Enable system DPI awareness before creating the first Tk window."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
