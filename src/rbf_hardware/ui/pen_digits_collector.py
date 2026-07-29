"""Reusable Tkinter drawing pad for 16-value Pen Digits trajectories."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class CollectorSettings:
    canvas_size: int = 500
    point_count: int = 8
    minimum_move: float = 1.5


class PenDigitDrawingPad(ttk.Frame):
    """Draw a digit and expose the normalized 16-value Pen Digits feature row."""

    CANVAS_BACKGROUND = "#FCFDFE"
    GRID_COLOR = "#E8EEF5"
    STROKE_COLOR = "#102A43"
    SAMPLE_COLOR = "#0B92A8"

    def __init__(
        self,
        master: tk.Misc,
        *,
        settings: CollectorSettings | None = None,
        on_ready_changed: Callable[[bool], None] | None = None,
    ) -> None:
        super().__init__(master, style="Surface.TFrame")
        self.settings = settings or CollectorSettings()
        self.on_ready_changed = on_ready_changed
        self.strokes: list[list[Point]] = []
        self.current_stroke: list[Point] = []
        self.sampled_points: list[Point] = []
        self.enabled = True

        self.canvas = tk.Canvas(
            self,
            width=self.settings.canvas_size,
            height=self.settings.canvas_size,
            background=self.CANVAS_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#CCD8E5",
            cursor="pencil",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.begin_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        self.redraw()

    @property
    def is_ready(self) -> bool:
        return len(self.sampled_points) == self.settings.point_count

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.canvas.configure(cursor="pencil" if enabled else "arrow")

    def event_point(self, event: tk.Event) -> Point:
        limit = self.settings.canvas_size - 1
        return (
            float(max(0, min(limit, event.x))),
            float(max(0, min(limit, event.y))),
        )

    def begin_stroke(self, event: tk.Event) -> None:
        if not self.enabled:
            return
        self.current_stroke = [self.event_point(event)]

    def continue_stroke(self, event: tk.Event) -> None:
        if not self.enabled or not self.current_stroke:
            return
        point = self.event_point(event)
        previous = self.current_stroke[-1]
        if math.dist(previous, point) < self.settings.minimum_move:
            return
        self.current_stroke.append(point)
        self.canvas.create_line(
            previous[0],
            previous[1],
            point[0],
            point[1],
            fill=self.STROKE_COLOR,
            width=10,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
            tags=("stroke",),
        )

    def end_stroke(self, event: tk.Event) -> None:
        if not self.enabled or not self.current_stroke:
            return
        point = self.event_point(event)
        if math.dist(self.current_stroke[-1], point) >= self.settings.minimum_move:
            self.current_stroke.append(point)
        if len(self.current_stroke) >= 2:
            self.strokes.append(self.current_stroke)
        self.current_stroke = []
        self._recompute_samples()
        self.redraw()

    def undo(self) -> None:
        if not self.enabled or not self.strokes:
            return
        self.strokes.pop()
        self._recompute_samples()
        self.redraw()

    def clear(self) -> None:
        if not self.enabled:
            return
        self.strokes.clear()
        self.current_stroke.clear()
        self.sampled_points.clear()
        self.redraw()

    def normalized_features(self) -> np.ndarray:
        if not self.is_ready:
            raise ValueError("请先写下一个完整数字。")
        x_coordinates = [point[0] for point in self.sampled_points]
        y_coordinates = [point[1] for point in self.sampled_points]
        minimum_x, maximum_x = min(x_coordinates), max(x_coordinates)
        minimum_y, maximum_y = min(y_coordinates), max(y_coordinates)
        span_x = max(maximum_x - minimum_x, 1.0e-12)
        span_y = max(maximum_y - minimum_y, 1.0e-12)

        features: list[float] = []
        for x_coordinate, y_coordinate in self.sampled_points:
            normalized_x = round((x_coordinate - minimum_x) * 100 / span_x)
            normalized_y = round((maximum_y - y_coordinate) * 100 / span_y)
            features.extend((normalized_x, normalized_y))
        return np.asarray(features, dtype=np.float32)

    def _recompute_samples(self) -> None:
        segments: list[tuple[Point, Point, float]] = []
        total_length = 0.0
        for stroke in self.strokes:
            for start, end in zip(stroke, stroke[1:]):
                length = math.dist(start, end)
                if length > 0:
                    segments.append((start, end, length))
                    total_length += length
        if total_length <= 0 or not segments:
            self.sampled_points = []
            return

        targets = [
            total_length * index / (self.settings.point_count - 1)
            for index in range(self.settings.point_count)
        ]
        sampled: list[Point] = []
        segment_index = 0
        distance_before = 0.0
        for target in targets:
            while (
                segment_index < len(segments) - 1
                and distance_before + segments[segment_index][2] < target
            ):
                distance_before += segments[segment_index][2]
                segment_index += 1
            start, end, length = segments[segment_index]
            ratio = max(0.0, min(1.0, (target - distance_before) / length))
            sampled.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )
        self.sampled_points = sampled

    def redraw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        for stroke in self.strokes:
            coordinates = [coordinate for point in stroke for coordinate in point]
            if len(coordinates) >= 4:
                self.canvas.create_line(
                    *coordinates,
                    fill=self.STROKE_COLOR,
                    width=10,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True,
                    tags=("stroke",),
                )
        radius = 8
        for index, (x_coordinate, y_coordinate) in enumerate(
            self.sampled_points,
            start=1,
        ):
            self.canvas.create_oval(
                x_coordinate - radius,
                y_coordinate - radius,
                x_coordinate + radius,
                y_coordinate + radius,
                fill=self.SAMPLE_COLOR,
                outline="white",
                width=2,
                tags=("sample",),
            )
            self.canvas.create_text(
                x_coordinate,
                y_coordinate,
                text=str(index),
                fill="white",
                font=("Segoe UI", 8, "bold"),
                tags=("sample",),
            )
        if self.on_ready_changed is not None:
            self.on_ready_changed(self.is_ready)

    def _draw_grid(self) -> None:
        size = self.settings.canvas_size
        step = size // 10
        for coordinate in range(step, size, step):
            self.canvas.create_line(
                coordinate,
                0,
                coordinate,
                size,
                fill=self.GRID_COLOR,
                width=1,
                tags=("grid",),
            )
            self.canvas.create_line(
                0,
                coordinate,
                size,
                coordinate,
                fill=self.GRID_COLOR,
                width=1,
                tags=("grid",),
            )
        center = size // 2
        self.canvas.create_line(
            center,
            0,
            center,
            size,
            fill="#D5E1EC",
            dash=(4, 4),
            tags=("grid",),
        )
        self.canvas.create_line(
            0,
            center,
            size,
            center,
            fill="#D5E1EC",
            dash=(4, 4),
            tags=("grid",),
        )
