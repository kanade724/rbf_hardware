"""Tkinter collector for append-only 16-value Pen Digits samples."""

from __future__ import annotations

import logging
import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox

import numpy as np

from ..data.csv_store import NumericCsvStore


Point = tuple[float, float]


@dataclass(frozen=True)
class CollectorSettings:
    canvas_size: int = 500
    point_count: int = 8
    minimum_move: float = 1.5


class PenDigitsCollector:
    def __init__(
        self,
        root: tk.Tk,
        sample_store: NumericCsvStore,
        logger: logging.Logger,
        settings: CollectorSettings | None = None,
    ) -> None:
        self.root = root
        self.sample_store = sample_store
        self.logger = logger
        self.settings = settings or CollectorSettings()
        self.strokes: list[list[Point]] = []
        self.current_stroke: list[Point] = []
        self.sampled_points: list[Point] = []
        self.saved_count = sample_store.row_count()

        root.title("Pen Digits 轨迹采集器")
        root.resizable(False, False)
        tk.Label(
            root,
            text="按住鼠标连续写下一个数字；松开后自动等距选择 8 个点",
            font=("Microsoft YaHei UI", 11),
        ).pack(padx=12, pady=(12, 6))

        self.canvas = tk.Canvas(
            root,
            width=self.settings.canvas_size,
            height=self.settings.canvas_size,
            bg="white",
            highlightthickness=2,
            highlightbackground="#333333",
            cursor="pencil",
        )
        self.canvas.pack(padx=14)
        self.canvas.bind("<ButtonPress-1>", self.begin_stroke)
        self.canvas.bind("<B1-Motion>", self.continue_stroke)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=14, pady=10)
        tk.Button(controls, text="撤销上一笔", command=self.undo, width=12).pack(side="left")
        tk.Button(controls, text="清空", command=self.clear, width=9).pack(side="left", padx=8)
        self.save_button = tk.Button(
            controls,
            text="保存这一条",
            command=self.save_sample,
            width=12,
            state="disabled",
        )
        self.save_button.pack(side="left")

        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w").pack(
            fill="x", padx=14, pady=(0, 12)
        )
        root.bind("<Control-z>", lambda _event: self.undo())
        root.bind("<Return>", lambda _event: self.save_sample())
        self.update_status()

    def event_point(self, event: tk.Event) -> Point:
        limit = self.settings.canvas_size - 1
        return float(max(0, min(limit, event.x))), float(max(0, min(limit, event.y)))

    def begin_stroke(self, event: tk.Event) -> None:
        self.current_stroke = [self.event_point(event)]

    def continue_stroke(self, event: tk.Event) -> None:
        if not self.current_stroke:
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
            fill="#171717",
            width=9,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND,
        )

    def end_stroke(self, event: tk.Event) -> None:
        if not self.current_stroke:
            return
        point = self.event_point(event)
        if math.dist(self.current_stroke[-1], point) >= self.settings.minimum_move:
            self.current_stroke.append(point)
        if len(self.current_stroke) >= 2:
            self.strokes.append(self.current_stroke)
        self.current_stroke = []
        self.recompute_samples()
        self.redraw()

    def recompute_samples(self) -> None:
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

    def undo(self) -> None:
        if self.strokes:
            self.strokes.pop()
            self.recompute_samples()
            self.redraw()

    def clear(self) -> None:
        self.strokes.clear()
        self.current_stroke.clear()
        self.sampled_points.clear()
        self.redraw()

    def redraw(self) -> None:
        self.canvas.delete("all")
        for stroke in self.strokes:
            coordinates = [coordinate for point in stroke for coordinate in point]
            if len(coordinates) >= 4:
                self.canvas.create_line(
                    *coordinates,
                    fill="#171717",
                    width=9,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=True,
                )
        radius = 8
        for index, (x_coordinate, y_coordinate) in enumerate(
            self.sampled_points, start=1
        ):
            self.canvas.create_oval(
                x_coordinate - radius,
                y_coordinate - radius,
                x_coordinate + radius,
                y_coordinate + radius,
                fill="#e53935",
                outline="white",
                width=2,
            )
            self.canvas.create_text(
                x_coordinate,
                y_coordinate,
                text=str(index),
                fill="white",
                font=("Arial", 8, "bold"),
            )
        self.save_button.configure(state="normal" if self.sampled_points else "disabled")
        self.update_status()

    def normalized_features(self) -> np.ndarray:
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

    def save_sample(self) -> None:
        if len(self.sampled_points) != self.settings.point_count:
            messagebox.showwarning("无法保存", "请先连续写下一个完整数字。")
            return
        features = self.normalized_features()
        self.sample_store.append_rows(features)
        self.saved_count += 1
        self.logger.info(
            "[采集] 已追加原始样本，行号=%d，共享文件=%s",
            self.saved_count,
            self.sample_store.path,
        )
        self.clear()
        self.status.set(
            f"已保存第 {self.saved_count} 条样本：{self.sample_store.path}"
        )

    def update_status(self) -> None:
        if self.sampled_points:
            message = f"已自动选择 {self.settings.point_count} 个等距点，可以保存"
        else:
            message = "请在白色区域按住鼠标书写"
        self.status.set(f"{message}；文件中已有 {self.saved_count} 条样本")
