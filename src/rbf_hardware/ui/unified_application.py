"""Unified scientific desktop interface for collection and hardware inference."""

from __future__ import annotations

import csv
import logging
import queue
import threading
import tkinter as tk
from collections.abc import Iterable
from pathlib import Path
from tkinter import messagebox, ttk

from ..data.csv_store import NumericCsvStore
from ..inference.pipeline import (
    PipelineProgress,
    PredictionSummary,
    StreamingInferencePipeline,
)
from .experiment_workflow import ExperimentWorkflowState
from .pen_digits_collector import PenDigitDrawingPad
from .responsive_layout import (
    ApplicationLayoutMode,
    resolve_application_layout_mode,
)


class UnifiedPenDigitsApplication:
    """Coordinate drawing, append-only persistence, inference, and presentation."""

    BACKGROUND = "#EEF3F8"
    SURFACE = "#FFFFFF"
    NAVY = "#102A43"
    MUTED = "#627D98"
    ACCENT = "#087F8C"
    ACCENT_BRIGHT = "#20B8C5"
    SUCCESS = "#138A72"
    WARNING = "#D97706"
    BORDER = "#D8E2EC"

    def __init__(
        self,
        root: tk.Tk,
        *,
        sample_store: NumericCsvStore,
        pipeline: StreamingInferencePipeline,
        logger: logging.Logger,
        sampling_mode: str,
        monitor_interval_ms: int = 500,
    ) -> None:
        self.root = root
        self.sample_store = sample_store
        self.pipeline = pipeline
        self.logger = logger
        self.sampling_mode = sampling_mode
        self.monitor_interval_ms = monitor_interval_ms
        self.saved_count = sample_store.row_count()
        self.inference_busy = False
        self.inference_thread: threading.Thread | None = None
        self.workflow_state = ExperimentWorkflowState.IDLE
        self.close_requested = False
        self._root_destroyed = False
        self.last_reported_error: str | None = None
        self.last_raw_signature = self._file_signature(sample_store.path)
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.status_text = tk.StringVar(
            value="Experiment 1 — draw and save a digit",
        )
        self.status_detail = tk.StringVar(
            value="Experiment 2 runs hardware inference",
        )
        self.predicted_digit = tk.StringVar(value="—")
        self.top_score = tk.StringVar(value="—")
        self.sample_index = tk.StringVar(value="—")
        self.sample_count_text = tk.StringVar(value=str(self.saved_count))
        self.pipeline_state = tk.StringVar(value="READY")
        self.save_button: ttk.Button
        self.inference_button: ttk.Button
        self.layout_mode: ApplicationLayoutMode | None = None
        self._layout_refresh_after_id: str | None = None
        self._page_extent_after_id: str | None = None
        self._result_queue_after_id: str | None = None
        self._monitor_after_id: str | None = None
        self._close_poll_after_id: str | None = None
        self._page_scroll_required = False
        self._last_page_extent: tuple[int, int] | None = None

        self._configure_root()
        self._configure_styles()
        self._build_layout()
        self._load_recent_history()
        self._load_latest_hardware_output()
        self._refresh_manual_workflow_status()
        self.root.bind("<Control-z>", lambda _event: self.drawing_pad.undo())
        self.root.bind("<Return>", lambda _event: self.save_sample())
        self.root.bind("<F5>", lambda _event: self._request_inference())
        self.root.protocol("WM_DELETE_WINDOW", self._request_close)
        self._result_queue_after_id = self.root.after(
            80,
            self._drain_result_queue,
        )
        self._monitor_after_id = self.root.after(
            self.monitor_interval_ms,
            self._monitor_new_rows,
        )

    def _configure_root(self) -> None:
        self.root.title("RBF Hardware · Pen Digits Research Console")
        self.root.configure(background=self.BACKGROUND)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        available_width = max(320, screen_width - 64)
        available_height = max(420, screen_height - 96)
        minimum_width = min(640, available_width)
        minimum_height = min(720, available_height)

        if screen_height > screen_width:
            initial_width = min(900, available_width)
            initial_height = min(1500, available_height)
        else:
            initial_width = min(1220, available_width)
            initial_height = min(930, available_height)

        self.initial_layout_mode = resolve_application_layout_mode(
            initial_width,
            initial_height,
        )
        self.root.geometry(f"{initial_width}x{initial_height}")
        self.root.minsize(minimum_width, minimum_height)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Surface.TFrame", background=self.SURFACE)
        style.configure("Page.TFrame", background=self.BACKGROUND)
        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="white",
            borderwidth=0,
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#096C77"), ("disabled", "#A9BCC7")],
            foreground=[("disabled", "#EEF3F8")],
        )
        style.configure(
            "Inference.TButton",
            background=self.NAVY,
            foreground="white",
            borderwidth=0,
            padding=(18, 11),
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "Inference.TButton",
            background=[("active", "#183E59"), ("disabled", "#A9BCC7")],
            foreground=[("disabled", "#EEF3F8")],
        )
        style.configure(
            "Secondary.TButton",
            background="#E8F0F5",
            foreground=self.NAVY,
            borderwidth=0,
            padding=(15, 10),
            font=("Segoe UI", 10),
        )
        style.map("Secondary.TButton", background=[("active", "#D8E6EE")])
        style.configure(
            "Scientific.Horizontal.TProgressbar",
            troughcolor="#DCE7EE",
            background=self.ACCENT_BRIGHT,
            borderwidth=0,
        )
        style.configure(
            "Treeview",
            background=self.SURFACE,
            fieldbackground=self.SURFACE,
            foreground=self.NAVY,
            rowheight=28,
            borderwidth=0,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#EAF1F6",
            foreground=self.MUTED,
            relief="flat",
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#CDEFF2")])

    def _build_layout(self) -> None:
        self._build_header()
        self.page_viewport = tk.Frame(self.root, background=self.BACKGROUND)
        self.page_viewport.pack(fill="both", expand=True)
        self.page_viewport.columnconfigure(0, weight=1)
        self.page_viewport.rowconfigure(0, weight=1)

        self.page_canvas = tk.Canvas(
            self.page_viewport,
            background=self.BACKGROUND,
            borderwidth=0,
            highlightthickness=0,
            yscrollincrement=28,
        )
        self.page_canvas.grid(row=0, column=0, sticky="nsew")
        self.page_scrollbar = ttk.Scrollbar(
            self.page_viewport,
            orient="vertical",
            command=self.page_canvas.yview,
        )
        self.page_canvas.configure(yscrollcommand=self.page_scrollbar.set)

        self.page = ttk.Frame(
            self.page_canvas,
            style="Page.TFrame",
            padding=(22, 18, 22, 16),
        )
        self.page_window_id = self.page_canvas.create_window(
            (0, 0),
            window=self.page,
            anchor="nw",
        )

        self.drawing_card = tk.Frame(
            self.page,
            background=self.SURFACE,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        self.result_card = tk.Frame(
            self.page,
            background=self.SURFACE,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )

        self._build_drawing_panel(self.drawing_card)
        self._build_result_panel(self.result_card)
        self._build_status_bar()
        self._apply_layout_mode(self.initial_layout_mode)

        self.page.bind("<Configure>", self._on_page_content_configure)
        self.page_canvas.bind("<Configure>", self._on_page_canvas_configure)
        self.root.bind("<Configure>", self._on_root_configure, add="+")
        self.root.bind("<MouseWheel>", self._on_page_mousewheel, add="+")
        self._schedule_page_extent_refresh()

    def _build_header(self) -> None:
        self.header = tk.Frame(self.root, background=self.NAVY, height=92)
        self.header.pack(fill="x")
        self.header.grid_propagate(False)
        self.header_title_group = tk.Frame(self.header, background=self.NAVY)
        tk.Label(
            self.header_title_group,
            text="RBF HARDWARE",
            background=self.NAVY,
            foreground="#7FE3EA",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            self.header_title_group,
            text="Pen Digits Research Console",
            background=self.NAVY,
            foreground="white",
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        self.mode_badge = tk.Frame(self.header, background="#183E59")
        tk.Label(
            self.mode_badge,
            text="●",
            background="#183E59",
            foreground="#4EE1B2",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(12, 5), pady=7)
        tk.Label(
            self.mode_badge,
            text=f"MEASURED RESPONSE · {self.sampling_mode.upper()}",
            background="#183E59",
            foreground="#D9F6F8",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(0, 12), pady=7)

    def _build_drawing_panel(self, parent: tk.Frame) -> None:
        self.drawing_header = tk.Frame(parent, background=self.SURFACE)
        self.drawing_header.pack(fill="x", padx=20, pady=(18, 10))
        self.drawing_title_label = tk.Label(
            self.drawing_header,
            text="01  HANDWRITING DATA COLLECTION",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Segoe UI", 14, "bold"),
        )
        self.drawing_subtitle_label = tk.Label(
            self.drawing_header,
            text="8 equidistant trajectory points",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )

        canvas_host = ttk.Frame(parent, style="Surface.TFrame")
        canvas_host.pack(fill="both", expand=True, padx=20)
        self.drawing_pad = PenDigitDrawingPad(
            canvas_host,
            on_ready_changed=self._on_drawing_ready,
        )
        self.drawing_pad.pack(fill="both", expand=True)

        instruction = tk.Label(
            parent,
            text=(
                "Experiment 1 saves the digit. Experiment 2 runs hardware "
                "inference when you choose."
            ),
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )
        instruction.pack(fill="x", padx=20, pady=(10, 6))

        self.drawing_controls = tk.Frame(parent, background=self.SURFACE)
        self.drawing_controls.pack(fill="x", padx=20, pady=(2, 18))
        utility_controls = tk.Frame(
            self.drawing_controls,
            background=self.SURFACE,
        )
        utility_controls.pack(fill="x", pady=(0, 8))
        ttk.Button(
            utility_controls,
            text="Undo  Ctrl+Z",
            command=self.drawing_pad.undo,
            style="Secondary.TButton",
        ).pack(side="left")
        ttk.Button(
            utility_controls,
            text="Clear",
            command=self.drawing_pad.clear,
            style="Secondary.TButton",
        ).pack(side="left", padx=8)

        experiment_controls = tk.Frame(
            self.drawing_controls,
            background=self.SURFACE,
        )
        experiment_controls.pack(fill="x")
        self.save_button = ttk.Button(
            experiment_controls,
            text="Experiment 1 · Save  Enter",
            command=self.save_sample,
            style="Accent.TButton",
            state="disabled",
        )
        self.save_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.inference_button = ttk.Button(
            experiment_controls,
            text="Experiment 2 · Infer  F5",
            command=self._request_inference,
            style="Inference.TButton",
            state="disabled",
        )
        self.inference_button.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(5, 0),
        )

    def _build_result_panel(self, parent: tk.Frame) -> None:
        result_header = tk.Frame(parent, background=self.SURFACE)
        result_header.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            result_header,
            text="02  HARDWARE INFERENCE",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Segoe UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            result_header,
            textvariable=self.pipeline_state,
            background="#E3F5F3",
            foreground=self.SUCCESS,
            padx=10,
            pady=4,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="right")

        result_card = tk.Frame(parent, background=self.NAVY, height=172)
        result_card.pack(fill="x", padx=20)
        result_card.pack_propagate(False)
        result_left = tk.Frame(result_card, background=self.NAVY)
        result_left.pack(side="left", fill="both", expand=True, padx=(22, 8), pady=18)
        tk.Label(
            result_left,
            text="PREDICTION",
            background=self.NAVY,
            foreground="#9FB9CC",
            font=("Segoe UI", 10),
        ).pack(anchor="w")
        tk.Label(
            result_left,
            textvariable=self.predicted_digit,
            background=self.NAVY,
            foreground="#7FE3EA",
            font=("Segoe UI", 60, "bold"),
        ).pack(anchor="w")

        metrics = tk.Frame(result_card, background="#183E59")
        metrics.pack(side="right", fill="y", padx=14, pady=14)
        self._metric_row(metrics, "SAMPLE INDEX", self.sample_index)
        self._metric_row(metrics, "TOP SCORE", self.top_score)

        stage_section = tk.Frame(parent, background=self.SURFACE)
        stage_section.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            stage_section,
            text="PIPELINE STATUS",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        stages = tk.Frame(stage_section, background=self.SURFACE)
        stages.pack(fill="x")
        self.stage_container = stages
        self.stage_labels: list[tk.Label] = []
        for index, label in enumerate(
            ("CAPTURE", "QUANTIZE", "16×16 HARDWARE", "CLASSIFY")
        ):
            stage = tk.Label(
                stages,
                text=f"{index + 1:02d}  {label}",
                background="#EAF1F6",
                foreground=self.MUTED,
                padx=8,
                pady=7,
                font=("Segoe UI", 8, "bold"),
            )
            self.stage_labels.append(stage)

        info_grid = tk.Frame(parent, background=self.SURFACE)
        info_grid.pack(fill="x", padx=20, pady=(4, 12))
        self._small_info_card(info_grid, "LOCAL SAMPLES", self.sample_count_text, 0)
        mode_value = tk.StringVar(value=self.sampling_mode.upper())
        self._small_info_card(info_grid, "RESPONSE MODE", mode_value, 1)

        hardware_section = tk.Frame(parent, background="#F5F8FB")
        hardware_section.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(
            hardware_section,
            text="HARDWARE OUTPUT · 256D SCIENTIFIC NOTATION",
            background="#F5F8FB",
            foreground=self.MUTED,
            font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", padx=12, pady=(9, 2))
        output_container = tk.Frame(hardware_section, background="#F5F8FB")
        output_container.pack(fill="x", padx=12, pady=(0, 9))
        self.hardware_output_text = tk.Text(
            output_container,
            height=2,
            wrap="none",
            background="#F5F8FB",
            foreground=self.NAVY,
            borderwidth=0,
            highlightthickness=0,
            font=("Consolas", 9),
            state="disabled",
        )
        hardware_scrollbar = ttk.Scrollbar(
            output_container,
            orient="horizontal",
            command=self.hardware_output_text.xview,
        )
        self.hardware_output_text.configure(
            xscrollcommand=hardware_scrollbar.set,
        )
        self.hardware_output_text.pack(fill="x")
        hardware_scrollbar.pack(fill="x", pady=(3, 0))
        self._set_hardware_output(None)

        history_header = tk.Frame(parent, background=self.SURFACE)
        history_header.pack(fill="x", padx=20, pady=(0, 6))
        tk.Label(
            history_header,
            text="RECENT PREDICTIONS",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")

        self.history = ttk.Treeview(
            parent,
            columns=("sample", "digit"),
            show="headings",
            height=3,
        )
        self.history.heading("sample", text="SAMPLE")
        self.history.heading("digit", text="RESULT")
        self.history.column("sample", width=150, anchor="center")
        self.history.column("digit", width=150, anchor="center")
        self.history.pack(fill="both", expand=True, padx=20, pady=(0, 18))

    def _metric_row(self, parent: tk.Frame, label: str, variable: tk.StringVar) -> None:
        row = tk.Frame(parent, background="#183E59")
        row.pack(fill="x", padx=12, pady=2)
        tk.Label(
            row,
            text=label,
            background="#183E59",
            foreground="#9FB9CC",
            font=("Segoe UI", 8),
        ).pack(side="left")
        tk.Label(
            row,
            textvariable=variable,
            background="#183E59",
            foreground="white",
            font=("Segoe UI", 10, "bold"),
        ).pack(side="right")

    def _small_info_card(
        self,
        parent: tk.Frame,
        label: str,
        variable: tk.StringVar,
        column: int,
    ) -> None:
        parent.columnconfigure(column, weight=1)
        card = tk.Frame(
            parent,
            background="#F5F8FB",
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        card.grid(
            row=0,
            column=column,
            sticky="ew",
            padx=(0, 5) if column == 0 else (5, 0),
        )
        tk.Label(
            card,
            text=label,
            background="#F5F8FB",
            foreground=self.MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", padx=12, pady=(8, 1))
        tk.Label(
            card,
            textvariable=variable,
            background="#F5F8FB",
            foreground=self.NAVY,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 8))

    def _build_status_bar(self) -> None:
        self.status_bar = tk.Frame(
            self.root,
            background="#DDE7EF",
            height=42,
        )
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.grid_propagate(False)
        self.activity = ttk.Progressbar(
            self.status_bar,
            mode="indeterminate",
            length=90,
            style="Scientific.Horizontal.TProgressbar",
        )
        self.status_text_label = tk.Label(
            self.status_bar,
            textvariable=self.status_text,
            background="#DDE7EF",
            foreground=self.NAVY,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_detail_label = tk.Label(
            self.status_bar,
            textvariable=self.status_detail,
            background="#DDE7EF",
            foreground=self.MUTED,
            font=("Segoe UI", 8),
        )

    def _on_root_configure(self, event: tk.Event) -> None:
        if event.widget is not self.root or self.close_requested:
            return
        if self._layout_refresh_after_id is not None:
            self.root.after_cancel(self._layout_refresh_after_id)
        self._layout_refresh_after_id = self.root.after(
            80,
            self._refresh_responsive_layout,
        )

    def _refresh_responsive_layout(self) -> None:
        self._layout_refresh_after_id = None
        if self.close_requested or self._root_destroyed:
            return
        viewport_width = max(1, self.root.winfo_width())
        viewport_height = max(1, self.root.winfo_height())
        target_mode = resolve_application_layout_mode(
            viewport_width,
            viewport_height,
            self.layout_mode,
        )
        if target_mode is not self.layout_mode:
            self._apply_layout_mode(target_mode)
        self._update_responsive_text_width(viewport_width)
        self._schedule_page_extent_refresh()

    def _apply_layout_mode(self, layout_mode: ApplicationLayoutMode) -> None:
        previous_mode = self.layout_mode
        self.layout_mode = layout_mode
        for column_index in (0, 1):
            self.page.columnconfigure(column_index, weight=0, minsize=0)
        for row_index in (0, 1):
            self.page.rowconfigure(row_index, weight=0, minsize=0)

        if layout_mode is ApplicationLayoutMode.LANDSCAPE:
            self.page.columnconfigure(0, weight=5)
            self.page.columnconfigure(1, weight=4)
            self.page.rowconfigure(0, weight=1)
            self.drawing_card.grid(
                row=0,
                column=0,
                sticky="nsew",
                padx=(0, 10),
                pady=0,
            )
            self.result_card.grid(
                row=0,
                column=1,
                sticky="nsew",
                padx=(10, 0),
                pady=0,
            )
        else:
            self.page.columnconfigure(0, weight=1)
            self.page.rowconfigure(0, weight=1)
            self.drawing_card.grid(
                row=0,
                column=0,
                sticky="nsew",
                padx=0,
                pady=(0, 10),
            )
            self.result_card.grid(
                row=1,
                column=0,
                sticky="nsew",
                padx=0,
                pady=(10, 0),
            )

        self._layout_header(layout_mode)
        self._layout_drawing_header(layout_mode)
        self._layout_stage_labels(layout_mode)
        self._layout_status_bar(layout_mode)
        self._update_responsive_text_width(max(1, self.root.winfo_width()))
        self.page_canvas.yview_moveto(0)
        self._schedule_page_extent_refresh()

        if previous_mode is None:
            self.logger.info(
                "[GUI] 自适应排版已启用，模式=%s",
                layout_mode.value,
            )
        elif previous_mode is not layout_mode:
            self.logger.info(
                "[GUI] 自适应排版已切换，模式=%s，窗口=%dx%d",
                layout_mode.value,
                self.root.winfo_width(),
                self.root.winfo_height(),
            )

    def _layout_header(self, layout_mode: ApplicationLayoutMode) -> None:
        for column_index in (0, 1):
            self.header.columnconfigure(column_index, weight=0)
        for row_index in (0, 1):
            self.header.rowconfigure(row_index, weight=0)
        self.header.columnconfigure(0, weight=1)

        if layout_mode is ApplicationLayoutMode.LANDSCAPE:
            self.header.configure(height=92)
            self.header_title_group.grid(
                row=0,
                column=0,
                sticky="w",
                padx=24,
                pady=14,
            )
            self.mode_badge.grid(
                row=0,
                column=1,
                sticky="e",
                padx=24,
                pady=22,
            )
        else:
            self.header.configure(height=156)
            self.header_title_group.grid(
                row=0,
                column=0,
                sticky="w",
                padx=20,
                pady=(16, 6),
            )
            self.mode_badge.grid(
                row=1,
                column=0,
                sticky="w",
                padx=20,
                pady=(2, 16),
            )

    def _layout_drawing_header(
        self,
        layout_mode: ApplicationLayoutMode,
    ) -> None:
        for column_index in (0, 1):
            self.drawing_header.columnconfigure(column_index, weight=0)
        self.drawing_header.columnconfigure(0, weight=1)
        self.drawing_title_label.grid(
            row=0,
            column=0,
            sticky="w",
        )
        if layout_mode is ApplicationLayoutMode.LANDSCAPE:
            self.drawing_subtitle_label.grid(
                row=0,
                column=1,
                sticky="e",
                pady=0,
            )
        else:
            self.drawing_subtitle_label.grid(
                row=1,
                column=0,
                sticky="w",
                pady=(3, 0),
            )

    def _layout_stage_labels(
        self,
        layout_mode: ApplicationLayoutMode,
    ) -> None:
        for column_index in range(4):
            self.stage_container.columnconfigure(column_index, weight=0)
        for row_index in (0, 1):
            self.stage_container.rowconfigure(row_index, weight=0)

        if layout_mode is ApplicationLayoutMode.LANDSCAPE:
            for column_index, stage_label in enumerate(self.stage_labels):
                self.stage_container.columnconfigure(column_index, weight=1)
                stage_label.grid(
                    row=0,
                    column=column_index,
                    sticky="ew",
                    padx=3,
                    pady=0,
                )
        else:
            for column_index in (0, 1):
                self.stage_container.columnconfigure(column_index, weight=1)
            for stage_index, stage_label in enumerate(self.stage_labels):
                stage_label.grid(
                    row=stage_index // 2,
                    column=stage_index % 2,
                    sticky="ew",
                    padx=3,
                    pady=3,
                )

    def _layout_status_bar(
        self,
        layout_mode: ApplicationLayoutMode,
    ) -> None:
        for column_index in range(3):
            self.status_bar.columnconfigure(column_index, weight=0)
        for row_index in (0, 1):
            self.status_bar.rowconfigure(row_index, weight=0)

        if layout_mode is ApplicationLayoutMode.LANDSCAPE:
            self.status_bar.configure(height=42)
            self.status_bar.columnconfigure(1, weight=1)
            self.activity.grid(
                row=0,
                column=0,
                rowspan=1,
                sticky="w",
                padx=(22, 12),
                pady=14,
            )
            self.status_text_label.grid(
                row=0,
                column=1,
                rowspan=1,
                sticky="w",
                padx=0,
                pady=0,
            )
            self.status_detail_label.grid(
                row=0,
                column=2,
                rowspan=1,
                sticky="e",
                padx=22,
                pady=0,
            )
        else:
            self.status_bar.configure(height=66)
            self.status_bar.columnconfigure(1, weight=1)
            self.activity.grid(
                row=0,
                column=0,
                rowspan=2,
                sticky="w",
                padx=(18, 12),
                pady=18,
            )
            self.status_text_label.grid(
                row=0,
                column=1,
                rowspan=1,
                sticky="sw",
                padx=(0, 18),
                pady=(8, 1),
            )
            self.status_detail_label.grid(
                row=1,
                column=1,
                rowspan=1,
                sticky="nw",
                padx=(0, 18),
                pady=(1, 8),
            )

    def _update_responsive_text_width(self, viewport_width: int) -> None:
        wrap_length = (
            max(260, viewport_width - 170)
            if self.layout_mode is ApplicationLayoutMode.PORTRAIT
            else 0
        )
        self.status_detail_label.configure(
            wraplength=wrap_length,
            justify="left",
            anchor="w",
        )

    def _on_page_content_configure(self, _event: tk.Event) -> None:
        self._schedule_page_extent_refresh()

    def _on_page_canvas_configure(self, event: tk.Event) -> None:
        self.page_canvas.itemconfigure(
            self.page_window_id,
            width=max(1, event.width),
        )
        self._schedule_page_extent_refresh()

    def _schedule_page_extent_refresh(self) -> None:
        if (
            self._page_extent_after_id is None
            and not self.close_requested
            and not self._root_destroyed
        ):
            self._page_extent_after_id = self.root.after_idle(
                self._refresh_page_extent,
            )

    def _refresh_page_extent(self) -> None:
        self._page_extent_after_id = None
        if self.close_requested or self._root_destroyed:
            return
        viewport_width = max(1, self.page_canvas.winfo_width())
        viewport_height = max(1, self.page_canvas.winfo_height())
        requested_height = max(1, self.page.winfo_reqheight())
        content_height = max(viewport_height, requested_height)
        page_extent = (viewport_width, content_height)
        if page_extent != self._last_page_extent:
            self.page_canvas.itemconfigure(
                self.page_window_id,
                width=viewport_width,
                height=content_height,
            )
            self.page_canvas.configure(
                scrollregion=(0, 0, viewport_width, content_height),
            )
            self._last_page_extent = page_extent

        scroll_required = requested_height > viewport_height + 1
        if scroll_required and not self._page_scroll_required:
            self.page_scrollbar.grid(row=0, column=1, sticky="ns")
        elif not scroll_required and self._page_scroll_required:
            self.page_scrollbar.grid_remove()
            self.page_canvas.yview_moveto(0)
        self._page_scroll_required = scroll_required

    def _on_page_mousewheel(self, event: tk.Event) -> str | None:
        if (
            not self._page_scroll_required
            or event.state & 0x0001
            or event.widget is self.history
        ):
            return None
        wheel_delta = int(event.delta)
        if wheel_delta == 0:
            return None
        direction = -1 if wheel_delta > 0 else 1
        distance = max(1, abs(wheel_delta) // 120) * 3
        self.page_canvas.yview_scroll(direction * distance, "units")
        return "break"

    def _on_drawing_ready(self, ready: bool) -> None:
        state = "normal" if ready and not self.inference_busy else "disabled"
        if hasattr(self, "save_button"):
            self.save_button.configure(state=state)
        if self.inference_busy:
            return
        if ready and self.workflow_state in {
            ExperimentWorkflowState.IDLE,
            ExperimentWorkflowState.DRAWING,
            ExperimentWorkflowState.COMPLETE,
        }:
            self.workflow_state = ExperimentWorkflowState.DRAWING
            self.status_text.set(
                "Trajectory ready — run Experiment 1",
            )
            self.status_detail.set(
                "Press Enter to save without running inference",
            )
        elif (
            not ready
            and self.workflow_state is ExperimentWorkflowState.DRAWING
        ):
            self.workflow_state = ExperimentWorkflowState.IDLE
            self.status_text.set("Experiment 1 — draw and save a digit")
            self.status_detail.set(
                "Experiment 2 remains a separate manual action",
            )

    def save_sample(self) -> None:
        if self.inference_busy:
            return
        if not self.drawing_pad.is_ready:
            messagebox.showwarning(
                "Cannot Save",
                "Draw a complete digit on the handwriting pad first.",
            )
            return
        try:
            features = self.drawing_pad.normalized_features()
            self.sample_store.append_rows(features)
            self.saved_count = self.sample_store.row_count()
        except PermissionError:
            messagebox.showerror(
                "File In Use",
                "The raw data CSV is open in WPS, Excel, or another program. "
                "Close it and try again.",
            )
            return
        except Exception as error:
            self.logger.exception("[GUI] 保存手写样本失败")
            messagebox.showerror("Save Failed", str(error))
            return

        self.sample_count_text.set(str(self.saved_count))
        self.logger.info(
            "[实验一] 已保存手写样本但未启动推理，行号=%d，共享文件=%s",
            self.saved_count,
            self.sample_store.path,
        )
        self.drawing_pad.clear()
        self.last_raw_signature = self._file_signature(self.sample_store.path)
        self.workflow_state = ExperimentWorkflowState.PENDING
        self.pipeline_state.set("SAMPLE SAVED")
        self.status_text.set("Experiment 1 complete — sample saved")
        self.status_detail.set("Click Experiment 2 to run hardware inference")
        self._set_stage_state(1)
        self.inference_button.configure(state="normal")

    def _request_inference(self) -> None:
        if self.inference_busy or self.close_requested:
            return
        try:
            has_pending_work = self.pipeline.has_pending_work()
        except (OSError, ValueError) as error:
            self.logger.warning("[实验二] 无法检查待推理数据：%s", error)
            self.workflow_state = ExperimentWorkflowState.ERROR
            self.pipeline_state.set("ATTENTION")
            self.status_text.set("Experiment 2 cannot inspect pending data")
            self.status_detail.set("Review app.log, repair the data, and try again")
            self.inference_button.configure(state="disabled")
            return
        if not has_pending_work:
            self.workflow_state = ExperimentWorkflowState.IDLE
            self.pipeline_state.set("READY")
            self.status_text.set("Experiment 2 has no pending samples")
            self.status_detail.set("Save a digit with Experiment 1 first")
            self.inference_button.configure(state="disabled")
            return
        self.inference_busy = True
        self.workflow_state = ExperimentWorkflowState.PROCESSING
        self.drawing_pad.set_enabled(False)
        self.save_button.configure(state="disabled")
        self.inference_button.configure(state="disabled")
        self.pipeline_state.set("PROCESSING")
        self.status_text.set("Experiment 2 — running hardware inference…")
        self.status_detail.set(
            "Normalize → Quantize → 16×16 response → Classify"
        )
        self.logger.info("[实验二] 用户已启动硬件推理")
        self.activity.start(12)
        self._set_stage_state(0)
        worker = threading.Thread(
            target=self._run_inference_worker,
            name="pen-digits-inference",
            daemon=False,
        )
        self.inference_thread = worker
        worker.start()

    def _run_inference_worker(self) -> None:
        try:
            progress = self.pipeline.process_once()
        except Exception as error:
            self.logger.exception("[实验二] 硬件推理失败")
            self.result_queue.put(("error", error))
        else:
            self.result_queue.put(("success", progress))

    def _drain_result_queue(self) -> None:
        self._result_queue_after_id = None
        try:
            result_type, payload = self.result_queue.get_nowait()
        except queue.Empty:
            if not self._root_destroyed:
                self._result_queue_after_id = self.root.after(
                    80,
                    self._drain_result_queue,
                )
            return

        self.inference_busy = False
        self.activity.stop()
        if self.close_requested:
            self._finalize_close()
            return

        self.drawing_pad.set_enabled(True)
        if result_type == "success":
            self.last_raw_signature = self._file_signature(self.sample_store.path)
            self.last_reported_error = None
            self._handle_inference_success(payload)
            self._refresh_manual_workflow_status(
                preserve_when_no_pending=True,
            )
        else:
            self._handle_inference_error(payload)
            self.inference_button.configure(state="normal")
        self.save_button.configure(
            state="normal" if self.drawing_pad.is_ready else "disabled"
        )
        self._result_queue_after_id = self.root.after(
            80,
            self._drain_result_queue,
        )

    def _handle_inference_success(self, payload: object) -> None:
        if not isinstance(payload, PipelineProgress):
            raise TypeError("GUI inference worker returned an invalid result.")
        self.pipeline_state.set("ONLINE")
        if payload.changed:
            self.workflow_state = ExperimentWorkflowState.COMPLETE
            self._set_stage_state(4)
            self._load_latest_hardware_output()
            for summary in payload.predictions:
                self._show_prediction(summary)
            self.status_text.set("Experiment 2 complete — hardware inference finished")
            self.status_detail.set(
                f"New rows: differential {payload.normalized_rows} / "
                f"hardware {payload.simulated_rows} / "
                f"predictions {payload.predicted_rows}"
            )
        else:
            self.workflow_state = ExperimentWorkflowState.COMPLETE
            self._set_stage_state(4)
            self.status_text.set("Experiment 2 complete — no pending rows")
            self.status_detail.set("Save a new digit with Experiment 1")

    def _handle_inference_error(self, payload: object) -> None:
        error = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        self.workflow_state = ExperimentWorkflowState.ERROR
        self.pipeline_state.set("ATTENTION")
        self.status_text.set("Experiment 2 inference incomplete")
        self.status_detail.set(
            "Review the error dialog or app.log, then retry Experiment 2",
        )
        self._set_stage_error()
        if isinstance(error, PermissionError):
            message = (
                "A runtime CSV is open in WPS, Excel, or another program. "
                "Close it, then run Experiment 2 again."
            )
        else:
            message = str(error)
        if message != self.last_reported_error:
            self.last_reported_error = message
            messagebox.showerror("Inference Failed", message)

    def _show_prediction(self, summary: PredictionSummary) -> None:
        self.predicted_digit.set(str(summary.predicted_digit))
        self.sample_index.set(str(summary.sample_index))
        self.top_score.set(f"{summary.top_score:.4f}")
        self.history.insert(
            "",
            0,
            values=(
                summary.sample_index,
                summary.predicted_digit,
            ),
        )
        children = self.history.get_children()
        for item in children[8:]:
            self.history.delete(item)

    def _set_stage_state(self, completed_count: int) -> None:
        for index, label in enumerate(self.stage_labels):
            completed = index < completed_count
            label.configure(
                background="#DDF3EE" if completed else "#EAF1F6",
                foreground=self.SUCCESS if completed else self.MUTED,
            )

    def _set_stage_error(self) -> None:
        for label in self.stage_labels:
            label.configure(background="#FEF0E7", foreground=self.WARNING)

    def _refresh_manual_workflow_status(
        self,
        *,
        preserve_when_no_pending: bool = False,
    ) -> bool:
        try:
            has_pending_work = self.pipeline.has_pending_work()
            self.saved_count = self.sample_store.row_count()
        except (OSError, ValueError) as error:
            self.logger.warning("[GUI] 无法检查待推理数据：%s", error)
            self.workflow_state = ExperimentWorkflowState.ERROR
            self.pipeline_state.set("ATTENTION")
            self.status_text.set("Pipeline data requires attention")
            self.status_detail.set("Review app.log before running Experiment 2")
            self.inference_button.configure(state="disabled")
            return False

        self.sample_count_text.set(str(self.saved_count))
        if has_pending_work:
            if not self.inference_busy:
                self.workflow_state = ExperimentWorkflowState.PENDING
                self.pipeline_state.set("INFERENCE READY")
                self.status_text.set(
                    "Pending sample detected — Experiment 2 is ready",
                )
                self.status_detail.set(
                    "Inference waits until you click Experiment 2",
                )
                self._set_stage_state(1)
                self.inference_button.configure(state="normal")
            return True

        if not preserve_when_no_pending and not self.inference_busy:
            self.workflow_state = ExperimentWorkflowState.IDLE
            self.pipeline_state.set("READY")
            self.status_text.set("Experiment 1 — draw and save a digit")
            self.status_detail.set("Experiment 2 runs hardware inference")
            self._set_stage_state(0)
        if not self.inference_busy:
            self.inference_button.configure(state="disabled")
        return False

    def _monitor_new_rows(self) -> None:
        self._monitor_after_id = None
        if self.close_requested:
            return
        signature = self._file_signature(self.sample_store.path)
        if signature != self.last_raw_signature:
            self.last_raw_signature = signature
            if self.inference_busy:
                self.logger.info(
                    "[GUI] 推理期间侦测到原始数据变化，完成后将重新检查积压行",
                )
            else:
                has_pending_work = self._refresh_manual_workflow_status()
                if has_pending_work:
                    self.logger.info(
                        "[实验一] 已侦测外部新增样本，等待用户启动实验二，"
                        "共享文件=%s",
                        self.sample_store.path,
                    )
        self._monitor_after_id = self.root.after(
            self.monitor_interval_ms,
            self._monitor_new_rows,
        )

    def _request_close(self) -> None:
        if self.close_requested or self._root_destroyed:
            return
        self.close_requested = True
        worker_is_active = (
            self.inference_thread is not None
            and self.inference_thread.is_alive()
        )
        if self.inference_busy or worker_is_active:
            self.workflow_state = ExperimentWorkflowState.CLOSING
            self.pipeline_state.set("FINISHING")
            self.status_text.set("Finishing Experiment 2 before closing…")
            self.status_detail.set(
                "The window will close after all CSV writes are complete",
            )
            self.save_button.configure(state="disabled")
            self.inference_button.configure(state="disabled")
            self.drawing_pad.set_enabled(False)
            self.logger.info("[GUI] 收到关闭请求，等待实验二安全完成")
            self._close_poll_after_id = self.root.after(
                100,
                self._poll_inference_before_close,
            )
            return
        self._finalize_close()

    def _poll_inference_before_close(self) -> None:
        self._close_poll_after_id = None
        worker_is_active = (
            self.inference_thread is not None
            and self.inference_thread.is_alive()
        )
        if worker_is_active:
            self._close_poll_after_id = self.root.after(
                100,
                self._poll_inference_before_close,
            )
            return
        self._finalize_close()

    def _finalize_close(self) -> None:
        if self._root_destroyed:
            return
        self._root_destroyed = True
        callback_attributes = (
            "_layout_refresh_after_id",
            "_page_extent_after_id",
            "_result_queue_after_id",
            "_monitor_after_id",
            "_close_poll_after_id",
        )
        for attribute_name in callback_attributes:
            callback_id = getattr(self, attribute_name)
            if callback_id is None:
                continue
            try:
                self.root.after_cancel(callback_id)
            except tk.TclError:
                pass
            setattr(self, attribute_name, None)
        self.logger.info("[GUI] 两阶段实验窗口已安全关闭")
        self.root.destroy()

    def _load_recent_history(self) -> None:
        report_path = self.pipeline.paths.report_file
        if not report_path.exists():
            return
        try:
            with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))[-8:]
        except (OSError, csv.Error):
            self.logger.warning("[GUI] 无法读取已有推理记录：%s", report_path)
            return
        for row in reversed(rows):
            self.history.insert(
                "",
                "end",
                values=(
                    row.get("sample_index", "—"),
                    row.get("predicted_digit", "—"),
                ),
            )
        if rows:
            latest = rows[-1]
            self.predicted_digit.set(latest.get("predicted_digit", "—"))
            self.sample_index.set(latest.get("sample_index", "—"))
            self.top_score.set(self._format_number(latest.get("top_score")))

    def _load_latest_hardware_output(self) -> None:
        try:
            row_count = self.pipeline.hardware_store.row_count()
            if row_count == 0:
                self._set_hardware_output(None)
                return
            latest_rows = self.pipeline.hardware_store.read_rows(row_count - 1)
        except (OSError, ValueError) as error:
            self.logger.warning("[GUI] 无法读取最新硬件输出：%s", error)
            self._set_hardware_output(None)
            return
        self._set_hardware_output(latest_rows[-1])

    def _set_hardware_output(self, values: Iterable[float] | None) -> None:
        output = self.format_hardware_output(values)
        self.hardware_output_text.configure(state="normal")
        self.hardware_output_text.delete("1.0", "end")
        self.hardware_output_text.insert("1.0", output)
        self.hardware_output_text.configure(state="disabled")
        self.hardware_output_text.xview_moveto(0)

    @staticmethod
    def format_hardware_output(values: Iterable[float] | None) -> str:
        if values is None:
            return "No hardware output yet."
        return ", ".join(f"{float(value):.6e}" for value in values)

    @staticmethod
    def _format_number(value: str | None) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return "—"

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int] | None:
        try:
            metadata = path.stat()
        except FileNotFoundError:
            return None
        return metadata.st_mtime_ns, metadata.st_size
