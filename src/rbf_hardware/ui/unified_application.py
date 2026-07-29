"""Unified scientific desktop interface for collection and hardware inference."""

from __future__ import annotations

import csv
import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from ..data.csv_store import NumericCsvStore
from ..inference.pipeline import (
    PipelineProgress,
    PredictionSummary,
    StreamingInferencePipeline,
)
from .pen_digits_collector import PenDigitDrawingPad


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
        self.last_reported_error: str | None = None
        self.last_raw_signature = self._file_signature(sample_store.path)
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self.status_text = tk.StringVar(value="Initializing hardware response model...")
        self.status_detail = tk.StringVar(value="Please wait")
        self.predicted_digit = tk.StringVar(value="—")
        self.top_score = tk.StringVar(value="—")
        self.sample_index = tk.StringVar(value="—")
        self.experiment_file = tk.StringVar(value="No experiment table generated yet")
        self.sample_count_text = tk.StringVar(value=str(self.saved_count))
        self.pipeline_state = tk.StringVar(value="Initializing")
        self.save_button: ttk.Button

        self._configure_root()
        self._configure_styles()
        self._build_layout()
        self._load_recent_history()
        self.root.bind("<Control-z>", lambda _event: self.drawing_pad.undo())
        self.root.bind("<Return>", lambda _event: self.save_and_infer())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self.root.after(80, self._drain_result_queue)
        self.root.after(150, self._request_inference)
        self.root.after(self.monitor_interval_ms, self._monitor_new_rows)

    def _configure_root(self) -> None:
        self.root.title("RBF Hardware · Pen Digits Research Workstation")
        self.root.configure(background=self.BACKGROUND)
        self.root.geometry("1220x930")
        self.root.minsize(1120, 820)

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
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#096C77"), ("disabled", "#A9BCC7")],
            foreground=[("disabled", "#EEF3F8")],
        )
        style.configure(
            "Secondary.TButton",
            background="#E8F0F5",
            foreground=self.NAVY,
            borderwidth=0,
            padding=(15, 10),
            font=("Microsoft YaHei UI", 10),
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
            font=("Microsoft YaHei UI", 9),
        )
        style.configure(
            "Treeview.Heading",
            background="#EAF1F6",
            foreground=self.MUTED,
            relief="flat",
            font=("Microsoft YaHei UI", 9, "bold"),
        )
        style.map("Treeview", background=[("selected", "#CDEFF2")])

    def _build_layout(self) -> None:
        self._build_header()
        page = ttk.Frame(self.root, style="Page.TFrame", padding=(22, 18, 22, 16))
        page.pack(fill="both", expand=True)
        page.columnconfigure(0, weight=5)
        page.columnconfigure(1, weight=4)
        page.rowconfigure(0, weight=1)

        left_card = tk.Frame(
            page,
            background=self.SURFACE,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        right_card = tk.Frame(
            page,
            background=self.SURFACE,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        right_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self._build_drawing_panel(left_card)
        self._build_result_panel(right_card)
        self._build_status_bar()

    def _build_header(self) -> None:
        header = tk.Frame(self.root, background=self.NAVY, height=92)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_group = tk.Frame(header, background=self.NAVY)
        title_group.pack(side="left", padx=24, pady=14)
        tk.Label(
            title_group,
            text="RBF HARDWARE",
            background=self.NAVY,
            foreground="#7FE3EA",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Pen Digits Research Workstation",
            background=self.NAVY,
            foreground="white",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")

        mode_badge = tk.Frame(header, background="#183E59")
        mode_badge.pack(side="right", padx=24, pady=22)
        tk.Label(
            mode_badge,
            text="●",
            background="#183E59",
            foreground="#4EE1B2",
            font=("Segoe UI", 10),
        ).pack(side="left", padx=(12, 5), pady=7)
        tk.Label(
            mode_badge,
            text=f"Measured Response · {self.sampling_mode.upper()}",
            background="#183E59",
            foreground="#D9F6F8",
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side="left", padx=(0, 12), pady=7)

    def _build_drawing_panel(self, parent: tk.Frame) -> None:
        title = tk.Frame(parent, background=self.SURFACE)
        title.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            title,
            text="01  Handwriting Data Collection",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            title,
            text="Automatically extracts 8 evenly spaced trajectory points",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right")

        canvas_host = ttk.Frame(parent, style="Surface.TFrame")
        canvas_host.pack(fill="both", expand=True, padx=20)
        self.drawing_pad = PenDigitDrawingPad(
            canvas_host,
            on_ready_changed=self._on_drawing_ready,
        )
        self.drawing_pad.pack(fill="both", expand=True)

        instruction = tk.Label(
            parent,
            text="Hold the mouse button to draw. The colored markers are the 8 points sent to the model.",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        )
        instruction.pack(fill="x", padx=20, pady=(10, 6))

        controls = tk.Frame(parent, background=self.SURFACE)
        controls.pack(fill="x", padx=20, pady=(2, 18))
        ttk.Button(
            controls,
            text="Undo  Ctrl+Z",
            command=self.drawing_pad.undo,
            style="Secondary.TButton",
        ).pack(side="left")
        ttk.Button(
            controls,
            text="Clear",
            command=self.drawing_pad.clear,
            style="Secondary.TButton",
        ).pack(side="left", padx=8)
        self.save_button = ttk.Button(
            controls,
            text="Save and Recognize  Enter",
            command=self.save_and_infer,
            style="Accent.TButton",
            state="disabled",
        )
        self.save_button.pack(side="right")

    def _build_result_panel(self, parent: tk.Frame) -> None:
        result_header = tk.Frame(parent, background=self.SURFACE)
        result_header.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            result_header,
            text="02  Hardware Inference",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            result_header,
            textvariable=self.pipeline_state,
            background="#E3F5F3",
            foreground=self.SUCCESS,
            padx=10,
            pady=4,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side="right")

        result_card = tk.Frame(parent, background=self.NAVY, height=172)
        result_card.pack(fill="x", padx=20)
        result_card.pack_propagate(False)
        result_left = tk.Frame(result_card, background=self.NAVY)
        result_left.pack(side="left", fill="both", expand=True, padx=(22, 8), pady=18)
        tk.Label(
            result_left,
            text="Recognition Result",
            background=self.NAVY,
            foreground="#9FB9CC",
            font=("Microsoft YaHei UI", 10),
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
        self._metric_row(metrics, "Sample Number", self.sample_index)
        self._metric_row(metrics, "Top Score", self.top_score)

        stage_section = tk.Frame(parent, background=self.SURFACE)
        stage_section.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            stage_section,
            text="Pipeline Status",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(anchor="w", pady=(0, 8))
        stages = tk.Frame(stage_section, background=self.SURFACE)
        stages.pack(fill="x")
        self.stage_labels: list[tk.Label] = []
        for index, label in enumerate(
            ("Collection", "Quantization", "16×16 Hardware", "Recognition")
        ):
            stage = tk.Label(
                stages,
                text=f"{index + 1:02d}  {label}",
                background="#EAF1F6",
                foreground=self.MUTED,
                padx=8,
                pady=7,
                font=("Microsoft YaHei UI", 8, "bold"),
            )
            stage.pack(side="left", fill="x", expand=True, padx=(0, 5 if index < 3 else 0))
            self.stage_labels.append(stage)

        info_grid = tk.Frame(parent, background=self.SURFACE)
        info_grid.pack(fill="x", padx=20, pady=(4, 12))
        self._small_info_card(info_grid, "Local Samples", self.sample_count_text, 0)
        mode_value = tk.StringVar(value=self.sampling_mode.upper())
        self._small_info_card(info_grid, "Response Mode", mode_value, 1)

        experiment_section = tk.Frame(parent, background="#F5F8FB")
        experiment_section.pack(fill="x", padx=20, pady=(0, 12))
        tk.Label(
            experiment_section,
            text="Latest Experiment Table",
            background="#F5F8FB",
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 8, "bold"),
        ).pack(anchor="w", padx=12, pady=(9, 2))
        tk.Label(
            experiment_section,
            textvariable=self.experiment_file,
            background="#F5F8FB",
            foreground=self.NAVY,
            anchor="w",
            font=("Consolas", 8),
        ).pack(fill="x", padx=12, pady=(0, 9))

        history_header = tk.Frame(parent, background=self.SURFACE)
        history_header.pack(fill="x", padx=20, pady=(0, 6))
        tk.Label(
            history_header,
            text="Recent Recognition History",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 11, "bold"),
        ).pack(side="left")
        ttk.Button(
            history_header,
            text="Open Experiment Folder",
            command=self.open_experiment_directory,
            style="Secondary.TButton",
        ).pack(side="right")

        self.history = ttk.Treeview(
            parent,
            columns=("sample", "digit"),
            show="headings",
            height=3,
        )
        self.history.heading("sample", text="Sample")
        self.history.heading("digit", text="Result")
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
            font=("Microsoft YaHei UI", 8),
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
            font=("Microsoft YaHei UI", 8),
        ).pack(anchor="w", padx=12, pady=(8, 1))
        tk.Label(
            card,
            textvariable=variable,
            background="#F5F8FB",
            foreground=self.NAVY,
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", padx=12, pady=(0, 8))

    def _build_status_bar(self) -> None:
        status_bar = tk.Frame(self.root, background="#DDE7EF", height=42)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        self.activity = ttk.Progressbar(
            status_bar,
            mode="indeterminate",
            length=90,
            style="Scientific.Horizontal.TProgressbar",
        )
        self.activity.pack(side="left", padx=(22, 12), pady=14)
        tk.Label(
            status_bar,
            textvariable=self.status_text,
            background="#DDE7EF",
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side="left")
        tk.Label(
            status_bar,
            textvariable=self.status_detail,
            background="#DDE7EF",
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 8),
        ).pack(side="right", padx=22)

    def _on_drawing_ready(self, ready: bool) -> None:
        state = "normal" if ready and not self.inference_busy else "disabled"
        if hasattr(self, "save_button"):
            self.save_button.configure(state=state)
        if not self.inference_busy:
            self.status_text.set(
                "Trajectory sampled; ready to save and recognize"
                if ready
                else "System Ready"
            )
            self.status_detail.set(
                "Press Enter to save" if ready else "Draw a digit on the pad"
            )

    def save_and_infer(self) -> None:
        if self.inference_busy:
            return
        if not self.drawing_pad.is_ready:
            messagebox.showwarning(
                "Cannot Save", "Please draw a complete digit first."
            )
            return
        try:
            features = self.drawing_pad.normalized_features()
            self.sample_store.append_rows(features)
        except PermissionError:
            messagebox.showerror(
                "File In Use",
                "The raw data CSV is open in WPS, Excel, or another program. "
                "Close it and try again.",
            )
            return
        except Exception as error:
            self.logger.exception("[GUI] Failed to save handwriting sample")
            messagebox.showerror("Save Failed", str(error))
            return

        self.saved_count += 1
        self.sample_count_text.set(str(self.saved_count))
        self.logger.info(
            "[Collection] GUI appended raw sample, row=%d, shared_file=%s",
            self.saved_count,
            self.sample_store.path,
        )
        self.drawing_pad.clear()
        self._request_inference()

    def _request_inference(self) -> None:
        if self.inference_busy:
            return
        self.inference_busy = True
        self.drawing_pad.set_enabled(False)
        self.save_button.configure(state="disabled")
        self.pipeline_state.set("Processing")
        self.status_text.set("Running simulated hardware inference...")
        self.status_detail.set(
            "Normalization → Quantization → 16×16 Response → Recognition"
        )
        self.activity.start(12)
        self._set_stage_state(0)
        worker = threading.Thread(
            target=self._run_inference_worker,
            name="pen-digits-inference",
            daemon=True,
        )
        worker.start()

    def _run_inference_worker(self) -> None:
        try:
            progress = self.pipeline.process_once()
        except Exception as error:
            self.logger.exception("[GUI] Automatic inference failed")
            self.result_queue.put(("error", error))
        else:
            self.result_queue.put(("success", progress))

    def _drain_result_queue(self) -> None:
        try:
            result_type, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(80, self._drain_result_queue)
            return

        self.inference_busy = False
        self.drawing_pad.set_enabled(True)
        self.activity.stop()
        if result_type == "success":
            self.last_raw_signature = self._file_signature(self.sample_store.path)
            self.last_reported_error = None
            self._handle_inference_success(payload)
        else:
            self._handle_inference_error(payload)
            self.root.after(1500, self._request_inference)
        self.save_button.configure(
            state="normal" if self.drawing_pad.is_ready else "disabled"
        )
        self.root.after(80, self._drain_result_queue)

    def _handle_inference_success(self, payload: object) -> None:
        if not isinstance(payload, PipelineProgress):
            raise TypeError("GUI inference worker returned an invalid result.")
        self.pipeline_state.set("Online")
        if payload.changed:
            self._set_stage_state(4)
            if payload.experiment_files:
                self.experiment_file.set(payload.experiment_files[-1].name)
            for summary in payload.predictions:
                self._show_prediction(summary)
            self.status_text.set("Hardware simulation and recognition completed")
            self.status_detail.set(
                f"Added: quantized {payload.normalized_rows} / "
                f"hardware {payload.simulated_rows} / "
                f"predictions {payload.predicted_rows}"
            )
        else:
            self._set_stage_state(4)
            self.status_text.set("System Ready")
            self.status_detail.set("Automatically monitoring for new data rows")

    def _handle_inference_error(self, payload: object) -> None:
        error = payload if isinstance(payload, Exception) else RuntimeError(str(payload))
        self.pipeline_state.set("Check Required")
        self.status_text.set("Automatic inference did not complete")
        self.status_detail.set(str(error))
        self._set_stage_error()
        if isinstance(error, PermissionError):
            message = (
                "A runtime CSV is open in WPS, Excel, or another program. "
                "Close it and wait for the automatic retry."
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

    def _monitor_new_rows(self) -> None:
        if not self.inference_busy:
            signature = self._file_signature(self.sample_store.path)
            if signature != self.last_raw_signature:
                self._request_inference()
        self.root.after(self.monitor_interval_ms, self._monitor_new_rows)

    def _load_recent_history(self) -> None:
        report_path = self.pipeline.paths.report_file
        if not report_path.exists():
            return
        try:
            with report_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))[-8:]
        except (OSError, csv.Error):
            self.logger.warning(
                "[GUI] Unable to read existing inference history: %s", report_path
            )
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

    def open_experiment_directory(self) -> None:
        directory = self.pipeline.paths.experiment_output_dir
        directory.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(directory)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Experiment Folder", str(directory))

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
