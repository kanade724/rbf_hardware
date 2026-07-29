"""Standalone scientific GUI for collecting Pen Digits rows without inference."""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

from ..data.csv_store import NumericCsvStore
from .pen_digits_collector import PenDigitDrawingPad


class PenDigitsCollectionApplication:
    """Persist hand-drawn samples while leaving inference to another process."""

    BACKGROUND = "#EEF3F8"
    SURFACE = "#FFFFFF"
    NAVY = "#102A43"
    MUTED = "#627D98"
    ACCENT = "#087F8C"
    BORDER = "#D8E2EC"

    def __init__(
        self,
        root: tk.Tk,
        *,
        sample_store: NumericCsvStore,
        logger: logging.Logger,
    ) -> None:
        self.root = root
        self.sample_store = sample_store
        self.logger = logger
        self.saved_count = sample_store.row_count()
        self.status_text = tk.StringVar()

        self._configure_root()
        self._configure_styles()
        self._build_layout()
        self.root.bind("<Control-z>", lambda _event: self.drawing_pad.undo())
        self.root.bind("<Return>", lambda _event: self.save_sample())
        self._update_status(False)

    def _configure_root(self) -> None:
        self.root.title("RBF Hardware · Pen Digits 独立采集器")
        self.root.configure(background=self.BACKGROUND)
        self.root.geometry("720x800")
        self.root.minsize(660, 760)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Surface.TFrame", background=self.SURFACE)
        style.configure(
            "CollectorAccent.TButton",
            background=self.ACCENT,
            foreground="white",
            borderwidth=0,
            padding=(18, 11),
            font=("Microsoft YaHei UI", 10, "bold"),
        )
        style.map(
            "CollectorAccent.TButton",
            background=[("active", "#096C77"), ("disabled", "#A9BCC7")],
            foreground=[("disabled", "#EEF3F8")],
        )
        style.configure(
            "CollectorSecondary.TButton",
            background="#E8F0F5",
            foreground=self.NAVY,
            borderwidth=0,
            padding=(15, 10),
            font=("Microsoft YaHei UI", 10),
        )

    def _build_layout(self) -> None:
        header = tk.Frame(self.root, background=self.NAVY, height=92)
        header.pack(fill="x")
        header.pack_propagate(False)
        title_group = tk.Frame(header, background=self.NAVY)
        title_group.pack(side="left", padx=28, pady=13)
        tk.Label(
            title_group,
            text="RBF HARDWARE",
            background=self.NAVY,
            foreground="#7FE3EA",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            title_group,
            text="Pen Digits 独立数据采集",
            background=self.NAVY,
            foreground="white",
            font=("Microsoft YaHei UI", 18, "bold"),
        ).pack(anchor="w")

        card = tk.Frame(
            self.root,
            background=self.SURFACE,
            highlightbackground=self.BORDER,
            highlightthickness=1,
        )
        card.pack(fill="both", expand=True, padx=24, pady=20)

        heading = tk.Frame(card, background=self.SURFACE)
        heading.pack(fill="x", padx=20, pady=(18, 10))
        tk.Label(
            heading,
            text="手写板",
            background=self.SURFACE,
            foreground=self.NAVY,
            font=("Microsoft YaHei UI", 14, "bold"),
        ).pack(side="left")
        tk.Label(
            heading,
            text="仅采集，不在本窗口执行推理",
            background=self.SURFACE,
            foreground=self.MUTED,
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right")

        self.drawing_pad = PenDigitDrawingPad(
            card,
            on_ready_changed=self._on_drawing_ready,
        )
        self.drawing_pad.pack(anchor="center", padx=20)

        tk.Label(
            card,
            textvariable=self.status_text,
            background=self.SURFACE,
            foreground=self.MUTED,
            anchor="w",
            font=("Microsoft YaHei UI", 9),
        ).pack(fill="x", padx=20, pady=(10, 6))

        controls = tk.Frame(card, background=self.SURFACE)
        controls.pack(fill="x", padx=20, pady=(2, 18))
        ttk.Button(
            controls,
            text="撤销  Ctrl+Z",
            command=self.drawing_pad.undo,
            style="CollectorSecondary.TButton",
        ).pack(side="left")
        ttk.Button(
            controls,
            text="清空",
            command=self.drawing_pad.clear,
            style="CollectorSecondary.TButton",
        ).pack(side="left", padx=8)
        self.save_button = ttk.Button(
            controls,
            text="保存样本  Enter",
            command=self.save_sample,
            style="CollectorAccent.TButton",
            state="disabled",
        )
        self.save_button.pack(side="right")

    def _on_drawing_ready(self, ready: bool) -> None:
        if hasattr(self, "save_button"):
            self.save_button.configure(state="normal" if ready else "disabled")
        self._update_status(ready)

    def save_sample(self) -> None:
        if not self.drawing_pad.is_ready:
            messagebox.showwarning("无法保存", "请先在手写板中写下一个完整数字。")
            return
        try:
            self.sample_store.append_rows(self.drawing_pad.normalized_features())
        except PermissionError:
            messagebox.showerror(
                "文件被占用",
                "原始数据表正在被 WPS、Excel 或其他程序占用，请关闭后重试。",
            )
            return
        except Exception as error:
            self.logger.exception("[采集] 独立采集器保存样本失败")
            messagebox.showerror("保存失败", str(error))
            return

        self.saved_count += 1
        self.logger.info(
            "[采集] 独立采集器已追加原始样本，行号=%d，共享文件=%s",
            self.saved_count,
            self.sample_store.path,
        )
        self.drawing_pad.clear()
        self.status_text.set(
            f"已保存第 {self.saved_count} 条样本 · {self.sample_store.path}"
        )

    def _update_status(self, ready: bool) -> None:
        if ready:
            self.status_text.set("已提取 8 个等距点，可以保存样本")
        else:
            self.status_text.set(
                f"请写下一个数字 · 文件中已有 {self.saved_count} 条样本"
            )
