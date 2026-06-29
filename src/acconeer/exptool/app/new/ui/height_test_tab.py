# Copyright (c) Acconeer AB, 2022-2024
# All rights reserved
# Modified: precision testing panel for A121 radar

from __future__ import annotations

import collections
import typing as t

import numpy as np

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


WINDOW_SIZE = 100

# ---- Apple HIG color palette ----
C_PRIMARY = "#007AFF"
C_SUCCESS = "#34C759"
C_WARN = "#FF9500"
C_ERROR = "#FF3B30"
C_BG_CARD = "#FFFFFF"
C_BG_PAGE = "#F2F2F7"
C_BORDER = "#C6C6C8"
C_TEXT_PRIMARY = "#1C1C1E"
C_TEXT_SECONDARY = "#8E8E93"
C_TEXT_MUTED = "#AEAEB2"


class _MeasurementWorker(QThread):
    """Worker thread: raw readings for display, rolling window for stats."""

    sig_distance = Signal(float)
    sig_stats = Signal(dict)
    sig_error = Signal(str)
    sig_finished = Signal()

    def __init__(self, detector: t.Any, parent: t.Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.detector = detector
        self._running = False
        self._window: collections.deque[float] = collections.deque(maxlen=WINDOW_SIZE)

    def run(self) -> None:
        self._running = True
        try:
            while self._running:
                result = self.detector.get_next()
                sensor_result = result[1]
                if sensor_result.distances is not None and len(sensor_result.distances) > 0:
                    d = sensor_result.distances[0]
                    self._window.append(d)
                    self.sig_distance.emit(d)
                    self.sig_stats.emit(self._compute_stats())
        except Exception as e:
            self.sig_error.emit(str(e))
        finally:
            self.sig_finished.emit()

    def stop(self) -> None:
        self._running = False

    def clear(self) -> None:
        self._window.clear()

    def _compute_stats(self) -> dict:
        arr = np.array(list(self._window)) * 100.0
        n = len(arr)
        if n < 2:
            v = arr[0] if n else 0.0
            return {"n": n, "mean": v, "std": 0.0, "min": v, "max": v, "pct_2mm": 100.0}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        within_2mm = np.mean(np.abs(arr - mean) < 0.2) * 100.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "pct_2mm": float(within_2mm),
        }


class HeightTestWidget(QWidget):
    """Precision testing panel. High Accuracy preset: Profile 3, step=2, quality=20."""

    def __init__(self, parent: t.Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._client: t.Any = None
        self._detector: t.Any = None
        self._worker: t.Optional[_MeasurementWorker] = None
        self._connected = False
        self._measuring = False
        self._last_mean: float = 0.0
        self._last_stats: dict = {}
        self._snap_count = 0
        self._unit_cm = True  # True=cm, False=mm
        self._last_raw_m: float = 0.0  # cached raw distance in meters for unit switch
        self._snapshots: list[dict] = []  # structured records for export

        self._build_ui()
        self._update_ui_state()

    # ==================== STYLESHEET HELPERS ====================

    _CARD_CSS = (
        "QGroupBox {"
        "  border: none; border-radius: 10px;"
        "  background: %s; padding: 12px;"
        "}"
    ) % C_BG_CARD

    _BTN_PRIMARY_CSS = (
        "QPushButton {"
        "  background: %s; color: white; border: none; border-radius: 8px;"
        "  padding: 8px 20px; font-size: 13px; font-weight: 600;"
        "  letter-spacing: -0.01em;"
        "}"
        "QPushButton:hover  { background: #0066D6; }"
        "QPushButton:pressed { background: #004CAA; }"
        "QPushButton:disabled { background: %s; }"
    ) % (C_PRIMARY, C_TEXT_MUTED)

    _BTN_SUCCESS_CSS = (
        "QPushButton {"
        "  background: %s; color: white; border: none; border-radius: 8px;"
        "  padding: 8px 20px; font-size: 13px; font-weight: 600;"
        "  letter-spacing: -0.01em;"
        "}"
        "QPushButton:hover  { background: #2DB84D; }"
        "QPushButton:disabled { background: %s; }"
    ) % (C_SUCCESS, C_TEXT_MUTED)

    _BTN_DANGER_CSS = (
        "QPushButton {"
        "  background: rgba(255,59,48,0.10); color: %s; border: none; border-radius: 8px;"
        "  padding: 8px 20px; font-size: 13px; font-weight: 600;"
        "  letter-spacing: -0.01em;"
        "}"
        "QPushButton:hover  { background: rgba(255,59,48,0.18); }"
        "QPushButton:disabled { color: %s; background: rgba(0,0,0,0.05); }"
    ) % (C_ERROR, C_TEXT_MUTED)

    _BTN_OUTLINE_CSS = (
        "QPushButton {"
        "  background: rgba(0,122,255,0.08); color: %s; border: none; border-radius: 8px;"
        "  padding: 8px 20px; font-size: 13px; font-weight: 600;"
        "  letter-spacing: -0.01em;"
        "}"
        "QPushButton:hover  { background: rgba(0,122,255,0.15); }"
        "QPushButton:disabled { color: %s; background: rgba(0,0,0,0.05); }"
    ) % (C_PRIMARY, C_TEXT_MUTED)

    # ==================== UI construction ====================

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        self.setStyleSheet(f"background: {C_BG_PAGE};")

        # --- Top bar (compact) ---
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)

        self._btn_connect = QPushButton("连接设备")
        self._btn_connect.setStyleSheet(self._BTN_PRIMARY_CSS)
        self._btn_connect.setFixedHeight(32)
        self._btn_connect.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_connect.clicked.connect(self._on_connect)
        top_bar.addWidget(self._btn_connect)

        self._lbl_status = QLabel("未连接")
        self._lbl_status.setStyleSheet(
            f"color: {C_TEXT_MUTED}; font-weight: 500; font-size: 14px; border: none;"
        )
        top_bar.addWidget(self._lbl_status)
        top_bar.addStretch()

        self._lbl_config = QLabel("")
        self._lbl_config.setStyleSheet(
            f"color: {C_TEXT_SECONDARY}; font-size: 12px; border: none;"
            "background: #ECEFF1; border-radius: 4px; padding: 2px 8px;"
        )
        top_bar.addWidget(self._lbl_config)
        layout.addLayout(top_bar)

        # ===== Main content: left (1/2) + right (1/2) block layout =====
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(12)

        # ==================== LEFT COLUMN: Distance + Stats (1/2) ====================
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        # -- Distance card (top-left, 1/4 of total) --
        dist_card = QGroupBox()
        dist_card.setMinimumWidth(300)
        dist_card.setStyleSheet(self._CARD_CSS)
        dist_inner = QVBoxLayout(dist_card)
        dist_inner.setContentsMargins(12, 16, 12, 8)
        dist_inner.setSpacing(4)

        self._lbl_distance = QLabel("---- cm")
        self._lbl_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_distance.setStyleSheet(
            f"font-size: 52px; font-weight: 500; color: {C_TEXT_PRIMARY};"
            "border: none; background: transparent;"
            "letter-spacing: -0.02em;"
        )
        self._lbl_distance.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        dist_inner.addWidget(self._lbl_distance)

        # Bottom row: grade + deviation + unit switch
        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(8)

        self._lbl_grade = QLabel("")
        self._lbl_grade.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_grade.setStyleSheet(
            "font-size: 13px; font-weight: 600; padding: 4px 10px;"
            "border-radius: 8px; border: none; letter-spacing: -0.01em;"
        )
        self._lbl_grade.hide()
        info_row.addWidget(self._lbl_grade)

        self._lbl_dev_text = QLabel("")
        self._lbl_dev_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_dev_text.setStyleSheet(
            f"font-size: 13px; color: {C_TEXT_SECONDARY}; border: none; font-weight: 500;"
        )
        info_row.addWidget(self._lbl_dev_text)
        info_row.addStretch()

        # Unit switch (cm | mm) bottom-right
        seg_layout = QHBoxLayout()
        seg_layout.setContentsMargins(0, 0, 0, 0)
        seg_layout.setSpacing(0)

        self._btn_cm = QPushButton("cm")
        self._btn_cm.setCheckable(True)
        self._btn_cm.setChecked(True)
        self._btn_cm.setFixedSize(50, 28)
        self._btn_cm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_cm.clicked.connect(lambda: self._on_toggle_unit(True))
        seg_layout.addWidget(self._btn_cm)

        self._btn_mm = QPushButton("mm")
        self._btn_mm.setCheckable(True)
        self._btn_mm.setChecked(False)
        self._btn_mm.setFixedSize(50, 28)
        self._btn_mm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_mm.clicked.connect(lambda: self._on_toggle_unit(False))
        seg_layout.addWidget(self._btn_mm)

        _BSTYLE_CM_ACTIVE = (
            f"background:{C_SUCCESS}; color:white; border:none;"
            "border-top-left-radius:8px; border-bottom-left-radius:8px;"
            "border-top-right-radius:0px; border-bottom-right-radius:0px;"
            "font-size:13px; font-weight:600; padding:0px;"
        )
        _BSTYLE_CM_INACTIVE = (
            "background:#E5E5EA; color:#8E8E93; border:none;"
            "border-top-left-radius:8px; border-bottom-left-radius:8px;"
            "border-top-right-radius:0px; border-bottom-right-radius:0px;"
            "font-size:13px; font-weight:500; padding:0px;"
        )
        _BSTYLE_MM_ACTIVE = (
            f"background:{C_SUCCESS}; color:white; border:none;"
            "border-top-left-radius:0px; border-bottom-left-radius:0px;"
            "border-top-right-radius:8px; border-bottom-right-radius:8px;"
            "font-size:13px; font-weight:600; padding:0px;"
        )
        _BSTYLE_MM_INACTIVE = (
            "background:#E5E5EA; color:#8E8E93; border:none;"
            "border-top-left-radius:0px; border-bottom-left-radius:0px;"
            "border-top-right-radius:8px; border-bottom-right-radius:8px;"
            "font-size:13px; font-weight:500; padding:0px;"
        )
        self._btn_cm.setStyleSheet(f"QPushButton {{{_BSTYLE_CM_ACTIVE}}}")
        self._btn_mm.setStyleSheet(f"QPushButton {{{_BSTYLE_MM_INACTIVE}}}")

        info_row.addLayout(seg_layout)
        dist_inner.addLayout(info_row)

        left_col.addWidget(dist_card, stretch=1)

        # -- Stats card (bottom-left, 1/4 of total) --
        stats_card = QGroupBox("实时统计")
        stats_card.setMinimumWidth(300)
        stats_card.setStyleSheet(self._CARD_CSS)
        stats_grid = QGridLayout(stats_card)
        stats_grid.setSpacing(4)
        stats_grid.setVerticalSpacing(2)

        self._lbl_mean = QLabel("—")
        self._lbl_std = QLabel("—")
        self._lbl_pct = QLabel("—")
        self._lbl_min = QLabel("—")
        self._lbl_max = QLabel("—")
        self._lbl_range = QLabel("—")
        self._lbl_count = QLabel("0")
        self._lbl_dev = QLabel("—")

        self._make_stat_card(stats_grid, 0, 0, "均值", self._lbl_mean, "cm")
        self._make_stat_card(stats_grid, 0, 1, "σ 稳定度", self._lbl_std, "cm")
        self._make_stat_card(stats_grid, 1, 0, "±0.2cm 命中率", self._lbl_pct, "%")
        self._make_stat_card(stats_grid, 1, 1, "极差", self._lbl_range, "cm")
        self._make_stat_card(stats_grid, 2, 0, "最小", self._lbl_min, "cm")
        self._make_stat_card(stats_grid, 2, 1, "最大", self._lbl_max, "cm")
        self._make_stat_card(stats_grid, 3, 0, f"采样数 ({WINDOW_SIZE})", self._lbl_count, "")
        self._make_stat_card(stats_grid, 3, 1, "当前偏差", self._lbl_dev, "cm")

        left_col.addWidget(stats_card, stretch=1)

        main_layout.addLayout(left_col, stretch=1)

        # ==================== RIGHT COLUMN: Records (1/2) ====================
        records_card = QGroupBox()
        records_card.setStyleSheet(self._CARD_CSS)
        records_inner = QVBoxLayout(records_card)
        records_inner.setContentsMargins(0, 0, 0, 0)
        records_inner.setSpacing(6)

        log_label = QLabel("多点精度记录")
        log_label.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {C_TEXT_PRIMARY}; border: none;"
        )
        records_inner.addWidget(log_label)

        # Laser reference input row
        laser_row = QHBoxLayout()
        laser_row.setContentsMargins(0, 0, 0, 0)
        laser_row.setSpacing(6)

        self._lbl_laser = QLabel("激光参考值:")
        self._lbl_laser.setStyleSheet(
            f"font-size: 12px; color: {C_TEXT_SECONDARY}; border: none; font-weight: 500;"
        )
        laser_row.addWidget(self._lbl_laser)

        self._laser_input = QLineEdit()
        self._laser_input.setPlaceholderText("输入激光记录仪测量值")
        self._laser_input.setFixedHeight(28)
        self._laser_input.setStyleSheet(
            "QLineEdit {"
            "  font-size: 13px; color: #1C1C1E;"
            "  border: 1px solid #C6C6C8; border-radius: 6px;"
            "  padding: 2px 8px; background: white;"
            "}"
            "QLineEdit:focus { border-color: #007AFF; }"
        )
        laser_row.addWidget(self._laser_input, stretch=1)

        self._lbl_laser_unit = QLabel("cm")
        self._lbl_laser_unit.setStyleSheet(
            f"font-size: 12px; color: {C_TEXT_SECONDARY}; border: none; font-weight: 500;"
        )
        laser_row.addWidget(self._lbl_laser_unit)

        self._lbl_laser_hint = QLabel("输完后点「记录当前」")
        self._lbl_laser_hint.setStyleSheet(
            f"font-size: 11px; color: {C_TEXT_MUTED}; border: none;"
        )
        laser_row.addWidget(self._lbl_laser_hint)

        records_inner.addLayout(laser_row)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        self._log_view.setStyleSheet(
            "QPlainTextEdit {"
            "  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; font-size: 13px;"
            f"  color: {C_TEXT_PRIMARY}; background: rgba(0,0,0,0.02);"
            f"  border: none; border-radius: 8px;"
            "  padding: 10px;"
            "}"
        )
        self._log_view.setPlaceholderText("对准目标 → 稳定后点「记录」")
        records_inner.addWidget(self._log_view, stretch=1)

        # Button row at bottom
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_snap = QPushButton("记录当前")
        self._btn_snap.setStyleSheet(
            "QPushButton {"
            f"  background: {C_SUCCESS}; color: white; border: none; border-radius: 8px;"
            "  padding: 8px 16px; font-size: 12px; font-weight: 600;"
            "  letter-spacing: -0.01em;"
            "}"
            "QPushButton:hover  { background: #2DB84D; }"
            "QPushButton:disabled { background: rgba(0,0,0,0.15); color: rgba(0,0,0,0.3); }"
        )
        self._btn_snap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_snap.clicked.connect(self._on_snapshot)
        btn_row.addWidget(self._btn_snap)

        self._btn_clear_log = QPushButton("清空记录")
        self._btn_clear_log.setStyleSheet(
            "QPushButton {"
            f"  background: rgba(0,0,0,0.04); color: {C_TEXT_SECONDARY};"
            "  border: none; border-radius: 8px;"
            "  padding: 8px 16px; font-size: 12px; font-weight: 500;"
            "}"
            f"QPushButton:hover {{ color: {C_ERROR}; background: rgba(255,59,48,0.08); }}"
        )
        self._btn_clear_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear_log.clicked.connect(self._on_clear_log)
        btn_row.addWidget(self._btn_clear_log)

        self._btn_export = QPushButton("导出数据")
        self._btn_export.setStyleSheet(
            "QPushButton {"
            f"  background: rgba(0,122,255,0.08); color: {C_PRIMARY};"
            "  border: none; border-radius: 8px;"
            "  padding: 8px 16px; font-size: 12px; font-weight: 600;"
            "}"
            "QPushButton:hover  { background: rgba(0,122,255,0.15); }"
        )
        self._btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_export.clicked.connect(self._on_export)
        btn_row.addWidget(self._btn_export)

        btn_row.addStretch()
        records_inner.addLayout(btn_row)

        main_layout.addWidget(records_card, stretch=1)

        layout.addLayout(main_layout, stretch=1)

        # --- Control bar ---
        ctrl_group = QGroupBox()
        ctrl_group.setStyleSheet(self._CARD_CSS)
        ctrl_layout = QHBoxLayout(ctrl_group)

        self._btn_start = QPushButton("开始测量")
        self._btn_start.setStyleSheet(self._BTN_SUCCESS_CSS)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.setMinimumHeight(42)
        self._btn_start.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("停止")
        self._btn_stop.setStyleSheet(self._BTN_DANGER_CSS)
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.setMinimumHeight(42)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_stop)

        self._btn_reset = QPushButton("清零窗口")
        self._btn_reset.setStyleSheet(self._BTN_OUTLINE_CSS)
        self._btn_reset.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_reset.setMinimumHeight(42)
        self._btn_reset.clicked.connect(self._on_reset)
        ctrl_layout.addWidget(self._btn_reset)

        layout.addWidget(ctrl_group)

    def _make_stat_card(
        self,
        grid: QGridLayout,
        row: int,
        col: int,
        label: str,
        value_label: QLabel,
        unit: str,
    ) -> None:
        wrapper = QWidget()
        wrapper.setMinimumWidth(140)
        wrapper.setStyleSheet(
            "QWidget {"
            "  border: 1px solid rgba(0,0,0,0.06); border-radius: 6px;"
            "  background: transparent;"
            "}"
        )
        inner = QHBoxLayout(wrapper)
        inner.setContentsMargins(8, 4, 8, 4)
        inner.setSpacing(4)

        title = QLabel(f"{label}:")
        title.setStyleSheet(
            f"font-size: 12px; color: {C_TEXT_PRIMARY}; border: none; font-weight: 500;"
        )
        inner.addWidget(title)

        val_text = f"— {unit}" if unit else "—"
        value_label.setText(val_text)
        value_label.setMinimumWidth(60)
        value_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {C_TEXT_PRIMARY}; border: none;"
        )
        inner.addWidget(value_label)
        inner.addStretch()
        grid.addWidget(wrapper, row, col)

    # ==================== unit helpers ====================

    def _unit_info(self) -> tuple:
        """Return (scale, unit_str, std_good, std_ok, acc_threshold, fmt) for current unit."""
        if self._unit_cm:
            return 1.0, "cm", 0.1, 0.2, 0.2, ".1f"
        else:
            return 10, "mm", 0.1, 0.2, 2, ".0f"

    def _on_toggle_unit(self, to_cm: bool) -> None:
        if self._unit_cm == to_cm:
            return

        # Convert laser input value along with unit switch
        old_scale, _, _, _, _, _ = self._unit_info()
        self._unit_cm = to_cm
        new_scale, new_unit, _, _, _, _ = self._unit_info()

        laser_text = self._laser_input.text().strip()
        if laser_text:
            try:
                laser_val = float(laser_text)
                converted = laser_val * (new_scale / old_scale) if old_scale != 0 else laser_val
                self._laser_input.setText(f"{converted:.1f}")
            except ValueError:
                pass

        self._lbl_laser_unit.setText(new_unit)

        self._btn_cm.setChecked(to_cm)
        self._btn_mm.setChecked(not to_cm)

        _ACT_CM = (
            f"background:{C_SUCCESS}; color:white; border:none;"
            "border-top-left-radius:8px; border-bottom-left-radius:8px;"
            "border-top-right-radius:0px; border-bottom-right-radius:0px;"
            "font-size:13px; font-weight:600; padding:0px;"
        )
        _INACT_CM = (
            "background:#E5E5EA; color:#8E8E93; border:none;"
            "border-top-left-radius:8px; border-bottom-left-radius:8px;"
            "border-top-right-radius:0px; border-bottom-right-radius:0px;"
            "font-size:13px; font-weight:500; padding:0px;"
        )
        _ACT_MM = (
            f"background:{C_SUCCESS}; color:white; border:none;"
            "border-top-left-radius:0px; border-bottom-left-radius:0px;"
            "border-top-right-radius:8px; border-bottom-right-radius:8px;"
            "font-size:13px; font-weight:600; padding:0px;"
        )
        _INACT_MM = (
            "background:#E5E5EA; color:#8E8E93; border:none;"
            "border-top-left-radius:0px; border-bottom-left-radius:0px;"
            "border-top-right-radius:8px; border-bottom-right-radius:8px;"
            "font-size:13px; font-weight:500; padding:0px;"
        )
        if to_cm:
            self._btn_cm.setStyleSheet(f"QPushButton {{{_ACT_CM}}}")
            self._btn_mm.setStyleSheet(f"QPushButton {{{_INACT_MM}}}")
        else:
            self._btn_cm.setStyleSheet(f"QPushButton {{{_INACT_CM}}}")
            self._btn_mm.setStyleSheet(f"QPushButton {{{_ACT_MM}}}")

        _, unit, _, _, _, _ = self._unit_info()
        self._lbl_distance.setText(f"---- {unit}")
        self._lbl_mean.setText(f"— {unit}")
        self._lbl_std.setText(f"— {unit}")
        self._lbl_min.setText(f"— {unit}")
        self._lbl_max.setText(f"— {unit}")
        self._lbl_range.setText(f"— {unit}")
        self._lbl_dev.setText(f"— {unit}")
        if self._last_raw_m > 0:
            self._on_new_distance(self._last_raw_m)
        if self._last_stats:
            self._on_stats_update(self._last_stats)

    # ==================== state ====================

    def _update_ui_state(self) -> None:
        if self._connected:
            self._btn_connect.setText("断开连接")
            self._btn_connect.setStyleSheet(
                "QPushButton {"
                f"  background: rgba(255,59,48,0.10); color: {C_ERROR};"
                "  border: none; border-radius: 8px;"
                "  padding: 6px 18px; font-size: 13px; font-weight: 600;"
                "  letter-spacing: -0.01em;"
                "}"
                f"QPushButton:hover {{ background: rgba(255,59,48,0.18); }}"
            )
        else:
            self._btn_connect.setText("连接设备")
            self._btn_connect.setStyleSheet(self._BTN_PRIMARY_CSS)

        self._btn_connect.setEnabled(not self._measuring)
        self._btn_start.setEnabled(self._connected and not self._measuring)
        self._btn_stop.setEnabled(self._measuring)
        self._btn_reset.setEnabled(not self._measuring)
        self._btn_snap.setEnabled(self._measuring and self._last_stats.get("n", 0) >= 10)

    # ==================== slots ====================

    def _on_connect(self) -> None:
        if self._connected:
            self._do_disconnect()
        else:
            self._do_connect()

    def _do_connect(self) -> None:
        try:
            from acconeer.exptool import a121
            from acconeer.exptool.a121.algo.distance import (
                Detector,
                DetectorConfig,
                ThresholdMethod,
            )

            self._lbl_status.setText("正在连接…")
            self._lbl_status.setStyleSheet(
                f"color: {C_WARN}; font-weight: 500; font-size: 14px; border: none;"
            )
            self._btn_connect.setEnabled(False)

            self._client = a121.Client.open(usb_device=True)

            detector_config = DetectorConfig(
                start_m=0.1,
                end_m=2.5,
                max_profile=a121.Profile.PROFILE_1,
                max_step_length=1,
                signal_quality=30.0,
                threshold_method=ThresholdMethod.CFAR,
            )
            self._detector = Detector(
                client=self._client,
                sensor_ids=[1],
                detector_config=detector_config,
            )
            self._detector.calibrate_detector()

            self._connected = True
            self._lbl_status.setText("已连接")
            self._lbl_status.setStyleSheet(
                f"color: {C_SUCCESS}; font-weight: 500; font-size: 14px; border: none;"
            )
            self._lbl_config.setText("Profile 1 | 采样间距 2.5mm | 信号质量 30 | 范围 10-250cm")
        except Exception as e:
            self._lbl_status.setText(f"连接失败: {e}")
            self._lbl_status.setStyleSheet(
                f"color: {C_ERROR}; font-weight: 500; font-size: 14px; border: none;"
            )
            self._client = None
            self._detector = None
        finally:
            self._update_ui_state()

    def _do_disconnect(self) -> None:
        self._on_stop()
        try:
            if self._detector is not None:
                self._detector.stop()
        except Exception:
            pass
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = None
        self._detector = None
        self._connected = False
        self._lbl_status.setText("未连接")
        self._lbl_status.setStyleSheet(
            f"color: {C_TEXT_MUTED}; font-weight: 500; font-size: 14px; border: none;"
        )
        self._lbl_config.setText("")
        self._lbl_grade.hide()
        _, unit, _, _, _, _ = self._unit_info()
        self._lbl_distance.setText(f"---- {unit}")
        self._lbl_dev_text.hide()
        self._update_ui_state()

    def _on_start(self) -> None:
        if not self._connected or self._detector is None:
            return
        try:
            self._detector.start()
        except Exception as e:
            self._lbl_status.setText(f"启动失败: {e}")
            self._lbl_status.setStyleSheet(
                f"color: {C_ERROR}; font-weight: 500; font-size: 14px; border: none;"
            )
            return

        self._measuring = True
        self._lbl_status.setText("测量中 ...")
        self._lbl_status.setStyleSheet(
            f"color: {C_PRIMARY}; font-weight: 500; font-size: 14px; border: none;"
        )
        self._update_ui_state()

        self._worker = _MeasurementWorker(self._detector, self)
        self._worker.sig_distance.connect(self._on_new_distance)
        self._worker.sig_stats.connect(self._on_stats_update)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.sig_finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_stop(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(2000)
            self._worker = None
        if self._detector is not None and self._measuring:
            try:
                self._detector.stop()
            except Exception:
                pass
        self._measuring = False
        if self._connected:
            self._lbl_status.setText("已连接")
            self._lbl_status.setStyleSheet(
                f"color: {C_SUCCESS}; font-weight: 500; font-size: 14px; border: none;"
            )
        self._update_ui_state()

    def _on_reset(self) -> None:
        if self._worker is not None:
            self._worker.clear()
        self._last_mean = 0.0
        self._last_stats = {}
        self._last_raw_m = 0.0
        _, unit, _, _, _, _ = self._unit_info()
        self._lbl_distance.setText(f"---- {unit}")
        self._lbl_dev_text.hide()
        self._lbl_grade.hide()
        self._lbl_mean.setText(f"— {unit}")
        self._lbl_std.setText(f"— {unit}")
        self._lbl_pct.setText("—")
        self._lbl_min.setText(f"— {unit}")
        self._lbl_max.setText(f"— {unit}")
        self._lbl_range.setText(f"— {unit}")
        self._lbl_count.setText("0")
        self._lbl_dev.setText(f"— {unit}")

    def _on_snapshot(self) -> None:
        s = self._last_stats
        if not s or s.get("n", 0) < 10:
            return

        self._snap_count += 1
        n = self._snap_count
        scale, unit, _, _, acc_thresh, fmt = self._unit_info()
        mean = s["mean"] * scale
        std = s["std"] * scale
        rng = (s["max"] - s["min"]) * scale
        pct = s["pct_2mm"]

        raw_text = self._lbl_distance.text().replace(f" {unit}", "").strip()
        try:
            raw_val = float(raw_text)
            raw_str = f"{raw_val:{fmt}}"
        except ValueError:
            raw_val = 0.0
            raw_str = "—"

        # Read laser reference value
        laser_str = self._laser_input.text().strip()
        laser_val: t.Optional[float] = None
        laser_display = "—"
        if laser_str:
            try:
                laser_val = float(laser_str)
                laser_display = f"{laser_val:{fmt}}"
            except ValueError:
                laser_val = None

        # Calculate differences (sensor - laser)
        diff_raw_val: t.Optional[float] = None   # instant - laser
        diff_mean_val: t.Optional[float] = None  # mean - laser
        diff_raw_pct_val: t.Optional[float] = None
        diff_mean_pct_val: t.Optional[float] = None

        if laser_val is not None:
            diff_mean_val = mean - laser_val
            if raw_val is not None and raw_val != 0.0:
                diff_raw_val = raw_val - laser_val
            if abs(laser_val) > 0.001:
                diff_mean_pct_val = diff_mean_val / laser_val * 100.0
                if diff_raw_val is not None:
                    diff_raw_pct_val = diff_raw_val / laser_val * 100.0

        # Build log line
        line = (
            f"#{n:2d}  |  传感器={raw_str} {unit}  |  "
            f"均值={mean:7.2f} {unit}  |  "
            f"标准差σ={std:.2f} {unit}  |  "
            f"极差={rng:.2f} {unit}  |  "
            f"±{acc_thresh:.1f}{unit}={pct:.0f}%"
        )
        if laser_val is not None:
            d_raw = f"{diff_raw_val:{fmt}}" if diff_raw_val is not None else "—"
            d_mean = f"{diff_mean_val:{fmt}}"
            p_raw = f"{diff_raw_pct_val:.2f}%" if diff_raw_pct_val is not None else "—"
            p_mean = f"{diff_mean_pct_val:.2f}%"
            line += (
                f"\n      激光参考={laser_display} {unit}  |  "
                f"瞬时差值={d_raw} {unit}({p_raw})  |  "
                f"均值差值={d_mean} {unit}({p_mean})"
            )
        self._log_view.appendPlainText(line)

        # Store structured record for export
        self._snapshots.append({
            "index": n,
            "sensor_raw": raw_val,
            "sensor_mean": mean,
            "sensor_std": std,
            "sensor_range": rng,
            "hit_rate": pct,
            "laser": laser_val,
            "diff_raw": diff_raw_val,
            "diff_raw_pct": diff_raw_pct_val,
            "diff_mean": diff_mean_val,
            "diff_mean_pct": diff_mean_pct_val,
            "unit": unit,
            "samples": s["n"],
        })

        scrollbar = self._log_view.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _on_clear_log(self) -> None:
        self._log_view.clear()
        self._snapshots.clear()
        self._snap_count = 0

    def _on_export(self) -> None:
        if not self._snapshots:
            text = self._log_view.toPlainText()
            if not text.strip():
                return
        else:
            text = self._log_view.toPlainText()

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出精度记录",
            "precision_log.csv",
            "CSV (*.csv);;文本文件 (*.txt);;所有文件 (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return

        _, unit, _, _, acc_thresh, fmt = self._unit_info()
        is_csv = path.lower().endswith(".csv") or "CSV" in selected_filter

        if is_csv:
            self._export_csv(path, unit, acc_thresh, fmt)
        else:
            self._export_txt(path, unit, acc_thresh, text)

    def _export_csv(self, path: str, unit: str, acc_thresh: float, fmt: str) -> None:
        """Export structured CSV with laser comparison columns."""
        has_laser = any(s["laser"] is not None for s in self._snapshots)

        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            import csv
            writer = csv.writer(f)

            # Metadata header
            writer.writerow(["# A121 精度测试记录 — 传感器 vs 激光记录仪对比"])
            writer.writerow(["# 配置: Profile 1, step=1 (2.5mm), sq=30, 0.1-2.5m"])
            writer.writerow([f"# 单位: {unit}"])
            writer.writerow([f"# 总计 {len(self._snapshots)} 个记录点"])
            writer.writerow([])

            # Column headers: 序号, 瞬时, 均值, 激光, 瞬时差值, 瞬时偏差率, 均值差值, 均值偏差率, σ, 极差, 命中率, 采样数
            header = [
                "序号",
                f"传感器瞬时({unit})",
                f"均值({unit})",
            ]
            if has_laser:
                header.extend([
                    f"激光参考({unit})",
                    f"瞬时差值({unit})",
                    "瞬时偏差率(%)",
                    f"均值差值({unit})",
                    "均值偏差率(%)",
                ])
            header.extend([
                f"标准差σ({unit})",
                f"极差({unit})",
                f"±{acc_thresh:.1f}{unit}命中率(%)",
                f"采样数",
            ])

            # Extra empty columns for user manual fill-in
            header.extend(["备注", "测试点位置"])

            writer.writerow(header)

            # Data rows
            for s in self._snapshots:
                row = [
                    s["index"],
                    f"{s['sensor_raw']:{fmt}}",
                    f"{s['sensor_mean']:.2f}",
                ]
                if has_laser:
                    if s["laser"] is not None:
                        row.extend([
                            f"{s['laser']:{fmt}}",
                            f"{s['diff_raw']:{fmt}}" if s["diff_raw"] is not None else "",
                            f"{s['diff_raw_pct']:.2f}" if s["diff_raw_pct"] is not None else "",
                            f"{s['diff_mean']:{fmt}}" if s["diff_mean"] is not None else "",
                            f"{s['diff_mean_pct']:.2f}" if s["diff_mean_pct"] is not None else "",
                        ])
                    else:
                        row.extend(["", "", "", "", ""])
                row.extend([
                    f"{s['sensor_std']:.2f}",
                    f"{s['sensor_range']:.2f}",
                    f"{s['hit_rate']:.0f}",
                    s["samples"],
                ])
                row.extend(["", ""])
                writer.writerow(row)

    def _export_txt(self, path: str, unit: str, acc_thresh: float, text: str) -> None:
        """Legacy plain-text export."""
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write(f"# A121 精度测试记录\n")
            f.write(f"# 配置: Profile 1, step=1 (2.5mm), sq=30, 0.1-2.5m\n")
            f.write(f"# 单位: {unit}\n")
            f.write(f"#\n")
            f.write(f"#  序号  |  传感器({unit})  |  均值({unit})  |  标准差σ({unit})  |  极差({unit})  |  ±{acc_thresh:.1f}{unit}命中率\n")
            f.write(f"{text}\n")

    def _on_new_distance(self, d_m: float) -> None:
        self._last_raw_m = d_m
        scale, unit, _, _, _, fmt = self._unit_info()
        d_val = d_m * 100.0 * scale
        self._lbl_distance.setText(f"{d_val:{fmt}} {unit}")

        if self._last_mean:
            dev = (d_m * 100.0 - self._last_mean) * scale
            self._lbl_dev_text.setText(f"偏离: {dev:{fmt}} {unit}")
            self._lbl_dev_text.setStyleSheet(
                f"font-size: 14px; color: {C_WARN}; border: none;"
                if abs(dev) > 0.1 * scale
                else f"font-size: 14px; color: {C_SUCCESS}; border: none;"
            )
            self._lbl_dev_text.show()
            self._lbl_dev.setText(f"{dev:{fmt}} {unit}")

    def _on_stats_update(self, stats: dict) -> None:
        self._last_stats = stats
        self._last_mean = stats["mean"]
        scale, unit, good_std, ok_std, _, fmt = self._unit_info()
        self._lbl_count.setText(str(stats["n"]))
        self._lbl_mean.setText(f"{stats['mean'] * scale:{fmt}} {unit}")
        self._lbl_std.setText(f"{stats['std'] * scale:{fmt}} {unit}")
        self._lbl_pct.setText(f"{stats['pct_2mm']:.0f}%")
        self._lbl_min.setText(f"{stats['min'] * scale:{fmt}} {unit}")
        self._lbl_max.setText(f"{stats['max'] * scale:{fmt}} {unit}")
        self._lbl_range.setText(f"{(stats['max'] - stats['min']) * scale:{fmt}} {unit}")

        n = stats["n"]
        std = stats["std"]
        if n < 10:
            self._lbl_grade.hide()
        else:
            if std < good_std:
                label = "优"
                fg = C_SUCCESS
                bg = "#E8F5E9"
            elif std < ok_std:
                label = "良"
                fg = C_WARN
                bg = "#FFF3E0"
            else:
                label = "差"
                fg = C_ERROR
                bg = "#FFEBEE"
            self._lbl_grade.setText(f"精度 {label}  ·  σ = {std * scale:{fmt}} {unit}")
            self._lbl_grade.setStyleSheet(
                f"font-size: 13px; font-weight: 600; padding: 4px 10px;"
                f"border-radius: 8px; border: none;"
                f"color: {fg}; background: {bg};"
            )
            self._lbl_grade.show()

        self._update_ui_state()

    def _on_worker_error(self, msg: str) -> None:
        self._lbl_status.setText(f"错误: {msg}")
        self._lbl_status.setStyleSheet(
            f"color: {C_ERROR}; font-weight: 500; font-size: 14px; border: none;"
        )
        self._on_stop()

    def _on_worker_finished(self) -> None:
        pass
