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

        # --- Distance display ---
        dist_group = QGroupBox()
        dist_group.setStyleSheet(
            "QGroupBox {"
            f"  border: none; border-radius: 12px;"
            f"  background: {C_BG_CARD}; padding: 12px;"
            "}"
        )
        dist_layout = QVBoxLayout(dist_group)
        dist_layout.setContentsMargins(0, 0, 0, 0)
        dist_layout.setSpacing(0)

        self._lbl_distance = QLabel("---- cm")
        self._lbl_distance.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_distance.setStyleSheet(
            f"font-size: 52px; font-weight: 500; color: {C_TEXT_PRIMARY};"
            "border: none; background: transparent;"
            "padding: 16px 16px 0px 16px; letter-spacing: -0.02em;"
        )
        self._lbl_distance.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        dist_layout.addWidget(self._lbl_distance)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(16, 4, 16, 10)

        self._lbl_grade = QLabel("")
        self._lbl_grade.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_grade.setStyleSheet(
            "font-size: 13px; font-weight: 600; padding: 4px 10px;"
            "border-radius: 8px; border: none; letter-spacing: -0.01em;"
        )
        self._lbl_grade.hide()
        info_row.addWidget(self._lbl_grade, stretch=1)

        self._lbl_dev_text = QLabel("")
        self._lbl_dev_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_dev_text.setStyleSheet(
            f"font-size: 13px; color: {C_TEXT_SECONDARY}; border: none; font-weight: 500;"
        )
        info_row.addWidget(self._lbl_dev_text)
        dist_layout.addLayout(info_row)
        layout.addWidget(dist_group, stretch=1)

        # --- Stats cards ---
        stats_group = QGroupBox("实时统计")
        stats_group.setStyleSheet(self._CARD_CSS)
        stats_outer = QVBoxLayout(stats_group)
        stats_outer.setSpacing(4)

        stats_grid = QGridLayout()
        stats_grid.setSpacing(4)

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
        self._make_stat_card(stats_grid, 0, 2, "±0.2cm 命中率", self._lbl_pct, "%")
        self._make_stat_card(stats_grid, 0, 3, "极差 (含偶发跳变)", self._lbl_range, "cm")
        self._make_stat_card(stats_grid, 1, 0, "最小 (最偏近)", self._lbl_min, "cm")
        self._make_stat_card(stats_grid, 1, 1, "最大 (最偏远)", self._lbl_max, "cm")
        self._make_stat_card(stats_grid, 1, 2, f"采样数 ({WINDOW_SIZE})", self._lbl_count, "")
        self._make_stat_card(stats_grid, 1, 3, "当前偏差", self._lbl_dev, "cm")

        for c in range(4):
            stats_grid.setColumnStretch(c, 1)
        stats_outer.addLayout(stats_grid)
        layout.addWidget(stats_group)

        # --- Log area: text left, buttons stacked right ---
        log_card = QGroupBox()
        log_card.setStyleSheet(self._CARD_CSS)
        log_layout = QVBoxLayout(log_card)
        log_layout.setSpacing(4)

        log_label = QLabel("多点精度记录")
        log_label.setStyleSheet(
            f"font-size: 13px; font-weight: 600; color: {C_TEXT_PRIMARY}; border: none;"
        )
        log_layout.addWidget(log_label)

        log_body = QHBoxLayout()
        log_body.setSpacing(8)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(500)
        self._log_view.setStyleSheet(
            "QPlainTextEdit {"
            f"  font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; font-size: 13px;"
            f"  color: {C_TEXT_PRIMARY}; background: rgba(0,0,0,0.02);"
            f"  border: none; border-radius: 8px;"
            "  padding: 10px;"
            "}"
        )
        self._log_view.setPlaceholderText("对准目标 → 稳定后点「记录」")
        log_body.addWidget(self._log_view, stretch=1)

        btn_col = QVBoxLayout()
        btn_col.setSpacing(4)

        self._btn_snap = QPushButton("记录当前")
        self._btn_snap.setStyleSheet(
            "QPushButton {"
            f"  background: {C_SUCCESS}; color: white; border: none; border-radius: 8px;"
            "  padding: 7px 12px; font-size: 12px; font-weight: 600;"
            "  letter-spacing: -0.01em;"
            "}"
            "QPushButton:hover  { background: #2DB84D; }"
            "QPushButton:disabled { background: rgba(0,0,0,0.15); color: rgba(0,0,0,0.3); }"
        )
        self._btn_snap.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_snap.setFixedWidth(100)
        self._btn_snap.clicked.connect(self._on_snapshot)
        btn_col.addWidget(self._btn_snap)

        self._btn_clear_log = QPushButton("清空记录")
        self._btn_clear_log.setStyleSheet(
            "QPushButton {"
            f"  background: rgba(0,0,0,0.04); color: {C_TEXT_SECONDARY};"
            "  border: none; border-radius: 8px;"
            "  padding: 7px 12px; font-size: 12px; font-weight: 500;"
            "}"
            f"QPushButton:hover {{ color: {C_ERROR}; background: rgba(255,59,48,0.08); }}"
        )
        self._btn_clear_log.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_clear_log.setFixedWidth(100)
        self._btn_clear_log.clicked.connect(self._on_clear_log)
        btn_col.addWidget(self._btn_clear_log)

        self._btn_export = QPushButton("导出数据")
        self._btn_export.setStyleSheet(
            "QPushButton {"
            f"  background: rgba(0,122,255,0.08); color: {C_PRIMARY};"
            "  border: none; border-radius: 8px;"
            "  padding: 7px 12px; font-size: 12px; font-weight: 600;"
            "}"
            "QPushButton:hover  { background: rgba(0,122,255,0.15); }"
        )
        self._btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_export.setFixedWidth(100)
        self._btn_export.clicked.connect(self._on_export)
        btn_col.addWidget(self._btn_export)

        btn_col.addStretch()
        log_body.addLayout(btn_col)
        log_layout.addLayout(log_body)
        layout.addWidget(log_card, stretch=4)

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
        frame = QGroupBox()
        frame.setStyleSheet(
            "QGroupBox {"
            f"  border: none; border-radius: 8px;"
            f"  background: rgba(0,0,0,0.03); padding: 2px;"
            "}"
        )
        inner = QHBoxLayout(frame)
        inner.setContentsMargins(8, 5, 8, 5)
        inner.setSpacing(6)

        title = QLabel(f"{label}:")
        title.setStyleSheet(
            f"font-size: 12px; color: {C_TEXT_PRIMARY}; border: none; font-weight: 500;"
        )
        inner.addWidget(title)

        val_text = f"— {unit}" if unit else "—"
        value_label.setText(val_text)
        value_label.setStyleSheet(
            f"font-size: 14px; font-weight: 600; color: {C_TEXT_PRIMARY}; border: none;"
        )
        inner.addWidget(value_label)
        inner.addStretch()
        grid.addWidget(frame, row, col)

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
            self._lbl_config.setText("Profile 1 │ 采样间距 2.5mm │ 信号质量 30 │ 范围 10–250cm")
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
        self._lbl_distance.setText("---- cm")
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
        self._lbl_status.setText("测量中 …")
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
        self._lbl_distance.setText("---- cm")
        self._lbl_dev_text.hide()
        self._lbl_grade.hide()
        self._lbl_mean.setText("— cm")
        self._lbl_std.setText("— cm")
        self._lbl_pct.setText("—")
        self._lbl_min.setText("— cm")
        self._lbl_max.setText("— cm")
        self._lbl_range.setText("— cm")
        self._lbl_count.setText("0")
        self._lbl_dev.setText("— cm")

    def _on_snapshot(self) -> None:
        s = self._last_stats
        if not s or s.get("n", 0) < 10:
            return

        self._snap_count += 1
        n = self._snap_count
        mean = s["mean"]
        std = s["std"]
        rng = s["max"] - s["min"]
        pct = s["pct_2mm"]

        raw_text = self._lbl_distance.text().replace(" cm", "").strip()
        try:
            raw_val = float(raw_text)
            raw_str = f"{raw_val:.1f}"
        except ValueError:
            raw_str = "—"

        line = (
            f"#{n:2d}  |  当前={raw_str} cm  |  "
            f"均值={mean:7.2f} cm  |  "
            f"σ={std:.2f} cm  |  "
            f"极差={rng:.2f} cm  |  "
            f"±0.2cm={pct:.0f}%"
        )
        self._log_view.appendPlainText(line)

        scrollbar = self._log_view.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _on_clear_log(self) -> None:
        self._log_view.clear()
        self._snap_count = 0

    def _on_export(self) -> None:
        text = self._log_view.toPlainText()
        if not text.strip():
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出精度记录", "precision_log.txt",
            "文本文件 (*.txt);;CSV (*.csv);;所有文件 (*)"
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# A121 精度测试记录\n")
            f.write(f"# 配置: Profile 1, step=1 (2.5mm), sq=30, 0.1-2.5m\n")
            f.write(f"#\n")
            f.write(f"#  序号  |  当前(cm)  |  均值(cm)  |  σ(cm)  |  极差(cm)  |  ±0.2cm命中率\n")
            f.write(f"{text}\n")

    def _on_new_distance(self, d_m: float) -> None:
        d_cm = d_m * 100.0
        self._lbl_distance.setText(f"{d_cm:.1f} cm")

        if self._last_mean:
            dev = d_cm - self._last_mean
            self._lbl_dev_text.setText(f"偏离: {dev:+.2f} cm")
            self._lbl_dev_text.setStyleSheet(
                f"font-size: 14px; color: {C_WARN}; border: none;"
                if abs(dev) > 0.1
                else f"font-size: 14px; color: {C_SUCCESS}; border: none;"
            )
            self._lbl_dev_text.show()
            self._lbl_dev.setText(f"{dev:+.2f} cm")

    def _on_stats_update(self, stats: dict) -> None:
        self._last_stats = stats
        self._last_mean = stats["mean"]
        self._lbl_count.setText(str(stats["n"]))
        self._lbl_mean.setText(f"{stats['mean']:.2f} cm")
        self._lbl_std.setText(f"{stats['std']:.2f} cm")
        self._lbl_pct.setText(f"{stats['pct_2mm']:.0f}%")
        self._lbl_min.setText(f"{stats['min']:.2f} cm")
        self._lbl_max.setText(f"{stats['max']:.2f} cm")
        self._lbl_range.setText(f"{stats['max'] - stats['min']:.2f} cm")

        n = stats["n"]
        std = stats["std"]
        if n < 10:
            self._lbl_grade.hide()
        else:
            if std < 0.1:
                label = "优"
                fg = C_SUCCESS
                bg = "#E8F5E9"
            elif std < 0.2:
                label = "良"
                fg = C_WARN
                bg = "#FFF3E0"
            else:
                label = "差"
                fg = C_ERROR
                bg = "#FFEBEE"
            self._lbl_grade.setText(f"精度 {label}  ·  σ = {std:.2f} cm")
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
