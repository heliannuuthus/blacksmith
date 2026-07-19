from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import cast

from PIL import Image, ImageOps
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QIcon,
    QImage,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QShortcut,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from raster_recast.ai_backend import (
    MODEL_PROFILE,
    FluxTextBackend,
    ai_environment_status,
)
from raster_recast.core import Box, save_image
from raster_recast.worker import WorkerUnavailable, runtime_socket_path, worker_launch

IMAGE_FILTER = "图片文件 (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;所有文件 (*)"

APP_STYLE = """
* {
    font-family: "SF Pro Text", "Inter", "Segoe UI", "Noto Sans CJK SC", sans-serif;
    font-size: 14px;
    color: #1d1d1f;
}
QMainWindow, QWidget#appRoot {
    background: #f5f5f7;
}
QFrame#topBar {
    background: rgba(255, 255, 255, 0.92);
    border-bottom: 1px solid #dedee3;
}
QLabel#brandTitle {
    font-family: "SF Pro Display", "Inter", "Segoe UI", sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #111113;
}
QLabel#brandSubtitle, QLabel[role="muted"] {
    color: #77777d;
    font-size: 12px;
}
QLabel[role="sectionTitle"] {
    color: #111113;
    font-size: 15px;
    font-weight: 650;
}
QLabel[role="eyebrow"] {
    color: #85858b;
    font-size: 11px;
    font-weight: 650;
}
QLabel#selectionBadge {
    color: #626269;
    background: #f2f2f7;
    border-radius: 8px;
    padding: 7px 9px;
}
QFrame#aiModelPanel {
    background: #eef5ff;
    border: 1px solid #d6e8ff;
    border-radius: 12px;
}
QLabel#aiModelName {
    color: #0a3f7a;
    font-size: 14px;
    font-weight: 700;
}
QLabel#aiModelBadge {
    color: #0067d8;
    background: #ffffff;
    border: 1px solid #c8e0ff;
    border-radius: 7px;
    padding: 4px 7px;
    font-size: 10px;
    font-weight: 700;
}
QLabel#statusBadge[status="ready"] {
    color: #176b3a;
    background: #e8f7ee;
    border-radius: 8px;
    padding: 6px 10px;
}
QLabel#statusBadge[status="idle"] {
    color: #66666d;
    background: #ededf2;
    border-radius: 8px;
    padding: 6px 10px;
}
QFrame#card {
    background: #ffffff;
    border: 1px solid #e1e1e6;
    border-radius: 16px;
}
QFrame#canvasCard {
    background: #ffffff;
    border: 1px solid #dedee3;
    border-radius: 18px;
}
QLineEdit, QSpinBox {
    background: #f2f2f7;
    border: 1px solid transparent;
    border-radius: 10px;
    min-height: 38px;
    padding: 0 11px;
    selection-background-color: #007aff;
}
QLineEdit:hover, QSpinBox:hover {
    background: #ededf2;
}
QLineEdit:focus, QSpinBox:focus {
    background: #ffffff;
    border: 1px solid #007aff;
}
QPushButton {
    min-height: 38px;
    padding: 0 14px;
    background: #f2f2f7;
    border: 1px solid transparent;
    border-radius: 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #e8e8ed;
}
QPushButton:pressed {
    background: #dedee4;
}
QPushButton[role="primary"] {
    color: #ffffff;
    background: #007aff;
}
QPushButton[role="primary"]:hover {
    background: #1687ff;
}
QPushButton[role="primary"]:pressed {
    background: #0067d8;
}
QPushButton[role="danger"] {
    color: #ffffff;
    background: #ff3b30;
}
QPushButton[role="danger"]:hover {
    background: #ff5148;
}
QPushButton[role="danger"]:pressed {
    background: #d92d24;
}
QPushButton[role="quiet"] {
    color: #007aff;
    background: transparent;
    padding: 0 8px;
}
QPushButton[role="quiet"]:hover {
    background: #eef5ff;
}
QToolButton {
    background: transparent;
    border: none;
    border-radius: 8px;
    min-width: 32px;
    min-height: 32px;
    font-weight: 600;
}
QToolButton:hover {
    background: #f0f0f4;
}
QGraphicsView {
    background: #1c1c1e;
    border: none;
    border-radius: 12px;
}
QScrollArea {
    background: transparent;
    border: none;
}
QScrollArea > QWidget > QWidget {
    background: transparent;
}
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 3px 0;
}
QScrollBar::handle:vertical {
    background: #c7c7cc;
    border-radius: 4px;
    min-height: 28px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QMessageBox {
    background: #f5f5f7;
}
QProgressBar {
    min-height: 4px;
    max-height: 4px;
    border: none;
    border-radius: 2px;
    background: #e5e5ea;
}
QProgressBar::chunk {
    border-radius: 2px;
    background: #007aff;
}
"""


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    rgba = image.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimage)


def create_app_icon(size: int = 64) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#007aff"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(QRectF(2, 2, size - 4, size - 4), 15, 15)
    font = QFont("SF Pro Display", int(size * 0.43), QFont.Weight.Bold)
    painter.setFont(font)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(QRectF(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, "R")
    painter.end()
    return QIcon(pixmap)


class ImageCanvas(QGraphicsView):
    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
            | QPainter.RenderHint.TextAntialiasing
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Alt + 滚轮缩放 · 空格 + 拖动平移 · 左键框选")

        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.selection_item: QGraphicsRectItem | None = None
        self.drag_start: QPointF | None = None
        self.image_rect = QRectF()
        self.fit_mode = True
        self.selection_enabled = True
        self._space_panning = False
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        scene = self.scene()
        scene.clear()
        label = scene.addText("选择或拖入图片后，在这里框选文字区域")
        label.setDefaultTextColor(QColor("#9a9aa0"))
        label.setFont(QFont("SF Pro Text", 14))
        bounds = label.boundingRect()
        label.setPos(-bounds.width() / 2, -bounds.height() / 2)
        scene.setSceneRect(-360, -220, 720, 440)
        self.pixmap_item = None
        self.selection_item = None
        self.image_rect = QRectF()

    def set_image(self, image: Image.Image) -> None:
        scene = self.scene()
        scene.clear()
        pixmap = pil_to_pixmap(image)
        self.pixmap_item = scene.addPixmap(pixmap)
        self.image_rect = QRectF(0, 0, pixmap.width(), pixmap.height())
        scene.setSceneRect(self.image_rect)
        self.selection_item = None
        self.fit_to_view()

    def clear_selection(self) -> None:
        if self.selection_item is not None:
            self.scene().removeItem(self.selection_item)
            self.selection_item = None
        self.drag_start = None
        self.selection_changed.emit(None)

    def fit_to_view(self) -> None:
        if self.pixmap_item is None:
            return
        self.fit_mode = True
        self.resetTransform()
        self.fitInView(self.image_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def actual_size(self) -> None:
        if self.pixmap_item is None:
            return
        self.fit_mode = False
        self.resetTransform()
        self.centerOn(self.image_rect.center())

    def zoom(self, factor: float, anchor_position: QPointF | None = None) -> None:
        if self.pixmap_item is None:
            return
        scene_anchor = (
            self.mapToScene(anchor_position.toPoint()) if anchor_position is not None else None
        )
        current = self.transform().m11()
        target = max(0.05, min(16.0, current * factor))
        if abs(target - current) < 1e-9:
            return
        self.fit_mode = False
        applied_factor = target / current
        self.scale(applied_factor, applied_factor)
        if anchor_position is not None and scene_anchor is not None:
            moved_anchor = self.mapToScene(anchor_position.toPoint())
            scene_center = self.mapToScene(self.viewport().rect().center())
            self.centerOn(scene_center + scene_anchor - moved_anchor)

    def wheelEvent(self, event: QWheelEvent) -> None:
        zoom_modifiers = Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier
        if event.modifiers() & zoom_modifiers:
            delta = event.pixelDelta().y() or event.angleDelta().y()
            if delta == 0:
                return super().wheelEvent(event)
            # One traditional wheel notch is 120 units. Fractional deltas keep
            # touchpads smooth instead of snapping in fixed 15% jumps.
            self.zoom(1.2 ** (delta / 120), event.position())
            event.accept()
            return
        super().wheelEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.key() == Qt.Key.Key_Space
            and not event.isAutoRepeat()
            and self.pixmap_item is not None
        ):
            self._space_panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.fit_mode and self.pixmap_item is not None:
            self.fitInView(self.image_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.pixmap_item is None:
            return super().mousePressEvent(event)
        if self._space_panning and event.button() == Qt.MouseButton.LeftButton:
            self.fit_mode = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if not self.selection_enabled:
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        if not self.image_rect.contains(point):
            return
        self.clear_selection()
        self.drag_start = point
        pen = QPen(QColor("#30d158"), 2)
        pen.setCosmetic(True)
        pen.setDashPattern([6.0, 4.0])
        self.selection_item = self.scene().addRect(QRectF(point, point), pen)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag_start is None or self.selection_item is None:
            return super().mouseMoveEvent(event)
        point = self.mapToScene(event.position().toPoint())
        point.setX(min(self.image_rect.right(), max(self.image_rect.left(), point.x())))
        point.setY(min(self.image_rect.bottom(), max(self.image_rect.top(), point.y())))
        self.selection_item.setRect(QRectF(self.drag_start, point).normalized())
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._space_panning and event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return super().mouseReleaseEvent(event)
        if self.drag_start is None or self.selection_item is None:
            return super().mouseReleaseEvent(event)
        rect = self.selection_item.rect().intersected(self.image_rect)
        self.drag_start = None
        if rect.width() < 2 or rect.height() < 2:
            self.clear_selection()
            return
        box: Box = (
            round(rect.left()),
            round(rect.top()),
            max(1, round(rect.width())),
            max(1, round(rect.height())),
        )
        self.selection_changed.emit(box)
        event.accept()


class RasterRecastWindow(QMainWindow):
    ai_completed = Signal(object)
    ai_failed = Signal(object)
    ai_progress = Signal(object)

    def __init__(self, initial_image: Path | None = None) -> None:
        super().__init__()
        self.image: Image.Image | None = None
        self.selected_box: Box | None = None
        self.history: list[Image.Image] = []
        self.edit_count = 0
        self.ai_socket_path = runtime_socket_path()
        self.ai_backend = FluxTextBackend(self.ai_socket_path)
        self.ai_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="raster-recast-ai")
        self.ai_worker_process: subprocess.Popen[bytes] | None = None
        self.ai_worker_ready = False
        self.ai_worker_starting = False
        self.ai_worker_deadline = 0.0
        self.ai_worker_stopping: subprocess.Popen[bytes] | None = None
        self.ai_worker_stop_deadline = 0.0
        self.closing = False
        self.ai_locked_controls: list[QWidget] = []
        self.ai_job_generation = 0
        self.ai_pending_previous: tuple[int, Image.Image] | None = None

        self.setWindowTitle("RasterRecast · 图片文字替换")
        self.setWindowIcon(create_app_icon())
        self.resize(1440, 900)
        self.setMinimumSize(1060, 700)
        self.setAcceptDrops(True)
        self._build_ui()
        self._install_shortcuts()
        self.ai_completed.connect(self._ai_completed)
        self.ai_failed.connect(self._ai_failed)
        self.ai_progress.connect(self._ai_progress)
        QTimer.singleShot(0, self._ensure_ai_worker)

        if initial_image is not None:
            self.load_image(initial_image)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_top_bar())

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(20, 18, 20, 20)
        content_layout.setSpacing(18)
        content_layout.addWidget(self._build_sidebar())
        content_layout.addWidget(self._build_canvas_card(), 1)
        root_layout.addWidget(content, 1)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(72)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(22, 0, 22, 0)
        layout.setSpacing(12)

        icon = QLabel("R")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setFixedSize(38, 38)
        icon.setStyleSheet(
            "background:#007aff;color:white;border-radius:10px;font-size:20px;font-weight:700;"
        )
        layout.addWidget(icon)

        brand = QVBoxLayout()
        brand.setSpacing(0)
        title = QLabel("RasterRecast")
        title.setObjectName("brandTitle")
        subtitle = QLabel("本机图片文字精修")
        subtitle.setObjectName("brandSubtitle")
        brand.addWidget(title)
        brand.addWidget(subtitle)
        layout.addLayout(brand)
        layout.addStretch()

        open_button = QPushButton("打开图片")
        open_button.clicked.connect(self.choose_image)
        layout.addWidget(open_button)

        save_button = QPushButton("保存结果")
        save_button.setProperty("role", "primary")
        save_button.clicked.connect(self.save_result)
        layout.addWidget(save_button)
        self.ai_locked_controls.extend((open_button, save_button))
        return bar

    def _build_sidebar(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(350)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(12)
        layout.addWidget(self._build_source_card())
        layout.addWidget(self._build_edit_card())
        layout.addWidget(self._build_output_card())
        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def _card(self, title: str, eyebrow: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 15, 16, 16)
        layout.setSpacing(9)
        eyebrow_label = QLabel(eyebrow.upper())
        eyebrow_label.setProperty("role", "eyebrow")
        title_label = QLabel(title)
        title_label.setProperty("role", "sectionTitle")
        layout.addWidget(eyebrow_label)
        layout.addWidget(title_label)
        return card, layout

    def _build_source_card(self) -> QWidget:
        card, layout = self._card("选择原始图片", "步骤 1")
        self.source_path_label = QLabel("尚未选择文件")
        self.source_path_label.setProperty("role", "muted")
        self.source_path_label.setWordWrap(True)
        layout.addWidget(self.source_path_label)
        button = QPushButton("从本机选择图片…")
        button.clicked.connect(self.choose_image)
        layout.addWidget(button)
        self.ai_locked_controls.append(button)
        hint = QLabel("也可以直接把图片拖到窗口中")
        hint.setProperty("role", "muted")
        layout.addWidget(hint)
        return card

    def _build_edit_card(self) -> QWidget:
        card, layout = self._card("大模型智能换字", "步骤 2")

        model_panel = QFrame()
        model_panel.setObjectName("aiModelPanel")
        model_layout = QVBoxLayout(model_panel)
        model_layout.setContentsMargins(12, 11, 12, 11)
        model_layout.setSpacing(5)
        model_header = QHBoxLayout()
        model_name = QLabel(MODEL_PROFILE.label)
        model_name.setObjectName("aiModelName")
        model_badge = QLabel("固定引擎")
        model_badge.setObjectName("aiModelBadge")
        model_header.addWidget(model_name, 1)
        model_header.addWidget(model_badge)
        model_layout.addLayout(model_header)
        model_description = QLabel(MODEL_PROFILE.description)
        model_description.setProperty("role", "muted")
        model_description.setWordWrap(True)
        model_layout.addWidget(model_description)
        layout.addWidget(model_panel)

        hint = QLabel("在右侧框住原文字，模型会保留字体、颜色、透视和背景质感。")
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.selection_badge = QLabel("尚未框选区域")
        self.selection_badge.setObjectName("selectionBadge")
        layout.addWidget(self.selection_badge)

        layout.addWidget(self._field_label("新文字"))
        self.replacement_input = QLineEdit()
        self.replacement_input.setPlaceholderText("输入要替换成的汉字或数字")
        self.replacement_input.returnPressed.connect(self.apply_replacement)
        layout.addWidget(self.replacement_input)

        context_row = QHBoxLayout()
        context_row.setSpacing(8)
        context_label = self._field_label("参考上下文")
        self.context_spin = self._spinbox(32, 256, 96)
        self.context_spin.setSuffix(" px")
        self.context_spin.setFixedWidth(106)
        context_row.addWidget(context_label)
        context_row.addStretch()
        context_row.addWidget(self.context_spin)
        layout.addLayout(context_row)
        context_hint = QLabel("扩大模型观察范围；复杂纹理可适当提高。")
        context_hint.setProperty("role", "muted")
        layout.addWidget(context_hint)

        self.ai_progress_bar = QProgressBar()
        self.ai_progress_bar.setRange(0, 0)
        self.ai_progress_bar.setTextVisible(False)
        self.ai_progress_bar.hide()
        layout.addWidget(self.ai_progress_bar)

        actions = QHBoxLayout()
        self.apply_button = QPushButton("开始 AI 替换")
        self.apply_button.setProperty("role", "primary")
        self.apply_button.clicked.connect(self.apply_replacement)
        undo_button = QPushButton("撤销")
        undo_button.clicked.connect(self.undo)
        actions.addWidget(self.apply_button, 1)
        actions.addWidget(undo_button)
        layout.addLayout(actions)
        self.ai_locked_controls.append(undo_button)
        return card

    def _build_output_card(self) -> QWidget:
        card, layout = self._card("导出处理结果", "步骤 3")
        self.output_dir_label = QLabel("默认保存到原图片目录")
        self.output_dir_label.setProperty("role", "muted")
        self.output_dir_label.setWordWrap(True)
        layout.addWidget(self.output_dir_label)
        choose_button = QPushButton("选择输出目录…")
        choose_button.clicked.connect(self.choose_output_directory)
        layout.addWidget(choose_button)
        layout.addWidget(self._field_label("文件名"))
        self.output_name_input = QLineEdit("output.png")
        layout.addWidget(self.output_name_input)
        save_button = QPushButton("保存图片")
        save_button.setProperty("role", "primary")
        save_button.clicked.connect(self.save_result)
        layout.addWidget(save_button)
        self.ai_locked_controls.extend((choose_button, self.output_name_input, save_button))
        return card

    def _build_canvas_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("canvasCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(9)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        image_title = QLabel("编辑预览")
        image_title.setProperty("role", "sectionTitle")
        toolbar.addWidget(image_title)
        toolbar.addStretch()
        toolbar.addWidget(
            self._tool_button("−", lambda: self.canvas.zoom(1 / 1.2), "缩小（Ctrl + -）")
        )
        toolbar.addWidget(self._tool_button("+", lambda: self.canvas.zoom(1.2), "放大（Ctrl + +）"))
        toolbar.addWidget(self._tool_button("适合", self._fit_canvas, "适合窗口（Ctrl + 0）"))
        toolbar.addWidget(self._tool_button("100%", self._actual_canvas, "原始大小（Ctrl + 1）"))
        layout.addLayout(toolbar)

        self.canvas = ImageCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.selection_changed.connect(self._selection_changed)
        layout.addWidget(self.canvas, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("请选择图片开始")
        self.status_label.setProperty("role", "muted")
        status_row.addWidget(self.status_label, 1)
        self.status_badge = QLabel("本机处理")
        self.status_badge.setObjectName("statusBadge")
        self.status_badge.setProperty("status", "idle")
        status_row.addWidget(self.status_badge)
        layout.addLayout(status_row)
        return card

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("role", "eyebrow")
        return label

    def _spinbox(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        return spinbox

    def _tool_button(self, text: str, callback: Callable[[], object], tooltip: str) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tooltip)
        button.clicked.connect(callback)
        return button

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence.StandardKey.Open, self, self.choose_image)
        QShortcut(QKeySequence.StandardKey.Save, self, self.save_result)
        QShortcut(QKeySequence.StandardKey.Undo, self, self.undo)
        QShortcut(QKeySequence.StandardKey.ZoomIn, self, lambda: self.canvas.zoom(1.2))
        QShortcut(QKeySequence.StandardKey.ZoomOut, self, lambda: self.canvas.zoom(1 / 1.2))
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self.canvas.zoom(1.2))
        QShortcut(QKeySequence("Ctrl+0"), self, self._fit_canvas)
        QShortcut(QKeySequence("Ctrl+1"), self, self._actual_canvas)

    def _fit_canvas(self) -> None:
        self.canvas.fit_to_view()

    def _actual_canvas(self) -> None:
        self.canvas.actual_size()

    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", IMAGE_FILTER)
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        try:
            with Image.open(path) as source:
                transposed = ImageOps.exif_transpose(source)
                mode = "RGBA" if "A" in transposed.getbands() else "RGB"
                image = transposed.convert(mode)
        except (OSError, ValueError) as exc:
            self._show_error("无法打开图片", str(exc))
            return

        self.image = image
        self.ai_job_generation += 1
        self.selected_box = None
        self.history.clear()
        self.edit_count = 0
        self.source_path = path
        self.output_dir = path.parent
        self.source_path_label.setText(str(path))
        self.output_dir_label.setText(str(path.parent))
        self.output_name_input.setText(f"{path.stem}-replaced.png")
        self.selection_badge.setText("尚未框选区域")
        self.canvas.set_image(image)
        self._set_status(f"{path.name} · {image.width} × {image.height}", ready=True)

    def choose_output_directory(self) -> None:
        initial = str(getattr(self, "output_dir", Path.cwd()))
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", initial)
        if path:
            self.output_dir = Path(path)
            self.output_dir_label.setText(path)

    def _selection_changed(self, box: object) -> None:
        self.selected_box = box if isinstance(box, tuple) and len(box) == 4 else None  # type: ignore[assignment]
        if self.selected_box is None:
            self.selection_badge.setText("尚未框选区域")
            return
        x, y, width, height = self.selected_box
        self.selection_badge.setText(f"x {x} · y {y} · {width} × {height} px")
        self.replacement_input.setFocus()

    def apply_replacement(self) -> None:
        if self.ai_pending_previous is not None:
            self.cancel_ai_replacement()
            return
        if not self.ai_worker_ready:
            self._ensure_ai_worker()
            return
        if self.image is None:
            self._show_error("缺少图片", "请先选择一张图片。")
            return
        if self.selected_box is None:
            self._show_error("缺少区域", "请先在右侧图片上框选需要替换的文字区域。")
            return
        if not self.replacement_input.text().strip():
            self._show_error("缺少新文字", "FLUX-Text 精确替换需要填写新的汉字或数字。")
            return

        self._start_ai_replacement()

    def _ensure_ai_worker(self) -> None:
        if self.closing:
            return
        available, _ = ai_environment_status(self.ai_socket_path)
        if available:
            self._ai_worker_started()
            return
        if self.ai_worker_starting:
            return

        try:
            launch = worker_launch(self.ai_socket_path)
            environment = os.environ.copy()
            environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            current_python_path = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = str(launch.python_path)
            if current_python_path:
                environment["PYTHONPATH"] += os.pathsep + current_python_path
            self.ai_worker_process = subprocess.Popen(
                launch.command,
                cwd=launch.working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except (OSError, WorkerUnavailable) as exc:
            self._ai_worker_failed(str(exc))
            return

        self.ai_worker_starting = True
        self.ai_worker_deadline = time.monotonic() + 90
        self._set_ai_busy("正在启动内置 AI 引擎…", cancellable=False)
        QTimer.singleShot(500, self._poll_ai_worker)

    def _poll_ai_worker(self) -> None:
        if self.closing:
            return
        available, reason = ai_environment_status(self.ai_socket_path)
        if available:
            self._ai_worker_started()
            return
        if self.ai_worker_process is not None and self.ai_worker_process.poll() is not None:
            return_code = self.ai_worker_process.returncode
            self._ai_worker_failed(f"内置 AI 引擎提前退出（代码 {return_code}）")
            return
        if time.monotonic() >= self.ai_worker_deadline:
            self._ai_worker_failed(f"内置 AI 引擎启动超时：{reason}")
            return
        QTimer.singleShot(500, self._poll_ai_worker)

    def _ai_worker_started(self) -> None:
        self.ai_worker_ready = True
        self.ai_worker_starting = False
        if self.ai_pending_previous is None:
            self._set_apply_button("开始 AI 替换", role="primary", enabled=True)
            self.ai_progress_bar.hide()
        if self.image is None:
            self._set_status("AI 引擎已就绪，请选择图片", ready=True)
        else:
            self._set_status("AI 引擎已就绪，可以开始替换", ready=True)

    def _ai_worker_failed(self, message: str) -> None:
        self.ai_worker_ready = False
        self.ai_worker_starting = False
        self.ai_progress_bar.hide()
        self._set_apply_button("重试 AI 引擎", role="primary", enabled=True)
        self._set_status(f"AI 引擎启动失败：{message}", ready=False)

    def _start_ai_replacement(self) -> None:
        assert self.image is not None
        assert self.selected_box is not None
        available, reason = ai_environment_status(self.ai_socket_path)
        if not available:
            self.ai_worker_ready = False
            self._ensure_ai_worker()
            self._set_ai_busy(f"正在等待内置 AI 引擎：{reason}", cancellable=False)
            return
        self.ai_worker_ready = True

        previous = self.image.copy()
        selection = self.selected_box
        replacement = self.replacement_input.text()
        context = self.context_spin.value()
        self.ai_job_generation += 1
        job_id = self.ai_job_generation
        self.ai_pending_previous = (job_id, previous)
        self._set_edit_controls_enabled(False)
        self._set_ai_busy("正在准备本地 AI 推理…")

        future = self.ai_executor.submit(
            self.ai_backend.edit_region,
            previous,
            selection,
            replacement,
            context=context,
            progress=lambda message: self.ai_progress.emit((job_id, message)),
        )
        future.add_done_callback(lambda completed: self._emit_ai_future(job_id, completed))

    def _emit_ai_future(self, job_id: int, future: Future[Image.Image]) -> None:
        try:
            result = future.result()
        except Exception as exc:  # Qt displays this user-facing error on the main thread.
            self.ai_failed.emit((job_id, str(exc)))
        else:
            self.ai_completed.emit((job_id, result))

    def _ai_progress(self, payload: object) -> None:
        job_id, message = cast(tuple[int, str], payload)
        if job_id == self.ai_job_generation:
            self._set_ai_busy(str(message))

    def _ai_completed(self, payload: object) -> None:
        job_id, updated = cast(tuple[int, Image.Image], payload)
        if job_id != self.ai_job_generation or self.ai_pending_previous is None:
            return
        pending_id, previous = self.ai_pending_previous
        if pending_id != job_id:
            return
        self._finish_ai_job()
        self._commit_edit(previous, updated)

    def _ai_failed(self, payload: object) -> None:
        job_id, message = cast(tuple[int, str], payload)
        if job_id != self.ai_job_generation:
            return
        self._finish_ai_job()
        self._show_error("AI 替换失败", str(message))
        self._set_status("AI 处理失败，原图未发生变化", ready=False)

    def _finish_ai_job(self) -> None:
        self.ai_pending_previous = None
        self._set_edit_controls_enabled(True)
        self.ai_progress_bar.hide()
        self._set_apply_button("开始 AI 替换", role="primary", enabled=self.ai_worker_ready)

    def _set_apply_button(self, text: str, *, role: str, enabled: bool) -> None:
        self.apply_button.setText(text)
        self.apply_button.setProperty("role", role)
        self.apply_button.setEnabled(enabled)
        self.apply_button.style().unpolish(self.apply_button)
        self.apply_button.style().polish(self.apply_button)

    def _set_edit_controls_enabled(self, enabled: bool) -> None:
        self.replacement_input.setEnabled(enabled)
        self.context_spin.setEnabled(enabled)
        for control in self.ai_locked_controls:
            control.setEnabled(enabled)
        self.canvas.selection_enabled = enabled
        self.canvas.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _set_ai_busy(self, message: str, *, cancellable: bool = True) -> None:
        self.status_label.setText(message)
        self.status_badge.setText("AI 处理中")
        self.status_badge.setProperty("status", "idle")
        self.ai_progress_bar.show()
        if cancellable:
            self._set_apply_button("取消处理", role="danger", enabled=True)
        else:
            self._set_apply_button("AI 引擎启动中…", role="primary", enabled=False)
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def cancel_ai_replacement(self) -> None:
        """Invalidate the current edit and terminate inference to release GPU memory."""
        if self.ai_pending_previous is None:
            return
        self.ai_job_generation += 1
        self.ai_pending_previous = None
        self.ai_worker_ready = False
        self.ai_worker_starting = False
        self._set_edit_controls_enabled(True)
        self._set_ai_busy("正在取消并释放显存…", cancellable=False)
        self._set_apply_button("正在取消…", role="danger", enabled=False)

        process = self.ai_worker_process
        self.ai_worker_process = None
        if process is None or process.poll() is not None:
            self._finish_cancelled_worker()
            return
        process.terminate()
        self.ai_worker_stopping = process
        self.ai_worker_stop_deadline = time.monotonic() + 3
        QTimer.singleShot(100, self._poll_cancelled_worker)

    def _poll_cancelled_worker(self) -> None:
        process = self.ai_worker_stopping
        if process is None:
            self._finish_cancelled_worker()
            return
        if process.poll() is None and time.monotonic() < self.ai_worker_stop_deadline:
            QTimer.singleShot(100, self._poll_cancelled_worker)
            return
        if process.poll() is None:
            process.kill()
        self.ai_worker_stopping = None
        self._finish_cancelled_worker()

    def _finish_cancelled_worker(self) -> None:
        self.ai_socket_path.unlink(missing_ok=True)
        self.ai_progress_bar.hide()
        self._set_status("已取消处理，原图未发生变化", ready=False)
        self._set_apply_button("正在恢复 AI 引擎…", role="primary", enabled=False)
        QTimer.singleShot(150, self._ensure_ai_worker)

    def _commit_edit(self, previous: Image.Image, updated: Image.Image) -> None:
        self.history.append(previous)
        if len(self.history) > 10:
            self.history.pop(0)
        self.image = updated
        self.edit_count += 1
        self.selected_box = None
        self.replacement_input.clear()
        self.canvas.set_image(updated)
        self.selection_badge.setText("替换完成，可继续框选")
        self._set_status(f"已应用 {self.edit_count} 次修改 · 尚未保存", ready=True)

    def undo(self) -> None:
        if not self.history:
            self._set_status("当前没有可以撤销的修改", ready=False)
            return
        self.image = self.history.pop()
        self.edit_count = max(0, self.edit_count - 1)
        self.selected_box = None
        self.canvas.set_image(self.image)
        self.selection_badge.setText("已撤销上一次修改")
        self._set_status(f"当前保留 {self.edit_count} 次修改 · 尚未保存", ready=True)

    def save_result(self) -> None:
        if self.image is None:
            self._show_error("缺少图片", "请先选择并处理一张图片。")
            return
        output_name = self.output_name_input.text().strip()
        if not output_name:
            self._show_error("缺少文件名", "请填写输出文件名。")
            return
        output_dir = getattr(self, "output_dir", Path.cwd())
        output_path = Path(output_dir) / Path(output_name).name
        if not output_path.suffix:
            output_path = output_path.with_suffix(".png")
        if output_path.exists():
            answer = QMessageBox.question(
                self,
                "覆盖文件",
                f"文件已存在：\n{output_path}\n\n是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            save_image(self.image, output_path)
        except (OSError, ValueError) as exc:
            self._show_error("保存失败", str(exc))
            return
        self._set_status(f"已保存到 {output_path}", ready=True)

    def _set_status(self, text: str, *, ready: bool) -> None:
        self.status_label.setText(text)
        self.status_badge.setText("已就绪" if ready else "本机处理")
        self.status_badge.setProperty("status", "ready" if ready else "idle")
        self.status_badge.style().unpolish(self.status_badge)
        self.status_badge.style().polish(self.status_badge)

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.warning(self, title, message)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            suffix = Path(urls[0].toLocalFile()).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls and urls[0].isLocalFile():
            self.load_image(Path(urls[0].toLocalFile()))
            event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.closing = True
        self.ai_executor.shutdown(wait=False, cancel_futures=True)
        if self.ai_worker_process is not None and self.ai_worker_process.poll() is None:
            self.ai_worker_process.terminate()
            try:
                self.ai_worker_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.ai_worker_process.kill()
        if self.ai_worker_stopping is not None and self.ai_worker_stopping.poll() is None:
            self.ai_worker_stopping.kill()
        self.ai_socket_path.unlink(missing_ok=True)
        super().closeEvent(event)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 RasterRecast 本机图片文字替换 GUI。")
    parser.add_argument("image", nargs="?", type=Path, help="可选：启动时直接打开的图片")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("RasterRecast")
    app.setApplicationDisplayName("RasterRecast")
    app.setWindowIcon(create_app_icon())
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = RasterRecastWindow(args.image.expanduser() if args.image else None)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
