import pytest
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QGraphicsView
from raster_recast.gui import ImageCanvas, RasterRecastWindow


@pytest.fixture(scope="module")
def app() -> QApplication:
    instance = QApplication.instance()
    if isinstance(instance, QApplication):
        return instance
    return QApplication([])


def make_canvas(app: QApplication) -> ImageCanvas:
    canvas = ImageCanvas()
    canvas.resize(640, 480)
    canvas.set_image(Image.new("RGB", (1200, 800), "white"))
    canvas.actual_size()
    canvas.show()
    app.processEvents()
    return canvas


def test_zoom_keeps_scene_point_under_pointer(app: QApplication) -> None:
    canvas = make_canvas(app)
    pointer = QPointF(470, 180)
    scene_before = canvas.mapToScene(pointer.toPoint())

    canvas.zoom(1.2, pointer)

    scene_after = canvas.mapToScene(pointer.toPoint())
    assert canvas.transform().m11() == pytest.approx(1.2)
    assert scene_after.x() == pytest.approx(scene_before.x(), abs=1.5)
    assert scene_after.y() == pytest.approx(scene_before.y(), abs=1.5)


def test_zoom_clamps_to_photoshop_style_limits(app: QApplication) -> None:
    canvas = make_canvas(app)

    canvas.zoom(100)
    assert canvas.transform().m11() == pytest.approx(16)

    canvas.zoom(0.0001)
    assert canvas.transform().m11() == pytest.approx(0.05)


def test_space_toggles_hand_pan_mode(app: QApplication) -> None:
    canvas = make_canvas(app)
    press = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    )
    release = QKeyEvent(
        QEvent.Type.KeyRelease,
        Qt.Key.Key_Space,
        Qt.KeyboardModifier.NoModifier,
    )

    canvas.keyPressEvent(press)
    assert canvas.dragMode() == QGraphicsView.DragMode.ScrollHandDrag
    assert canvas.cursor().shape() == Qt.CursorShape.OpenHandCursor

    canvas.keyReleaseEvent(release)
    assert canvas.dragMode() == QGraphicsView.DragMode.NoDrag
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor


def test_cancel_ai_replacement_invalidates_job_and_stops_worker(app: QApplication) -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    window = RasterRecastWindow()
    process = FakeProcess()
    window.ai_job_generation = 4
    window.ai_pending_previous = (4, Image.new("RGB", (20, 20)))
    window.ai_worker_process = process  # type: ignore[assignment]

    window.cancel_ai_replacement()

    assert process.terminated
    assert window.ai_job_generation == 5
    assert window.ai_pending_previous is None
    assert window.apply_button.text() == "正在取消…"
    assert window.apply_button.property("role") == "danger"

    window.closing = True
    window._poll_cancelled_worker()
    assert window.ai_worker_stopping is None
    assert not window.ai_progress_bar.isVisible()
    assert window.status_label.text() == "已取消处理，原图未发生变化"
    window.ai_worker_process = None
    window.close()
