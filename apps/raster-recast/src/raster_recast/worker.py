"""Resolve and launch RasterRecast's private FLUX-Text inference process."""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


class WorkerUnavailable(RuntimeError):
    """Raised when the local inference runtime is incomplete."""


@dataclass(frozen=True)
class WorkerLayout:
    """Paths required by the standalone FLUX-Text runtime."""

    root: Path
    python: Path
    model_directory: Path
    fluxtext_source: Path


@dataclass(frozen=True)
class WorkerLaunch:
    """A fully resolved command for the hidden inference process."""

    command: tuple[str, ...]
    working_directory: Path
    python_path: Path


def runtime_socket_path() -> Path:
    """Create a unique local IPC address for this GUI process."""
    name = f"raster-recast-{os.getpid()}-{uuid.uuid4().hex}.sock"
    return Path(tempfile.gettempdir()) / name


def worker_root() -> Path:
    """Locate the isolated runtime installed beside the blacksmith workspace."""
    configured = os.environ.get("RASTER_RECAST_WORKER_DIR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path.cwd() / "models" / "fluxtext-worker",
        Path.cwd() / "blacksmith" / "models" / "fluxtext-worker",
        Path(__file__).resolve().parents[4] / "models" / "fluxtext-worker",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise WorkerUnavailable(
        "未找到 models/fluxtext-worker；可用 RASTER_RECAST_WORKER_DIR 指定运行时目录。"
    )


def worker_layout() -> WorkerLayout:
    """Resolve the standalone model cache and inference source."""
    root = worker_root()
    python = root / ".venv" / "bin" / "python"
    model_directory = root / "models"
    fluxtext_source = root / "fluxtext-runtime"

    missing = [
        path
        for path in (python, model_directory, fluxtext_source / "flux_text")
        if not path.exists()
    ]
    if missing:
        joined = "、".join(str(path) for path in missing)
        raise WorkerUnavailable(f"AI 运行环境不完整，缺少：{joined}")
    return WorkerLayout(root, python, model_directory, fluxtext_source)


def worker_launch(socket_path: Path) -> WorkerLaunch:
    """Build the standalone worker command managed by the GUI."""
    layout = worker_layout()
    source_root = Path(__file__).resolve().parents[1]
    command = (
        str(layout.python),
        "-m",
        "raster_recast.inference_worker",
        "--socket",
        str(socket_path),
        "--runtime-root",
        str(layout.root),
    )
    return WorkerLaunch(command, layout.root, source_root)
