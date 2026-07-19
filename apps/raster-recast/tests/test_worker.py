from pathlib import Path

from pytest import MonkeyPatch
from raster_recast.worker import worker_launch


def test_worker_launch_resolves_project_runtime(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    root = tmp_path / "fluxtext-worker"
    models = root / "models"
    source = root / "fluxtext-runtime" / "flux_text"
    python = root / ".venv" / "bin" / "python"
    models.mkdir(parents=True)
    source.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setenv("RASTER_RECAST_WORKER_DIR", str(root))
    socket_path = tmp_path / "worker.sock"

    launch = worker_launch(socket_path)

    assert launch.working_directory == root
    assert launch.command[0] == str(python)
    assert "raster_recast.inference_worker" in launch.command
    assert str(socket_path) in launch.command
