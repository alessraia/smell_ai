import shutil
import sys
from pathlib import Path

import pytest


class ImmediateThread:
    """Sostituto di threading.Thread: esegue subito target() nel test (no async)."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    # test/gui_system_spec/conftest.py -> repo root
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def fixtures_root(repo_root: Path) -> Path:
    return repo_root / "test" / "system_testing"


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture()
def force_sync_threads(monkeypatch):
    import threading

    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    return ImmediateThread


@pytest.fixture()
def tk_root():
    """Crea Tk root; se non disponibile (headless senza Xvfb), skip."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
    except Exception as e:
        pytest.skip(f"Tk not available for GUI system tests: {e}")
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture()
def gui_app(repo_root: Path, tk_root, monkeypatch):
    """
    Istanzia la GUI senza redirezione stdout sulla textbox,
    così pytest cattura i print() con capsys.
    """
    sys.path.insert(0, str(repo_root))
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI

    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)

    app = CodeSmellDetectorGUI(tk_root)
    yield app

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


def _wait_for_daemon_threads(timeout: int = 60) -> None:
    """
    Aspetta che tutti i thread daemon completino.
    Utile per test async/parallel che usano thread daemon.
    """
    import threading
    import time
    
    start = time.time()
    main_thread = threading.current_thread()
    
    while time.time() - start < timeout:
        daemon_threads = [
            t for t in threading.enumerate()
            if t != main_thread and t.daemon
        ]
        if not daemon_threads:
            return  # Tutti i daemon thread hanno terminato
        time.sleep(0.01)  # Aspetta 10ms prima di controllare di nuovo
    
    # Se arriviamo qui, è timeout
    raise TimeoutError(f"Daemon threads didn't complete within {timeout}s")


def copy_project(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


@pytest.fixture()
def project_factory(tmp_path: Path, fixtures_root: Path):
    """
    Crea directory input ad-hoc usando i file presenti in test/system_testing/TC*
    Ritorna path della directory creata.
    """

    def _mk_empty(name="empty_project") -> Path:
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _mk_single_file_no_py(name="single_no_py") -> Path:
        # Usa TC15 (contiene sum.c) come sorgente “nessun .py”
        src = fixtures_root / "TC15"
        dst = tmp_path / name
        copy_project(src, dst)
        return dst

    def _mk_single_project_from_tc(tc_folder: str, name: str) -> Path:
        src = fixtures_root / tc_folder
        dst = tmp_path / name
        copy_project(src, dst)
        return dst

    def _mk_single_project_custom(files: list[Path], name: str) -> Path:
        dst = tmp_path / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dst / f.name)
        return dst

    def _mk_multi_base(projects: dict[str, Path], name="multi_base") -> Path:
        base = tmp_path / name
        base.mkdir(parents=True, exist_ok=True)
        for proj_name, proj_src in projects.items():
            proj_dst = base / proj_name
            copy_project(proj_src, proj_dst)
        return base

    return {
        "empty": _mk_empty,
        "single_no_py": _mk_single_file_no_py,
        "single_from_tc": _mk_single_project_from_tc,
        "single_custom": _mk_single_project_custom,
        "multi_base": _mk_multi_base,
    }
