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


@pytest.fixture(scope="session")
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


# ==================== LLM Testing Fixtures ====================

@pytest.fixture()
def mock_llm_catalog():
    """
    Crea un catalog LLM standard per i test con provider e smell di default.
    """
    from llm_detection.types import (
        LLMCatalog,
        LLMSmellDefinition,
        LLMProviderDefinition,
        ProviderKind,
    )

    return LLMCatalog(
        schema_version=1,
        smells=[
            LLMSmellDefinition(
                smell_id="test_smell_1",
                display_name="Test Smell 1",
                description="Test description",
                default_prompt="Test prompt",
                enabled=True,
            ),
            LLMSmellDefinition(
                smell_id="test_smell_2",
                display_name="Test Smell 2",
                description="Test description 2",
                default_prompt="Test prompt 2",
                enabled=True,
            ),
        ],
        providers=[
            LLMProviderDefinition(
                provider_id="local-ollama",
                kind=ProviderKind.LOCAL,
                display_name="Ollama Local",
                config={"host": "http://localhost:11434", "model_name": "qwen2.5-coder:7b"},
            ),
            LLMProviderDefinition(
                provider_id="api-openai",
                kind=ProviderKind.API,
                display_name="OpenAI API",
                config={"base_url": "https://api.openai.com"},
            ),
        ],
    )


@pytest.fixture()
def mock_llm_catalog_no_smells():
    """
    Catalog con provider ma senza smell detectabili (per TC_4.44).
    """
    from llm_detection.types import (
        LLMCatalog,
        LLMProviderDefinition,
        ProviderKind,
    )

    return LLMCatalog(
        schema_version=1,
        smells=[],  # Nessuno smell
        providers=[
            LLMProviderDefinition(
                provider_id="local-ollama",
                kind=ProviderKind.LOCAL,
                display_name="Ollama Local",
                config={"host": "http://localhost:11434", "model_name": "qwen2.5-coder:7b"},
            ),
        ],
    )


@pytest.fixture()
def mock_llm_catalog_no_providers():
    """
    Catalog con smell ma senza provider locali (per TC_4.49).
    """
    from llm_detection.types import (
        LLMCatalog,
        LLMSmellDefinition,
        ProviderKind,
    )

    return LLMCatalog(
        schema_version=1,
        smells=[
            LLMSmellDefinition(
                smell_id="test_smell_1",
                display_name="Test Smell 1",
                description="Test description",
                default_prompt="Test prompt",
                enabled=True,
            ),
        ],
        providers=[],  # Nessun provider
    )


@pytest.fixture()
def mock_catalog_service(mock_llm_catalog):
    """
    Mock del LLMCatalogService che ritorna il catalog standard.
    """
    from unittest.mock import MagicMock

    service = MagicMock()
    service.load.return_value = mock_llm_catalog
    service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog.smells if s.is_ready_for_detection()
    ]
    service.get_provider.side_effect = lambda pid: next(
        (p for p in mock_llm_catalog.providers if p.provider_id == pid), None
    )
    return service


@pytest.fixture()
def gui_app_with_llm_mock(repo_root: Path, tk_root, monkeypatch, mock_catalog_service):
    """
    GUI app con catalog service mockato per i test LLM.
    """
    sys.path.insert(0, str(repo_root))
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    from unittest.mock import MagicMock

    # Mock il catalog service
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_catalog_service
    )

    # Disabilita stdout redirect
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)

    # Mock dell'orchestrator per evitare chiamate LLM reali
    mock_orchestrator = MagicMock()
    mock_orchestrator.analyze_project.return_value = None  # Simuliamo successo
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMOrchestrator",
        lambda *args, **kwargs: mock_orchestrator
    )

    app = CodeSmellDetectorGUI(tk_root)
    yield app

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))