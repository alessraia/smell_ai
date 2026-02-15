from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


class ImmediateThread:
    """Esegue subito il target (sincrono), come nel vostro system testing."""
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
    # test/sys_test_prompt_engineering/conftest.py -> repo root
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def tk_root():
    """GUI reale: se Tk non è disponibile (headless), salta i system test GUI."""
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
def force_sync_threads(monkeypatch):
    """Allineato all’altro system testing: niente thread reali."""
    import threading
    monkeypatch.setattr(threading, "Thread", ImmediateThread)
    return ImmediateThread


@pytest.fixture()
def fake_messagebox(monkeypatch):
    """
    Intercetta messagebox per evitare popup bloccanti e per poter fare assert.
    """
    from tkinter import messagebox

    calls = {"showerror": [], "askyesno": []}

    def _showerror(title, message):
        calls["showerror"].append((title, message))
        return None

    def _askyesno(title, message):
        calls["askyesno"].append((title, message))
        return True  # default: conferma

    monkeypatch.setattr(messagebox, "showerror", _showerror)
    monkeypatch.setattr(messagebox, "askyesno", _askyesno)

    return calls


@pytest.fixture()
def tmp_output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture()
def tmp_input_single_py_dir(tmp_path: Path) -> Path:
    """
    Input path valido per TC_2.2:
    - directory
    - contiene almeno 1 .py
    - “unico progetto” (file .py direttamente in root)
    """
    proj = tmp_path / "proj"
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "main.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return proj


@pytest.fixture()
def tmp_catalog_json_one_smell(tmp_path: Path) -> Path:
    """
    Catalogo valido per TC_2.2:
    - 1 smell disponibile
    - 1 provider LOCAL disponibile
    """
    catalog = {
        "schema_version": 1,
        "smells": [
            {
                "smell_id": "chain_index",
                "display_name": "Chain Indexing",
                "description": "Pandas chained indexing smell",
                "default_prompt": "Return JSON with findings.",
                "draft_prompt": "Return JSON with findings.",
                "created_by_user": False,
                "enabled": True,
            }
        ],
        "providers": [
            {
                "provider_id": "local-ollama",
                "kind": "local",
                "display_name": "Ollama (local)",
                "config": {"model_name": "qwen2.5-coder:7b"},
            }
        ],
    }

    p = tmp_path / "llm_catalog_one_smell.json"
    p.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@pytest.fixture()
def pe_gui_app(repo_root: Path, tk_root, tmp_catalog_json_one_smell: Path):
    """
    GUI Prompt Engineering reale con catalog valido (1 smell).
    """
    sys.path.insert(0, str(repo_root))

    from llm_detection.catalog_store import LLMCatalogStore
    from llm_detection.catalog_service import LLMCatalogService
    from prompt_engineering.prompt_engineering_gui import PromptEngineeringGUI

    store = LLMCatalogStore(file_path=str(tmp_catalog_json_one_smell))
    service = LLMCatalogService(store=store)

    app = PromptEngineeringGUI(tk_root, catalog_service=service)
    yield app

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ---- TC_2.1 specific fixtures (catalogo smell vuoto) ----

@pytest.fixture()
def tmp_empty_smell_catalog_json(tmp_path: Path) -> Path:
    """
    Catalogo per TC_2.1: smells=[] (VUOTO).
    """
    catalog = {
        "schema_version": 1,
        "smells": [],  # <-- CONDIZIONE del TC_2.1
        "providers": [
            {
                "provider_id": "local-ollama",
                "kind": "local",
                "display_name": "Ollama (local)",
                "config": {"model_name": "qwen2.5-coder:7b"},
            }
        ],
    }

    p = tmp_path / "llm_catalog_empty_smells.json"
    p.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@pytest.fixture()
def pe_gui_app_empty_smells(repo_root: Path, tk_root, tmp_empty_smell_catalog_json: Path):
    """
    GUI Prompt Engineering reale con catalog smells vuoto.
    """
    sys.path.insert(0, str(repo_root))

    from llm_detection.catalog_store import LLMCatalogStore
    from llm_detection.catalog_service import LLMCatalogService
    from prompt_engineering.prompt_engineering_gui import PromptEngineeringGUI

    store = LLMCatalogStore(file_path=str(tmp_empty_smell_catalog_json))
    service = LLMCatalogService(store=store)

    app = PromptEngineeringGUI(tk_root, catalog_service=service)
    yield app

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


def flush_tk_events(root, max_iter: int = 50) -> None:
    """
    Fa girare la coda eventi Tk (utile se la GUI usa after()).
    """
    for _ in range(max_iter):
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            break

@pytest.fixture()
def tmp_catalog_no_local_provider(tmp_path: Path) -> Path:
    """
    Catalogo per TC_2.7:
    - 1 smell presente
    - provider presenti ma NESSUNO con kind="local"
    """
    catalog = {
        "schema_version": 1,
        "smells": [
            {
                "smell_id": "chain_index",
                "display_name": "Chain Indexing",
                "description": "Pandas chained indexing smell",
                "default_prompt": "Return JSON with findings.",
                "draft_prompt": None,
                "created_by_user": False,
                "enabled": True,
            }
        ],
        "providers": [
            {
                "provider_id": "openai-api",
                "kind": "api",  # <-- NON local
                "display_name": "OpenAI API",
                "config": {},
            }
        ],
    }

    p = tmp_path / "llm_catalog_no_local.json"
    p.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


@pytest.fixture()
def pe_gui_app_no_local_provider(repo_root: Path, tk_root, tmp_catalog_no_local_provider: Path):
    sys.path.insert(0, str(repo_root))

    from llm_detection.catalog_store import LLMCatalogStore
    from llm_detection.catalog_service import LLMCatalogService
    from prompt_engineering.prompt_engineering_gui import PromptEngineeringGUI

    store = LLMCatalogStore(file_path=str(tmp_catalog_no_local_provider))
    service = LLMCatalogService(store=store)

    app = PromptEngineeringGUI(tk_root, catalog_service=service)
    yield app

    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
