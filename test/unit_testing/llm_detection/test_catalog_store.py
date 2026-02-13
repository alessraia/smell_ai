import json
import os

import pytest

from llm_detection.catalog_store import LLMCatalogStore
from llm_detection.types import LLMCatalog, LLMSmellDefinition, LLMProviderDefinition, ProviderKind


def test_catalog_store_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "llm_catalog.json"
    store = LLMCatalogStore(file_path=str(path))

    catalog = LLMCatalog(
        schema_version=1,
        smells=[
            LLMSmellDefinition(
                smell_id="s1",
                display_name="S1",
                description="desc",
                default_prompt="prompt",
                draft_prompt="draft",
                created_by_user=True,
                enabled=True,
            )
        ],
        providers=[
            LLMProviderDefinition(
                provider_id="p1",
                kind=ProviderKind.LOCAL,
                display_name="Local",
                config={"model_name": "x"},
            )
        ],
    )

    assert store.exists() is False
    store.save(catalog)
    assert store.exists() is True

    loaded = store.load()
    assert loaded.schema_version == 1
    assert len(loaded.smells) == 1
    assert loaded.smells[0].smell_id == "s1"
    assert loaded.smells[0].enabled is True
    assert len(loaded.providers) == 1
    assert loaded.providers[0].kind == ProviderKind.LOCAL
    assert loaded.providers[0].config["model_name"] == "x"


def test_catalog_store_load_defaults_enabled_and_created_by_user(tmp_path):
    path = tmp_path / "llm_catalog.json"
    data = {
        "schema_version": 1,
        "smells": [
            {
                "smell_id": "s1",
                "display_name": "S1",
                "description": "d",
                "default_prompt": "p",
            }
        ],
        "providers": [
            {"provider_id": "p1", "kind": "local", "display_name": "Local"}
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    store = LLMCatalogStore(file_path=str(path))

    loaded = store.load()
    assert loaded.smells[0].enabled is True
    assert loaded.smells[0].created_by_user is False


def test_catalog_store_ensure_exists_seeds_default_provider(tmp_path):
    path = tmp_path / "sub" / "llm_catalog.json"
    store = LLMCatalogStore(file_path=str(path))
    assert store.exists() is False

    created = store.ensure_exists()
    assert created.schema_version == 1
    assert store.exists() is True
    assert len(created.providers) == 1
    assert created.providers[0].provider_id == "local-ollama"
    assert created.providers[0].kind == ProviderKind.LOCAL


def test_catalog_store_ensure_exists_uses_seed(tmp_path):
    path = tmp_path / "llm_catalog.json"
    store = LLMCatalogStore(file_path=str(path))

    seed = LLMCatalog(
        schema_version=1,
        smells=[],
        providers=[
            LLMProviderDefinition(
                provider_id="pseed",
                kind=ProviderKind.API,
                display_name="API",
                config={"base_url": "http://example"},
            )
        ],
    )

    loaded = store.ensure_exists(seed=seed)
    assert loaded.providers[0].provider_id == "pseed"
    assert loaded.providers[0].kind == ProviderKind.API


def test_catalog_store_ensure_exists_when_file_already_exists_returns_loaded(tmp_path):
    path = tmp_path / "llm_catalog.json"
    store = LLMCatalogStore(file_path=str(path))

    seed = LLMCatalog(
        schema_version=1,
        smells=[],
        providers=[
            LLMProviderDefinition(
                provider_id="p1",
                kind=ProviderKind.LOCAL,
                display_name="Local",
                config={"model_name": "x"},
            )
        ],
    )
    store.save(seed)

    loaded = store.ensure_exists(seed=LLMCatalog(schema_version=1, smells=[], providers=[]))
    assert loaded.providers[0].provider_id == "p1"
    assert loaded.providers[0].kind == ProviderKind.LOCAL


def test_catalog_store_save_supports_file_path_without_directory(tmp_path, monkeypatch):
    # Exercise the `os.path.dirname(self.file_path) or "."` branch.
    monkeypatch.chdir(tmp_path)

    store = LLMCatalogStore(file_path="llm_catalog.json")
    catalog = LLMCatalog(schema_version=1, smells=[], providers=[])
    store.save(catalog)
    assert os.path.exists(os.path.join(str(tmp_path), "llm_catalog.json"))


def test_catalog_store_save_does_not_rewrite_kind_if_already_serializable(tmp_path):
    # Cover the branch where provider.kind is NOT an Enum instance.
    # (This can happen if a caller constructs the dataclass with a str at runtime.)
    path = tmp_path / "llm_catalog.json"
    store = LLMCatalogStore(file_path=str(path))

    provider = LLMProviderDefinition(
        provider_id="p1",
        kind="local",  # type: ignore[arg-type]
        display_name="Local",
        config={},
    )
    store.save(LLMCatalog(schema_version=1, smells=[], providers=[provider]))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["providers"][0]["kind"] == "local"
