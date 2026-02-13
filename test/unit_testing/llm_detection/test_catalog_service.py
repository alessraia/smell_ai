import os
from dataclasses import replace

import pytest

from llm_detection.catalog_service import CatalogValidationError, LLMCatalogService
from llm_detection.types import (
    DetectionTarget,
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    PromptMode,
    ProviderKind,
)


class InMemoryStore:
    def __init__(self, catalog: LLMCatalog):
        self._catalog = catalog
        self.saved = 0

    def ensure_exists(self) -> LLMCatalog:
        return self._catalog

    def save(self, catalog: LLMCatalog) -> None:
        self._catalog = catalog
        self.saved += 1


@pytest.fixture
def base_catalog() -> LLMCatalog:
    return LLMCatalog(
        schema_version=1,
        smells=[],
        providers=[
            LLMProviderDefinition(
                provider_id="local",
                kind=ProviderKind.LOCAL,
                display_name="Local",
                config={"model_name": "x"},
            )
        ],
    )


def test_add_smell_validates_inputs(base_catalog):
    svc = LLMCatalogService(store=InMemoryStore(base_catalog))
    with pytest.raises(CatalogValidationError):
        svc.add_smell("", "desc")
    with pytest.raises(CatalogValidationError):
        svc.add_smell("name", "")


def test_add_smell_creates_slug_and_avoids_collisions(base_catalog):
    svc = LLMCatalogService(store=InMemoryStore(base_catalog))

    smell_id_1 = svc.add_smell("My Smell!", "desc")
    assert smell_id_1 == "my-smell"

    smell_id_2 = svc.add_smell("My Smell", "desc2")
    assert smell_id_2 == "my-smell-2"


def test_add_smell_enforces_unique_display_name_case_insensitive(base_catalog):
    store = InMemoryStore(base_catalog)
    svc = LLMCatalogService(store=store)

    svc.add_smell("Smell", "desc")
    with pytest.raises(CatalogValidationError):
        svc.add_smell("sMeLl", "desc2")


def test_remove_smell_raises_on_unknown(base_catalog):
    svc = LLMCatalogService(store=InMemoryStore(base_catalog))
    with pytest.raises(KeyError):
        svc.remove_smell("missing")


def test_update_smell_description_validates_and_persists(base_catalog):
    store = InMemoryStore(base_catalog)
    svc = LLMCatalogService(store=store)

    smell_id = svc.add_smell("A", "desc")
    with pytest.raises(CatalogValidationError):
        svc.update_smell_description(smell_id, "")

    svc.update_smell_description(smell_id, "new")
    smell = svc.load().get_smell(smell_id)
    assert smell.description == "new"
    assert store.saved >= 2


def test_prompt_engineering_save_draft_and_promote(base_catalog):
    store = InMemoryStore(base_catalog)
    svc = LLMCatalogService(store=store)
    smell_id = svc.add_smell("A", "desc")

    with pytest.raises(CatalogValidationError):
        svc.save_draft_prompt(smell_id, "  ")

    svc.save_draft_prompt(smell_id, "draft")
    assert svc.load().get_smell(smell_id).draft_prompt == "draft"

    svc.promote_draft_to_default(smell_id)
    smell = svc.load().get_smell(smell_id)
    assert smell.default_prompt == "draft"
    assert smell.enabled is True


def test_get_prompt_delegates_to_smell(base_catalog):
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    cat = replace(base_catalog, smells=[smell])
    svc = LLMCatalogService(store=InMemoryStore(cat))
    assert svc.get_prompt("s1", PromptMode.DEFAULT) == "default"
    assert svc.get_prompt("s1", PromptMode.DRAFT) == "draft"


def test_list_detectable_smells_filters_is_ready(base_catalog):
    smell_ready = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="p",
        draft_prompt=None,
        enabled=True,
    )
    smell_not_ready = LLMSmellDefinition(
        smell_id="s2",
        display_name="S2",
        description="d",
        default_prompt="",
        draft_prompt=None,
        enabled=True,
    )
    svc = LLMCatalogService(
        store=InMemoryStore(replace(base_catalog, smells=[smell_ready, smell_not_ready]))
    )
    detectable = svc.list_detectable_smells()
    assert [s.smell_id for s in detectable] == ["s1"]


def test_provider_listing_and_lookup(base_catalog):
    svc = LLMCatalogService(store=InMemoryStore(base_catalog))
    providers = svc.list_providers()
    assert len(providers) == 1
    assert svc.get_provider("local").provider_id == "local"
    with pytest.raises(KeyError):
        svc.get_provider("missing")


def test_build_targets_from_input_path_reads_python_files(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    (project / "a.py").write_text("print('hi')\n", encoding="utf-8")
    (project / "b.txt").write_text("no", encoding="utf-8")

    targets = LLMCatalogService.build_targets_from_input_path(str(project))
    assert len(targets) == 1
    assert isinstance(targets[0], DetectionTarget)
    assert targets[0].filename.endswith(os.path.join("proj", "a.py"))
    assert "print('hi')" in targets[0].code


def test_build_targets_from_input_path_raises_if_no_py(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CatalogValidationError):
        LLMCatalogService.build_targets_from_input_path(str(d))


def test_validate_prompt_engineering_input_path_valid_root_py(tmp_path):
    d = tmp_path / "proj"
    d.mkdir()
    (d / "main.py").write_text("x=1\n", encoding="utf-8")
    LLMCatalogService.validate_prompt_engineering_input_path(str(d))


def test_validate_prompt_engineering_input_path_detects_multi_project(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    p1 = root / "p1"
    p2 = root / "p2"
    p1.mkdir()
    p2.mkdir()
    (p1 / "a.py").write_text("x=1\n", encoding="utf-8")
    (p2 / "b.py").write_text("x=2\n", encoding="utf-8")

    with pytest.raises(CatalogValidationError):
        LLMCatalogService.validate_prompt_engineering_input_path(str(root))


def test_validate_prompt_engineering_input_path_allows_single_project_dir(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    p1 = root / "p1"
    p1.mkdir()
    (p1 / "a.py").write_text("x=1\n", encoding="utf-8")

    LLMCatalogService.validate_prompt_engineering_input_path(str(root))


def test_validate_prompt_engineering_input_path_validates_missing_or_invalid(tmp_path):
    with pytest.raises(CatalogValidationError):
        LLMCatalogService.validate_prompt_engineering_input_path("")

    with pytest.raises(CatalogValidationError):
        LLMCatalogService.validate_prompt_engineering_input_path(str(tmp_path / "nope"))

    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(CatalogValidationError):
        LLMCatalogService.validate_prompt_engineering_input_path(str(d))
