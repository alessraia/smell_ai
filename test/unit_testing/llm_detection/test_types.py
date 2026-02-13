import pytest

from llm_detection.types import LLMCatalog, LLMSmellDefinition, PromptMode


def test_smell_is_ready_for_detection_requires_enabled_and_default_prompt():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="",
        draft_prompt=None,
        enabled=True,
    )
    assert smell.is_ready_for_detection() is False

    smell.default_prompt = "  \n  "
    assert smell.is_ready_for_detection() is False

    smell.default_prompt = "prompt"
    smell.enabled = False
    assert smell.is_ready_for_detection() is False

    smell.enabled = True
    assert smell.is_ready_for_detection() is True


def test_smell_get_prompt_default_raises_if_empty():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt=" ",
        draft_prompt=None,
        enabled=False,
    )
    with pytest.raises(ValueError):
        smell.get_prompt(PromptMode.DEFAULT)


def test_smell_get_prompt_draft_raises_if_missing_or_empty():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="prompt",
        draft_prompt=None,
        enabled=False,
    )
    with pytest.raises(ValueError):
        smell.get_prompt(PromptMode.DRAFT)

    smell.draft_prompt = "   "
    with pytest.raises(ValueError):
        smell.get_prompt(PromptMode.DRAFT)


def test_smell_get_prompt_draft_if_available_falls_back_to_default():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="default",
        draft_prompt="",
        enabled=False,
    )
    assert smell.get_prompt(PromptMode.DRAFT_IF_AVAILABLE) == "default"

    smell.draft_prompt = "draft"
    assert smell.get_prompt(PromptMode.DRAFT_IF_AVAILABLE) == "draft"


def test_smell_get_prompt_unknown_raises():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    with pytest.raises(ValueError):
        smell.get_prompt("unknown")


def test_save_draft_as_default_promotes_and_enables():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="old",
        draft_prompt="new",
        enabled=False,
    )
    smell.save_draft_as_default()
    assert smell.default_prompt == "new"
    assert smell.enabled is True


def test_save_draft_as_default_raises_on_empty_draft():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="old",
        draft_prompt=" ",
        enabled=False,
    )
    with pytest.raises(ValueError):
        smell.save_draft_as_default()


def test_catalog_get_smell_and_upsert():
    catalog = LLMCatalog(schema_version=1, smells=[], providers=[])
    with pytest.raises(KeyError):
        catalog.get_smell("missing")

    s1 = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="p",
        draft_prompt=None,
        enabled=False,
    )
    catalog.upsert_smell(s1)
    assert catalog.get_smell("s1").display_name == "S1"

    s1_updated = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1-new",
        description="d2",
        default_prompt="p",
        draft_prompt=None,
        enabled=True,
    )
    catalog.upsert_smell(s1_updated)
    assert len(catalog.smells) == 1
    assert catalog.get_smell("s1").display_name == "S1-new"
