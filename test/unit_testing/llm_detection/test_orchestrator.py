import pytest

from llm_detection.orchestrator import LLMOrchestrator
from llm_detection.providers import MockLLMProvider
from llm_detection.types import (
    DetectionTarget,
    LLMCatalog,
    LLMSmellDefinition,
    NormalizationMode,
    PromptMode,
)


@pytest.fixture
def catalog_with_smells() -> LLMCatalog:
    ready = LLMSmellDefinition(
        smell_id="s_ready",
        display_name="Ready",
        description="desc",
        default_prompt="Detect stuff.",
        draft_prompt=None,
        enabled=True,
    )
    not_ready = LLMSmellDefinition(
        smell_id="s_not",
        display_name="NotReady",
        description="d",
        default_prompt="",
        draft_prompt=None,
        enabled=True,
    )
    return LLMCatalog(schema_version=1, smells=[ready, not_ready], providers=[])


def test_code_with_line_numbers_is_1_based_and_preserves_text():
    provider = MockLLMProvider(fixed_response='{ "findings": [] }')
    cat = LLMCatalog(schema_version=1, smells=[], providers=[])
    orch = LLMOrchestrator(provider=provider, catalog=cat)
    numbered = orch._code_with_line_numbers("a\n\n  b")
    assert numbered.splitlines()[0] == "1: a"
    assert numbered.splitlines()[1] == "2: "
    assert numbered.splitlines()[2] == "3:   b"


def test_build_prompt_includes_filename_and_numbered_code(catalog_with_smells):
    provider = MockLLMProvider(fixed_response='{ "findings": [] }')
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)
    target = DetectionTarget(filename="f.py", code="print('x')\n")

    prompt = orch.build_prompt("s_ready", target, PromptMode.DEFAULT)
    assert "FILENAME: f.py" in prompt
    assert "CODE (numbered):" in prompt
    assert "1: print('x')" in prompt
    assert "OUTPUT FORMAT (STRICT)" in prompt


def test_detect_skips_non_ready_smells_and_counts_prompts(catalog_with_smells):
    provider = MockLLMProvider(fixed_response='{ "findings": [] }')
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)
    targets = [DetectionTarget(filename="a.py", code="x=1\n")]

    findings, stats = orch.detect(
        targets=targets,
        smell_ids=["s_ready", "s_not"],
        prompt_mode=PromptMode.DEFAULT,
        normalize_mode=NormalizationMode.STRICT,
    )
    assert findings == []
    assert stats.prompts_sent == 1
    assert stats.targets_processed == 1
    assert stats.smells_processed == 2


def test_try_parse_json_payload_handles_fences_and_embedded_json(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    assert orch._try_parse_json_payload('{"a": 1}') == {"a": 1}
    fenced = "```json\n{\"a\": 2}\n```"
    assert orch._try_parse_json_payload(fenced) == {"a": 2}
    embedded = "blah blah {\"a\": 3} trailing"
    assert orch._try_parse_json_payload(embedded) == {"a": 3}
    assert orch._try_parse_json_payload("not json") is None


def test_normalize_response_strict_accepts_only_findings_array(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"findings": [{"function_name": "f", "line": 2, "description": "d", "additional_info": "i", "confidence": 0.9}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.STRICT,
    )
    assert len(out) == 1
    assert out[0].filename == "a.py"
    assert out[0].function_name == "f"
    assert out[0].line == 2
    assert out[0].smell_name == "Ready"
    assert out[0].confidence == 0.9
    assert out[0].raw_response == raw

    # Wrong schema => empty
    out2 = orch._normalize_response(
        '{"wrong": []}',
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.STRICT,
    )
    assert out2 == []


def test_normalize_response_salvage_unparseable_returns_diagnostic(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    out = orch._normalize_response(
        "not json",
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == -1
    assert "Unparseable" in out[0].additional_info
    assert out[0].raw_response == "not json"


def test_normalize_response_salvage_schema_invalid_returns_diagnostic(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    out = orch._normalize_response(
        '{"something": "else"}',
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == -1
    assert "Invalid LLM response schema" in out[0].description


def test_normalize_response_salvage_recovers_from_fallback_key(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"results": [{"function_name": "f", "line_number": 3, "code_snippet": "x=1"}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == 3
    assert "Recovered from non-standard" in out[0].description
    assert out[0].additional_info == "x=1"
    assert out[0].raw_response == raw


def test_normalize_response_salvage_recovers_single_object_under_key(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"one": {"function_name": "f", "line": 5, "description": "d"}}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == 5
    assert out[0].description == "d"


def test_findings_to_dataframe_columns(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)
    raw = '{"findings": [{"function_name": "f", "line": 2, "description": "d", "additional_info": "i"}]}'
    findings = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.STRICT,
    )
    df = orch.findings_to_dataframe(findings)
    assert list(df.columns) == [
        "filename",
        "function_name",
        "smell_name",
        "line",
        "description",
        "additional_info",
    ]
    assert df.iloc[0]["filename"] == "a.py"


def test_detect_for_prompt_engineering_runs_even_if_smell_not_enabled(tmp_path):
    # detect_for_prompt_engineering does not check is_ready_for_detection;
    # it is used to experiment with prompts before enabling.
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="Prompt",
        draft_prompt=None,
        enabled=False,
    )
    catalog = LLMCatalog(schema_version=1, smells=[smell], providers=[])
    provider = MockLLMProvider(fixed_response="not json")
    orch = LLMOrchestrator(provider=provider, catalog=catalog)

    targets = [
        DetectionTarget(filename="a.py", code="x=1\n"),
        DetectionTarget(filename="b.py", code="x=2\n"),
    ]
    findings, stats = orch.detect_for_prompt_engineering(
        targets=targets,
        smell_id="s1",
        prompt_mode=PromptMode.DEFAULT,
        normalize_mode=NormalizationMode.SALVAGE,
    )

    assert stats.prompts_sent == 2
    assert stats.targets_processed == 2
    assert stats.smells_processed == 1
    # unparseable -> diagnostic row per target
    assert len(findings) == 2
    assert all(f.line == -1 for f in findings)


def test_detect_for_prompt_engineering_with_raw_returns_mapping_per_file():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="desc",
        default_prompt="Prompt",
        draft_prompt=None,
        enabled=False,
    )
    catalog = LLMCatalog(schema_version=1, smells=[smell], providers=[])

    def factory(prompt: str) -> str:
        if "FILENAME: a.py" in prompt:
            return '{"findings": [{"function_name": "fa", "line": 1, "description": "d"}]}'
        return '{"findings": []}'

    provider = MockLLMProvider(response_factory=factory)
    orch = LLMOrchestrator(provider=provider, catalog=catalog)
    targets = [
        DetectionTarget(filename="a.py", code="x=1\n"),
        DetectionTarget(filename="b.py", code="x=2\n"),
    ]

    findings, stats, raw_by_filename = orch.detect_for_prompt_engineering_with_raw(
        targets=targets,
        smell_id="s1",
        prompt_mode=PromptMode.DEFAULT,
        normalize_mode=NormalizationMode.SALVAGE,
    )

    assert stats.prompts_sent == 2
    assert raw_by_filename["a.py"].startswith("{\"findings\"")
    assert raw_by_filename["b.py"].startswith("{\"findings\"")
    assert any(f.filename == "a.py" and f.line == 1 for f in findings)


def test_try_parse_json_payload_extracts_first_json_array(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    # The scanner tries to extract an object ({...}) before an array ([...]).
    # To exercise the '[' branch, avoid '{' in the payload.
    embedded = "prefix [1, 2, 3] suffix"
    assert orch._try_parse_json_payload(embedded) == [1, 2, 3]


def test_try_parse_json_payload_does_not_break_on_braces_in_strings(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    # includes a '}' inside a JSON string and an escaped quote to exercise
    # in_string/escape logic in the scanner.
    embedded = (
        "xxx {\"a\": \"brace } and quote \\\" ok\", \"b\": 1} yyy"
    )
    assert orch._try_parse_json_payload(embedded) == {
        "a": 'brace } and quote " ok',
        "b": 1,
    }


def test_normalize_response_strict_line_number_fallback_and_confidence_parse_error(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"findings": [{"function_name": null, "line_number": "7", "confidence": "abc"}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.STRICT,
    )
    assert len(out) == 1
    assert out[0].line == 7
    assert out[0].confidence is None
    # description fallback in STRICT uses smell.description
    assert out[0].description == "desc"


def test_normalize_response_salvage_schema_invalid_when_lists_without_line(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"results": [{"foo": 1}, {"bar": 2}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == -1
    assert "Invalid LLM response schema" in out[0].description


def test_normalize_response_salvage_findings_missing_description_uses_smell_description_and_code_fallback(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"findings": [{"line": 2, "code": "x=1"}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].description == "desc"
    assert out[0].additional_info == "x=1"


def test_normalize_response_salvage_skips_non_dict_and_invalid_lines(catalog_with_smells):
    provider = MockLLMProvider(fixed_response="")
    orch = LLMOrchestrator(provider=provider, catalog=catalog_with_smells)

    raw = '{"findings": [1, {"line": 0}, {"line": "bad"}, {"line": 3, "description": "ok"}]}'
    out = orch._normalize_response(
        raw,
        filename="a.py",
        smell_id="s_ready",
        normalize_mode=NormalizationMode.SALVAGE,
    )
    assert len(out) == 1
    assert out[0].line == 3
    assert out[0].description == "ok"
