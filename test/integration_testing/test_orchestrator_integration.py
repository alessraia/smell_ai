"""
Test minimale: LLMOrchestrator ↔ LLMProvider
Verifica che orchestrator possa chiamare provider e processare risposta
"""

from llm_detection.orchestrator import LLMOrchestrator
from llm_detection.providers import MockLLMProvider
from llm_detection.types import LLMCatalog, LLMSmellDefinition, DetectionTarget, PromptMode


def test_orchestrator_can_detect_with_mock_provider():
    """Verifica che orchestrator chiami provider.generate() e processi JSON response"""
    
    # 1. Setup catalog con uno smell
    smell = LLMSmellDefinition(
        smell_id="test-smell",
        display_name="Test Smell",
        description="Test description",
        default_prompt="Detect the smell in:\n{code}",
        draft_prompt=None,
        created_by_user=True,
        enabled=True
    )
    catalog = LLMCatalog(schema_version=1, smells=[smell], providers=[])
    
    # 2. Setup mock provider con risposta JSON valida
    mock_response = """{
        "findings": [
            {
                "function_name": "test_func",
                "line": 5,
                "description": "Test smell detected",
                "additional_info": "Refactor suggested"
            }
        ]
    }"""
    provider = MockLLMProvider(fixed_response=mock_response)
    
    # 3. Setup target
    target = DetectionTarget(
        filename="test.py",
        code="def test_func():\n    x = 1\n    y = 2\n    return x + y"
    )
    
    # 4. Crea orchestrator e esegui detection
    orchestrator = LLMOrchestrator(provider=provider, catalog=catalog)
    findings, stats = orchestrator.detect(
        targets=[target],
        smell_ids=["test-smell"],
        prompt_mode=PromptMode.DEFAULT
    )
    
    # 5. Verifica risultati
    assert stats.prompts_sent == 1, "Expected 1 prompt sent"
    assert stats.targets_processed == 1, "Expected 1 target processed"
    assert len(findings) == 1, "Expected 1 finding"
    
    # 6. Verifica finding content
    finding = findings[0]
    assert finding.filename == "test.py"
    assert finding.function_name == "test_func"
    assert finding.line == 5
    assert "Test smell detected" in finding.description


def test_orchestrator_handles_empty_findings():
    """Verifica che orchestrator gestisca correttamente risposta senza findings"""
    
    # Setup catalog
    smell = LLMSmellDefinition(
        smell_id="test-smell",
        display_name="Test Smell",
        description="Test",
        default_prompt="Detect:\n{code}",
        draft_prompt=None,
        created_by_user=True,
        enabled=True
    )
    catalog = LLMCatalog(schema_version=1, smells=[smell], providers=[])
    
    # Provider che restituisce array vuoto (no smell found)
    provider = MockLLMProvider(fixed_response='{"findings": []}')
    
    target = DetectionTarget(filename="test.py", code="def clean_func():\n    return 42")
    
    orchestrator = LLMOrchestrator(provider=provider, catalog=catalog)
    findings, stats = orchestrator.detect(
        targets=[target],
        smell_ids=["test-smell"],
        prompt_mode=PromptMode.DEFAULT
    )
    
    # Verifica nessun finding ma stats corretti
    assert len(findings) == 0, "Expected no findings"
    assert stats.prompts_sent == 1, "Expected 1 prompt sent"
    assert stats.targets_processed == 1


def test_orchestrator_processes_multiple_targets():
    """Verifica che orchestrator processi correttamente più targets"""
    
    smell = LLMSmellDefinition(
        smell_id="test-smell",
        display_name="Test",
        description="Test",
        default_prompt="Detect:\n{code}",
        draft_prompt=None,
        created_by_user=True,
        enabled=True
    )
    catalog = LLMCatalog(schema_version=1, smells=[smell], providers=[])
    
    # Provider che restituisce sempre un finding alla linea 1
    provider = MockLLMProvider(
        fixed_response='{"findings": [{"function_name": "func", "line": 1, "description": "smell", "additional_info": ""}]}'
    )
    
    # Tre targets diversi
    targets = [
        DetectionTarget(filename="file1.py", code="def func1():\n    pass"),
        DetectionTarget(filename="file2.py", code="def func2():\n    pass"),
        DetectionTarget(filename="file3.py", code="def func3():\n    pass"),
    ]
    
    orchestrator = LLMOrchestrator(provider=provider, catalog=catalog)
    findings, stats = orchestrator.detect(
        targets=targets,
        smell_ids=["test-smell"],
        prompt_mode=PromptMode.DEFAULT
    )
    
    # Verifica stats
    assert stats.prompts_sent == 3, "Expected 3 prompts (one per target)"
    assert stats.targets_processed == 3
    assert len(findings) == 3, "Expected 3 findings (one per target)"
    
    # Verifica che ogni finding abbia filename corretto
    filenames = {f.filename for f in findings}
    assert filenames == {"file1.py", "file2.py", "file3.py"}
