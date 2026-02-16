"""
System tests for LLM detection functionality - TC_4.44 to TC_4.49
Post-modification tests based on Category Partition analysis.

Test Frame Reference:
- FI2: campo input non vuoto (property FILE_SINGOLO if #FILE_PY)
- NF3: >1 (property FILE_MULTIPLI)
- EF1: almeno un file .py (property FILE_SINGOLO | FILE_MULTIPLI)
- NP1: 1 (if ESISTE_INPUT)
- FO2: campo output non vuoto (property ESISTE_OUTPUT)
- M2: false (property NOT_MULTIPLE)
- P2: false (property NOT_PARALLEL)
- NW2: >=1 (error)
- R2: false (if RESUME)
- NCS2: >0 (if CODE_SMELL_PRESENTI)
- TCS3: misto (if CODE_SMELL_PRESENTI) [single]
- LD1: true (property LLM)
- P1/P2: Locale/API (if LLM)
- NumP1/NumP2: 0 / >=1 (if LLM)
- NS1/NS2: 0 / >=1 (if LLM)
- SS1/SS2/SS3: 0 / 1 / >1 (if LLM && SMELL_SELEZIONABILI)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Helper per configurare GUI e simulare selezione smell
def _set_label(label, value: str) -> None:
    label.configure(text=value)


def _set_spinbox(spinbox, value: str) -> None:
    spinbox.delete(0, "end")
    spinbox.insert(0, value)


def _run(gui_app, capsys) -> str:
    """Esegue l'analisi e cattura l'output."""
    from test.gui_system_spec.conftest import _wait_for_daemon_threads
    
    gui_app.run_program()
    
    # Aspetta che TUTTI i thread daemon completino
    _wait_for_daemon_threads(timeout=60)
    
    return capsys.readouterr().out


def _select_smells(gui_app, smell_indices: list[int]) -> None:
    """Seleziona smell nella listbox per indice (0-based)."""
    gui_app.smell_listbox.selection_clear(0, "end")
    for idx in smell_indices:
        gui_app.smell_listbox.selection_set(idx)


def _configure_llm_gui(
    gui_app,
    provider_type: str = "local",
    provider_index: int = 0,
    smell_indices: list[int] = None
):
    """
    Configura la sezione LLM della GUI.
    
    Args:
        gui_app: GUI instance
        provider_type: "local" o "api"
        provider_index: indice del provider da selezionare nel combo
        smell_indices: lista di indici degli smell da selezionare (None = nessuno)
    """
    # Abilita LLM
    gui_app.llm_var.set(True)
    gui_app.toggle_llm_controls()
    
    # Imposta tipo provider
    gui_app.provider_type_var.set(provider_type)
    gui_app.update_provider_list()
    
    # Seleziona provider se disponibile
    if gui_app.provider_combo['values']:
        gui_app.provider_combo.current(provider_index)
    
    # Seleziona smell se richiesto
    if smell_indices is not None:
        _select_smells(gui_app, smell_indices)


# ==================== TC_4.44 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_44_llm_no_detectable_smells(
    tk_root, project_factory, tmp_output_dir, fixtures_root, 
    capsys, monkeypatch, repo_root, mock_llm_catalog_no_smells
):
    """
    TC_4.44: LLM enabled con provider locale ma 0 smell detectabili.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P1-NS1
    
    Oracolo: Warning "Non sono presenti Code Smell detectabili tramite LLM, 
             l'analisi procederà in modo statico" + analisi statica con smell misti.
    """
    # Arrange: progetto con smell misti
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_44_mixed")
    
    # Mock catalog service con 0 smell
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog_no_smells
    mock_service.list_detectable_smells.return_value = []
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    # Configura paths
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: provider locale, ma 0 smell disponibili
    _configure_llm_gui(gui_app, provider_type="local", provider_index=0)
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert
    assert "Warning: Non sono presenti Code Smell detectabili tramite LLM" in out
    assert "l'analisi procederà in modo statico" in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ==================== TC_4.45 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_45_llm_no_smell_selected(
    tk_root, project_factory, tmp_output_dir, fixtures_root,
    capsys, monkeypatch, repo_root, mock_llm_catalog
):
    """
    TC_4.45: LLM enabled, smell disponibili ma 0 selezionati.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P1-NumP2-NS2-SS1
    
    Oracolo: Error "Please select at least one code smell."
    """
    # Arrange
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_45_mixed")
    
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog
    mock_service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog.smells if s.is_ready_for_detection()
    ]
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: provider locale, smell disponibili ma NESSUNO selezionato
    _configure_llm_gui(gui_app, provider_type="local", provider_index=0, smell_indices=[])
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert
    assert "Error: Please select at least one code smell." in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ==================== TC_4.46 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_46_llm_one_smell_selected_local(
    tk_root, project_factory, tmp_output_dir, fixtures_root,
    capsys, monkeypatch, repo_root, mock_llm_catalog
):
    """
    TC_4.46: LLM enabled con provider locale e 1 smell selezionato.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P1-NumP2-NS2-SS2
    
    Oracolo: Analisi completa (statica + LLM locale) con smell misti.
    """
    # Arrange
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_46_mixed")
    
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog
    mock_service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog.smells if s.is_ready_for_detection()
    ]
    
    # Mock orchestrator e provider per evitare chiamate LLM reali
    mock_orchestrator = MagicMock()
    mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=2, smells_processed=1, prompts_sent=1))
    
    mock_provider = MagicMock()
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LocalLLMProvider",
        lambda *args, **kwargs: mock_provider
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMOrchestrator",
        lambda *args, **kwargs: mock_orchestrator
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: provider locale, 1 smell selezionato
    _configure_llm_gui(gui_app, provider_type="local", provider_index=0, smell_indices=[0])
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert: analisi completata con successo
    assert "Error:" not in out or "Error" not in out.split("Warning:")[0]
    assert "Starting LLM Detection" in out
    assert "LLM Detection completed" in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ==================== TC_4.47 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_47_llm_two_smells_selected_local(
    tk_root, project_factory, tmp_output_dir, fixtures_root,
    capsys, monkeypatch, repo_root, mock_llm_catalog
):
    """
    TC_4.47: LLM enabled con provider locale e 2 smell selezionati.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P1-NumP2-NS2-SS3
    
    Oracolo: Analisi completa (statica + LLM locale) con smell misti.
    """
    # Arrange
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_47_mixed")
    
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog
    mock_service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog.smells if s.is_ready_for_detection()
    ]
    
    # Mock orchestrator e provider per evitare chiamate LLM reali
    mock_orchestrator = MagicMock()
    mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=2, smells_processed=2, prompts_sent=2))
    
    mock_provider = MagicMock()
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LocalLLMProvider",
        lambda *args, **kwargs: mock_provider
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMOrchestrator",
        lambda *args, **kwargs: mock_orchestrator
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: provider locale, 2 smell selezionati
    _configure_llm_gui(gui_app, provider_type="local", provider_index=0, smell_indices=[0, 1])
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert: analisi completata con successo
    assert "Error:" not in out or "Error" not in out.split("Warning:")[0]
    assert "Starting LLM Detection" in out
    assert "LLM Detection completed" in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ==================== TC_4.48 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_48_llm_one_smell_selected_api(
    tk_root, project_factory, tmp_output_dir, fixtures_root,
    capsys, monkeypatch, repo_root, mock_llm_catalog
):
    """
    TC_4.48: LLM enabled con provider API e 1 smell selezionato.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P2-NumP2-NS2-SS2
    
    Oracolo: Analisi completa (statica + LLM API) con smell misti.
    """
    # Arrange
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_48_mixed")
    
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog
    mock_service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog.smells if s.is_ready_for_detection()
    ]
    
    # Mock orchestrator e provider per evitare chiamate LLM reali
    mock_orchestrator = MagicMock()
    mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=2, smells_processed=1, prompts_sent=1))
    
    mock_api_provider = MagicMock()
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.ApiLLMProvider",
        lambda *args, **kwargs: mock_api_provider
    )
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMOrchestrator",
        lambda *args, **kwargs: mock_orchestrator
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: provider API, 1 smell selezionato
    _configure_llm_gui(gui_app, provider_type="api", provider_index=0, smell_indices=[0])
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert: analisi completata con successo
    assert "Error:" not in out or "Error" not in out.split("Warning:")[0]
    assert "Starting LLM Detection" in out
    assert "LLM Detection completed" in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))


# ==================== TC_4.49 ====================
@pytest.mark.usefixtures("force_sync_threads")
def test_TC_4_49_llm_no_providers_available(
    tk_root, project_factory, tmp_output_dir, fixtures_root,
    capsys, monkeypatch, repo_root, mock_llm_catalog_no_providers
):
    """
    TC_4.49: LLM enabled ma 0 provider locali disponibili.
    
    Test Frame: FI2-NF3-EF1-NP1-FO2-M2-P2-NW2-R2-NCS2-TCS3-LD1-P1-NumP1-NS2-SS2
    
    Oracolo: Error "Please select an LLM provider."
    """
    # Arrange
    import sys
    sys.path.insert(0, str(repo_root))
    
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_4_49_mixed")
    
    from gui.code_smell_detector_gui import CodeSmellDetectorGUI
    
    mock_service = MagicMock()
    mock_service.load.return_value = mock_llm_catalog_no_providers
    mock_service.list_detectable_smells.return_value = [
        s for s in mock_llm_catalog_no_providers.smells if s.is_ready_for_detection()
    ]
    
    monkeypatch.setattr(
        "gui.code_smell_detector_gui.LLMCatalogService",
        lambda: mock_service
    )
    monkeypatch.setattr(CodeSmellDetectorGUI, "configure_stdout", lambda self: None)
    
    gui_app = CodeSmellDetectorGUI(tk_root)
    
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "1")
    
    # Configura opzioni (M2-P2-R2 dal test frame)
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)
    
    # Configura LLM: nessun provider disponibile ma smell detectabili presenti
    gui_app.llm_var.set(True)
    gui_app.toggle_llm_controls()
    gui_app.provider_type_var.set("local")
    gui_app.update_provider_list()
    # Seleziona 1 smell come da test frame SS2 (anche se poi fallisce per mancanza provider)
    _select_smells(gui_app, [0])
    
    # Act
    out = _run(gui_app, capsys)
    
    # Assert
    assert "Error: Please select an LLM provider." in out
    
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
