"""
White-box unit tests for LLM detection functionality in CodeSmellDetectorGUI.
Tests cover all decision branches introduced by CR01 modifications.

Test Strategy: White-box testing with branch coverage focus
Coverage Target: All conditional branches in LLM-related methods
"""

import os
import tkinter as tk
from unittest.mock import MagicMock, Mock, patch, call
import pytest

from gui.code_smell_detector_gui import CodeSmellDetectorGUI
from llm_detection.types import (
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    ProviderKind,
)


@pytest.fixture
def mock_catalog_service():
    """Mock LLMCatalogService for testing."""
    service = MagicMock()
    
    # Default catalog with providers and smells
    catalog = LLMCatalog(
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
            LLMSmellDefinition(
                smell_id="disabled_smell",
                display_name="Disabled Smell",
                description="Disabled",
                default_prompt="",
                enabled=False,
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
                provider_id="local-gemma",
                kind=ProviderKind.LOCAL,
                display_name="Gemma Local",
                config={"host": "http://localhost:11434", "model_name": "gemma3:4b"},
            ),
            LLMProviderDefinition(
                provider_id="api-openai",
                kind=ProviderKind.API,
                display_name="OpenAI API",
                config={"base_url": "https://api.openai.com"},
            ),
        ],
    )
    
    service.load.return_value = catalog
    service.list_detectable_smells.return_value = [
        s for s in catalog.smells if s.is_ready_for_detection()
    ]
    service.get_provider.side_effect = lambda pid: next(
        (p for p in catalog.providers if p.provider_id == pid), None
    )
    
    return service


@pytest.fixture
def gui(mock_catalog_service):
    """Create GUI instance with mocked dependencies."""
    with patch('gui.code_smell_detector_gui.LLMCatalogService', return_value=mock_catalog_service):
        root = tk.Tk()
        gui_instance = CodeSmellDetectorGUI(root)
        yield gui_instance
        try:
            root.destroy()
        except:
            pass


class TestToggleLLMControls:
    """Test toggle_llm_controls() method - Branch coverage for frame visibility."""
    
    def test_toggle_llm_controls_enable(self, gui):
        """Branch: llm_var.get() == True -> grid() called"""
        # Arrange
        gui.llm_var.set(False)
        gui.llm_frame.grid_remove()
        
        # Act
        gui.llm_var.set(True)
        gui.toggle_llm_controls()
        gui.master.update_idletasks()  # Force update
        
        # Assert - widget is gridded (not removed)
        # Note: winfo_ismapped() may return 0 in test environment, check grid_info instead
        assert gui.llm_frame.grid_info() != {}
    
    def test_toggle_llm_controls_disable(self, gui):
        """Branch: llm_var.get() == False -> grid_remove() called"""
        # Arrange
        gui.llm_var.set(True)
        gui.llm_frame.grid()
        gui.master.update_idletasks()
        
        # Act
        gui.llm_var.set(False)
        gui.toggle_llm_controls()
        gui.master.update_idletasks()
        
        # Assert - widget is not gridded (removed)
        assert gui.llm_frame.grid_info() == {}


class TestUpdateProviderList:
    """Test update_provider_list() method - Branch coverage for provider filtering."""
    
    def test_update_provider_list_local(self, gui):
        """Branch: provider_type == 'local' -> filter LOCAL providers"""
        # Arrange
        gui.provider_type_var.set("local")
        
        # Act
        gui.update_provider_list()
        
        # Assert
        providers = gui.provider_combo['values']
        assert len(providers) == 2
        assert "Ollama Local" in providers
        assert "Gemma Local" in providers
        assert "OpenAI API" not in providers
    
    def test_update_provider_list_api(self, gui):
        """Branch: provider_type == 'api' -> filter API providers"""
        # Arrange
        gui.provider_type_var.set("api")
        
        # Act
        gui.update_provider_list()
        
        # Assert
        providers = gui.provider_combo['values']
        assert len(providers) == 1
        assert "OpenAI API" in providers
        assert "Ollama Local" not in providers
    
    def test_update_provider_list_with_providers_sets_first(self, gui):
        """Branch: if provider_names -> current(0)"""
        # Arrange
        gui.provider_type_var.set("local")
        
        # Act
        gui.update_provider_list()
        
        # Assert
        assert gui.provider_combo.current() == 0
    
    def test_update_provider_list_no_providers_clears_selection(self, gui, mock_catalog_service):
        """Branch: else (no providers) -> set('')"""
        # Arrange
        gui.catalog.providers = []  # No providers
        gui.provider_type_var.set("local")
        
        # Act
        gui.update_provider_list()
        
        # Assert
        assert gui.provider_combo.get() == ''
    
    def test_update_provider_list_exception_handling(self, gui):
        """Branch: except Exception -> error printed"""
        # Arrange
        gui.catalog = None  # Force exception
        
        # Act & Assert (should not raise, only print)
        with patch('builtins.print') as mock_print:
            gui.update_provider_list()
            assert any('Error updating provider list' in str(call) for call in mock_print.call_args_list)


class TestUpdateSmellList:
    """Test update_smell_list() method - Branch coverage for smell population."""
    
    def test_update_smell_list_populates_listbox(self, gui, mock_catalog_service):
        """Branch: Success path -> smells added to listbox"""
        # Act
        gui.update_smell_list()
        
        # Assert
        assert gui.smell_listbox.size() == 2  # Only enabled smells
        items = [gui.smell_listbox.get(i) for i in range(gui.smell_listbox.size())]
        assert "Test Smell 1" in items
        assert "Test Smell 2" in items
        assert "Disabled Smell" not in items
    
    def test_update_smell_list_exception_handling(self, gui, mock_catalog_service):
        """Branch: except Exception -> error printed"""
        # Arrange
        mock_catalog_service.list_detectable_smells.side_effect = Exception("Catalog error")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.update_smell_list()
            assert any('Error updating smell list' in str(call) for call in mock_print.call_args_list)


class TestRunProgramValidation:
    """Test run_program() validation logic - Branch coverage for input validation."""
    
    def test_run_program_missing_input_path(self, gui):
        """Branch: input_path == 'No path selected' -> error"""
        # Arrange
        gui.input_path.configure(text="No path selected")
        gui.output_path.configure(text="/valid/path")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('Please select valid input' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_missing_output_path(self, gui):
        """Branch: output_path == 'No path selected' -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/path")
        gui.output_path.configure(text="No path selected")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('Please select valid input' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_invalid_num_walkers_non_numeric(self, gui):
        """Branch: ValueError on num_walkers -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "invalid")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('max_workers must be a number' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_invalid_num_walkers_zero(self, gui):
        """Branch: num_walkers <= 0 -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "0")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('max_workers must be greater than 0' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_invalid_num_walkers_negative(self, gui):
        """Branch: num_walkers <= 0 -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "-5")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('max_workers must be greater than 0' in str(call) for call in mock_print.call_args_list)


class TestRunProgramLLMValidation:
    """Test run_program() LLM-specific validation - Branch coverage for LLM path."""
    
    def test_run_program_llm_enabled_no_provider_selected(self, gui):
        """Branch: use_llm and not provider_name -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_combo.set('')  # No provider
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('Please select an LLM provider' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_llm_enabled_provider_not_found(self, gui):
        """Branch: use_llm and not matching_providers -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_combo.set("Non-existent Provider")
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('Provider' in str(call) and 'not found' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_llm_enabled_no_smells_selected_with_detectable_smells(self, gui, mock_catalog_service):
        """Branch: use_llm and not selected_indices and detectable_smells exist -> error"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_type_var.set("local")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        # Don't select any smells
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            gui.run_program()
            assert any('Please select at least one code smell' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_llm_enabled_no_smells_selected_no_detectable_smells(self, gui, mock_catalog_service):
        """Branch: use_llm and not selected_indices and no detectable_smells -> warning, disable LLM"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_type_var.set("local")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        
        # Mock no detectable smells
        mock_catalog_service.list_detectable_smells.return_value = []
        gui.update_smell_list()  # Refresh
        
        # Act & Assert
        with patch('builtins.print') as mock_print:
            with patch.object(gui, 'run_analysis'):  # Don't actually run
                gui.run_program()
                assert any('Non sono presenti Code Smell detectabili' in str(call) for call in mock_print.call_args_list)
    
    def test_run_program_llm_enabled_smells_selected_success(self, gui):
        """Branch: use_llm and selected_indices -> success path"""
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_type_var.set("local")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        gui.smell_listbox.selection_set(0)  # Select first smell
        
        # Act & Assert
        with patch('threading.Thread') as mock_thread:
            gui.run_program()
            mock_thread.assert_called_once()
            args = mock_thread.call_args[1]['args']
            assert args[6] == True  # use_llm
            assert args[7] is not None  # llm_provider_id
            assert len(args[8]) > 0  # selected_smell_ids
    
    def test_run_program_smell_not_in_catalog(self, gui, mock_catalog_service):
        """Branch: if matching_smell (line 351) -> None path not covered yet
        
        This test covers the edge case where a smell in the listbox
        doesn't match any smell in the catalog (should never happen in practice,
        but tests the defensive programming).
        """
        # Arrange
        gui.input_path.configure(text="/valid/input")
        gui.output_path.configure(text="/valid/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(True)
        gui.provider_type_var.set("local")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        
        # Manually insert a smell that won't be in catalog
        gui.smell_listbox.insert(tk.END, "Non-Existent Smell")
        gui.smell_listbox.selection_set(gui.smell_listbox.size() - 1)
        
        # Mock empty detectable_smells for the lookup
        mock_catalog_service.list_detectable_smells.return_value = []
        
        # Act
        with patch('threading.Thread') as mock_thread:
            gui.run_program()
            
            # Assert - thread should still be called but with empty smell_ids
            mock_thread.assert_called_once()
            args = mock_thread.call_args[1]['args']
            assert args[8] == []  # selected_smell_ids should be empty


class TestRunAnalysisLLMExecution:
    """Test run_analysis() LLM execution - Branch coverage for LLM detection invocation."""
    
    def test_run_analysis_without_llm(self, gui):
        """Branch: not use_llm -> skip LLM detection"""
        # Arrange
        with patch.object(gui, '_run_llm_detection') as mock_llm:
            with patch.object(gui, 'project_analyzer') as mock_analyzer:
                mock_analyzer.analyze_project.return_value = 5
                
                # Act
                gui.run_analysis(
                    input_path="/test/input",
                    output_path="/test/output",
                    num_walkers=2,
                    is_parallel=False,
                    is_resume=False,
                    is_multiple=False,
                    use_llm=False,
                    llm_provider_id=None,
                    selected_smell_ids=None,
                )
                
                # Assert
                mock_llm.assert_not_called()
    
    def test_run_analysis_with_llm_calls_detection(self, gui):
        """Branch: use_llm and provider_id and smell_ids -> _run_llm_detection called"""
        # Arrange
        with patch.object(gui, '_run_llm_detection') as mock_llm:
            # Mock ProjectAnalyzer class to avoid file system checks
            with patch('gui.code_smell_detector_gui.ProjectAnalyzer') as mock_analyzer_class:
                mock_analyzer = MagicMock()
                mock_analyzer.analyze_project.return_value = 5
                mock_analyzer_class.return_value = mock_analyzer
                
                # Act
                gui.run_analysis(
                    input_path="/test/input",
                    output_path="/test/output",
                    num_walkers=2,
                    is_parallel=False,
                    is_resume=False,
                    is_multiple=False,
                    use_llm=True,
                    llm_provider_id="local-ollama",
                    selected_smell_ids=["test_smell_1"],
                )
                
                # Assert
                mock_llm.assert_called_once_with(
                    "/test/input",
                    "/test/output",
                    "local-ollama",
                    ["test_smell_1"]
                )


class TestRunLLMDetection:
    """Test _run_llm_detection() - Branch coverage for provider creation and execution."""
    
    def test_run_llm_detection_local_provider(self, gui, mock_catalog_service):
        """Branch: provider_def.kind == LOCAL -> LocalLLMProvider created"""
        # Arrange
        mock_provider = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=1, smells_processed=1, prompts_sent=1))
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider', return_value=mock_provider) as mock_local:
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator):
                with patch('os.walk', return_value=[("/test", [], ["file.py"])]):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        
                        # Act
                        gui._run_llm_detection(
                            input_path="/test",
                            output_path="/output",
                            provider_id="local-ollama",
                            smell_ids=["test_smell_1"]
                        )
                        
                        # Assert
                        mock_local.assert_called_once()
                        assert mock_local.call_args[1]['model_name'] == "qwen2.5-coder:7b"
    
    def test_run_llm_detection_api_provider(self, gui, mock_catalog_service):
        """Branch: provider_def.kind == API -> ApiLLMProvider created"""
        # Arrange
        mock_provider = MagicMock()
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=1, smells_processed=1, prompts_sent=1))
        
        with patch('gui.code_smell_detector_gui.ApiLLMProvider', return_value=mock_provider) as mock_api:
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator):
                with patch('os.walk', return_value=[("/test", [], ["file.py"])]):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        
                        # Act
                        gui._run_llm_detection(
                            input_path="/test",
                            output_path="/output",
                            provider_id="api-openai",
                            smell_ids=["test_smell_1"]
                        )
                        
                        # Assert
                        mock_api.assert_called_once()
                        assert mock_api.call_args[1]['base_url'] == "https://api.openai.com"
    
    def test_run_llm_detection_single_file_input(self, gui, mock_catalog_service):
        """Branch: os.path.isfile(input_path) -> single file target"""
        # Arrange
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=1, smells_processed=1, prompts_sent=1))
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator) as mock_orch_cls:
                with patch('os.path.isfile', return_value=True):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        
                        # Act
                        gui._run_llm_detection(
                            input_path="/test/file.py",
                            output_path="/output",
                            provider_id="local-ollama",
                            smell_ids=["test_smell_1"]
                        )
                        
                        # Assert
                        orchestrator = mock_orch_cls.return_value
                        call_args = orchestrator.detect.call_args[1]
                        targets = call_args['targets']
                        assert len(targets) == 1
                        assert targets[0].filename == "file.py"
    
    def test_run_llm_detection_directory_walk(self, gui, mock_catalog_service):
        """Branch: else (directory) -> os.walk for multiple files"""
        # Arrange
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = ([], MagicMock(targets_processed=2, smells_processed=1, prompts_sent=2))
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator) as mock_orch_cls:
                with patch('os.path.isfile', return_value=False):
                    with patch('os.walk', return_value=[
                        ("/test", [], ["file1.py", "file2.py", "other.txt"])
                    ]):
                        with patch('builtins.open', create=True) as mock_open:
                            mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                            
                            # Act
                            gui._run_llm_detection(
                                input_path="/test",
                                output_path="/output",
                                provider_id="local-ollama",
                                smell_ids=["test_smell_1"]
                            )
                            
                            # Assert
                            orchestrator = mock_orch_cls.return_value
                            call_args = orchestrator.detect.call_args[1]
                            targets = call_args['targets']
                            assert len(targets) == 2  # Only .py files
    
    def test_run_llm_detection_no_python_files(self, gui, mock_catalog_service):
        """Branch: not targets -> early return with message"""
        # Arrange
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator'):
                with patch('os.path.isfile', return_value=False):
                    with patch('os.walk', return_value=[("/test", [], ["file.txt"])]):
                        
                        # Act & Assert
                        with patch('builtins.print') as mock_print:
                            gui._run_llm_detection(
                                input_path="/test",
                                output_path="/output",
                                provider_id="local-ollama",
                                smell_ids=["test_smell_1"]
                            )
                            assert any('No Python files found' in str(call) for call in mock_print.call_args_list)
    
    def test_run_llm_detection_with_findings(self, gui, mock_catalog_service):
        """Branch: if findings -> save results"""
        # Arrange
        from llm_detection.types import LLMSmellFinding
        mock_findings = [
            MagicMock(
                filename="test.py",
                smell_id="test_smell_1",
                line=10,
                to_overview_row=MagicMock(return_value={"file": "test.py", "line": 10})
            )
        ]
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = (
            mock_findings,
            MagicMock(targets_processed=1, smells_processed=1, prompts_sent=1)
        )
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator):
                with patch('os.walk', return_value=[("/test", [], ["file.py"])]):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        with patch.object(gui, '_save_llm_findings') as mock_save:
                            
                            # Act
                            gui._run_llm_detection(
                                input_path="/test",
                                output_path="/output",
                                provider_id="local-ollama",
                                smell_ids=["test_smell_1"]
                            )
                            
                            # Assert
                            mock_save.assert_called_once_with(mock_findings, "/output")
    
    def test_run_llm_detection_no_findings(self, gui, mock_catalog_service):
        """Branch: else (no findings) -> message only"""
        # Arrange
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = (
            [],
            MagicMock(targets_processed=1, smells_processed=1, prompts_sent=1)
        )
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator):
                with patch('os.walk', return_value=[("/test", [], ["file.py"])]):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        with patch.object(gui, '_save_llm_findings') as mock_save:
                            with patch('builtins.print') as mock_print:
                                
                                # Act
                                gui._run_llm_detection(
                                    input_path="/test",
                                    output_path="/output",
                                    provider_id="local-ollama",
                                    smell_ids=["test_smell_1"]
                                )
                                
                                # Assert
                                mock_save.assert_not_called()
                                assert any('No LLM findings detected' in str(call) for call in mock_print.call_args_list)
    
    def test_run_llm_detection_exception_handling(self, gui, mock_catalog_service):
        """Branch: except Exception -> error printed with traceback"""
        # Arrange
        with patch('gui.code_smell_detector_gui.LocalLLMProvider', side_effect=Exception("Provider error")):
            with patch('builtins.print') as mock_print:
                with patch('traceback.print_exc') as mock_traceback:
                    
                    # Act
                    gui._run_llm_detection(
                        input_path="/test",
                        output_path="/output",
                        provider_id="local-ollama",
                        smell_ids=["test_smell_1"]
                    )
                    
                    # Assert
                    assert any('Error during LLM detection' in str(call) for call in mock_print.call_args_list)
                    mock_traceback.assert_called_once()
    
    def test_run_llm_detection_non_python_file(self, gui, mock_catalog_service):
        """Branch: if input_path.endswith('.py') (line 511) -> False path
        
        Tests that when a single non-Python file is provided as input,
        no targets are created and appropriate message is shown.
        """
        # Arrange
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator'):
                with patch('os.path.isfile', return_value=True):
                    with patch('builtins.print') as mock_print:
                        
                        # Act
                        gui._run_llm_detection(
                            input_path="/test/data.txt",  # Non-Python file
                            output_path="/output",
                            provider_id="local-ollama",
                            smell_ids=["test_smell_1"]
                        )
                        
                        # Assert
                        assert any('No Python files found' in str(call) for call in mock_print.call_args_list)
    
    def test_run_llm_detection_multiple_findings_same_file(self, gui, mock_catalog_service):
        """Branch: if filename not in findings_by_file (line 559) -> False path
        
        Tests that when multiple findings are detected in the same file,
        they are correctly aggregated in findings_by_file dictionary.
        """
        # Arrange
        mock_finding1 = MagicMock(
            filename="test.py",
            smell_id="smell_1",
            line=10,
            to_overview_row=MagicMock(return_value={"file": "test.py", "line": 10})
        )
        mock_finding2 = MagicMock(
            filename="test.py",  # Same file
            smell_id="smell_2",
            line=20,
            to_overview_row=MagicMock(return_value={"file": "test.py", "line": 20})
        )
        
        mock_orchestrator = MagicMock()
        mock_orchestrator.detect.return_value = (
            [mock_finding1, mock_finding2],
            MagicMock(targets_processed=1, smells_processed=2, prompts_sent=2)
        )
        
        with patch('gui.code_smell_detector_gui.LocalLLMProvider'):
            with patch('gui.code_smell_detector_gui.LLMOrchestrator', return_value=mock_orchestrator):
                with patch('os.walk', return_value=[("/test", [], ["file.py"])]):
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = "print('test')"
                        with patch.object(gui, '_save_llm_findings') as mock_save:
                            with patch('builtins.print') as mock_print:
                                
                                # Act
                                gui._run_llm_detection(
                                    input_path="/test",
                                    output_path="/output",
                                    provider_id="local-ollama",
                                    smell_ids=["smell_1", "smell_2"]
                                )
                                
                                # Assert - verify both findings are reported for same file
                                assert any('test.py: 2 code smell(s)' in str(call) for call in mock_print.call_args_list)


class TestLoadLLMData:
    """Test load_llm_data() - Branch coverage for initialization."""
    
    def test_load_llm_data_success(self, gui):
        """Branch: Success path -> catalog loaded, lists updated"""
        # Arrange is done in fixture
        
        # Assert
        assert gui.catalog is not None
        assert gui.provider_combo['values']
        assert gui.smell_listbox.size() > 0
    
    def test_load_llm_data_exception(self, mock_catalog_service):
        """Branch: except Exception -> warning printed"""
        # Arrange
        mock_catalog_service.load.side_effect = Exception("Catalog load error")
        
        with patch('gui.code_smell_detector_gui.LLMCatalogService', return_value=mock_catalog_service):
            with patch('builtins.print') as mock_print:
                root = tk.Tk()
                
                # Act
                gui = CodeSmellDetectorGUI(root)
                
                # Assert
                assert any('Could not load LLM catalog' in str(call) for call in mock_print.call_args_list)
                
                root.destroy()


class TestIntegrationScenarios:
    """Integration tests for complete user scenarios."""
    
    def test_complete_llm_detection_workflow_local_provider(self, gui):
        """Integration: Complete workflow with local provider"""
        # Arrange
        gui.input_path.configure(text="/test/input")
        gui.output_path.configure(text="/test/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        
        # Enable LLM
        gui.llm_var.set(True)
        gui.toggle_llm_controls()
        
        # Select local provider
        gui.provider_type_var.set("local")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        
        # Select smell
        gui.smell_listbox.selection_set(0)
        
        # Act & Assert
        with patch('threading.Thread') as mock_thread:
            gui.run_program()
            
            # Verify thread started with correct parameters
            mock_thread.assert_called_once()
            args = mock_thread.call_args[1]['args']
            assert args[6] == True  # use_llm
            assert args[7] == "local-ollama"  # provider_id
            assert "test_smell_1" in args[8]  # smell_ids
    
    def test_complete_llm_detection_workflow_api_provider(self, gui):
        """Integration: Complete workflow with API provider"""
        # Arrange
        gui.input_path.configure(text="/test/input")
        gui.output_path.configure(text="/test/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        
        # Enable LLM
        gui.llm_var.set(True)
        gui.toggle_llm_controls()
        
        # Select API provider
        gui.provider_type_var.set("api")
        gui.update_provider_list()
        gui.provider_combo.current(0)
        
        # Select smell
        gui.smell_listbox.selection_set(0)
        
        # Act & Assert
        with patch('threading.Thread') as mock_thread:
            gui.run_program()
            
            # Verify thread started with API provider
            mock_thread.assert_called_once()
            args = mock_thread.call_args[1]['args']
            assert args[6] == True  # use_llm
            assert args[7] == "api-openai"  # provider_id
    
    def test_llm_disabled_standard_analysis_only(self, gui):
        """Integration: LLM disabled -> standard analysis only"""
        # Arrange
        gui.input_path.configure(text="/test/input")
        gui.output_path.configure(text="/test/output")
        gui.walker_picker.delete(0, tk.END)
        gui.walker_picker.insert(0, "2")
        gui.llm_var.set(False)
        
        # Act & Assert
        with patch('threading.Thread') as mock_thread:
            gui.run_program()
            
            # Verify LLM parameters are False/None
            mock_thread.assert_called_once()
            args = mock_thread.call_args[1]['args']
            assert args[6] == False  # use_llm
            assert args[7] is None  # provider_id
            assert args[8] == []  # smell_ids
