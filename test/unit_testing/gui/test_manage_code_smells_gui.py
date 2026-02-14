"""
Unit tests for manage_code_smells_gui.py with maximum code coverage.
Tests both ManageCodeSmellsGUI and AddSmellDialog classes.
Uses comprehensive mocking to avoid Tkinter initialization issues.
"""
import unittest
from unittest.mock import MagicMock, Mock, patch, call, PropertyMock

from gui.manage_code_smells_gui import ManageCodeSmellsGUI, AddSmellDialog
from llm_detection.catalog_service import CatalogValidationError, LLMCatalogService
from llm_detection.types import LLMCatalog, LLMSmellDefinition


class TestManageCodeSmellsGUI(unittest.TestCase):
    """Test suite for ManageCodeSmellsGUI class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock Tkinter completely to avoid initialization issues
        self.mock_root = MagicMock()
        
        # Create mock catalog service
        self.mock_catalog_service = MagicMock(spec=LLMCatalogService)
        
        # Create sample smells for testing
        self.smell1 = LLMSmellDefinition(
            smell_id="smell-1",
            display_name="Test Smell 1",
            description="Description 1",
            default_prompt="",
            draft_prompt=None,
            created_by_user=True,
            enabled=False
        )
        
        self.smell2 = LLMSmellDefinition(
            smell_id="smell-2",
            display_name="Another Smell",
            description="Description 2",
            default_prompt="",
            draft_prompt=None,
            created_by_user=True,
            enabled=False
        )
        
        self.mock_catalog = LLMCatalog(
            schema_version=1,
            smells=[self.smell1, self.smell2],
            providers=[]
        )
        
        self.mock_catalog_service.load.return_value = self.mock_catalog

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_init_with_smells(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test GUI initialization with smells in catalog."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Verify service was called to load catalog
        self.mock_catalog_service.load.assert_called()
        
        # Verify smell mapping was created
        self.assertEqual(len(gui._smell_display_to_id), 2)
        self.assertIn("Another Smell", gui._smell_display_to_id)
        self.assertIn("Test Smell 1", gui._smell_display_to_id)

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_init_empty_catalog(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test GUI initialization with empty catalog."""
        self.mock_catalog.smells = []
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Verify controls are disabled when no smells
        self.assertIsNone(gui._current_smell_id)

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_load_smells_error_handling(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test error handling when loading smells fails."""
        error_service = MagicMock(spec=LLMCatalogService)
        error_service.load.side_effect = Exception("Load error")
        
        gui = ManageCodeSmellsGUI(self.mock_root, error_service)
        
        # Verify error messagebox was shown
        mock_msgbox.showerror.assert_called_once()
        args = mock_msgbox.showerror.call_args[0]
        self.assertIn("Errore Caricamento", args[0])
        self.assertIn("Load error", args[1])

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_smell_selection(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test selecting a smell from dropdown."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Mock combo get
        gui._smell_combo.get.return_value = "Test Smell 1"
        gui._on_smell_selected()
        
        # Verify current smell was set
        self.assertEqual(gui._current_smell_id, "smell-1")

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_smell_selection_empty(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test selecting nothing from dropdown."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        gui._smell_combo.get.return_value = ""
        gui._on_smell_selected()
        
        # Should return early without error

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_smell_selection_invalid_id(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test selecting smell with invalid ID."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Add a smell to display map but break the ID
        gui._smell_display_to_id["Invalid"] = None
        gui._smell_combo.get.return_value = "Invalid"
        gui._on_smell_selected()
        
        # Should return early without error

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_smell_selection_error(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test error handling when smell details can't be loaded."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Mock catalog.get_smell to raise error
        self.mock_catalog.get_smell = MagicMock(side_effect=KeyError("Not found"))
        
        gui._smell_combo.get.return_value = "Test Smell 1"
        gui._on_smell_selected()
        
        # Verify error was shown
        mock_msgbox.showerror.assert_called()

    @patch('gui.manage_code_smells_gui.AddSmellDialog')
    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_on_add_smell(self, mock_st, mock_tk, mock_ttk, mock_msgbox, mock_dialog):
        """Test opening add smell dialog."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        gui._on_add_smell()
        
        # Verify dialog was created
        mock_dialog.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_on_smell_added_callback(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test callback after adding a smell."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Add a new smell to the mock catalog
        new_smell = LLMSmellDefinition(
            smell_id="new-smell-id",
            display_name="New Smell",
            description="New description",
            default_prompt="",
            draft_prompt=None,
            created_by_user=True,
            enabled=False
        )
        self.mock_catalog.smells.append(new_smell)
        
        # Call callback with new smell id
        gui._on_smell_added_callback("new-smell-id")
        
        # Verify reload was triggered
        self.assertTrue(self.mock_catalog_service.load.call_count > 1)
        
        # Verify dropdown was updated
        self.assertIn("new-smell-id", gui._smell_display_to_id.values())

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_remove_smell_no_selection(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test remove without selection does nothing."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = None
        
        gui._on_remove_smell()
        
        # Verify no messagebox shown
        mock_msgbox.askyesno.assert_not_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_remove_smell_confirmed(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test removing smell when user confirms."""
        mock_msgbox.askyesno.return_value = True
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        original_smell_id = "smell-1"
        gui._current_smell_id = original_smell_id
        gui._name_var.set = MagicMock()
        gui._name_var.get = MagicMock(return_value="Test Smell 1")
        
        # Remove smell-1 from catalog to simulate deletion
        self.mock_catalog.smells = [self.smell2]
        
        gui._on_remove_smell()
        
        # Verify confirmation dialog shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify smell was removed
        self.mock_catalog_service.remove_smell.assert_called_once_with(original_smell_id)
        
        # Verify success message shown
        mock_msgbox.showinfo.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_remove_smell_cancelled(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test removing smell when user cancels."""
        mock_msgbox.askyesno.return_value = False
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        gui._name_var.get = MagicMock(return_value="Test Smell 1")
        
        gui._on_remove_smell()
        
        # Verify confirmation dialog shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify smell was NOT removed
        self.mock_catalog_service.remove_smell.assert_not_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_remove_smell_error(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test error handling when removing smell fails."""
        mock_msgbox.askyesno.return_value = True
        self.mock_catalog_service.remove_smell.side_effect = Exception("Delete error")
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        gui._name_var.get = MagicMock(return_value="Test Smell 1")
        
        gui._on_remove_smell()
        
        # Verify error shown
        mock_msgbox.showerror.assert_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_no_selection(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save without selection does nothing."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = None
        
        gui._on_save_changes()
        
        # Should return early
        mock_msgbox.assert_not_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_empty_description(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save with empty description shows warning."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        # Mock empty description
        gui._desc_text.get = MagicMock(return_value="   ")
        
        gui._on_save_changes()
        
        # Verify warning shown
        mock_msgbox.showwarning.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_no_change(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save when description hasn't changed."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        # Mock description same as original
        gui._desc_text.get = MagicMock(return_value="Description 1")
        
        gui._on_save_changes()
        
        # Verify info shown
        mock_msgbox.showinfo.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_load_error(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save when loading original description fails."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        # Mock load to fail on second call
        self.mock_catalog_service.load.side_effect = Exception("Load error")
        
        gui._desc_text.get = MagicMock(return_value="New description")
        
        gui._on_save_changes()
        
        # Verify error shown
        mock_msgbox.showerror.assert_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_confirmed(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save when user confirms changes."""
        mock_msgbox.askyesno.return_value = True
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        # Mock changed description
        gui._desc_text.get = MagicMock(return_value="New description")
        
        gui._on_save_changes()
        
        # Verify confirmation shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify update called
        self.mock_catalog_service.update_smell_description.assert_called_once_with(
            "smell-1", "New description"
        )

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_cancelled(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test save when user cancels changes."""
        mock_msgbox.askyesno.return_value = False
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        # Mock changed description
        gui._desc_text.get = MagicMock(return_value="New description")
        gui._desc_text.delete = MagicMock()
        gui._desc_text.insert = MagicMock()
        
        gui._on_save_changes()
        
        # Verify confirmation shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify update NOT called
        self.mock_catalog_service.update_smell_description.assert_not_called()
        
        # Verify description reverted
        gui._desc_text.delete.assert_called_once()
        gui._desc_text.insert.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_save_changes_error(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test error handling when save fails."""
        mock_msgbox.askyesno.return_value = True
        self.mock_catalog_service.update_smell_description.side_effect = Exception("Save error")
        
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        gui._current_smell_id = "smell-1"
        
        gui._desc_text.get = MagicMock(return_value="New description")
        
        gui._on_save_changes()
        
        # Verify error shown
        self.assertTrue(len(mock_msgbox.showerror.call_args_list) > 0)

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_load_smells_preserve_selection(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test that reload preserves current selection."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Select first smell
        gui._current_smell_id = "smell-1"
        
        # Reload
        gui._load_smells_into_dropdown()
        
        # Verify selection preserved
        self.assertTrue(gui._smell_combo.set.called)

    @patch('gui.manage_code_smells_gui.messagebox')
    @patch('gui.manage_code_smells_gui.ttk')
    @patch('gui.manage_code_smells_gui.tk')
    @patch('gui.manage_code_smells_gui.ScrolledText')
    def test_load_smells_invalid_selection(self, mock_st, mock_tk, mock_ttk, mock_msgbox):
        """Test reload when current selection no longer exists."""
        gui = ManageCodeSmellsGUI(self.mock_root, self.mock_catalog_service)
        
        # Select a smell that will be removed
        gui._current_smell_id = "non-existent"
        
        # Reload
        gui._load_smells_into_dropdown()
        
        # Verify defaults to first smell
        self.assertTrue(gui._smell_combo.current.called)


class TestAddSmellDialog(unittest.TestCase):
    """Test suite for AddSmellDialog class using direct method testing."""

    def _create_dialog_instance(self):
        """Create an AddSmellDialog instance without calling __init__."""
        # Use __new__ to create instance without initialization
        dialog = AddSmellDialog.__new__(AddSmellDialog)
        
        # Manually set required attributes
        dialog.catalog_service = MagicMock(spec=LLMCatalogService)
        dialog.on_success = MagicMock()
        dialog.name_entry = MagicMock()
        dialog.desc_text = MagicMock()
        dialog.destroy = MagicMock()
        
        return dialog

    @patch('gui.manage_code_smells_gui.ScrolledText')
    @patch('gui.manage_code_smells_gui.ttk.Button')
    @patch('gui.manage_code_smells_gui.ttk.Frame')
    @patch('gui.manage_code_smells_gui.ttk.Label')
    @patch('gui.manage_code_smells_gui.ttk.Entry')
    def test_dialog_init_and_ui_build(self, mock_entry, mock_label, mock_frame, mock_button, mock_scrolled):
        """Test AddSmellDialog initialization and UI building."""
        # Create a partial mock of AddSmellDialog that allows __init__ to run
        # but mocks all Tkinter Toplevel methods
        with patch.object(AddSmellDialog, '__init__', lambda self, parent, catalog, callback: None):
            dialog = AddSmellDialog(None, None, None)
            
        # Manually set attributes as __init__ would
        dialog.catalog_service = MagicMock(spec=LLMCatalogService)
        dialog.on_success = MagicMock()
        
        # Mock self methods that Toplevel would provide  
        dialog.title = MagicMock()
        dialog.geometry = MagicMock()
        dialog.transient = MagicMock()
        dialog.grab_set = MagicMock()
        dialog.update_idletasks = MagicMock()
        dialog.winfo_width = MagicMock(return_value=400)
        dialog.winfo_height = MagicMock(return_value=300)
        dialog.winfo_x = MagicMock(return_value=100)
        dialog.winfo_y = MagicMock(return_value=100)
        
        # Setup mock returns for UI elements
        mock_frame_instance = MagicMock()
        mock_frame.return_value = mock_frame_instance
        mock_entry.return_value = MagicMock()
        mock_scrolled.return_value = MagicMock()
        
        # Call the UI building method directly (this is what __init__ calls)
        dialog._build_dialog_ui()
        
        # Verify UI components were created
        self.assertTrue(mock_frame.called, "Frame should be created")
        self.assertTrue(mock_label.called, "Labels should be created")
        self.assertTrue(mock_entry.called, "Entry should be created")
        self.assertTrue(mock_scrolled.called, "ScrolledText should be created")
        self.assertTrue(mock_button.called, "Buttons should be created")
        
        # Verify at least 2 buttons created (Save and Cancel)
        self.assertGreaterEqual(mock_button.call_count, 2, "At least 2 buttons should be created")

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_missing_name(self, mock_msgbox):
        """Test save with missing name shows warning."""
        dialog = self._create_dialog_instance()
        
        # Mock empty name
        dialog.name_entry.get.return_value = "  "
        dialog.desc_text.get.return_value = "Some description"
        
        dialog._on_save()
        
        # Verify warning shown
        mock_msgbox.showwarning.assert_called_once()
        args = mock_msgbox.showwarning.call_args[0]
        self.assertIn("nome del code smell", args[1])

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_missing_description(self, mock_msgbox):
        """Test save with missing description shows warning."""
        dialog = self._create_dialog_instance()
        
        dialog.name_entry.get.return_value = "Test Smell"
        dialog.desc_text.get.return_value = "  "
        
        dialog._on_save()
        
        # Verify warning shown
        mock_msgbox.showwarning.assert_called_once()
        args = mock_msgbox.showwarning.call_args[0]
        self.assertIn("descrizione", args[1])

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_success(self, mock_msgbox):
        """Test successful save."""
        dialog = self._create_dialog_instance()
        dialog.catalog_service.add_smell.return_value = "new-smell-id"
        
        dialog.name_entry.get.return_value = "New Smell"
        dialog.desc_text.get.return_value = "New description"
        
        dialog._on_save()
        
        # Verify service called
        dialog.catalog_service.add_smell.assert_called_once_with(
            "New Smell", "New description"
        )
        
        # Verify success message shown
        mock_msgbox.showinfo.assert_called_once()
        
        # Verify callback called
        dialog.on_success.assert_called_once_with("new-smell-id")
        
        # Verify dialog destroyed
        dialog.destroy.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_success_without_callback(self, mock_msgbox):
        """Test successful save when callback is None."""
        dialog = self._create_dialog_instance()
        dialog.catalog_service.add_smell.return_value = "new-smell-id"
        dialog.on_success = None  # No callback
        
        dialog.name_entry.get.return_value = "New Smell"
        dialog.desc_text.get.return_value = "New description"
        
        dialog._on_save()
        
        # Verify service called
        dialog.catalog_service.add_smell.assert_called_once_with(
            "New Smell", "New description"
        )
        
        # Verify success message shown
        mock_msgbox.showinfo.assert_called_once()
        
        # Verify dialog destroyed (even without callback)
        dialog.destroy.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_validation_error(self, mock_msgbox):
        """Test save with validation error."""
        dialog = self._create_dialog_instance()
        dialog.catalog_service.add_smell.side_effect = CatalogValidationError("Duplicate name")
        
        dialog.name_entry.get.return_value = "Existing Smell"
        dialog.desc_text.get.return_value = "Description"
        
        dialog._on_save()
        
        # Verify error shown
        mock_msgbox.showerror.assert_called_once()
        args = mock_msgbox.showerror.call_args[0]
        self.assertIn("Duplicate name", args[1])
        
        # Verify dialog NOT destroyed
        dialog.destroy.assert_not_called()

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_save_generic_error(self, mock_msgbox):
        """Test save with generic error."""
        dialog = self._create_dialog_instance()
        dialog.catalog_service.add_smell.side_effect = Exception("Generic error")
        
        dialog.name_entry.get.return_value = "New Smell"
        dialog.desc_text.get.return_value = "Description"
        
        dialog._on_save()
        
        # Verify error shown
        mock_msgbox.showerror.assert_called()
        args = mock_msgbox.showerror.call_args[0]
        self.assertIn("Generic error", args[1])

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_cancel_empty_fields(self, mock_msgbox):
        """Test cancel with empty fields closes without confirmation."""
        dialog = self._create_dialog_instance()
        
        dialog.name_entry.get.return_value = ""
        dialog.desc_text.get.return_value = ""
        
        dialog._on_cancel()
        
        # Verify no confirmation shown
        mock_msgbox.askyesno.assert_not_called()
        
        # Verify destroyed
        dialog.destroy.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_cancel_with_data_confirmed(self, mock_msgbox):
        """Test cancel with data when user confirms."""
        dialog = self._create_dialog_instance()
        mock_msgbox.askyesno.return_value = True
        
        # Add some data
        dialog.name_entry.get.return_value = "Some name"
        dialog.desc_text.get.return_value = ""
        
        dialog._on_cancel()
        
        # Verify confirmation shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify destroyed
        dialog.destroy.assert_called_once()

    @patch('gui.manage_code_smells_gui.messagebox')
    def test_cancel_with_data_declined(self, mock_msgbox):
        """Test cancel with data when user declines."""
        dialog = self._create_dialog_instance()
        mock_msgbox.askyesno.return_value = False
        
        # Add some data
        dialog.name_entry.get.return_value = ""
        dialog.desc_text.get.return_value = "Some description"
        
        dialog._on_cancel()
        
        # Verify confirmation shown
        mock_msgbox.askyesno.assert_called_once()
        
        # Verify NOT destroyed
        dialog.destroy.assert_not_called()


if __name__ == '__main__':
    unittest.main()

