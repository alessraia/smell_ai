from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from gui.manage_code_smells_gui import ManageCodeSmellsGUI, AddSmellDialog
from llm_detection.catalog_service import LLMCatalogService
from llm_detection.catalog_store import LLMCatalogStore

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture
def mock_catalog_file(tmp_path) -> Path:
    """Creates a temporary catalog JSON file."""
    f = tmp_path / "llm_catalog.json"
    data = {
        "schema_version": 1, 
        "smells": [], 
        "providers": []
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    return f

@pytest.fixture
def catalog_service(mock_catalog_file) -> LLMCatalogService:
    store = LLMCatalogStore(str(mock_catalog_file))
    return LLMCatalogService(store)

@pytest.fixture
def manage_gui(tk_root, catalog_service):
    """
    Instantiates the GUI.
    Patches AddSmellDialog.wait_window to prevent blocking.
    """
    with patch('gui.manage_code_smells_gui.AddSmellDialog.wait_window'):
        app = ManageCodeSmellsGUI(tk_root, catalog_service)
        yield app
        try:
            app.destroy()
        except:
            pass

def _get_dialog(app) -> AddSmellDialog:
    """Helper to get the dialog instance since we mock wait_window."""
    # Since AddSmellDialog is a Toplevel, it should be in children
    for widget in app.master.winfo_children():
        if isinstance(widget, AddSmellDialog) and widget.winfo_exists():
            return widget
    return None

# -------------------------------------------------------------------------
# TC_3.1, TC_3.11, TC_3.16: Empty Catalog State
# -------------------------------------------------------------------------
def test_TC_3_1_empty_state(manage_gui, catalog_service):
    """
    CS1: Catalogo vuoto -> pulsanti disabilitati.
    Covers TC_3.1, TC_3.11, TC_3.16 (Startup state)
    """
    # Ensure empty
    assert len(catalog_service.list_smells()) == 0
    manage_gui._load_smells_into_dropdown()

    # Check GUI State
    assert str(manage_gui._remove_btn['state']) == 'disabled'
    assert str(manage_gui._save_btn['state']) == 'disabled'
    assert str(manage_gui._add_btn['state']) == 'normal'
    assert len(manage_gui._smell_combo['values']) == 0

# -------------------------------------------------------------------------
# ADD FLOW (TC_3.2 - TC_3.10)
# -------------------------------------------------------------------------

def test_TC_3_2_add_missing_name(manage_gui):
    """CS2-AN1-AD2-AA1: Save with empty name -> Error."""
    with patch('tkinter.messagebox.showwarning') as mock_warn:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        # Fill only description
        dlg.desc_text.insert("1.0", "Valid Desc")
        
        # Click Save
        dlg._on_save()
        
        mock_warn.assert_called_with("Dati mancanti", "Inserisci il nome del code smell.")
        dlg.destroy()

def test_TC_3_3_add_duplicate_name(manage_gui, catalog_service):
    """CS2-AN2-AD2-AA1: Save duplicate name -> Error."""
    # Pre-condition: existing smell
    catalog_service.add_smell("God Class", "Desc")
    
    with patch('tkinter.messagebox.showerror') as mock_err:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "God Class")
        dlg.desc_text.insert("1.0", "New Desc")
        
        dlg._on_save()
        
        # Assert error call
        assert mock_err.called
        assert "already exists" in str(mock_err.call_args) or "esistente" in str(mock_err.call_args)
        dlg.destroy()

def test_TC_3_4_add_missing_desc(manage_gui):
    """CS2-AN3-AD1-AA1: Save empty description -> Error."""
    with patch('tkinter.messagebox.showwarning') as mock_warn:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "New Smell")
        # Desc empty by default
        
        dlg._on_save()
        
        mock_warn.assert_called_with("Dati mancanti", "Inserisci una descrizione.")
        dlg.destroy()

def test_TC_3_5_cancel_name_empty_desc_valid(manage_gui):
    """CS2-AN1-AD2-AA2-AU1: Name Empty, Desc Valid -> Cancel -> Confirm Exit."""
    with patch('tkinter.messagebox.askyesno', return_value=True) as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        # Name Empty (default)
        dlg.desc_text.insert("1.0", "Valid Description")
        
        dlg._on_cancel()
        
        mock_ask.assert_called()
        assert not dlg.winfo_exists()

def test_TC_3_6_cancel_name_duplicate_desc_valid(manage_gui, catalog_service):
    """CS2-AN2-AD2-AA2-AU1: Name Duplicate, Desc Valid -> Cancel -> Confirm Exit."""
    catalog_service.add_smell("Existing", "Desc")
    
    with patch('tkinter.messagebox.askyesno', return_value=True) as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "Existing")
        dlg.desc_text.insert("1.0", "Valid Description")

        dlg._on_cancel()
        
        mock_ask.assert_called()
        assert not dlg.winfo_exists()

def test_TC_3_7_cancel_name_valid_desc_empty(manage_gui):
    """CS2-AN3-AD1-AA2-AU1: Name Valid, Desc Empty -> Cancel -> Confirm Exit."""
    with patch('tkinter.messagebox.askyesno', return_value=True) as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "ValidName")
        # Desc Empty (default)

        dlg._on_cancel()
        
        mock_ask.assert_called()
        assert not dlg.winfo_exists()

def test_TC_3_8_cancel_retry(manage_gui):
    """CS2-AN3-AD1-AA2-AU2: Name Valid, Desc Empty -> Cancel -> No -> Stay."""
    with patch('tkinter.messagebox.askyesno', return_value=False) as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "ValidName")
        # Desc Empty
        
        dlg._on_cancel()
        
        mock_ask.assert_called()
        assert dlg.winfo_exists()
        dlg.destroy()

def test_TC_3_9_cancel_empty(manage_gui):
    """CS2-AN1-AD1-AA2: Cancel with no data -> Immediate Close."""
    with patch('tkinter.messagebox.askyesno') as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg._on_cancel()
        
        mock_ask.assert_not_called()
        assert not dlg.winfo_exists()

def test_TC_3_10_cancel_full(manage_gui):
    """CS2-AN3-AD2-AA2-AU1: Full Data -> Cancel -> Yes -> Close."""
    with patch('tkinter.messagebox.askyesno', return_value=True) as mock_ask:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "ValidName")
        dlg.desc_text.insert("1.0", "ValidDesc")
        
        dlg._on_cancel()
        
        mock_ask.assert_called()
        assert not dlg.winfo_exists()

def test_TC_3_11_add_success(manage_gui, catalog_service):
    """CS2-AN3-AD2-AA1: Valid Add -> Success."""
    with patch('tkinter.messagebox.showinfo') as mock_info:
        manage_gui._on_add_smell()
        dlg = _get_dialog(manage_gui)
        
        dlg.name_entry.insert(0, "New Feature")
        dlg.desc_text.insert("1.0", "My Description")
        
        dlg._on_save()
        
        assert "New Feature" in [s.display_name for s in catalog_service.list_smells()]
        assert not dlg.winfo_exists()

# -------------------------------------------------------------------------
# MODIFY FLOW (TC_3.12 - TC_3.16)
# -------------------------------------------------------------------------

def test_TC_3_12_modify_empty_desc(manage_gui, catalog_service):
    """CS2-MD1: Clean DB -> Load -> Empty Desc -> Save -> Error + Revert."""
    # Setup: 1 smell
    sid = catalog_service.add_smell("Target", "Original Desc")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("Target")
    manage_gui._on_smell_selected() # Load into UI
    
    # Act: Empty desc
    manage_gui._desc_text.delete("1.0", "end")
    
    with patch('tkinter.messagebox.showwarning') as mock_warn:
        manage_gui._on_save_changes()
        mock_warn.assert_called_with("Attenzione", "La descrizione non può essere vuota.")
    
    # Check Revert (TC 3.12 fixed behavior)
    assert manage_gui._desc_text.get("1.0", "end-1c").strip() == "Original Desc"

def test_TC_3_13_modify_no_change(manage_gui, catalog_service):
    """CS2-MD3: No changes -> Info."""
    sid = catalog_service.add_smell("Target", "Desc")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("Target")
    manage_gui._on_smell_selected()
    
    with patch('tkinter.messagebox.showinfo') as mock_info:
        manage_gui._on_save_changes()
        mock_info.assert_called_with("Info", "Nessuna modifica rilevata alla descrizione.")

def test_TC_3_14_modify_cancel_confirm(manage_gui, catalog_service):
    """CS2-MD2-CO2: Change -> Save -> Confirm NO -> Revert."""
    sid = catalog_service.add_smell("Target", "Original")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("Target")
    manage_gui._on_smell_selected()
    
    manage_gui._desc_text.insert("end", " EDITED")
    
    with patch('tkinter.messagebox.askyesno', return_value=False) as mock_ask:
        manage_gui._on_save_changes()
        mock_ask.assert_called()
    
    # Check Revert
    assert manage_gui._desc_text.get("1.0", "end-1c").strip() == "Original"

def test_TC_3_15_modify_success(manage_gui, catalog_service):
    """CS2-MD2-CO1: Change -> Save -> Confirm YES -> Success."""
    sid = catalog_service.add_smell("Target", "Original")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("Target")
    manage_gui._on_smell_selected()
    
    manage_gui._desc_text.delete("1.0", "end")
    manage_gui._desc_text.insert("1.0", "New Value")
    
    with patch('tkinter.messagebox.askyesno', return_value=True), \
         patch('tkinter.messagebox.showinfo'):
        manage_gui._on_save_changes()
    
    # Check DB
    s = catalog_service.load().get_smell(sid)
    assert s.description == "New Value"

# -------------------------------------------------------------------------
# REMOVE FLOW (TC_3.17 - TC_3.19)
# -------------------------------------------------------------------------

def test_TC_3_17_remove_cancel(manage_gui, catalog_service):
    """CS2-RC2: Remove -> Confirm NO -> No Delete."""
    sid = catalog_service.add_smell("ToKeep", "Desc")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("ToKeep")
    manage_gui._on_smell_selected()
    
    with patch('tkinter.messagebox.askyesno', return_value=False):
        manage_gui._on_remove_smell()
        
    assert len(catalog_service.list_smells()) == 1

def test_TC_3_18_remove_success(manage_gui, catalog_service):
    """CS2-RC1: Remove -> Confirm YES -> Delete."""
    sid = catalog_service.add_smell("ToDelete", "Desc")
    manage_gui._load_smells_into_dropdown()
    manage_gui._smell_combo.set("ToDelete")
    manage_gui._on_smell_selected()
    
    with patch('tkinter.messagebox.askyesno', return_value=True), \
         patch('tkinter.messagebox.showinfo'):
        manage_gui._on_remove_smell()
        
    assert len(catalog_service.list_smells()) == 0
