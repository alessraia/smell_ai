"""
Test minimale: LLMCatalogService ↔ LLMCatalogStore
Verifica solo che service possa salvare/caricare tramite store
"""

import tempfile
import os
from llm_detection.catalog_service import LLMCatalogService
from llm_detection.catalog_store import LLMCatalogStore


def test_service_can_add_smell_and_persist():
    """Verifica che aggiungere uno smell tramite service lo persiste su disco"""
    
    # Usa un file temporaneo per non toccare il catalog reale
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_catalog.json")
        
        # 1. Crea service con store che usa file temporaneo
        store = LLMCatalogStore(file_path=test_file)
        service = LLMCatalogService(store=store)
        
        # 2. Aggiungi uno smell
        smell_id = service.add_smell("Test Smell", "This is a test")
        
        # 3. Crea un NUOVO service (simula restart) e verifica persisted
        new_store = LLMCatalogStore(file_path=test_file)
        new_service = LLMCatalogService(store=new_store)
        catalog = new_service.load()
        
        # 4. Verifica che lo smell esiste nel catalog ricaricato
        smell_found = any(s.smell_id == smell_id for s in catalog.smells)
        assert smell_found, f"Smell {smell_id} not found after reload"
        
        # 5. Verifica contenuto
        smell = next(s for s in catalog.smells if s.smell_id == smell_id)
        assert smell.display_name == "Test Smell"
        assert smell.description == "This is a test"
        assert smell.created_by_user is True


def test_service_can_remove_smell_and_persist():
    """Verifica che rimuovere uno smell tramite service persiste la rimozione"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_catalog.json")
        
        # 1. Crea service e aggiungi due smells
        store = LLMCatalogStore(file_path=test_file)
        service = LLMCatalogService(store=store)
        
        smell_id_1 = service.add_smell("Smell One", "First smell")
        smell_id_2 = service.add_smell("Smell Two", "Second smell")
        
        # 2. Rimuovi il primo
        service.remove_smell(smell_id_1)
        
        # 3. Ricarica da nuovo service
        new_store = LLMCatalogStore(file_path=test_file)
        new_service = LLMCatalogService(store=new_store)
        catalog = new_service.load()
        
        # 4. Verifica: smell_id_1 rimosso, smell_id_2 ancora presente
        smell_ids = [s.smell_id for s in catalog.smells]
        assert smell_id_1 not in smell_ids, "Removed smell still present"
        assert smell_id_2 in smell_ids, "Remaining smell was deleted"


def test_service_can_save_draft_prompt():
    """Verifica che salvare draft prompt tramite service persiste"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_catalog.json")
        
        # 1. Aggiungi smell
        store = LLMCatalogStore(file_path=test_file)
        service = LLMCatalogService(store=store)
        smell_id = service.add_smell("Test Smell", "Description")
        
        # 2. Salva draft prompt
        draft_text = "Detect smell in code:\n{code}"
        service.save_draft_prompt(smell_id, draft_text)
        
        # 3. Ricarica
        new_service = LLMCatalogService(store=LLMCatalogStore(file_path=test_file))
        catalog = new_service.load()
        
        # 4. Verifica draft prompt persisted
        smell = next(s for s in catalog.smells if s.smell_id == smell_id)
        assert smell.draft_prompt == draft_text
        assert smell.default_prompt == ""  # Non ancora promosso
