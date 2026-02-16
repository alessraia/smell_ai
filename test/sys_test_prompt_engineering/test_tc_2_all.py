from __future__ import annotations

from test.sys_test_prompt_engineering.conftest import flush_tk_events


# -----------------------
# Helper “realistici” (simulano interazione utente su widget)
# -----------------------

def _select_first_smell(app) -> str:
    app._load_smells_into_dropdown()
    values = list(app._smell_combo["values"])
    assert values, "Expected at least one smell in catalog"
    app._smell_combo.current(0)
    app._on_smell_selected()
    assert app._current_smell_id is not None
    return app._current_smell_id


def _select_first_local_provider(app) -> str:
    app._load_local_providers_into_dropdown()
    values = list(app._local_provider_combo["values"])
    assert values, "Expected at least one LOCAL provider in catalog"
    app._local_provider_combo.current(0)
    app._on_local_provider_selected()
    assert app._selected_local_provider_id is not None
    return app._selected_local_provider_id


def _set_input_path(app, path: str) -> None:
    app._input_path_value = path
    app._input_path_label.configure(text=path)


def _set_output_path(app, path: str) -> None:
    app._output_path_value = path
    app._output_path_label.configure(text=path)


def _set_mode_draft(app) -> None:
    # Se la GUI ha handler di mode-change, chiamalo; altrimenti basta settare.
    app._mode_var.set("draft")
    if hasattr(app, "_on_prompt_mode_changed"):
        app._on_prompt_mode_changed()


def _set_prompt_text(app, text: str) -> None:
    # Use the real GUI helper (private) used by the app itself.
    app._set_prompt_text(text, editable=True)
    app._draft_dirty = True


# -----------------------
# TC_2.1
# -----------------------

def test_tc_2_1(pe_gui_app_empty_smells):
    """
    TC_2.1 ORACOLO:
    - L’utente non può avviare il Test con LLM locale
    - log: "Catalogo smell vuoto: aggiungi almeno uno smell con '+' per procedere."
    """
    flush_tk_events(pe_gui_app_empty_smells.master)

    # Tk can return a Tcl string object; normalize to str for stable asserts.
    assert str(pe_gui_app_empty_smells._test_btn.cget("state")) == "disabled"
    assert str(pe_gui_app_empty_smells._smell_combo.cget("state")) == "disabled"
    assert pe_gui_app_empty_smells._smell_combo.get() == ""

    # Il + deve rimanere disponibile per aggiungere il primo smell
    assert str(pe_gui_app_empty_smells._add_smell_btn.cget("state")) == "normal"

    log_text = pe_gui_app_empty_smells._output_text.get("1.0", "end")
    assert "Catalogo smell vuoto: aggiungi almeno uno smell con '+' per procedere." in log_text


# -----------------------
# TC_2.2
# -----------------------

def test_tc_2_2(
    pe_gui_app,
    fake_messagebox,
    force_sync_threads,
    tmp_input_single_py_dir,
    tmp_output_dir,
):
    """
    TC_2.2 ORACOLO:
    La message box mostra l'errore "Il prompt è vuoto.".
    """
    app = pe_gui_app

    flush_tk_events(app.master)

    # 1) Seleziona smell (esiste, perché catalog non è vuoto)
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità: prompt temporaneo (draft)
    _set_mode_draft(app)

    # 3) Modifica prompt temporaneo -> qui lo rendiamo vuoto (strip => vuoto)
    _set_prompt_text(app, "   \n  ")

    # 4) Input path valido (1 file .py)
    _set_input_path(app, str(tmp_input_single_py_dir))

    # 5) Output path valido
    _set_output_path(app, str(tmp_output_dir))

    # 6) Provider locale disponibile e selezionato
    _select_first_local_provider(app)

    # 7) Avvia test (click su “Test con LLM locale”)
    #    Il sistema deve fermarsi PRIMA dell'LLM e mostrare l'errore sul prompt vuoto.
    app._on_test_clicked()
    flush_tk_events(app.master)

    assert fake_messagebox["showerror"], "Expected an error messagebox"
    title, msg = fake_messagebox["showerror"][-1]

    # Oracolo specifico (stringa precisa)
    assert "Il prompt è vuoto." in msg

# -----------------------
# TC_2.3
# -----------------------

def test_tc_2_3(
    pe_gui_app,
    fake_messagebox,
):
    """
    TC_2.3 ORACOLO:
    Il sistema stampa nei Risultati/Log:
    "Prompt salvato come default."
    """

    app = pe_gui_app

    flush_tk_events(app.master)

    # 1) Seleziona smell (esiste nel catalog di test)
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft (prompt temporaneo)
    _set_mode_draft(app)

    # 3) Modifica il prompt temporaneo con testo NON vuoto
    nuovo_prompt = "Nuovo prompt personalizzato"
    _set_prompt_text(app, nuovo_prompt)

    # 4) Click su "Salva come default"
    #    La GUI chiede conferma (askyesno)
    app._on_save_default_clicked()
    flush_tk_events(app.master)

    # 5) Verifica che sia stata richiesta conferma
    assert fake_messagebox["askyesno"], "Expected confirmation dialog"

    # 6) ORACOLO: log contiene messaggio corretto
    log_text = app._output_text.get("1.0", "end")
    assert "Prompt salvato come default." in log_text

    # 7) Verifica ulteriore (system-level realistico):
    #    La modalità deve tornare a DEFAULT
    assert app._mode_var.get() == "default"

    # 8) Verifica che il catalog sia stato aggiornato realmente
    smell = app.catalog_service.load().get_smell(smell_id)
    assert smell.default_prompt == nuovo_prompt

# -----------------------
# TC_2.4
# -----------------------

def test_tc_2_4(
    pe_gui_app,
    fake_messagebox,
    force_sync_threads,
    tmp_output_dir,
):
    """
    TC_2.4 ORACOLO:
    MessageBox titolo: "Input path non valido"
    Messaggio: "Input path must not be empty"
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 3) Imposto un prompt non vuoto per arrivare alla validazione path
    _set_prompt_text(app, "Draft prompt")

    # 4) INPUT PATH NON VALIDO (vuoto)
    app._input_path_value = ""
    app._input_path_label.configure(text="")

    # 5) OUTPUT PATH valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider locale
    _select_first_local_provider(app)

    # 7) Click su "Test con LLM locale"
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 8) ORACOLO: MessageBox errore corretto
    assert fake_messagebox["showerror"], "Expected error messagebox"

    title, msg = fake_messagebox["showerror"][-1]

    assert title == "Input path non valido"
    assert "Input path must not be empty" in msg

# -----------------------
# TC_2.5
# -----------------------

def test_tc_2_5(
    pe_gui_app,
    fake_messagebox,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.5 ORACOLO:
    Titolo: "Input path non valido"
    Messaggio: "Input path must contain at least one .py file"
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 2b) Imposto un prompt non vuoto per arrivare alla validazione path
    _set_prompt_text(app, "Draft prompt")

    # 3) Directory SENZA file .py
    no_py_dir = tmp_path / "no_py_project"
    no_py_dir.mkdir()
    (no_py_dir / "readme.txt").write_text("hello", encoding="utf-8")

    app._input_path_value = str(no_py_dir)
    app._input_path_label.configure(text=str(no_py_dir))

    # 4) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 5) Seleziona provider
    _select_first_local_provider(app)

    # 6) Click su Test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 7) ORACOLO
    assert fake_messagebox["showerror"], "Expected error messagebox"

    title, msg = fake_messagebox["showerror"][-1]

    assert title == "Input path non valido"
    assert "Input path must contain at least one .py file" in msg

# -----------------------
# TC_2.6
# -----------------------

def test_tc_2_6(
    pe_gui_app,
    fake_messagebox,
    force_sync_threads,
    tmp_input_single_py_dir,
):
    """
    TC_2.6 ORACOLO:
    MessageBox mostra l’errore "Output path mancante."
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 3) Prompt valido (usa default)

    # 4) Input valido (contiene un file .py)
    app._input_path_value = str(tmp_input_single_py_dir)
    app._input_path_label.configure(text=str(tmp_input_single_py_dir))

    # 5) OUTPUT NON VALIDO (vuoto)
    app._output_path_value = ""
    app._output_path_label.configure(text="")

    # 6) Provider selezionato
    _select_first_local_provider(app)

    # 7) Click su Test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 8) ORACOLO
    assert fake_messagebox["showerror"], "Expected error messagebox"

    title, msg = fake_messagebox["showerror"][-1]

    assert title == "Errore"
    assert "Output path mancante." in msg

# -----------------------
# TC_2.7
# -----------------------

def test_tc_2_7(
    pe_gui_app_no_local_provider,
    tmp_input_single_py_dir,
    tmp_output_dir,
):
    """
    TC_2.7
    Test frame: PS2-P1-PT2-SP2-FI2-NF2-NP1-FO2-LLM1-SD2

    ORACOLO:
    Il sistema stampa:
    "Nessun provider LLM locale configurato nel catalogo.
     Aggiungi almeno un provider con kind='local' in config/llm_catalog.json."
    """

    app = pe_gui_app_no_local_provider
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 3) Input valido
    app._input_path_value = str(tmp_input_single_py_dir)
    app._input_path_label.configure(text=str(tmp_input_single_py_dir))

    # 4) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # NOTA: non possiamo selezionare provider perché non ce ne sono di tipo local

    # ORACOLO: messaggio nel log GUI
    log_text = app._output_text.get("1.0", "end")

    assert "Nessun provider LLM locale configurato nel catalogo." in log_text
    assert "kind='local'" in log_text

# -----------------------
# TC_2.8
# -----------------------

def test_tc_2_8(
    pe_gui_app,
    monkeypatch,
    force_sync_threads,
    tmp_input_single_py_dir,
    tmp_output_dir,
):
    """
    TC_2.8
    Test frame: PS2-P1-PT2-SP2-FI2-NF2-NP1-FO2-LLM2-SOM1-SD2

    ORACOLO:
    Il sistema informa l’utente dell’errore sul provider.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 3) Input valido
    app._input_path_value = str(tmp_input_single_py_dir)
    app._input_path_label.configure(text=str(tmp_input_single_py_dir))

    # 4) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 5) Seleziona provider locale
    provider_id = _select_first_local_provider(app)

    # 6) Patch provider per simulare errore runtime
    class FailingProvider:
        def generate(self, prompt: str):
            raise RuntimeError("Provider connection error")

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: FailingProvider()
    )

    # 7) Avvia test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 8) ORACOLO: errore informato nel log
    log_text = app._output_text.get("1.0", "end")

    assert "Errore durante il test:" in log_text
    assert "Provider connection error" in log_text

# -----------------------
# TC_2.9
# -----------------------

def test_tc_2_9(
    pe_gui_app,
    monkeypatch,
    force_sync_threads,
    tmp_input_single_py_dir,
    tmp_output_dir,
):
    """
    TC_2.9
    Test frame: PS2-P1-PT2-SP2-FI2-NF2-NP1-FO2-LLM2-SOM2-C2-SD2

    ORACOLO:
    Il sistema salva i due file output nella cartella indicata
    e stampa un resoconto dell’analisi.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft
    _set_mode_draft(app)

    # 3) Input valido
    app._input_path_value = str(tmp_input_single_py_dir)
    app._input_path_label.configure(text=str(tmp_input_single_py_dir))

    # 4) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 5) Seleziona provider
    _select_first_local_provider(app)

    # 6) Provider deterministico che ritorna JSON valido
    from llm_detection.providers import MockLLMProvider

    fixed_response = '{"findings": []}'
    mock_provider = MockLLMProvider(fixed_response=fixed_response)

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: mock_provider
    )

    # 7) Avvia test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 8) Verifica creazione file output
    output_dir = tmp_output_dir / "output"
    assert output_dir.exists()

    csv_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*.csv"))
    raw_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*_raw.jsonl"))

    assert len(csv_files) == 1
    assert len(raw_files) == 1

    # 9) ORACOLO: resoconto nel log
    log_text = app._output_text.get("1.0", "end")

    assert "Test completato. Prompts sent:" in log_text
    assert "Risultati salvati in:" in log_text
    assert "Raw responses salvate in:" in log_text
    assert "--- Fine test ---" in log_text

# -----------------------
# TC_2.10
# -----------------------

def test_tc_2_10(
    pe_gui_app,
    monkeypatch,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.10
    Test frame: PS2-P1-PT2-SP2-FI2-NF2-NP1-FO2-LLM2-SOM2-C1-SD2

    ORACOLO:
    Il sistema accoglie la richiesta dell’utente e stampa:
    "Richiesta cancellazione: il test si fermerà dopo il file corrente".
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Crea input con più file .py (così ha senso cancellare "dopo il file corrente")
    proj = tmp_path / "proj_multi"
    proj.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (proj / f"f{i}.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    # 2) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 3) Modalità draft
    _set_mode_draft(app)

    # 4) Input valido (directory con .py)
    app._input_path_value = str(proj)
    app._input_path_label.configure(text=str(proj))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider locale
    _select_first_local_provider(app)

    # 7) Provider deterministico che al primo file simula il click "Cancel"
    calls = {"n": 0}

    class ProviderCancelOnFirst:
        def generate(self, prompt: str) -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                # Simula utente che preme Cancel mentre l'analisi è in corso
                app._on_cancel_clicked()
            return '{"findings": []}'

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: ProviderCancelOnFirst()
    )

    # 8) Avvia test (analisi parte)
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 9) ORACOLO: messaggio preciso nel log
    log_text = app._output_text.get("1.0", "end")
    assert "Richiesta cancellazione: il test si fermerà dopo il file corrente" in log_text

    # 10) Verifica “realistica” aggiuntiva: non deve processare tutti i file
    #     (se il cancel viene rispettato, le generate() chiamate saranno < 3)
    assert calls["n"] < 3
# -----------------------
# TC_2.11
# -----------------------

def test_tc_2_11(
    pe_gui_app,
    fake_messagebox,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.11
    Test frame: PS2-P1-PT2-SP2-FI2-NF2-NP2-FO2-LLM2-SOM2-SD2

    ORACOLO:
    MessageBox titolo: "Input path non valido"
    Messaggio:
    "Prompt engineering expects a single project folder; input path looks like it contains multiple projects."
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Crea input path con più cartelle "progetto"
    # Heuristica del validatore:
    # - no .py a root
    # - più subdir con .py -> sembra contenere più progetti
    root = tmp_path / "multi_projects"
    root.mkdir(parents=True, exist_ok=True)

    p1 = root / "proj1"
    p2 = root / "proj2"
    p1.mkdir()
    p2.mkdir()

    (p1 / "a.py").write_text("x=1\n", encoding="utf-8")
    (p2 / "b.py").write_text("y=2\n", encoding="utf-8")

    # 2) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 3) Modalità draft
    _set_mode_draft(app)

    # 4) Input non valido (multi-project)
    app._input_path_value = str(root)
    app._input_path_label.configure(text=str(root))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider locale (anche se non verrà usato)
    _select_first_local_provider(app)

    # 7) Click su Test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 8) ORACOLO: messagebox
    assert fake_messagebox["showerror"], "Expected error messagebox"

    title, msg = fake_messagebox["showerror"][-1]

    assert title == "Input path non valido"
    assert "expects a single project folder" in msg
    assert "contains multiple projects" in msg
# -----------------------
# TC_2.12
# -----------------------

def test_tc_2_12(
    pe_gui_app,
    monkeypatch,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.12
    Test frame: PS2-P1-PT2-SP2-FI2-NF3-NP1-FO2-LLM2-SOM2-SD2

    ORACOLO:
    Il sistema salva i due file output nella cartella indicata
    e stampa un resoconto dell’analisi.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Crea progetto unico con PIÙ file .py (tutti in root)
    proj = tmp_path / "proj_multi_py"
    proj.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        (proj / f"file{i}.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    # 2) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 3) Modalità draft
    _set_mode_draft(app)

    # 4) Input valido (più .py ma progetto unico)
    app._input_path_value = str(proj)
    app._input_path_label.configure(text=str(proj))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider
    _select_first_local_provider(app)

    # 7) Provider deterministico
    from llm_detection.providers import MockLLMProvider

    mock_provider = MockLLMProvider(fixed_response='{"findings": []}')

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: mock_provider
    )

    # 8) Avvia analisi
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 9) Verifica file creati
    output_dir = tmp_output_dir / "output"
    assert output_dir.exists()

    csv_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*.csv"))
    raw_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*_raw.jsonl"))

    assert len(csv_files) == 1
    assert len(raw_files) == 1

    # 10) ORACOLO: resoconto nel log
    log_text = app._output_text.get("1.0", "end")

    assert "Test completato. Prompts sent:" in log_text
    assert "Risultati salvati in:" in log_text
    assert "Raw responses salvate in:" in log_text

# -----------------------
# TC_2.13
# -----------------------

def test_tc_2_13(
    pe_gui_app,
    monkeypatch,
    fake_messagebox,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.13
    Test frame: PS2-P1-PT2-SP2-FI2-NF4-NP1-SU1-FO2-LLM2-SOM2-SD2

    ORACOLO:
    Il sistema salva i due file output nella cartella indicata
    e stampa un resoconto dell’analisi.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Crea progetto unico con >15 file .py
    proj = tmp_path / "proj_many_py"
    proj.mkdir(parents=True, exist_ok=True)

    for i in range(16):
        (proj / f"file{i}.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    # 2) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 3) Modalità draft
    _set_mode_draft(app)

    # 4) Input valido
    app._input_path_value = str(proj)
    app._input_path_label.configure(text=str(proj))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider
    _select_first_local_provider(app)

    # 7) Simula utente che risponde YES alla conferma
    # fake_messagebox già ritorna True di default,
    # ma lo rendiamo esplicito per chiarezza:
    from tkinter import messagebox

    monkeypatch.setattr(messagebox, "askyesno", lambda *args, **kwargs: True)

    # 8) Provider deterministico
    from llm_detection.providers import MockLLMProvider

    mock_provider = MockLLMProvider(fixed_response='{"findings": []}')

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: mock_provider
    )

    # 9) Avvia test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 10) Verifica file output
    output_dir = tmp_output_dir / "output"
    assert output_dir.exists()

    csv_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*.csv"))
    raw_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*_raw.jsonl"))

    assert len(csv_files) == 1
    assert len(raw_files) == 1

    # 11) ORACOLO: resoconto nel log
    log_text = app._output_text.get("1.0", "end")

    assert "Test completato. Prompts sent:" in log_text
    assert "Risultati salvati in:" in log_text
    assert "Raw responses salvate in:" in log_text

# -----------------------
# TC_2.14
# -----------------------

def test_tc_2_14(
    pe_gui_app,
    monkeypatch,
    fake_messagebox,
    force_sync_threads,
    tmp_output_dir,
    tmp_path,
):
    """
    TC_2.14
    Test frame: PS2-P1-PT2-SP2-FI2-NF4-NP1-SU2-FO2-LLM2-SOM2-SD2

    ORACOLO:
    Il sistema accoglie il volere dell’utente e non procede con l’analisi.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Crea progetto con >15 file .py
    proj = tmp_path / "proj_many_py_no"
    proj.mkdir(parents=True, exist_ok=True)

    for i in range(16):
        (proj / f"file{i}.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    # 2) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 3) Modalità draft
    _set_mode_draft(app)

    # 4) Input valido
    app._input_path_value = str(proj)
    app._input_path_label.configure(text=str(proj))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider
    _select_first_local_provider(app)

    # 7) Simula utente che risponde NO alla conferma
    from tkinter import messagebox
    monkeypatch.setattr(messagebox, "askyesno", lambda *args, **kwargs: False)

    # 8) Provider (non dovrebbe mai essere chiamato)
    called = {"n": 0}

    class DummyProvider:
        def generate(self, prompt: str):
            called["n"] += 1
            return '{"findings": []}'

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: DummyProvider()
    )

    # 9) Avvia test
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 10) ORACOLO: nessuna analisi eseguita
    output_dir = tmp_output_dir / "output"
    assert not output_dir.exists() or not any(output_dir.iterdir())

    # Provider non deve essere stato chiamato
    assert called["n"] == 0

    # Log non deve contenere messaggi di completamento
    log_text = app._output_text.get("1.0", "end")
    assert "Test completato." not in log_text
    assert "Risultati salvati in:" not in log_text

# -----------------------
# TC_2.15
# -----------------------

def test_tc_2_15(
    pe_gui_app,
    monkeypatch,
    force_sync_threads,
    tmp_input_single_py_dir,
    tmp_output_dir,
):
    """
    TC_2.15
    Test frame: PS2-P2-PT2-SP2-FI2-NF2-NP1-FO2-LLM2-SOM2-SD2

    ORACOLO:
    Il sistema salva i due file output nella cartella indicata
    e stampa un resoconto dell’analisi.
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    smell_id = _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità DEFAULT (non draft)
    app._mode_var.set("default")
    if hasattr(app, "_on_prompt_mode_changed"):
        app._on_prompt_mode_changed()

    # 3) Non modificare il prompt (usa quello di default del catalogo)

    # 4) Input valido (1 file .py)
    app._input_path_value = str(tmp_input_single_py_dir)
    app._input_path_label.configure(text=str(tmp_input_single_py_dir))

    # 5) Output valido
    app._output_path_value = str(tmp_output_dir)
    app._output_path_label.configure(text=str(tmp_output_dir))

    # 6) Seleziona provider
    _select_first_local_provider(app)

    # 7) Provider deterministico
    from llm_detection.providers import MockLLMProvider

    mock_provider = MockLLMProvider(fixed_response='{"findings": []}')

    monkeypatch.setattr(
        app,
        "_build_local_provider_by_id",
        lambda catalog, pid: mock_provider
    )

    # 8) Avvia analisi
    app._on_test_clicked()
    flush_tk_events(app.master)

    # 9) Verifica file creati
    output_dir = tmp_output_dir / "output"
    assert output_dir.exists()

    csv_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*.csv"))
    raw_files = list(output_dir.glob(f"prompt_engineering_{smell_id}_*_raw.jsonl"))

    assert len(csv_files) == 1
    assert len(raw_files) == 1

    # 10) ORACOLO: resoconto nel log
    log_text = app._output_text.get("1.0", "end")

    assert "Test completato. Prompts sent:" in log_text
    assert "Risultati salvati in:" in log_text
    assert "Raw responses salvate in:" in log_text

# -----------------------
# TC_2.16
# -----------------------

def test_tc_2_16(
    pe_gui_app,
    fake_messagebox,
):
    """
    TC_2.16
    Test frame: PS2-P1-PT1-SP2-FI2-NF2-NP1-FO2-LLM2-SOM2-SD1

    ORACOLO:
    La message box mostra l’errore: "Il prompt temporaneo è vuoto."
    """

    app = pe_gui_app
    flush_tk_events(app.master)

    # 1) Seleziona smell
    _select_first_smell(app)
    flush_tk_events(app.master)

    # 2) Modalità draft (prompt temporaneo)
    _set_mode_draft(app)

    # 3) Imposta prompt temporaneo vuoto
    _set_prompt_text(app, "   \n  ")

    # 4) Click su "Salva come default"
    app._on_save_default_clicked()
    flush_tk_events(app.master)

    # 5) ORACOLO: messagebox
    assert fake_messagebox["showerror"], "Expected error messagebox"

    title, msg = fake_messagebox["showerror"][-1]

    assert title == "Errore"
    assert "Il prompt temporaneo è vuoto." in msg
