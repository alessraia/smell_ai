import tkinter as tk
from types import SimpleNamespace

import pytest
import pandas as pd

from llm_detection.catalog_service import CatalogValidationError
from llm_detection.types import (
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    LLMSmellFinding,
    PromptMode,
    ProviderKind,
)
from prompt_engineering.prompt_engineering_gui import PromptEngineeringGUI


class FakeCatalogService:
    def __init__(self, catalog: LLMCatalog):
        self._catalog = catalog
        self.saved_drafts: list[tuple[str, str]] = []
        self.promoted: list[str] = []
        self.validated_paths: list[str] = []

    def load(self) -> LLMCatalog:
        return self._catalog

    def validate_prompt_engineering_input_path(self, input_path: str) -> None:
        self.validated_paths.append(input_path)
        if not input_path:
            raise CatalogValidationError("Input path missing")

    def save_draft_prompt(self, smell_id: str, prompt_text: str) -> None:
        if not prompt_text.strip():
            raise CatalogValidationError("Draft prompt empty")
        self.saved_drafts.append((smell_id, prompt_text))
        smell = self._catalog.get_smell(smell_id)
        updated = LLMSmellDefinition(
            smell_id=smell.smell_id,
            display_name=smell.display_name,
            description=smell.description,
            default_prompt=smell.default_prompt,
            draft_prompt=prompt_text,
            created_by_user=smell.created_by_user,
            enabled=smell.enabled,
        )
        self._catalog.upsert_smell(updated)

    def promote_draft_to_default(self, smell_id: str) -> None:
        smell = self._catalog.get_smell(smell_id)
        smell.save_draft_as_default()
        self.promoted.append(smell_id)

    def add_smell(self, smell_id: str, display_name: str = "New", description: str = "d") -> None:
        smell = LLMSmellDefinition(
            smell_id=smell_id,
            display_name=display_name,
            description=description,
            default_prompt="",
            draft_prompt="",
            enabled=False,
        )
        self._catalog.upsert_smell(smell)


def _catalog_with_smells_and_providers(
    *,
    smells: list[LLMSmellDefinition],
    providers: list[LLMProviderDefinition],
) -> LLMCatalog:
    return LLMCatalog(schema_version=1, smells=smells, providers=providers)


@pytest.fixture(scope="session")
def tk_app():
    """Single Tcl/Tk interpreter for the whole test session.

    Creating/destroying multiple `tk.Tk()` instances in one process can be flaky
    on some Windows setups; we keep one Tk app and spawn a fresh Toplevel per test.
    """

    try:
        app = tk.Tk()
    except tk.TclError as e:
        pytest.skip(
            "Tkinter/Tcl not available (cannot create tk.Tk()). "
            "This usually means you're running tests with a Python installation missing Tcl/Tk. "
            "Run pytest with the project venv interpreter (e.g. `.venv/Scripts/python.exe -m pytest ...`) "
            f"or fix the Python install. Original error: {e}"
        )

    app.withdraw()
    yield app
    app.quit()
    app.update()
    app.destroy()


@pytest.fixture
def tk_root(tk_app):
    win = tk.Toplevel(tk_app)
    win.withdraw()
    yield win
    win.destroy()


def test_empty_smell_catalog_disables_ui_and_logs_message(tk_root):
    svc = FakeCatalogService(_catalog_with_smells_and_providers(smells=[], providers=[]))
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)

    assert gui._ui_disabled_no_smells is True
    assert str(gui._smell_combo.cget("state")) == "disabled"
    # The + button stays enabled to allow adding the first smell from this UI.
    assert str(gui._add_smell_btn.cget("state")) == "normal"
    assert str(gui._test_btn.cget("state")) == "disabled"
    assert str(gui._local_provider_combo.cget("state")) == "disabled"

    out = gui._output_text.get("1.0", "end")
    assert "Catalogo smell vuoto" in out


def test_no_local_provider_disables_provider_combo_and_logs_message(tk_root):
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    api_provider = LLMProviderDefinition(
        provider_id="api1",
        kind=ProviderKind.API,
        display_name="Api",
        config={"base_url": "http://example"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell], providers=[api_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)

    assert gui._selected_local_provider_id is None
    assert str(gui._local_provider_combo.cget("state")) == "disabled"

    out = gui._output_text.get("1.0", "end")
    assert "Nessun provider LLM locale configurato" in out
    assert str(gui._test_btn.cget("state")) == "disabled"


def test_prompt_view_switches_between_draft_and_default(tk_root):
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="DEFAULT-PROMPT",
        draft_prompt="DRAFT-PROMPT",
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="local",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={"model_name": "x"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell], providers=[local_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)

    # Default mode on startup is DRAFT.
    assert gui._mode_var.get() == PromptMode.DRAFT.value
    assert gui._prompt_text.cget("state") == "normal"
    assert "DRAFT-PROMPT" in gui._prompt_text.get("1.0", "end")

    gui._mode_var.set(PromptMode.DEFAULT.value)
    gui._refresh_prompt_view()
    assert gui._prompt_text.cget("state") == "disabled"
    assert "DEFAULT-PROMPT" in gui._prompt_text.get("1.0", "end")


def test_prompt_mode_change_is_blocked_if_user_cancels_discard(tk_root, mocker):
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="local",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={"model_name": "x"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell], providers=[local_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)
    gui._draft_dirty = True

    mocker.patch("tkinter.messagebox.askyesno", return_value=False)

    gui._mode_var.set(PromptMode.DEFAULT.value)
    gui._on_prompt_mode_changed()

    assert gui._mode_var.get() == PromptMode.DRAFT.value


def test_sync_test_button_state_requires_smell_and_local_provider(tk_root):
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="local",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={"model_name": "x"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell], providers=[local_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)

    assert str(gui._test_btn.cget("state")) == "normal"

    gui._selected_local_provider_id = None
    gui._sync_test_button_state()
    assert str(gui._test_btn.cget("state")) == "disabled"

    gui._selected_local_provider_id = "local"
    gui._current_smell_id = None
    gui._sync_test_button_state()
    assert str(gui._test_btn.cget("state")) == "disabled"


def test_build_local_provider_by_id_uses_fallbacks_and_aliases():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt=None,
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="p1",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={
            "base_url": "http://localhost:11434",
            "options": {"temperature": 0.1},
            "response_format": "json",
        },
    )
    cat = _catalog_with_smells_and_providers(smells=[smell], providers=[local_provider])

    provider = PromptEngineeringGUI._build_local_provider_by_id(cat, "p1")
    assert provider.model_name == "qwen2.5-coder:7b"
    assert provider.host == "http://localhost:11434"
    assert provider.options == {"temperature": 0.1}
    assert provider.response_format == "json"


def _make_ready_gui(tk_root, *, mode: PromptMode = PromptMode.DRAFT) -> tuple[PromptEngineeringGUI, FakeCatalogService]:
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="local",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={"model_name": "x"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell], providers=[local_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)
    gui._current_smell_id = "s1"
    gui._selected_local_provider_id = "local"
    gui._mode_var.set(mode.value)
    gui._input_path_value = "C:/in"
    gui._output_path_value = "C:/out"
    gui._set_prompt_text("PROMPT", editable=True)
    return gui, svc


@pytest.mark.parametrize(
    "variant, expected_title, expected_message_substring",
    [
        ("missing_smell", "Errore", "Seleziona uno smell"),
        ("missing_provider", "Errore", "Nessun provider"),
        ("invalid_input", "Input path non valido", "bad"),
        ("missing_output", "Errore", "Output path mancante"),
        ("empty_prompt", "Errore", "Il prompt è vuoto"),
        ("no_py_files", "Errore", "no Python files"),
    ],
)
def test_on_test_clicked_validation_errors(tk_root, mocker, variant, expected_title, expected_message_substring):
    gui, svc = _make_ready_gui(tk_root)

    if variant == "missing_smell":
        gui._current_smell_id = None
    elif variant == "missing_provider":
        gui._selected_local_provider_id = ""
    elif variant == "invalid_input":
        svc.validate_prompt_engineering_input_path = lambda _p: (_ for _ in ()).throw(
            CatalogValidationError("bad")
        )
    elif variant == "missing_output":
        gui._output_path_value = ""
    elif variant == "empty_prompt":
        gui._set_prompt_text("   ", editable=True)
    elif variant == "no_py_files":
        mocker.patch(
            "prompt_engineering.prompt_engineering_gui.FileUtils.get_python_files",
            return_value=[],
        )
    else:
        raise AssertionError(f"Unknown variant: {variant}")

    showerror = mocker.patch("tkinter.messagebox.showerror")
    gui._on_test_clicked()
    showerror.assert_called()
    assert expected_title in showerror.call_args[0][0]
    assert expected_message_substring in showerror.call_args[0][1]


def test_on_test_clicked_many_files_user_cancels_does_not_start_thread(tk_root, mocker):
    gui, _svc = _make_ready_gui(tk_root)
    mocker.patch(
        "prompt_engineering.prompt_engineering_gui.FileUtils.get_python_files",
        return_value=[f"f{i}.py" for i in range(16)],
    )
    mocker.patch("tkinter.messagebox.askyesno", return_value=False)
    thread_ctor = mocker.patch("prompt_engineering.prompt_engineering_gui.threading.Thread")

    gui._on_test_clicked()

    thread_ctor.assert_not_called()


@pytest.mark.parametrize(
    "mode, expect_draft_saved",
    [
        (PromptMode.DRAFT, True),
        (PromptMode.DEFAULT, False),
    ],
)
def test_on_test_clicked_starts_thread_and_optional_draft_save(tk_root, mocker, mode, expect_draft_saved):
    gui, svc = _make_ready_gui(tk_root, mode=mode)
    mocker.patch(
        "prompt_engineering.prompt_engineering_gui.FileUtils.get_python_files",
        return_value=["a.py"],
    )
    mocker.patch("prompt_engineering.prompt_engineering_gui.os.makedirs")
    mocker.patch.object(gui, "_schedule_heartbeat")
    mocker.patch.object(gui, "_set_running_state")

    started = {"value": False}

    class FakeThread:
        def __init__(self, target=None, args=None, daemon=None):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started["value"] = True

    mocker.patch("prompt_engineering.prompt_engineering_gui.threading.Thread", FakeThread)
    gui._on_test_clicked()

    assert started["value"] is True
    if expect_draft_saved:
        assert ("s1", "PROMPT") in svc.saved_drafts
    else:
        assert svc.saved_drafts == []


def test_on_save_default_clicked_saves_and_promotes(tk_root, mocker):
    gui, svc = _make_ready_gui(tk_root, mode=PromptMode.DRAFT)
    gui._set_prompt_text("NEW DRAFT", editable=True)
    mocker.patch("tkinter.messagebox.askyesno", return_value=True)

    gui._on_save_default_clicked()

    assert ("s1", "NEW DRAFT") in svc.saved_drafts
    assert "s1" in svc.promoted
    assert gui._mode_var.get() == PromptMode.DEFAULT.value
    assert "NEW DRAFT" in gui._prompt_text.get("1.0", "end")


def test_on_cancel_clicked_sets_event_and_logs(tk_root):
    gui, _svc = _make_ready_gui(tk_root)
    assert gui._cancel_event.is_set() is False

    gui._on_cancel_clicked()

    assert gui._cancel_event.is_set() is True
    assert "Richiesta cancellazione" in gui._output_text.get("1.0", "end")


def test_run_test_thread_success_writes_outputs_and_updates_ui(tk_root, mocker, tmp_path):
    gui, _svc = _make_ready_gui(tk_root, mode=PromptMode.DRAFT)

    # Make after() execute callbacks immediately.
    def immediate_after(_ms, func=None, *args):
        if func is not None:
            func(*args)
        return "after-id"

    mocker.patch.object(gui.master, "after", side_effect=immediate_after)

    # Avoid building a real LocalLLMProvider.
    mocker.patch.object(PromptEngineeringGUI, "_build_local_provider_by_id", return_value=object())

    class FakeOrchestrator:
        def __init__(self, provider=None, catalog=None):
            self.provider = provider
            self.catalog = catalog

        def detect_for_prompt_engineering_with_raw(self, targets, smell_id, prompt_mode):
            findings = [
                LLMSmellFinding(
                    filename=targets[0].filename,
                    function_name="f",
                    smell_name="S",
                    line=10,
                    description="d",
                ),
                LLMSmellFinding(
                    filename=targets[0].filename,
                    function_name="f",
                    smell_name="S",
                    line=-1,
                    description="bad",
                ),
            ]
            stats = SimpleNamespace(prompts_sent=1)
            raw = {targets[0].filename: "RAW"}
            return findings, stats, raw

        def findings_to_dataframe(self, findings):
            return pd.DataFrame(
                [
                    {
                        "filename": f.filename,
                        "function_name": f.function_name,
                        "smell_name": f.smell_name,
                        "line": f.line,
                        "description": f.description,
                        "additional_info": "",
                    }
                    for f in findings
                ]
            )

    mocker.patch("prompt_engineering.prompt_engineering_gui.LLMOrchestrator", FakeOrchestrator)

    in_dir = tmp_path / "proj"
    in_dir.mkdir()
    py_file = in_dir / "a.py"
    py_file.write_text("print('hi')\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    gui._set_prompt_text("DRAFT RUN", editable=True)
    gui._run_test_thread(
        smell_id="s1",
        mode=PromptMode.DRAFT,
        python_files=[str(py_file)],
        output_path=str(out_dir),
        provider_id="local",
    )

    # Ensure output artifacts exist.
    output_folder = out_dir / "output"
    assert output_folder.exists()
    assert list(output_folder.glob("prompt_engineering_s1_*.csv"))
    assert list(output_folder.glob("prompt_engineering_s1_*_raw.jsonl"))

    # UI log contains completion.
    out_text = gui._output_text.get("1.0", "end")
    assert "Test completato" in out_text
    assert "parse_errors" in out_text


def test_run_test_thread_handles_exception_and_logs_traceback(tk_root, mocker, tmp_path):
    gui, _svc = _make_ready_gui(tk_root, mode=PromptMode.DEFAULT)

    def immediate_after(_ms, func=None, *args):
        if func is not None:
            func(*args)
        return "after-id"

    mocker.patch.object(gui.master, "after", side_effect=immediate_after)
    mocker.patch.object(PromptEngineeringGUI, "_build_local_provider_by_id", return_value=object())

    class BoomOrchestrator:
        def __init__(self, provider=None, catalog=None):
            pass

        def detect_for_prompt_engineering_with_raw(self, *args, **kwargs):
            raise RuntimeError("boom")

        def findings_to_dataframe(self, findings):
            return pd.DataFrame([])

    mocker.patch("prompt_engineering.prompt_engineering_gui.LLMOrchestrator", BoomOrchestrator)

    in_dir = tmp_path / "proj"
    in_dir.mkdir()
    py_file = in_dir / "a.py"
    py_file.write_text("print('hi')\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    gui._run_test_thread(
        smell_id="s1",
        mode=PromptMode.DEFAULT,
        python_files=[str(py_file)],
        output_path=str(out_dir),
        provider_id="local",
    )

    out_text = gui._output_text.get("1.0", "end")
    assert "Errore durante il test" in out_text
    assert "RuntimeError: boom" in out_text


def test_choose_input_and_output_path_update_labels(tk_root, mocker):
    gui, _svc = _make_ready_gui(tk_root)
    mocker.patch("tkinter.filedialog.askdirectory", return_value="C:/my/input")
    gui._choose_input_path()
    assert gui._input_path_value == "C:/my/input"
    assert "C:/my/input" in str(gui._input_path_label.cget("text"))

    mocker.patch("tkinter.filedialog.askdirectory", return_value="C:/my/output")
    gui._choose_output_path()
    assert gui._output_path_value == "C:/my/output"
    assert "C:/my/output" in str(gui._output_path_label.cget("text"))


def test_on_local_provider_selected_updates_selected_id(tk_root):
    gui, _svc = _make_ready_gui(tk_root)
    # The combobox is already populated; pick first display value.
    first_display = gui._local_provider_combo["values"][0]
    gui._local_provider_combo.set(first_display)
    gui._on_local_provider_selected()
    assert gui._selected_local_provider_id == "local"


@pytest.mark.parametrize(
    "dirty, askyesno_called",
    [
        (False, False),
        (True, True),
    ],
)
def test_confirm_discard_unsaved_draft_behaviour(tk_root, mocker, dirty, askyesno_called):
    gui, _svc = _make_ready_gui(tk_root)
    gui._draft_dirty = dirty
    ask = mocker.patch("tkinter.messagebox.askyesno", return_value=True)
    assert gui._confirm_discard_unsaved_draft_if_needed("ctx") is True
    assert ask.called is askyesno_called


def test_on_close_dirty_prompt_user_cancels_does_not_destroy(tk_root, mocker):
    gui, _svc = _make_ready_gui(tk_root)
    gui._draft_dirty = True
    mocker.patch("tkinter.messagebox.askyesno", return_value=False)
    destroy = mocker.patch.object(gui.master, "destroy")
    gui._on_close()
    destroy.assert_not_called()


def test_append_output_keeps_widget_disabled(tk_root):
    gui, _svc = _make_ready_gui(tk_root)
    assert str(gui._output_text.cget("state")) == "disabled"
    gui._append_output("hello")
    assert "hello" in gui._output_text.get("1.0", "end")
    assert str(gui._output_text.cget("state")) == "disabled"


def test_build_local_provider_by_id_raises_if_missing_or_not_local():
    smell = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt=None,
        enabled=False,
    )
    api_provider = LLMProviderDefinition(
        provider_id="p1",
        kind=ProviderKind.API,
        display_name="Api",
        config={},
    )
    cat = _catalog_with_smells_and_providers(smells=[smell], providers=[api_provider])
    with pytest.raises(RuntimeError):
        PromptEngineeringGUI._build_local_provider_by_id(cat, "p1")
    with pytest.raises(RuntimeError):
        PromptEngineeringGUI._build_local_provider_by_id(cat, "missing")


def test_set_running_state_disables_controls_and_restores_after(tk_root, mocker):
    gui, _svc = _make_ready_gui(tk_root)
    refresh = mocker.patch.object(gui, "_refresh_prompt_view")

    gui._set_running_state(True)
    assert str(gui._cancel_btn.cget("state")) == "normal"
    assert str(gui._prompt_text.cget("state")) == "disabled"
    assert str(gui._test_btn.cget("state")) == "disabled"

    gui._set_running_state(False)
    refresh.assert_called()
    assert str(gui._cancel_btn.cget("state")) == "disabled"


def test_schedule_and_stop_heartbeat_update_status_once(tk_root, mocker):
    gui, _svc = _make_ready_gui(tk_root)
    gui._running_total = 3
    gui._running_index = 1
    gui._running_filename = "a.py"

    calls = {"n": 0, "canceled": False}
    scheduled = {"tick": None}

    def fake_after(_ms, func=None, *args):
        # Store the first scheduled callback; let the test trigger it explicitly.
        if func is not None and scheduled["tick"] is None:
            scheduled["tick"] = lambda: func(*args)
        return "hb-id"

    def fake_after_cancel(_id):
        calls["canceled"] = True

    mocker.patch.object(gui.master, "after", side_effect=fake_after)
    mocker.patch.object(gui.master, "after_cancel", side_effect=fake_after_cancel)

    gui._schedule_heartbeat()

    # _schedule_heartbeat() calls _stop_heartbeat(), which clears _run_started_at.
    # Set it now, then trigger the scheduled tick once.
    gui._run_started_at = 0.0
    assert scheduled["tick"] is not None
    scheduled["tick"]()
    assert "Running:" in gui._status_var.get()
    assert "file: a.py" in gui._status_var.get()

    gui._stop_heartbeat()
    assert calls["canceled"] is True


def test_on_add_smell_calls_dialog_and_selects_new_smell(tk_root, mocker):
    gui, svc = _make_ready_gui(tk_root)
    # Start from a known smell.
    gui._current_smell_id = "s1"
    gui._draft_dirty = False

    def fake_dialog(_master, catalog_service, on_success):
        catalog_service.add_smell("s2", display_name="S2")
        on_success("s2")

    mocker.patch("prompt_engineering.prompt_engineering_gui.AddSmellDialog", side_effect=fake_dialog)
    gui._on_add_smell()

    assert gui._current_smell_id == "s2"
    assert "s2" in "".join(gui._smell_combo["values"])


def test_on_smell_selected_with_dirty_draft_and_user_cancels_restores_combo(tk_root, mocker):
    # Two smells.
    smell1 = LLMSmellDefinition(
        smell_id="s1",
        display_name="S1",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    smell2 = LLMSmellDefinition(
        smell_id="s2",
        display_name="S2",
        description="d",
        default_prompt="default",
        draft_prompt="draft",
        enabled=False,
    )
    local_provider = LLMProviderDefinition(
        provider_id="local",
        kind=ProviderKind.LOCAL,
        display_name="Local",
        config={"model_name": "x"},
    )
    svc = FakeCatalogService(
        _catalog_with_smells_and_providers(smells=[smell1, smell2], providers=[local_provider])
    )
    gui = PromptEngineeringGUI(tk_root, catalog_service=svc)

    # Initially selected smell is first one.
    assert gui._current_smell_id == "s1"
    initial_display = gui._smell_combo.get()

    # Try selecting second smell while dirty; user cancels.
    gui._draft_dirty = True
    mocker.patch("tkinter.messagebox.askyesno", return_value=False)
    second_display = [v for v in gui._smell_combo["values"] if "(s2)" in v][0]
    gui._smell_combo.set(second_display)
    gui._on_smell_selected()

    assert gui._current_smell_id == "s1"
    assert gui._smell_combo.get() == initial_display
