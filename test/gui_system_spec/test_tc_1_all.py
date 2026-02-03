from __future__ import annotations

from pathlib import Path

import pytest


# Smell name “veri” (detection_rules/*)
GENERIC_SMELLS = {
    "columns_and_datatype_not_explicitly_set",
    "empty_column_misinitialization",
    "in_place_apis_misused",
}

AI_SMELLS = {
    "Chain_Indexing",
    "matrix_multiplication_api_misused",
    "gradients_not_cleared_before_backward_propagation",
    "dataframe_conversion_misused",
}


# -----------------------
# Helper GUI utilities
# -----------------------
def _set_label(label, value: str) -> None:
    label.configure(text=value)


def _set_spinbox(spinbox, value: str) -> None:
    spinbox.delete(0, "end")
    spinbox.insert(0, value)


def _run(gui_app, capsys) -> str:
    from test.gui_system_spec.conftest import _wait_for_daemon_threads
    
    gui_app.run_program()
    
    # Aspetta che TUTTI i thread daemon completino (per test paralleli/async)
    _wait_for_daemon_threads(timeout=60)
    
    return capsys.readouterr().out


def _overview_path(output_dir: Path) -> Path:
    return output_dir / "output" / "overview.csv"


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _smell_names_from_overview(csv_path: Path) -> set[str]:
    # parsing “lite”: basta cercare la colonna smell_name (3a colonna)
    # e raccogliere tokens noti (robusto vs ordine/righe)
    text = _read_text(csv_path)
    found = set()
    for s in GENERIC_SMELLS | AI_SMELLS:
        if s in text:
            found.add(s)
    return found


def _assert_smells_generic_only(output_dir: Path):
    ov = _overview_path(output_dir)
    assert ov.exists(), f"Missing {ov}"
    found = _smell_names_from_overview(ov)
    assert found & GENERIC_SMELLS, "Expected at least one GENERIC smell"
    assert not (found & AI_SMELLS), "Did not expect AI smells here"


def _assert_smells_ai_only(output_dir: Path):
    ov = _overview_path(output_dir)
    assert ov.exists(), f"Missing {ov}"
    found = _smell_names_from_overview(ov)
    assert found & AI_SMELLS, "Expected at least one AI smell"
    assert not (found & GENERIC_SMELLS), "Did not expect GENERIC smells here"


def _assert_smells_mixed(output_dir: Path):
    ov = _overview_path(output_dir)
    assert ov.exists(), f"Missing {ov}"
    found = _smell_names_from_overview(ov)
    assert found & AI_SMELLS, "Expected at least one AI smell"
    assert found & GENERIC_SMELLS, "Expected at least one GENERIC smell"


# -----------------------
# TC_1.1 ... TC_1.43
# -----------------------

@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_1(gui_app, tmp_output_dir, capsys):
    # FI1 (input vuoto), FO2 (output valido)
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "Error: Please select valid input and output paths." in out
    assert gui_app.input_path.cget("text") == "No path selected"


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_2(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # FO1 (output vuoto)
    # input: 1 file .py con smell misti (costruiamo mix: chain_indexing + columns_dtype_not_set)
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_1_2_mixed_single")
    _set_label(gui_app.input_path, str(proj))
    # output lasciato "No path selected"
    out = _run(gui_app, capsys)
    assert "Error: Please select valid input and output paths." in out
    assert gui_app.output_path.cget("text") == "No path selected"


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_3(gui_app, project_factory, tmp_output_dir, capsys):
    # input valido ma cartella vuota -> no python files
    proj = project_factory["empty"]("tc_1_3_empty")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "An error occurred during analysis:" in out
    assert "contains no Python files." in out


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_4(gui_app, project_factory, tmp_output_dir, capsys):
    # input valido con 1 file ma nessun .py -> no python files
    proj = project_factory["single_no_py"]("tc_1_4_no_py")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "An error occurred during analysis:" in out
    assert "contains no Python files." in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non valida max_workers (spec TC_1.5)")
def test_TC_1_5(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # NW1 (<=0) -> spec: errore max_workers > 0
    proj = project_factory["single_from_tc"]("TC12", "tc_1_5_single_py")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "-1")
    out = _run(gui_app, capsys)
    assert "max_workers must be greater than 0" in out


def test_TC_1_6(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project + Parallel, NP2, smells misti
    p1 = project_factory["single_custom"](
        [fixtures_root / "TC12" / "chain_indexing.py", fixtures_root / "TC19" / "columns_dtype_not_set.py"],
        "tc_1_6_p1",
    )
    p2 = project_factory["single_custom"](
        [fixtures_root / "TC3" / "matrix_multiplication_misused.py", fixtures_root / "TC19" / "in_place_api_misused.py"],
        "tc_1_6_p2",
    )
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_6_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(False)

    _run(gui_app, capsys)
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_7(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project sequenziale, NP2, smells misti
    p1 = project_factory["single_custom"](
        [fixtures_root / "TC12" / "chain_indexing.py", fixtures_root / "TC19" / "columns_dtype_not_set.py"],
        "tc_1_7_p1",
    )
    p2 = project_factory["single_custom"](
        [fixtures_root / "TC3" / "matrix_multiplication_misused.py", fixtures_root / "TC19" / "in_place_api_misused.py"],
        "tc_1_7_p2",
    )
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_7_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)

    _run(gui_app, capsys)
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_8(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project + Resume (execution_log presente) -> salta già loggati
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_8_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_8_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_8_base")

    # execution_log: ultimo = project1 => skip project1
    (base / "execution_log.txt").write_text("project1\n", encoding="utf-8")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(True)

    out = _run(gui_app, capsys)

    # evidenza “skip”: non deve analizzare project1
    assert "Analyzing project 'project1' sequentially" not in out
    assert "Analyzing project 'project2' sequentially" in out

    # overview solo da project2 => generico-only
    _assert_smells_generic_only(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_9(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Resume (execution_log NON presente) -> non può saltare nulla, analizza tutti
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_9_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_9_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_9_base")

    # no execution_log.txt

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(True)

    out = _run(gui_app, capsys)
    assert "Analyzing project 'project1' sequentially" in out
    assert "Analyzing project 'project2' sequentially" in out
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_10(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Single project, smell generici
    proj = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_10_generic")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_generic_only(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_11(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Single project, smell AI-specific
    proj = project_factory["single_custom"]([fixtures_root / "TC3" / "matrix_multiplication_misused.py"], "tc_1_11_ai")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_ai_only(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_12(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Single project, smell misti
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_1_12_mixed")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_13(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # NP2 ma Multiple=OFF: l’analisi scansiona ricorsivamente tutto come “un progetto”
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_13_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_13_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_13_base")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(False)

    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_14(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Single project (NF2), smell misti
    ai_file = fixtures_root / "TC12" / "chain_indexing.py"
    gen_file = fixtures_root / "TC19" / "columns_dtype_not_set.py"
    proj = project_factory["single_custom"]([ai_file, gen_file], "tc_1_14")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_15(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project sequenziale (P2), smell misti
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_15_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_15_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_15_base")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)

    _run(gui_app, capsys)
    _assert_smells_mixed(tmp_output_dir)


def test_TC_1_16(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project parallel (P1)
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_16_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_16_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_16_base")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(False)

    _run(gui_app, capsys)
    _assert_smells_mixed(tmp_output_dir)


def test_TC_1_17(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project parallel con >1 walker (NW3)
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_17_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_17_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_17_base")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "2")
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(False)

    _run(gui_app, capsys)
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_18(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Multi-project + Resume (SF1)
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_18_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_18_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_18_base")
    (base / "execution_log.txt").write_text("project1\n", encoding="utf-8")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(True)

    out = _run(gui_app, capsys)
    assert "Analyzing project 'project1' sequentially" not in out
    assert "Analyzing project 'project2' sequentially" in out
    _assert_smells_generic_only(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non valida max_workers non numerico (spec TC_1.19)")
def test_TC_1_19(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC12", "tc_1_19")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    _set_spinbox(gui_app.walker_picker, "a")
    out = _run(gui_app, capsys)
    assert "max_workers must be a number" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa messaggio 'In parallel mode, resume mode is ignored' (spec TC_1.20)")
def test_TC_1_20(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_20_base_multi")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "In parallel mode, resume mode is ignored" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa messaggio 'In parallel mode, resume mode is ignored' (spec TC_1.21)")
def test_TC_1_21(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_21_base_multi")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "In parallel mode, resume mode is ignored" in out


# TC_1.22 .. TC_1.41: il documento richiede messaggi/controlli non presenti nella GUI attuale
# (Multiple con un solo progetto; Parallel/Resume “ignored without Multiple”; ecc.)
# Li lasciamo come singoli test case, ma xfail “as-spec”.

@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple con 1 solo progetto (spec TC_1.22)")
def test_TC_1_22(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # base con 1 progetto soltanto
    p1 = project_factory["single_from_tc"]("TC12", "tc_1_22_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_22_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Parallel con 1 solo progetto (spec TC_1.23)")
def test_TC_1_23(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC12", "tc_1_23_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_23_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Resume con 1 solo progetto (spec TC_1.24)")
def test_TC_1_24(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC12", "tc_1_24_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_24_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Parallel+Resume con 1 solo progetto (spec TC_1.25)")
def test_TC_1_25(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC12", "tc_1_25_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_25_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.26)")
def test_TC_1_26(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Parallel+Resume ma Multiple=OFF
    proj = project_factory["single_from_tc"]("TC12", "tc_1_26")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Parallel mode and Resume mode are ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.27)")
def test_TC_1_27(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC12", "tc_1_27")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.28)")
def test_TC_1_28(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC12", "tc_1_28")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Resume mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple con 1 progetto (spec TC_1.29)")
def test_TC_1_29(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC7", "tc_1_29_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_29_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Parallel con 1 progetto (spec TC_1.30)")
def test_TC_1_30(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC7", "tc_1_30_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_30_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Resume con 1 progetto (spec TC_1.31)")
def test_TC_1_31(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC7", "tc_1_31_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_31_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non blocca Multiple+Parallel+Resume con 1 progetto (spec TC_1.32)")
def test_TC_1_32(gui_app, project_factory, tmp_output_dir, capsys, fixtures_root):
    p1 = project_factory["single_from_tc"]("TC7", "tc_1_32_p1")
    base = project_factory["multi_base"]({"project1": p1}, "tc_1_32_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(True)
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    out = _run(gui_app, capsys)
    assert "Multiple mode isn’t available with only one project" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.33)")
def test_TC_1_33(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC7", "tc_1_33")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode and Resume mode are ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.34)")
def test_TC_1_34(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC7", "tc_1_34")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.35)")
def test_TC_1_35(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    proj = project_factory["single_from_tc"]("TC7", "tc_1_35")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Resume mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.36)")
def test_TC_1_36(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_36_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode and Resume mode are ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.37)")
def test_TC_1_37(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_37_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode and Resume mode are ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.38)")
def test_TC_1_38(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_38_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Resume mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.39)")
def test_TC_1_39(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_39_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.resume_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Resume mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.40)")
def test_TC_1_40(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_40_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
@pytest.mark.xfail(strict=True, reason="GUI attuale non stampa 'ignored without Multiple' (spec TC_1.41)")
def test_TC_1_41(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    base = project_factory["single_from_tc"]("TC11", "tc_1_41_base")
    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.parallel_var.set(True)
    gui_app.multiple_var.set(False)
    out = _run(gui_app, capsys)
    assert "Parallel mode is ignored without Multiple mode" in out


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_42(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # NP2 ma Multiple=OFF, smells presenti (misti)
    p1 = project_factory["single_custom"]([fixtures_root / "TC12" / "chain_indexing.py"], "tc_1_42_p1")
    p2 = project_factory["single_custom"]([fixtures_root / "TC19" / "columns_dtype_not_set.py"], "tc_1_42_p2")
    base = project_factory["multi_base"]({"project1": p1, "project2": p2}, "tc_1_42_base")

    _set_label(gui_app.input_path, str(base))
    _set_label(gui_app.output_path, str(tmp_output_dir))
    gui_app.multiple_var.set(False)
    gui_app.parallel_var.set(False)
    gui_app.resume_var.set(False)

    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    _assert_smells_mixed(tmp_output_dir)


@pytest.mark.usefixtures("force_sync_threads")
def test_TC_1_43(gui_app, project_factory, tmp_output_dir, fixtures_root, capsys):
    # Single project con 0 smell (NCS1)
    # prendo un file “banale” senza smell (TC6/MockDirectory2/checkeven.py)
    no_smell_file = fixtures_root / "TC6" / "MockDirectory2" / "checkeven.py"
    proj = project_factory["single_custom"]([no_smell_file], "tc_1_43_no_smells")
    _set_label(gui_app.input_path, str(proj))
    _set_label(gui_app.output_path, str(tmp_output_dir))

    out = _run(gui_app, capsys)
    assert "Analysis completed." in out
    assert "Total code smells found: 0" in out

    # l’implementazione attuale non salva overview.csv se df è vuoto
    assert not _overview_path(tmp_output_dir).exists()
