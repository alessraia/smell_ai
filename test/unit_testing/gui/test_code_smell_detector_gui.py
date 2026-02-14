import pytest
import tkinter as tk
from gui.code_smell_detector_gui import CodeSmellDetectorGUI


@pytest.fixture
def gui(mocker):
    """
    Fixture to create a tkinter root window
    and initialize the CodeSmellDetectorGUI.
    Mock Tk and other tkinter dialog
    components to avoid errors in headless environments.
    """

    mocker.patch("tkinter.Tk", return_value=tk.Tk())

    root = tk.Tk()
    gui = CodeSmellDetectorGUI(root)
    yield gui
    root.quit()
    root.update()
    root.destroy()


def test_choose_input_path(gui, mocker):
    """
    Test the `choose_input_path` method
    to ensure the input path label is updated.
    """

    mocker.patch(
        "tkinter.filedialog.askdirectory", return_value="/mock/input/path"
    )

    gui.choose_input_path()

    assert gui.input_path.cget("text") == "/mock/input/path"


def test_choose_output_path(gui, mocker):
    """
    Test the `choose_output_path` method to
    ensure the output path label is updated.
    """
    mocker.patch(
        "tkinter.filedialog.askdirectory", return_value="/mock/output/path"
    )

    gui.choose_output_path()

    assert gui.output_path.cget("text") == "/mock/output/path"


def test_run_program_missing_paths(gui, mocker):
    """
    Test the `run_program` method when input or output paths are missing.
    """
    gui.input_path.configure(text="No path selected")
    gui.output_path.configure(text="No path selected")

    mock_stdout = mocker.patch("sys.stdout", new_callable=mocker.MagicMock)

    gui.run_program()
    output = "".join([call[0][0] for call in mock_stdout.write.call_args_list])
    assert "Error: Please select valid input and output paths." in output


def test_disable_key_press(mocker, gui):
    """
    Test that key presses are disabled in the output Text widget.
    """
    event = mocker.MagicMock()
    result = gui.disable_key_press(event)

    assert result == "break"


def test_run_analysis_with_resume(gui, mocker):
    """
    Test run_analysis with resume mode enabled.
    Branch: if not is_resume (line 402) -> False path
    """
    # Arrange
    mock_project_analyzer = mocker.MagicMock()
    mocker.patch(
        'gui.code_smell_detector_gui.ProjectAnalyzer',
        return_value=mock_project_analyzer
    )
    
    # Act
    gui.run_analysis(
        input_path="/test/input",
        output_path="/test/output",
        num_walkers=1,
        is_parallel=False,
        is_resume=True,  # Resume mode enabled
        is_multiple=False,
        use_llm=False
    )
    
    # Assert
    # clean_output_directory should NOT be called when resume is True
    mock_project_analyzer.clean_output_directory.assert_not_called()
    mock_project_analyzer.analyze_project.assert_called_once_with("/test/input")


def test_gui_layout(gui):
    """
    Test that the GUI layout contains the expected widgets.
    """
    widgets = [
        gui.input_label,
        gui.input_button,
        gui.output_label,
        gui.output_button,
        gui.walker_picker,
        gui.parallel_check,
        gui.resume_check,
        gui.run_button,
        gui.exit_button,
        gui.output_textbox,
    ]

    for widget in widgets:
        assert widget.winfo_exists()
