import json
import re
from unittest.mock import mock_open, patch

import chess
import pytest

pytest.importorskip("PySide6")
pytestmark = pytest.mark.gui

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from gui import ChessGUI, PromotionDialog
from main import ReportingLevel

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub("", text)


def _button_with_text(gui: ChessGUI, label: str) -> QPushButton:
    for button in gui.findChildren(QPushButton):
        if button.text() == label:
            return button
    raise AssertionError(f"Button with text '{label}' not found")


@pytest.fixture(scope="session")
def app():
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application


@pytest.fixture()
def chess_gui(app):
    board = chess.Board()
    gui = ChessGUI(
        board,
        dev=True,
        reporting_level=ReportingLevel.VERBOSE,
    )
    gui.show()
    QTest.qWait(50)
    yield gui
    gui.close()
    QTest.qWait(20)


def test_window_properties(chess_gui):
    assert chess_gui.isVisible()
    assert chess_gui.windowTitle() == "JaskFish"
    assert chess_gui.minimumWidth() >= 450
    assert chess_gui.minimumHeight() >= 675
    assert chess_gui.turn_indicator.text() == "White's turn"
    assert chess_gui.info_indicator.text() == "Game Started"


def test_control_buttons_present(chess_gui):
    button_texts = {button.text() for button in chess_gui.findChildren(QPushButton)}
    expected = {
        "Undo",
        "Export",
        "Reset board",
        "Ready",
        "Go",
        "Start Self-Play",
        "Swap Colors",
        "Restart Engine",
        "Deactivate Engine 1",
        "Deactivate Engine 2",
    }
    assert expected.issubset(button_texts)


def test_board_interaction_moves_piece(chess_gui):
    e2_button = chess_gui.squares[chess.E2]
    e4_button = chess_gui.squares[chess.E4]

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.mouseClick(e4_button, Qt.LeftButton)
    QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.E4).symbol().lower() == "p"
    assert chess_gui.board.piece_at(chess.E2) is None
    assert chess_gui.turn_indicator.text() == "Black's turn"


def test_invalid_move_logs_message(chess_gui, capfd):
    capfd.readouterr()

    e2_button = chess_gui.squares[chess.E2]
    e5_button = chess_gui.squares[chess.E5]

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.mouseClick(e5_button, Qt.LeftButton)
    QTest.qWait(20)

    output = _strip_ansi(capfd.readouterr().out)
    assert "Invalid Move" in output


def test_promotion_dialog_selection(chess_gui):
    promotion_fen = "8/5P2/8/8/8/8/8/6k1 w - - 0 1"
    chess_gui.board.set_fen(promotion_fen)
    chess_gui.update_board()

    f7_button = chess_gui.squares[chess.F7]
    f8_button = chess_gui.squares[chess.F8]

    with (
        patch.object(PromotionDialog, "exec", return_value=True),
        patch.object(PromotionDialog, "get_promotion_piece", return_value="queen"),
    ):
        QTest.mouseClick(f7_button, Qt.LeftButton)
        QTest.mouseClick(f8_button, Qt.LeftButton)
        QTest.qWait(20)

    piece = chess_gui.board.piece_at(chess.F8)
    assert piece is not None and piece.symbol().lower() == "q"


def test_ready_go_restart_callbacks(chess_gui):
    ready_requests = []
    go_requests = []
    restart_requests = []

    chess_gui.ready_callback = lambda engine_id, label: ready_requests.append(
        (engine_id, label)
    )
    chess_gui.go_callback = lambda engine_id, label, fen: go_requests.append(
        (engine_id, label, fen)
    )
    chess_gui.restart_engine_callback = lambda: restart_requests.append("restart")

    chess_gui.set_manual_engine_provider(lambda: ("engine1", "Primary Engine"))

    chess_gui.ready_command()
    chess_gui.go_command()
    chess_gui.restart_engine()

    assert ready_requests == [("engine1", "Engine 1"), ("engine2", "Engine 2")]
    assert go_requests and go_requests[0][0] == "engine1"
    assert go_requests[0][1] == "Primary Engine"
    assert go_requests[0][2] == chess_gui.board.fen()
    assert chess_gui.manual_engine_busy is True
    assert chess_gui.info_indicator.text().endswith("Primary Engine evaluating…")
    assert restart_requests == ["restart"]


def test_self_play_toggle_uses_callbacks(chess_gui):
    events = []
    chess_gui.set_self_play_callbacks(
        lambda: events.append("start") or True,
        lambda: events.append("stop") or True,
    )

    chess_gui.toggle_self_play()
    assert events == ["start"]
    # Manager callback sets the state; simulate that transition.
    chess_gui.set_self_play_active(True)
    assert chess_gui.self_play_button.text() == "Stop Self-Play"
    chess_gui.toggle_self_play()
    assert events == ["start", "stop"]
    chess_gui.set_self_play_active(False)
    assert chess_gui.self_play_button.text() == "Start Self-Play"


def test_export_game_creates_expected_payload(chess_gui):
    fixed_time = "2025-11-04 10:00:00"
    chess_gui.board.push_san("e4")

    with (
        patch("time.strftime", return_value=fixed_time),
        patch("time.localtime"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open()) as mocked_file,
    ):
        chess_gui.export_game()

    mocked_file.assert_called_once_with(f"gamestates/chess_game_{fixed_time}.json", "w")
    handle = mocked_file()
    payload = json.loads(handle.write.call_args[0][0])
    assert payload["fen-final"] == chess_gui.board.fen()
    assert payload["san"] == "e4"
    assert payload["uci"] == "e2e4"


def test_engine_assignment_and_activation_helpers(chess_gui):
    chess_gui.set_engine_assignments("PVS", "Sunfish")
    assert "PVS" in chess_gui.engine_assignment_label.text()
    assert "Sunfish" in chess_gui.engine_assignment_label.text()

    chess_gui.set_engine_labels({"engine1": "Primary", "engine2": "Secondary"})
    chess_gui.set_engine_activation_states({"engine1": True, "engine2": False})
    button_states = {
        button.property("engineId"): button.text()
        for button in chess_gui.engine_toggle_buttons.values()
    }
    assert button_states["engine1"] == "Deactivate Primary"
    assert button_states["engine2"] == "Activate Secondary"


def test_update_board_triggers_game_over(chess_gui):
    checkmate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    chess_gui.board.set_fen(checkmate_fen)

    with patch.object(QMessageBox, "information") as mock_info:
        chess_gui.update_board()

    mock_info.assert_called_once()
    args, _ = mock_info.call_args
    assert args[0] is chess_gui
    assert args[1] == "Game Over"
    assert "Checkmate" in args[2]
