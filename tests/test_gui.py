import json
import os
import sys
import time
from unittest.mock import mock_open, patch

import chess
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QPushButton

from gui import ChessGUI, PromotionDialog


@pytest.fixture(scope="session")
def app():
    """Provide a single QApplication for all GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    app.quit()


@pytest.fixture
def chess_gui(app):
    """Create a ChessGUI instance in developer mode for deterministic behavior."""
    board = chess.Board()
    gui = ChessGUI(board, dev=True)
    gui.show()
    QTest.qWait(50)
    yield gui
    gui.close()
    QTest.qWait(50)


def _button_with_text(gui: ChessGUI, text: str) -> QPushButton:
    for button in gui.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"Button with text '{text}' not found")


def test_window_properties(chess_gui):
    assert chess_gui.isVisible()
    assert chess_gui.windowTitle() == "JaskFish"
    assert chess_gui.minimumWidth() >= 450
    assert chess_gui.minimumHeight() >= 600
    assert chess_gui.width() >= chess_gui.minimumWidth()
    assert chess_gui.height() >= chess_gui.minimumHeight()


def test_layout_structure(chess_gui):
    board_widget = chess_gui.squares[chess.A1].parentWidget()
    grid_layout = board_widget.layout()

    assert isinstance(chess_gui.turn_indicator, QLabel)
    assert isinstance(chess_gui.info_indicator, QLabel)

    files = "abcdefgh"
    for column, file_char in enumerate(files):
        widget = grid_layout.itemAtPosition(8, column + 1).widget()
        assert isinstance(widget, QLabel)
        assert widget.text() == file_char

    for row in range(8):
        widget = grid_layout.itemAtPosition(row, 0).widget()
        assert isinstance(widget, QLabel)
        assert widget.text() == str(8 - row)

    button_texts = {button.text() for button in chess_gui.findChildren(QPushButton)}
    assert {"Undo", "Export", "Reset board", "Ready", "Go", "Restart Engine"}.issubset(button_texts)


def test_board_squares_initialization(chess_gui):
    assert len(chess_gui.squares) == 64
    for button in chess_gui.squares.values():
        assert isinstance(button, QPushButton)


def test_selecting_and_moving_pieces(chess_gui):
    e2_button = chess_gui.squares[chess.E2]
    e4_button = chess_gui.squares[chess.E4]

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.qWait(20)
    assert chess_gui.selected_square == chess.E2

    QTest.mouseClick(e4_button, Qt.LeftButton)
    QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.E4).symbol().lower() == "p"
    assert chess_gui.board.piece_at(chess.E2) is None
    assert chess_gui.turn_indicator.text() == "Black's turn"


def test_invalid_move_rejection(chess_gui, capfd):
    capfd.readouterr()

    e2_button = chess_gui.squares[chess.E2]
    e5_button = chess_gui.squares[chess.E5]

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.mouseClick(e5_button, Qt.LeftButton)
    QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.E5) is None
    assert chess_gui.board.piece_at(chess.E2) is not None

    output = capfd.readouterr().out
    assert "Invalid Move" in output


def test_promotion_dialog_selection(chess_gui):
    promotion_fen = "8/5P2/8/8/8/8/8/7k w - - 0 1"
    chess_gui.board.set_fen(promotion_fen)
    chess_gui.update_board()

    f7_button = chess_gui.squares[chess.F7]
    f8_button = chess_gui.squares[chess.F8]

    with patch.object(PromotionDialog, "exec", return_value=True), \
         patch.object(PromotionDialog, "get_promotion_piece", return_value="queen"):
        QTest.mouseClick(f7_button, Qt.LeftButton)
        QTest.mouseClick(f8_button, Qt.LeftButton)
        QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.F8).symbol().lower() == "q"


def test_promotion_dialog_cancel(chess_gui):
    promotion_fen = "8/5P2/8/8/8/8/8/7k w - - 0 1"
    chess_gui.board.set_fen(promotion_fen)
    chess_gui.update_board()

    f7_button = chess_gui.squares[chess.F7]
    f8_button = chess_gui.squares[chess.F8]

    with patch.object(PromotionDialog, "exec", return_value=False):
        QTest.mouseClick(f7_button, Qt.LeftButton)
        QTest.mouseClick(f8_button, Qt.LeftButton)
        QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.F8) is None
    assert chess_gui.board.piece_at(chess.F7).symbol().lower() == "p"


def test_undo_button_reverts_last_move(chess_gui):
    e2_button = chess_gui.squares[chess.E2]
    e4_button = chess_gui.squares[chess.E4]
    undo_button = _button_with_text(chess_gui, "Undo")

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.mouseClick(e4_button, Qt.LeftButton)
    QTest.qWait(20)

    QTest.mouseClick(undo_button, Qt.LeftButton)
    QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.E4) is None
    assert chess_gui.board.piece_at(chess.E2).symbol().lower() == "p"


def test_reset_board_button_restores_starting_position(chess_gui):
    non_start_fen = "k7/2Q4P/8/8/8/8/8/K2R4 w - - 0 1"
    chess_gui.board.set_fen(non_start_fen)
    chess_gui.update_board()

    reset_button = _button_with_text(chess_gui, "Reset board")

    QTest.mouseClick(reset_button, Qt.LeftButton)
    QTest.qWait(20)

    assert chess_gui.board.fen() == chess.Board().fen()
    assert chess_gui.info_indicator.text() == "Game Reset"
    assert chess_gui.turn_indicator.text() == "White's turn"


def test_export_game_creates_expected_payload(chess_gui):
    fixed_time = "2024-09-15 12:00:00"

    with patch("time.strftime", return_value=fixed_time), \
         patch("time.localtime", return_value=time.localtime()), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()) as mocked_file:
        chess_gui.export_game()

    mocked_file.assert_called_once_with(f"gamestates/chess_game_{fixed_time}.json", "w")
    handle = mocked_file()
    expected_payload = {
        "export-time": fixed_time,
        "fen-init": chess.Board().fen(),
        "fen-final": chess_gui.board.fen(),
        "san": "",
        "uci": ""
    }
    handle.write.assert_called_once_with(json.dumps(expected_payload))


def test_ready_go_restart_without_callbacks(chess_gui, capfd):
    capfd.readouterr()

    for label in ("Ready", "Go", "Restart Engine"):
        button = _button_with_text(chess_gui, label)
        QTest.mouseClick(button, Qt.LeftButton)
        QTest.qWait(10)

    output = capfd.readouterr().out
    assert "Ready callback not set" in output
    assert "Go callback not set" in output
    assert "Restart engine callback not set" in output


def test_game_over_detection_triggers_message_box(chess_gui):
    checkmate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    chess_gui.board.set_fen(checkmate_fen)

    with patch.object(QMessageBox, "information") as mock_info:
        chess_gui.update_board()

    mock_info.assert_called_once()
    args, _ = mock_info.call_args
    assert args[0] is chess_gui
    assert args[1] == "Game Over"
    assert "Checkmate" in args[2]


def test_dev_mode_debug_information(chess_gui, capfd):
    capfd.readouterr()

    e2_button = chess_gui.squares[chess.E2]
    e4_button = chess_gui.squares[chess.E4]

    QTest.mouseClick(e2_button, Qt.LeftButton)
    QTest.mouseClick(e4_button, Qt.LeftButton)
    QTest.qWait(20)

    output = capfd.readouterr().out
    assert "Move attempted: e2 -> e4" in output


def test_developer_mode_player_color(chess_gui):
    assert chess_gui.player_is_white is True


def test_attempt_engine_move_uses_chess_logic(chess_gui):
    chess_gui.board.reset()
    chess_gui.update_board()

    chess_gui.attempt_engine_move("e2e4")
    QTest.qWait(20)

    assert chess_gui.board.piece_at(chess.E4).symbol().lower() == "p"


def test_gui_remains_responsive_during_clicks(chess_gui):
    squares = list(chess_gui.squares.values())[:10]
    for button in squares:
        QTest.mouseClick(button, Qt.LeftButton)
        QTest.qWait(5)

    assert chess_gui.isVisible()
