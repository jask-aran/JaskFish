import sys
import pytest
import chess
from unittest.mock import patch

pytest.importorskip("PySide2")

from PySide2.QtWidgets import QApplication, QPushButton, QLabel, QDialog, QMessageBox
from PySide2.QtTest import QTest
from PySide2.QtCore import Qt
from gui import ChessGUI, PromotionDialog

@pytest.fixture(scope="session")
def app():
    """Session-wide QApplication instance."""
    app = QApplication(sys.argv)
    yield app
    app.quit()

@pytest.fixture
def chess_gui(app):
    """Function-scoped ChessGUI instance."""
    board = chess.Board()
    gui = ChessGUI(board, dev=True)
    gui.show()
    yield gui
    gui.close()

def test_window_creation(chess_gui):
    """Test if the ChessGUI window is created and visible with the correct title."""
    assert chess_gui.isVisible(), "ChessGUI window should be visible"
    assert chess_gui.windowTitle() == 'JaskFish', "Window title should be 'JaskFish'"

def test_board_squares_initialization(chess_gui):
    """Test if the board squares are initialized correctly."""
    # Check that 64 squares are initialized (8x8 chess board)
    assert len(chess_gui.squares) == 64, "There should be 64 squares initialized for the chessboard"

    # Check that each square is a QPushButton
    for button in chess_gui.squares.values():
        assert isinstance(button, QPushButton), "Each square should be an instance of QPushButton"

def test_square_click(chess_gui):
    """Test if clicking on a square selects it correctly."""
    # Select a square (e.g., e2), which should select the piece on it
    e2_button = chess_gui.squares[chess.E2]

    # Simulate a click on the e2 square (white pawn's starting position)
    QTest.mouseClick(e2_button, Qt.LeftButton)

    # Assert the selected square is e2
    assert chess_gui.selected_square == chess.E2, "The e2 square should be selected after clicking it"

import sys

def test_reset_button(chess_gui, capfd):
    """Test if the 'Reset' button resets the board correctly from a non-starting position."""
    non_start_fen = "k7/2Q4P/8/8/8/8/8/K2R4 w - - 0 1"
    chess_gui.board = chess.Board(non_start_fen)
    chess_gui.update_board()

    assert chess_gui.board.fen() != chess.Board().fen(), "The board should be in a custom position before reset"

    chess_gui.reset_game()
    
    post_reset_fen = chess_gui.board.fen()
    print(f"Post reset FEN: {post_reset_fen}")

    captured = capfd.readouterr()
    print(captured.out.strip())  # Print the captured output

    assert chess_gui.board.fen() == chess.Board().fen(), "The board should be reset to the initial position"



def test_promotion_dialog(app):
    """Test if the promotion dialog works correctly."""
    dialog = PromotionDialog()

    # Ensure the dialog is a QDialog and contains a QComboBox
    assert isinstance(dialog, QDialog), "PromotionDialog should be an instance of QDialog"
    assert hasattr(dialog, 'combo'), "PromotionDialog should have a 'combo' attribute"
    assert dialog.combo.count() == 4, "The combo box should contain 4 promotion options"
    assert dialog.combo.itemText(0) == "Queen", "The first promotion option should be 'Queen'"
    assert dialog.combo.itemText(1) == "Rook", "The second promotion option should be 'Rook'"
    assert dialog.combo.itemText(2) == "Bishop", "The third promotion option should be 'Bishop'"
    assert dialog.combo.itemText(3) == "Knight", "The fourth promotion option should be 'Knight'"

    # Simulate selecting "Knight" and clicking OK
    dialog.combo.setCurrentIndex(3)  # Select "Knight"
    ok_button = dialog.findChild(QPushButton, "OK")
    if not ok_button:
        # If the OK button is not named, find it by text
        for button in dialog.findChildren(QPushButton):
            if button.text() == "OK":
                ok_button = button
                break
    assert ok_button is not None, "OK button should be present in the PromotionDialog"

    QTest.mouseClick(ok_button, Qt.LeftButton)

    # Assert that the promotion piece is 'knight'
    assert dialog.get_promotion_piece() == "knight", "The promotion piece should be 'knight' after selection"
