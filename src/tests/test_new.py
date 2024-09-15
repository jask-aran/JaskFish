# /home/jask/JaskFish/src/tests/test_gui.py

import pytest
import sys
import os
import json
import time
import chess
from unittest.mock import MagicMock, patch

from PySide2.QtWidgets import QApplication, QPushButton, QLabel, QDialog, QMessageBox
from PySide2.QtCore import Qt
from PySide2.QtTest import QTest

from gui import ChessGUI, PromotionDialog

pytestmark = pytest.mark.skip(reason="Skipping this test file for now")


@pytest.fixture(scope="session")
def app():
    """Fixture to initialize the QApplication."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app

@pytest.fixture
def chess_gui(qtbot):
    """Fixture to initialize ChessGUI with a starting board in developer mode."""
    board = chess.Board()
    gui = ChessGUI(board, dev=True)
    qtbot.addWidget(gui)
    gui.show()
    return gui

@pytest.fixture
def non_start_board(qtbot):
    """Fixture to initialize ChessGUI with a non-starting FEN."""
    fen = "k7/2Q4P/8/8/8/8/8/K2R4 w - - 0 1"
    board = chess.Board(fen)
    gui = ChessGUI(board, dev=True)
    qtbot.addWidget(gui)
    gui.show()
    return gui

@pytest.fixture
def mock_callbacks():
    """Fixture to provide mock callbacks for engine interactions."""
    go_callback = MagicMock()
    ready_callback = MagicMock()
    restart_engine_callback = MagicMock()
    return go_callback, ready_callback, restart_engine_callback

class TestChessGUI:
    """Test suite for ChessGUI using pytest and pytest-qt."""

    # #### **1. GUI Component Tests** ####

    @pytest.mark.gui
    def test_window_properties(self, chess_gui):
        """Test window visibility, title, and dimensions."""
        assert chess_gui.isVisible(), "ChessGUI window should be visible"
        assert chess_gui.windowTitle() == "JaskFish", "Window title should be 'JaskFish'"
        assert chess_gui.width() == 500, "Window width should be 500 pixels"
        assert chess_gui.height() == 550, "Window height should be 550 pixels"

    @pytest.mark.gui
    def test_layout_structure(self, chess_gui):
        """Test the layout structure of the main window."""
        central_widget = chess_gui.centralWidget()
        assert central_widget is not None, "Central widget should be set"

        main_layout = central_widget.layout()
        assert main_layout is not None, "Main layout should be a QVBoxLayout"

        # Check for turn and info indicators
        turn_indicator = chess_gui.turn_indicator
        info_indicator = chess_gui.info_indicator
        assert isinstance(turn_indicator, QLabel), "Turn indicator should be a QLabel"
        assert isinstance(info_indicator, QLabel), "Info indicator should be a QLabel"

        # Check for chessboard grid
        board_widget = main_layout.itemAt(1).widget()
        assert board_widget is not None, "Board widget should be present"
        grid_layout = board_widget.layout()
        assert grid_layout is not None, "Board should use a QGridLayout"

        # Check file and rank labels
        files = 'abcdefgh'
        for i in range(8):
            file_label = grid_layout.itemAtPosition(8, i + 1).widget()
            rank_label = grid_layout.itemAtPosition(i, 0).widget()
            assert isinstance(file_label, QLabel), f"File label at column {i} should be a QLabel"
            assert file_label.text() == files[i], f"File label should be '{files[i]}'"
            assert isinstance(rank_label, QLabel), f"Rank label at row {i} should be a QLabel"
            assert rank_label.text() == str(8 - i), f"Rank label should be '{8 - i}'"

        # Check control buttons
        button_layout = main_layout.itemAt(2).layout()
        assert button_layout is not None, "First button layout should be present"

        buttons = ["Undo", "Export", "Reset board"]
        for i, button_name in enumerate(buttons):
            button = button_layout.itemAt(i).widget()
            assert isinstance(button, QPushButton), f"'{button_name}' should be a QPushButton"
            assert button.text() == button_name, f"Button text should be '{button_name}'"

        button_layout2 = main_layout.itemAt(3).layout()
        buttons2 = ["Ready", "Go", "Restart Engine"]
        for i, button_name in enumerate(buttons2):
            button = button_layout2.itemAt(i).widget()
            assert isinstance(button, QPushButton), f"'{button_name}' should be a QPushButton"
            assert button.text() == button_name, f"Button text should be '{button_name}'"

    @pytest.mark.gui
    def test_chess_board_squares_initialization(self, chess_gui):
        """Test if the chessboard squares are correctly initialized."""
        assert len(chess_gui.squares) == 64, "There should be 64 squares initialized for the chessboard"
        for square, button in chess_gui.squares.items():
            assert isinstance(button, QPushButton), "Each square should be a QPushButton"
            assert hasattr(button, 'clicked'), "Each square should have a 'clicked' signal connected"

    @pytest.mark.gui
    def test_promotion_dialog_initialization(self, qtbot):
        """Test the initialization of the PromotionDialog."""
        dialog = PromotionDialog()
        qtbot.addWidget(dialog)
        dialog.show()

        assert dialog.windowTitle() == "Pawn Promotion", "PromotionDialog should have the correct title"

        combo = dialog.combo
        assert combo is not None, "PromotionDialog should contain a QComboBox"
        assert combo.count() == 4, "ComboBox should contain exactly four promotion options"
        expected_options = ["Queen", "Rook", "Bishop", "Knight"]
        for i, option in enumerate(expected_options):
            assert combo.itemText(i) == option, f"Promotion option {i} should be '{option}'"

        ok_button = dialog.findChild(QPushButton, "OK")
        assert ok_button is not None, "PromotionDialog should contain an 'OK' button"

    # #### **2. User Interaction Tests** ####

    @pytest.mark.gui
    def test_selecting_and_moving_pieces(self, chess_gui, qtbot):
        """Test selecting a piece and moving it to a valid square."""
        # Select a white pawn at e2
        e2_square = chess.E2
        e2_button = chess_gui.squares[e2_square]
        QTest.mouseClick(e2_button, Qt.LeftButton)
        qtbot.wait(100)  # Wait for the GUI to process the click

        assert chess_gui.selected_square == e2_square, "The e2 square should be selected after clicking it"

        # Move the pawn to e4
        e4_square = chess.E4
        e4_button = chess_gui.squares[e4_square]
        QTest.mouseClick(e4_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify that the move was executed
        assert chess_gui.board.piece_at(e4_square).symbol().lower() == 'p', "Pawn should have moved to e4"
        assert chess_gui.board.piece_at(e2_square) is None, "e2 square should be empty after the move"

        # Verify turn indicator
        assert chess_gui.turn_indicator.text() == "Black's turn", "Turn indicator should switch to Black"

    @pytest.mark.gui
    def test_invalid_move_rejection(self, chess_gui, qtbot, capsys):
        """Test attempting to move a piece to an invalid square."""
        # Select a white pawn at e2
        e2_square = chess.E2
        e2_button = chess_gui.squares[e2_square]
        QTest.mouseClick(e2_button, Qt.LeftButton)
        qtbot.wait(100)

        # Attempt to move the pawn to e5 (invalid in one move)
        e5_square = chess.E5
        e5_button = chess_gui.squares[e5_square]
        QTest.mouseClick(e5_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify that the move was not executed
        assert chess_gui.board.piece_at(e5_square) is None, "e5 square should remain empty after invalid move"
        assert chess_gui.board.piece_at(e2_square) is not None, "e2 square should still have the pawn"

        # Capture the printed output
        captured = capsys.readouterr()
        assert "Invalid Move" in captured.out, "An invalid move message should be printed"

    @pytest.mark.gui
    def test_promotion_dialog_invocation_and_selection(self, qtbot, chess_gui):
        """Test that the PromotionDialog is invoked during pawn promotion and handles selection."""
        # Set up a board state where a pawn is ready to promote
        promotion_fen = "8/5P2/8/8/8/8/8/7k w - - 0 1"
        chess_gui.board.set_fen(promotion_fen)
        chess_gui.update_board()

        # Select the pawn at f7
        f7_square = chess.F7
        f7_button = chess_gui.squares[f7_square]
        QTest.mouseClick(f7_button, Qt.LeftButton)
        qtbot.wait(100)

        # Move the pawn to f8, triggering promotion
        f8_square = chess.F8
        f8_button = chess_gui.squares[f8_square]

        with patch('PySide2.QtWidgets.QDialog.exec_') as mock_exec, \
             patch.object(PromotionDialog, 'get_promotion_piece', return_value='queen') as mock_get_piece:
            mock_exec.return_value = True  # Simulate clicking "OK"
            QTest.mouseClick(f8_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that the promotion dialog was invoked
            mock_exec.assert_called_once()
            mock_get_piece.assert_called_once()

        # Verify that the pawn was promoted to a queen
        promoted_piece = chess_gui.board.piece_at(f8_square)
        assert promoted_piece.symbol().lower() == 'q', "Pawn should be promoted to a queen"

    @pytest.mark.gui
    def test_cancel_promotion_dialog(self, qtbot, chess_gui):
        """Test cancelling the PromotionDialog during pawn promotion."""
        # Set up a board state where a pawn is ready to promote
        promotion_fen = "8/5P2/8/8/8/8/8/7k w - - 0 1"
        chess_gui.board.set_fen(promotion_fen)
        chess_gui.update_board()

        # Select the pawn at f7
        f7_square = chess.F7
        f7_button = chess_gui.squares[f7_square]
        QTest.mouseClick(f7_button, Qt.LeftButton)
        qtbot.wait(100)

        # Move the pawn to f8, triggering promotion
        f8_square = chess.F8
        f8_button = chess_gui.squares[f8_square]

        with patch('PySide2.QtWidgets.QDialog.exec_') as mock_exec, \
             patch.object(PromotionDialog, 'get_promotion_piece', return_value=None) as mock_get_piece:
            mock_exec.return_value = False  # Simulate cancelling the dialog
            QTest.mouseClick(f8_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that the promotion dialog was invoked
            mock_exec.assert_called_once()
            mock_get_piece.assert_called_once()

        # Verify that the pawn remains unpromoted (still a pawn at f8)
        piece = chess_gui.board.piece_at(f8_square)
        assert piece.symbol().lower() == 'p', "Pawn should remain unpromoted after cancelling the dialog"

    @pytest.mark.gui
    def test_undo_move(self, chess_gui, qtbot):
        """Test the Undo button functionality."""
        # Perform a move: e2 to e4
        e2_square = chess.E2
        e4_square = chess.E4
        e2_button = chess_gui.squares[e2_square]
        e4_button = chess_gui.squares[e4_square]

        QTest.mouseClick(e2_button, Qt.LeftButton)
        qtbot.wait(100)
        QTest.mouseClick(e4_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify the move
        assert chess_gui.board.piece_at(e4_square).symbol().lower() == 'p', "Pawn should have moved to e4"
        assert chess_gui.board.piece_at(e2_square) is None, "e2 square should be empty after the move"

        # Click the Undo button
        undo_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Undo":
                undo_button = button
                break
        assert undo_button is not None, "Undo button should be present"

        QTest.mouseClick(undo_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify that the move was undone
        assert chess_gui.board.piece_at(e4_square) is None, "e4 square should be empty after undo"
        assert chess_gui.board.piece_at(e2_square).symbol().lower() == 'p', "e2 square should have the pawn after undo"

    @pytest.mark.gui
    def test_export_game(self, chess_gui, qtbot, tmp_path, mocker):
        """Test exporting the game state to a JSON file."""
        # Mock the time to have a consistent filename
        fixed_time = "2024-09-15 12:00:00"
        mocker.patch('time.strftime', return_value=fixed_time)
        mocker.patch('time.localtime', return_value=time.localtime())

        # Set the initial and final FEN
        initial_fen = chess_gui.initial_fen
        final_fen = chess_gui.board.fen()

        # Perform a move: e2 to e4
        e2_square = chess.E2
        e4_square = chess.E4
        e2_button = chess_gui.squares[e2_square]
        e4_button = chess_gui.squares[e4_square]

        QTest.mouseClick(e2_button, Qt.LeftButton)
        qtbot.wait(100)
        QTest.mouseClick(e4_button, Qt.LeftButton)
        qtbot.wait(100)

        # Click the Export button
        export_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Export":
                export_button = button
                break
        assert export_button is not None, "Export button should be present"

        with patch('os.makedirs') as mock_makedirs, \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mocker.mock_open()) as mock_file:
            QTest.mouseClick(export_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that the JSON file was created with the correct data
            expected_filename = f'gamestates/chess_game_{fixed_time}.json'
            mock_file.assert_called_with(expected_filename, 'w')

            handle = mock_file()
            exported_data = {
                'export-time': fixed_time,
                'fen-init': initial_fen,
                'fen-final': chess_gui.board.fen(),
                'san': chess_gui.board.san(chess.Move.from_uci('e2e4')),
                'uci': 'e2e4'
            }
            handle.write.assert_called_once_with(json.dumps(exported_data))

    @pytest.mark.gui
    def test_reset_board(self, chess_gui, non_start_board, qtbot):
        """Test resetting the board from a non-starting position."""
        # Initialize with a non-starting position
        chess_gui.board = non_start_board.board
        chess_gui.update_board()

        # Verify that the board is not in the starting position
        assert chess_gui.board.fen() != chess.Board().fen(), "The board should be in a custom position before reset"

        # Click the Reset Board button
        reset_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Reset board":
                reset_button = button
                break
        assert reset_button is not None, "Reset Board button should be present"

        QTest.mouseClick(reset_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify that the board has been reset to the initial position
        assert chess_gui.board.fen() == chess.Board().fen(), "The board should be reset to the initial position"

        # Verify that game state indicators are updated
        assert chess_gui.info_indicator.text() == "Game Reset", "Info indicator should display 'Game Reset'"
        assert chess_gui.turn_indicator.text() == "White's turn", "Turn indicator should reset to White's turn"

    # #### **3. Game Logic Integration Tests** ####

    @pytest.mark.logic
    def test_move_validation(self, chess_gui, qtbot):
        """Test performing various valid and invalid moves."""
        # Example of a valid castling move
        # Set up a position where castling is possible
        castling_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
        chess_gui.board.set_fen(castling_fen)
        chess_gui.update_board()

        # Select the king at e1 and move it to g1 (king-side castling)
        e1_square = chess.E1
        g1_square = chess.G1
        e1_button = chess_gui.squares[e1_square]
        g1_button = chess_gui.squares[g1_square]

        QTest.mouseClick(e1_button, Qt.LeftButton)
        qtbot.wait(100)
        QTest.mouseClick(g1_button, Qt.LeftButton)
        qtbot.wait(100)

        # Verify that castling was performed
        # assert chess_gui.board.is_castling(chess.Move.from_square=e1_square, to_square=g1_square), "King-side castling should be performed"
        assert chess_gui.board.piece_at(g1_square).symbol().lower() == 'k', "King should be at g1 after castling"
        assert chess_gui.board.piece_at(f1_square := chess.F1) is not None, "Rook should have moved to f1 after castling"

    @pytest.mark.logic
    def test_game_over_detection(self, chess_gui, qtbot, capsys):
        """Test that game over conditions are detected and handled."""
        # Set up a checkmate position
        checkmate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
        chess_gui.board.set_fen(checkmate_fen)
        chess_gui.update_board()

        # Capture the information dialog
        with patch('PySide2.QtWidgets.QMessageBox.information') as mock_info:
            # Since the game is already over, update_board should detect it
            chess_gui.update_board()
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert args[0] == chess_gui, "MessageBox should be shown on the ChessGUI instance"
            assert args[1] == "Game Over", "MessageBox title should be 'Game Over'"
            assert "Checkmate" in args[2], "MessageBox message should indicate checkmate"

    # #### **4. Developer Mode Tests** ####

    @pytest.mark.dev
    def test_developer_mode_debug_information(self, chess_gui, capsys):
        """Test that developer mode prints debug information."""
        # Perform a move and check debug outputs
        e2_square = chess.E2
        e4_square = chess.E4
        e2_button = chess_gui.squares[e2_square]
        e4_button = chess_gui.squares[e4_square]

        QTest.mouseClick(e2_button, Qt.LeftButton)
        QTest.mouseClick(e4_button, Qt.LeftButton)

        captured = capsys.readouterr()
        assert "Move attempted: e2 -> e4" in captured.out, "Debug information should be printed for the move"

    @pytest.mark.dev
    def test_developer_mode_fixed_player_color(self, qtbot):
        """Test that in developer mode, the player is always assigned to White."""
        assert chess_gui.player_is_white is True, "Player should be assigned to White in developer mode"

    # #### **5. Edge Case and Error Handling Tests** ####

    @pytest.mark.edge_case
    def test_promotion_not_invoked_on_invalid_rank(self, chess_gui, qtbot, capsys):
        """Ensure that the promotion dialog is not invoked when a pawn is not on the promotion rank."""
        # Attempt to move a pawn from e2 to e3
        e2_square = chess.E2
        e3_square = chess.E3
        e2_button = chess_gui.squares[e2_square]
        e3_button = chess_gui.squares[e3_square]

        QTest.mouseClick(e2_button, Qt.LeftButton)
        QTest.mouseClick(e3_button, Qt.LeftButton)
        qtbot.wait(100)

        # Capture output to ensure promotion dialog was not invoked
        captured = capsys.readouterr()
        assert "Promotion Dialog" not in captured.out, "Promotion dialog should not be invoked for non-promotion moves"

    @pytest.mark.edge_case
    def test_multiple_pawn_promotions(self, chess_gui, qtbot):
        """Handle cases where multiple pawns are eligible for promotion simultaneously."""
        # Set up a position with two pawns ready to promote
        multi_promotion_fen = "P7/P7/8/8/8/8/7p/7k w - - 0 1"
        chess_gui.board.set_fen(multi_promotion_fen)
        chess_gui.update_board()

        # Select the first pawn at a8
        a8_square = chess.A8
        a8_button = chess_gui.squares[a8_square]
        QTest.mouseClick(a8_button, Qt.LeftButton)
        qtbot.wait(100)

        # Move the pawn to a9 (invalid square, but for the sake of promotion)
        # Since chess.Board doesn't allow a9, we simulate promotion by moving to a8
        move = chess.Move.from_uci('a7a8q')
        chess_gui.attempt_move(move)
        chess_gui.update_board()

        # Verify promotion
        promoted_piece = chess_gui.board.piece_at(a8_square)
        assert promoted_piece.symbol().lower() == 'q', "Pawn should be promoted to a queen"

        # Repeat for the second pawn at a7
        a7_square = chess.A7
        a7_button = chess_gui.squares[a7_square]
        QTest.mouseClick(a7_button, Qt.LeftButton)
        qtbot.wait(100)

        move = chess.Move.from_uci('a7a8r')
        chess_gui.attempt_move(move)
        chess_gui.update_board()

        promoted_piece = chess_gui.board.piece_at(a8_square)
        assert promoted_piece.symbol().lower() == 'r', "Second pawn should be promoted to a rook"

    @pytest.mark.edge_case
    def test_undo_beyond_initial_state(self, chess_gui, qtbot):
        """Attempt to undo moves beyond the initial game state."""
        # Initially, no moves have been made
        assert len(chess_gui.board.move_stack) == 0, "Move stack should be empty initially"

        # Click Undo button
        undo_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Undo":
                undo_button = button
                break
        assert undo_button is not None, "Undo button should be present"

        with patch('gui.ChessGUI.update_board') as mock_update:
            QTest.mouseClick(undo_button, Qt.LeftButton)
            qtbot.wait(100)
            mock_update.assert_called_once()

        # Ensure no error occurs and move stack remains empty
        assert len(chess_gui.board.move_stack) == 0, "Move stack should remain empty after undo"

    @pytest.mark.edge_case
    def test_export_empty_move_history(self, chess_gui, qtbot, mocker):
        """Export the game when the move history is empty."""
        # Ensure no moves have been made
        assert len(chess_gui.board.move_stack) == 0, "Move stack should be empty"

        # Mock the time to have a consistent filename
        fixed_time = "2024-09-15 12:00:00"
        mocker.patch('time.strftime', return_value=fixed_time)
        mocker.patch('time.localtime', return_value=time.localtime())

        # Click the Export button
        export_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Export":
                export_button = button
                break
        assert export_button is not None, "Export button should be present"

        with patch('os.makedirs') as mock_makedirs, \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mocker.mock_open()) as mock_file:
            QTest.mouseClick(export_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that the JSON file was created with the correct data
            expected_filename = f'gamestates/chess_game_{fixed_time}.json'
            mock_file.assert_called_with(expected_filename, 'w')

            handle = mock_file()
            exported_data = {
                'export-time': fixed_time,
                'fen-init': chess_gui.initial_fen,
                'fen-final': chess_gui.board.fen(),
                'san': '',
                'uci': ''
            }
            handle.write.assert_called_once_with(json.dumps(exported_data))

    @pytest.mark.edge_case
    def test_export_file_system_error(self, chess_gui, qtbot, mocker):
        """Handle file system errors gracefully during export."""
        # Click the Export button
        export_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Export":
                export_button = button
                break
        assert export_button is not None, "Export button should be present"

        # Simulate a file system error (e.g., permission denied)
        mocker.patch('os.makedirs', side_effect=PermissionError("Permission denied"))

        with patch('PySide2.QtWidgets.QMessageBox.critical') as mock_critical:
            QTest.mouseClick(export_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that an error message box is shown
            mock_critical.assert_called_once()
            args, kwargs = mock_critical.call_args
            assert "Export Error" in args[1], "Error message box should indicate an export error"

    @pytest.mark.edge_case
    def test_callbacks_absence(self, chess_gui, mock_callbacks, qtbot, capsys):
        """Ensure that clicking Ready/Go/Restart Engine buttons without setting callbacks does not crash."""
        go_callback, ready_callback, restart_engine_callback = mock_callbacks
        # Reset callbacks to None
        chess_gui.go_callback = None
        chess_gui.ready_callback = None
        chess_gui.restart_engine_callback = None

        # Click the Ready button
        ready_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Ready":
                ready_button = button
                break
        assert ready_button is not None, "Ready button should be present"

        QTest.mouseClick(ready_button, Qt.LeftButton)
        qtbot.wait(100)

        # Capture the printed debug message
        captured = capsys.readouterr()
        assert "Ready callback not set" in captured.out, "Should log that Ready callback is not set"

        # Similarly test Go and Restart Engine buttons
        go_button = None
        restart_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Go":
                go_button = button
            elif button.text() == "Restart Engine":
                restart_button = button

        assert go_button is not None, "Go button should be present"
        assert restart_button is not None, "Restart Engine button should be present"

        QTest.mouseClick(go_button, Qt.LeftButton)
        qtbot.wait(100)
        QTest.mouseClick(restart_button, Qt.LeftButton)
        qtbot.wait(100)

        captured = capsys.readouterr()
        assert "Go callback not set" in captured.out, "Should log that Go callback is not set"
        assert "Restart engine callback not set" in captured.out, "Should log that Restart Engine callback is not set"

    # #### **6. Performance and Responsiveness Tests** ####

    @pytest.mark.performance
    def test_gui_responsiveness_during_rapid_interactions(self, chess_gui, qtbot):
        """Ensure the GUI remains responsive during rapid clicking on multiple squares."""
        squares = list(chess_gui.squares.values())

        # Simulate rapid clicks on the first 10 squares
        for button in squares[:10]:
            QTest.mouseClick(button, Qt.LeftButton)
            qtbot.wait(10)  # Minimal wait to simulate rapid clicking

        # Verify that the application did not crash and remains responsive
        assert chess_gui.isVisible(), "ChessGUI should remain visible and responsive after rapid interactions"

    @pytest.mark.performance
    def test_export_with_large_move_history(self, chess_gui, qtbot, mocker):
        """Perform a long series of moves and export the game to ensure performance remains acceptable."""
        # Simulate a series of moves
        moves = [
            'e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'g8f6', 'd2d3', 'f8c5',
            'c2c3', 'd7d6', 'b1c3', 'c8g4', 'h2h3', 'g4h5', 'g2g4', 'h5g6',
            'f3g5', 'd8d7', 'g5e6', 'd7e6', 'c4b5', 'a7a6', 'b5a4', 'e6c6',
            # Add more moves as needed to simulate a large move history
        ]

        for move_uci in moves:
            move = chess.Move.from_uci(move_uci)
            if move in chess_gui.board.legal_moves:
                chess_gui.attempt_move(move)
                chess_gui.update_board()
            else:
                continue  # Skip illegal moves for the sake of the test

        # Mock the time to have a consistent filename
        fixed_time = "2024-09-15 12:00:00"
        mocker.patch('time.strftime', return_value=fixed_time)
        mocker.patch('time.localtime', return_value=time.localtime())

        # Click the Export button
        export_button = None
        for button in chess_gui.findChildren(QPushButton):
            if button.text() == "Export":
                export_button = button
                break
        assert export_button is not None, "Export button should be present"

        with patch('os.makedirs') as mock_makedirs, \
             patch('os.path.exists', return_value=True), \
             patch('builtins.open', mocker.mock_open()) as mock_file:
            QTest.mouseClick(export_button, Qt.LeftButton)
            qtbot.wait(100)

            # Verify that the JSON file was created
            expected_filename = f'gamestates/chess_game_{fixed_time}.json'
            mock_file.assert_called_with(expected_filename, 'w')

            handle = mock_file()
            # The SAN and UCI histories should reflect the move sequence
            expected_san = 'e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 c3 d6 Nc3 Bg4 h3 Bh5 g4 Bg6 Ng5 Qd7 g5 Qe6 Bc4'
            expected_uci = 'e2e4 e7e5 g1f3 b8c6 f1c4 g8f6 d2d3 f8c5 c2c3 d7d6 b1c3 c8g4 h2h3 g4h5 g2g4 h5g6 f3g5 d8d7 g5e6 d7e6 c4b5 a7a6 b5a4 e6c6'
            exported_data = {
                'export-time': fixed_time,
                'fen-init': chess_gui.initial_fen,
                'fen-final': chess_gui.board.fen(),
                'san': ' '.join([chess_gui.board.san(chess.Move.from_uci(m)) for m in moves if chess.Move.from_uci(m) in chess_gui.board.move_stack]),
                'uci': ' '.join(moves)
            }
            handle.write.assert_called_once_with(json.dumps(exported_data))

