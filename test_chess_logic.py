import unittest
import chess
from PySide2.QtWidgets import QApplication
from gui import ChessGUI

class TestChessLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication([])

    def setUp(self):
        self.board = chess.Board()
        self.gui = ChessGUI(self.board)

    def test_initial_board(self):
        self.assertFalse(self.board.is_game_over())
        self.assertEqual(self.gui.turn_indicator.text(), "White's turn")

    def test_valid_moves(self):
        e2e4 = chess.Move.from_uci("e2e4")
        self.assertTrue(chess.Move.from_uci("e2e4") in self.board.legal_moves)
        self.board.push(e2e4)
        self.gui.update_board()
        self.assertEqual(self.gui.turn_indicator.text(), "Black's turn")

    def test_checkmate(self):
        # Fool's mate
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        for move in moves:
            self.board.push(chess.Move.from_uci(move))
            self.gui.update_board()
        self.assertTrue(self.board.is_checkmate())
        self.assertEqual(self.gui.turn_indicator.text(), "White's turn")  # It's white's turn, but they're in checkmate

    def test_stalemate(self):
        # Set up a stalemate position
        self.board.set_fen("7k/5Q2/5K2/8/8/8/8/8 b - - 0 1")
        self.gui.update_board()
        self.assertTrue(self.board.is_stalemate())
        self.assertEqual(self.gui.turn_indicator.text(), "Black's turn")  # It's black's turn, but they're in stalemate

    def test_insufficient_material(self):
        # King vs King
        self.board.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        self.gui.update_board()
        self.assertTrue(self.board.is_insufficient_material())
        self.assertEqual(self.gui.turn_indicator.text(), "White's turn")

    def test_check(self):
        # Set up a check position
        self.board.set_fen("8/8/8/8/8/8/4Q3/4k3 b - - 0 1")
        self.gui.update_board()
        self.assertTrue(self.board.is_check())
        self.assertEqual(self.gui.turn_indicator.text(), "Black's turn")

    @classmethod
    def tearDownClass(cls):
        cls.app.quit()

if __name__ == '__main__':
    unittest.main()