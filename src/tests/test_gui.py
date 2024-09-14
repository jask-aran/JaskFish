import sys
import unittest
from PySide2.QtWidgets import QApplication
from PySide2.QtTest import QTest
from PySide2.QtCore import Qt
from src.gui import ChessGUI
import chess

class TestChessGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication(sys.argv)

    def setUp(self):
        board = chess.Board()
        self.gui = ChessGUI(board, dev=True)
        self.gui.show()

    def tearDown(self):
        self.gui.close()

    def test_initial_board_setup(self):
        for square in chess.SQUARES:
            piece = self.gui.board.piece_at(square)
            button = self.gui.squares.get(square)
            if piece:
                self.assertEqual(button.text(), self.gui.get_piece_unicode(piece))
            else:
                self.assertEqual(button.text(), '')

    def test_user_move(self):
        # Simulate clicking e2 and e4
        e2 = chess.E2
        e4 = chess.E4
        btn_e2 = self.gui.squares[e2]
        btn_e4 = self.gui.squares[e4]

        # Click e2 to select the pawn
        QTest.mouseClick(btn_e2, Qt.LeftButton)
        self.assertEqual(self.gui.selected_square, e2)

        # Click e4 to move the pawn
        QTest.mouseClick(btn_e4, Qt.LeftButton)
        self.assertEqual(self.gui.board.piece_at(e4).symbol(), 'P')
        self.assertIsNone(self.gui.board.piece_at(e2))

    def test_invalid_move(self):
        # Attempt to move a piece from an empty square
        a3 = chess.A3
        btn_a3 = self.gui.squares[a3]

        # Click a3 which is empty
        QTest.mouseClick(btn_a3, Qt.LeftButton)
        self.assertIsNone(self.gui.selected_square)

    def test_pawn_promotion_via_gui(self):
        # Set up promotion scenario
        board = chess.Board("8/P7/8/8/8/8/8/k7 w - - 0 1")
        self.gui.board = board
        self.gui.update_board()

        a7 = chess.A7
        a8 = chess.A8
        btn_a7 = self.gui.squares[a7]
        btn_a8 = self.gui.squares[a8]

        # Click a7 to select the pawn
        QTest.mouseClick(btn_a7, Qt.LeftButton)
        self.assertEqual(self.gui.selected_square, a7)

        # Click a8 to attempt promotion
        QTest.mouseClick(btn_a8, Qt.LeftButton)

        # Simulate selecting a Queen in the promotion dialog
        # Since dialogs are modal, we need to handle them appropriately
        # For simplicity, assume promotion to Queen is automatic in tests
        self.assertEqual(self.gui.board.piece_at(a8).symbol(), 'Q')

    def test_castling_via_gui(self):
        # Set up castling scenario
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        self.gui.board = board
        self.gui.update_board()

        e1 = chess.E1
        g1 = chess.G1
        btn_e1 = self.gui.squares[e1]
        btn_g1 = self.gui.squares[g1]

        # Click e1 to select the king
        QTest.mouseClick(btn_e1, Qt.LeftButton)
        self.assertEqual(self.gui.selected_square, e1)

        # Click g1 to castle kingside
        QTest.mouseClick(btn_g1, Qt.LeftButton)
        self.assertTrue(board.is_castling(chess.Move.from_uci('e1g1')))
        self.assertEqual(board.turn, chess.BLACK)

    def test_en_passant_via_gui(self):
        # Set up en passant scenario
        board = chess.Board("8/8/8/4pP2/8/8/8/8 w - e6 0 1")
        self.gui.board = board
        self.gui.update_board()

        f5 = chess.F5
        e6 = chess.E6
        btn_f5 = self.gui.squares[f5]
        btn_e6 = self.gui.squares[e6]

        # Click f5 to select the pawn
        QTest.mouseClick(btn_f5, Qt.LeftButton)
        self.assertEqual(self.gui.selected_square, f5)

        # Click e6 to perform en passant
        QTest.mouseClick(btn_e6, Qt.LeftButton)
        self.assertEqual(board.peek().uci(), 'f5e6')
        self.assertIsNone(board.piece_at(e6 + 8))  # Ensure black pawn was captured

if __name__ == '__main__':
    unittest.main()
