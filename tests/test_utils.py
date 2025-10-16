import chess
import pytest

pytest.importorskip("PySide6")

import main as utils


def test_color_text_wraps_ansi() -> None:
    text = utils.color_text("hello", "32")
    assert text.startswith("\033[32m")
    assert text.endswith("\033[0m")


def test_get_piece_unicode_values() -> None:
    white_queen = chess.Piece(chess.QUEEN, chess.WHITE)
    black_knight = chess.Piece(chess.KNIGHT, chess.BLACK)
    assert utils.get_piece_unicode(white_queen) == "♕"
    assert utils.get_piece_unicode(black_knight) == "♞"
