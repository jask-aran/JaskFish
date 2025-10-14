import chess
import pytest

import simple_engine


def test_handle_uci_outputs_identity(capsys: pytest.CaptureFixture[str]) -> None:
    engine = simple_engine.SimpleEngine()
    engine.handle_uci("")
    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        "id name SimpleEngine+",
        "id author JaskFish Project",
        "uciok",
    ]


def test_handle_position_startpos_and_fen(capsys: pytest.CaptureFixture[str]) -> None:
    engine = simple_engine.SimpleEngine()
    engine.handle_position("startpos moves e2e4 e7e5")
    assert engine.board.fullmove_number == 2
    assert engine.board.piece_at(chess.E5).piece_type == chess.PAWN

    engine.debug = True
    engine.handle_position("fen 6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1 moves g2g4")
    assert engine.board.turn is chess.BLACK
    assert engine.board.piece_at(chess.G4).piece_type == chess.PAWN

    engine.handle_position("fen invalid")
    output = capsys.readouterr().out
    assert "Invalid FEN" in output


def test_handle_go_emits_legal_move(capsys: pytest.CaptureFixture[str]) -> None:
    engine = simple_engine.SimpleEngine()
    engine.handle_go("")
    output = capsys.readouterr().out.strip()
    assert output.startswith("bestmove ")
    move = chess.Move.from_uci(output.split()[1])
    assert move in engine.board.legal_moves


def test_select_move_returns_none_when_game_over() -> None:
    engine = simple_engine.SimpleEngine()
    engine.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert engine._select_move() is None


def test_handle_debug_toggles_state() -> None:
    engine = simple_engine.SimpleEngine()
    engine.handle_debug("on")
    engine.handle_debug("off")
    assert engine.debug is False
