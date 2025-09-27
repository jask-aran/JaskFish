import chess

import simple_engine


def test_simple_engine_selects_legal_move_from_start():
    engine = simple_engine.SimpleEngine()
    move = engine._select_move()
    assert move is not None
    assert move in engine.board.legal_moves


def test_simple_engine_handles_position_startpos_moves():
    engine = simple_engine.SimpleEngine()
    engine.handle_position("startpos moves e2e4 e7e5")
    assert engine.board.fullmove_number == 2
    move = engine._select_move()
    assert move is not None
    engine.board.push(move)
    assert engine.board.is_valid()


def test_simple_engine_handles_fen_with_moves():
    engine = simple_engine.SimpleEngine()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.handle_position(f"fen {fen} moves g1f3")
    assert engine.board.piece_at(chess.G1) is None
    assert engine.board.piece_at(chess.F3).piece_type == chess.KNIGHT
    move = engine._select_move()
    assert move is not None
    assert move in engine.board.legal_moves


def test_simple_engine_go_outputs_move(capsys):
    engine = simple_engine.SimpleEngine()
    engine.handle_go("")
    output = capsys.readouterr().out.strip()
    assert output.startswith("bestmove ")
