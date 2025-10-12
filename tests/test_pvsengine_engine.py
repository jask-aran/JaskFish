import chess
import pytest

from pvsengine import ChessEngine, StrategyToggle


def make_engine() -> ChessEngine:
    # Disable heuristic strategy to keep tests deterministic and fast.
    return ChessEngine(meta_preset="balanced", toggles=(StrategyToggle.MATE_IN_ONE,))


def test_handle_uci_reports_identity(capsys: pytest.CaptureFixture[str]) -> None:
    engine = make_engine()
    capsys.readouterr()  # clear strategy registration logs
    engine.handle_uci()
    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        "id name JaskFish",
        "id author Jaskaran Singh",
        "uciok",
    ]


def test_handle_position_startpos_and_moves() -> None:
    engine = make_engine()
    engine.board.push(chess.Move.from_uci("e2e4"))
    engine.handle_position("startpos moves d2d4")
    assert engine.board.piece_at(chess.D4).piece_type == chess.PAWN
    assert engine.board.piece_at(chess.D2) is None
    assert engine.board.fullmove_number == 1


def test_handle_position_fen_and_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    engine = make_engine()
    valid_fen = "6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1"
    engine.handle_position(f"fen {valid_fen}")
    assert engine.board.fen() == valid_fen

    engine.handle_position("fen invalid-fen")
    output = capsys.readouterr().out
    assert "Invalid FEN" in output

    engine.handle_position("unsupported")
    output = capsys.readouterr().out
    assert "Unknown position command" in output


def test_parse_go_args_interprets_tokens() -> None:
    engine = make_engine()
    parsed = engine._parse_go_args(
        "wtime 1000 btime 2000 winc 10 binc 20 movestogo 30 movetime 40 depth 5 infinite ponder"
    )
    assert parsed == {
        "wtime": 1000,
        "btime": 2000,
        "winc": 10,
        "binc": 20,
        "movestogo": 30,
        "movetime": 40,
        "depth": 5,
        "infinite": True,
        "ponder": True,
    }


def test_build_context_reflects_board_state() -> None:
    engine = make_engine()
    fen = "r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 2 2"
    engine.board.set_fen(fen)
    time_controls = {"wtime": 120000, "winc": 2000}
    context = engine._build_context(engine.board.copy(stack=True), time_controls)

    assert context.fen == fen
    assert context.time_controls == time_controls
    assert context.legal_moves_count == engine.board.legal_moves.count()
    assert context.repetition_info["is_threefold_repetition"] is False
    assert context.in_check is False


def test_compute_move_uses_mate_strategy(capsys: pytest.CaptureFixture[str]) -> None:
    engine = make_engine()
    engine.board.set_fen("6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1")
    engine.move_calculating = True
    engine._compute_move({})
    output = capsys.readouterr().out.splitlines()
    assert any(line.startswith("bestmove") for line in output)
    best_line = next(line for line in output if line.startswith("bestmove"))
    move = chess.Move.from_uci(best_line.split()[1])
    board = engine.board.copy(stack=True)
    board.push(move)
    assert board.is_checkmate()
    assert output[-1] == "readyok"
    assert engine.move_calculating is False


def test_handle_go_prevents_reentrancy(capsys: pytest.CaptureFixture[str]) -> None:
    engine = make_engine()
    engine.move_calculating = True
    engine.handle_go("")
    output = capsys.readouterr().out
    assert "Please wait for computer move" in output


def test_material_imbalance_and_mate_threat_detection() -> None:
    engine = make_engine()
    board = chess.Board("8/8/8/8/4k3/8/3Q4/4K3 w - - 0 1")
    imbalance = engine._material_imbalance(board)
    assert imbalance == 9  # white queen advantage
    assert engine._mate_threat(board) is False

    mate_threat_board = chess.Board("6k1/5ppp/8/8/8/5Q2/5PPP/6K1 b - - 0 1")
    assert engine._mate_threat(mate_threat_board) is True
