import chess
import pytest

from src import engine as engine_module


class DummyStrategy(engine_module.MoveStrategy):
    """Deterministic strategy used to make engine responses predictable in tests."""

    def __init__(self, move: str):
        super().__init__(name="DummyStrategy", priority=100, short_circuit=True)
        self._move = move

    def is_applicable(self, context: engine_module.StrategyContext) -> bool:  # type: ignore[name-defined]
        return True

    def generate_move(
        self, board: chess.Board, context: engine_module.StrategyContext  # type: ignore[name-defined]
    ) -> engine_module.StrategyResult:
        return engine_module.StrategyResult(
            move=self._move,
            strategy_name=self.name,
            confidence=1.0,
        )


@pytest.fixture()
def deterministic_engine():
    engine = engine_module.ChessEngine()
    engine.strategy_selector.clear_strategies()
    engine.register_strategy(DummyStrategy("e2e4"))
    return engine


def test_handle_uci_reports_identity(capsys):
    engine = engine_module.ChessEngine()
    engine.handle_uci()
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert output_lines == [
        "id name JaskFish",
        "id author Jaskaran Singh",
        "uciok",
    ]


def test_handle_isready_reflects_engine_state(capsys, deterministic_engine):
    deterministic_engine.handle_isready("")
    assert capsys.readouterr().out.strip() == "readyok"

    deterministic_engine.move_calculating = True
    deterministic_engine.handle_isready("")
    busy_response = capsys.readouterr().out.strip()
    assert "Engine is busy" in busy_response


def test_handle_position_supports_startpos_and_fen(deterministic_engine):
    deterministic_engine.board.push_san("e4")
    deterministic_engine.handle_position("startpos")
    assert deterministic_engine.board.fen() == chess.STARTING_FEN

    target_fen = "rnbqkbnr/pp2pppp/2p5/3p4/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 0 4"
    deterministic_engine.handle_position(f"fen {target_fen}")
    assert deterministic_engine.board.fen() == target_fen


def test_parse_go_args_interprets_time_controls(deterministic_engine):
    parsed = deterministic_engine._parse_go_args(
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


def test_create_strategy_context_exposes_board_snapshot(deterministic_engine):
    deterministic_engine.board.set_fen(
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )
    time_controls = {"wtime": 2500, "btime": 1800}
    context = deterministic_engine.create_strategy_context(
        deterministic_engine.board.copy(stack=True), time_controls=time_controls
    )
    assert context.fen == deterministic_engine.board.fen()
    assert context.legal_moves_count == deterministic_engine.board.legal_moves.count()
    assert context.time_controls == time_controls
    assert context.piece_count == len(deterministic_engine.board.piece_map())


def test_process_go_command_uses_registered_strategy(capsys, deterministic_engine):
    deterministic_engine.move_calculating = True
    deterministic_engine.process_go_command({})
    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any(line.startswith("bestmove e2e4") for line in output_lines)
    assert "readyok" in output_lines
    assert deterministic_engine.move_calculating is False
