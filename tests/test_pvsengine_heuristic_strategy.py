import chess
import pytest

from pvsengine import HeuristicSearchStrategy, MetaRegistry, SearchOutcome, StrategyContext


def make_context(board: chess.Board) -> StrategyContext:
    return StrategyContext(
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=board.fen(),
        repetition_info={
            "is_threefold_repetition": False,
            "can_claim_threefold": False,
            "is_fivefold_repetition": False,
        },
        legal_moves_count=board.legal_moves.count(),
        time_controls=None,
        in_check=False,
        opponent_mate_in_one_threat=False,
    )


def test_generate_move_requires_configuration() -> None:
    strategy = HeuristicSearchStrategy()
    board = chess.Board()
    context = make_context(board)
    with pytest.raises(RuntimeError):
        strategy.generate_move(board, context)


def test_generate_move_uses_backend_and_normalises_metadata() -> None:
    messages = []
    strategy = HeuristicSearchStrategy(logger=messages.append)
    strategy.apply_config(MetaRegistry.resolve("fastblitz"))

    called = {}

    class StubBackend:
        def search(self, board, context, limits, reporter, budget, stop_event=None):
            called["board_fen"] = board.fen()
            called["budget"] = budget
            return SearchOutcome(
                move=chess.Move.from_uci("e2e4"),
                score=0.42,
                completed_depth=3,
                nodes=256,
                time_spent=0.05,
                principal_variation=(chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")),
                metadata={},
            )

    strategy._backend = StubBackend()  # type: ignore[assignment]

    board = chess.Board()
    context = make_context(board)
    result = strategy.generate_move(board, context)

    assert result is not None
    assert result.move == "e2e4"
    assert result.metadata["pv"][0] == "e2e4"
    assert result.metadata["label"] == strategy.log_tag
    assert result.score == pytest.approx(0.42)
    assert called["board_fen"] == board.fen()
    assert "budget" in called  # budget parameter was passed (may be None for infinite searches)
