import chess
import pytest

from pvsengine import MetaRegistry, MetaParams, SearchLimits, SearchReporter, StrategyContext


def make_context(
    *,
    turn: bool = chess.WHITE,
    legal_moves: int = 20,
    piece_count: int = 32,
    in_check: bool = False,
    mate_threat: bool = False,
    time_controls=None,
) -> StrategyContext:
    board = chess.Board()
    board.turn = turn
    return StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=piece_count,
        material_imbalance=0,
        turn=turn,
        fen=board.fen(),
        repetition_info={
            "is_threefold_repetition": False,
            "can_claim_threefold": False,
            "is_fivefold_repetition": False,
        },
        legal_moves_count=legal_moves,
        time_controls=time_controls,
        in_check=in_check,
        opponent_mate_in_one_threat=mate_threat,
    )


def test_resolve_budget_with_explicit_movetime() -> None:
    limits = SearchLimits(min_time=0.1, max_time=5.0, base_time=1.0, time_factor=0.5)
    context = make_context(time_controls={"movetime": 750})
    budget = limits.resolve_budget(context)
    assert pytest.approx(budget, rel=1e-3) == 0.75


def test_resolve_budget_with_incremental_clock() -> None:
    limits = SearchLimits(min_time=0.1, max_time=5.0, base_time=1.0, time_factor=0.25)
    tc = {"wtime": 4000, "winc": 500, "movestogo": 20}
    context = make_context(turn=chess.WHITE, time_controls=tc)
    budget = limits.resolve_budget(context)
    # (4000/20 + 500) / 1000 = 0.7 seconds
    assert pytest.approx(budget, rel=1e-3) == 0.7


def test_resolve_budget_without_time_controls_returns_none() -> None:
    limits = SearchLimits(min_time=0.1, max_time=5.0, base_time=1.0, time_factor=0.25)
    messages = []
    reporter = SearchReporter(logger=messages.append)
    context = make_context(legal_moves=45, piece_count=24)
    budget = limits.resolve_budget(context, reporter)
    assert budget is None
    assert messages and "infinite" in messages[0]


def test_resolve_budget_with_depth_only_is_infinite() -> None:
    limits = SearchLimits(min_time=0.1, max_time=5.0, base_time=0.5, time_factor=0.3)
    context = make_context(time_controls={"depth": 6})
    assert limits.resolve_budget(context) is None


def test_meta_registry_resolve_and_clamp() -> None:
    params = MetaRegistry.resolve("balanced")
    assert isinstance(params, MetaParams)
    assert 0.0 <= params.strength <= 1.0
    assert params.tt_budget_mb >= 1

    with pytest.raises(ValueError):
        MetaRegistry.resolve("unknown")
