import chess

from pvsengine import (
    ChessEngine,
    MetaParams,
    PVSearchBackend,
    SearchReporter,
    StrategyContext,
    StrategyToggle,
    build_search_tuning,
)


def _make_context(engine: ChessEngine, board: chess.Board) -> StrategyContext:
    return engine._build_context(board, None)


def _prepare_backend(threads: int):
    meta = MetaParams(
        strength=0.3,
        speed_bias=0.9,
        risk=0.1,
        stability=0.5,
        tt_budget_mb=16,
        style_tactical=0.3,
        endgame_focus=0.3,
    ).clamp()
    tuning, limits = build_search_tuning(meta)
    backend = PVSearchBackend(max_threads=threads)
    backend.configure(tuning, meta)
    return backend, tuning, limits


def test_pvsearch_single_thread_returns_move() -> None:
    engine = ChessEngine(meta_preset="fastblitz", toggles=(StrategyToggle.MATE_IN_ONE,))
    board = chess.Board()
    context = _make_context(engine, board)

    backend, tuning, limits = _prepare_backend(threads=1)
    reporter = SearchReporter(logger=lambda *_: None)
    budget = 1.0
    outcome = backend.search(board.copy(stack=True), context, limits, reporter, budget)

    assert outcome.move is not None
    assert outcome.nodes > 0
    assert outcome.time_spent <= budget + 0.5


def test_pvsearch_parallel_stays_consistent() -> None:
    engine = ChessEngine(meta_preset="fastblitz", toggles=(StrategyToggle.MATE_IN_ONE,))
    board = chess.Board("r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10")
    context = _make_context(engine, board)

    backend_single, tuning, limits = _prepare_backend(threads=1)
    backend_multi, _, _ = _prepare_backend(threads=4)

    reporter = SearchReporter(logger=lambda *_: None)
    budget = 1.0

    single = backend_single.search(board.copy(stack=True), context, limits, reporter, budget)
    parallel = backend_multi.search(board.copy(stack=True), context, limits, reporter, budget)

    assert single.move is not None
    assert parallel.move is not None
    assert parallel.completed_depth >= single.completed_depth
    assert parallel.nodes >= single.nodes // 2
