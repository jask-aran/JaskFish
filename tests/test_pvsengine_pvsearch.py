import chess
import pytest

from pvsengine import (
    ChessEngine,
    MetaRegistry,
    PVSearchBackend,
    SearchReporter,
    StrategyToggle,
    build_search_tuning,
)

pytestmark = pytest.mark.search_slow


def make_backend(max_threads: int = 1):
    meta = MetaRegistry.resolve("fastblitz")
    tuning, limits = build_search_tuning(meta)
    backend = PVSearchBackend(max_threads=max_threads)
    backend.configure(tuning, meta)
    return backend, tuning, limits


def make_engine():
    return ChessEngine(meta_preset="fastblitz", toggles=(StrategyToggle.HEURISTIC,))


def _run_search(backend, board, limits, *, budget=None):
    engine = make_engine()
    context = engine._build_context(board.copy(stack=True), None)
    reporter = SearchReporter(logger=lambda *_: None)
    outcome = backend.search(
        board.copy(stack=True),
        context,
        limits,
        reporter,
        budget_seconds=budget,
    )
    return outcome, board.fen()


def test_pvsearch_backend_smoke_returns_move() -> None:
    backend, _, limits = make_backend()
    board = chess.Board()
    outcome, fen_before = _run_search(backend, board, limits, budget=0.1)

    assert outcome.move is not None
    assert outcome.move in board.legal_moves
    assert outcome.completed_depth >= 1
    assert outcome.nodes > 0
    assert outcome.principal_variation
    assert board.fen() == fen_before


def test_pvsearch_backend_handles_complex_position() -> None:
    backend, _, limits = make_backend()
    fen = "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10"
    board = chess.Board(fen)
    outcome, fen_before = _run_search(backend, board, limits, budget=0.2)

    assert outcome.move is not None
    assert outcome.move in board.legal_moves
    assert outcome.nodes > 0
    assert outcome.completed_depth >= 1
    assert board.fen() == fen_before


@pytest.mark.parametrize(
    "fen,description",
    [
        ("8/8/8/2k5/8/3K4/3P4/8 w - - 0 1", "zugzwang-like king opposition"),
        ("6k1/5ppp/8/8/8/5Q2/5PPP/6K1 w - - 0 1", "forced mate pattern"),
        ("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1", "early pawn tension"),
    ],
)
def test_pvsearch_backend_edge_positions(fen: str, description: str) -> None:
    backend, _, limits = make_backend()
    board = chess.Board(fen)
    outcome, fen_before = _run_search(backend, board, limits, budget=0.25)

    assert outcome.move is not None, f"{description} produced no move"
    assert outcome.move in board.legal_moves, f"{description} suggested illegal move"
    assert outcome.completed_depth >= 1
    assert outcome.nodes > 0
    assert outcome.metadata["pv"], "principal variation missing metadata"
    assert board.fen() == fen_before


def test_pvsearch_backend_unbounded_budget_and_repeated_runs() -> None:
    backend, _, limits = make_backend()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3")

    first, fen_before = _run_search(backend, board, limits, budget=None)
    second, fen_after_first = _run_search(backend, board, limits, budget=0.05)

    assert first.move is not None
    assert second.move is not None
    assert board.fen() == fen_before == fen_after_first
    assert first.completed_depth >= 1 and second.completed_depth >= 1
    assert first.nodes > 0 and second.nodes > 0
