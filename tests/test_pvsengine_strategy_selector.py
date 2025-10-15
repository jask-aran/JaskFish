import chess

from engines.pvsengine import MoveStrategy, StrategyContext, StrategyResult, StrategySelector


def make_context(legal_moves: int = 20) -> StrategyContext:
    board = chess.Board()
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
        legal_moves_count=legal_moves,
        time_controls=None,
        in_check=False,
        opponent_mate_in_one_threat=False,
    )


class StaticStrategy(MoveStrategy):
    def __init__(self, name: str, priority: int, score: float, confidence: float, definitive: bool = False):
        super().__init__(name=name, priority=priority, confidence=confidence)
        self._score = score
        self._definitive = definitive

    def is_applicable(self, context: StrategyContext) -> bool:  # type: ignore[override]
        return True

    def generate_move(self, board: chess.Board, context: StrategyContext) -> StrategyResult:  # type: ignore[override]
        return StrategyResult(
            move="e2e4",
            strategy_name=self.name,
            score=self._score,
            confidence=self.confidence,
            definitive=self._definitive,
            metadata={"source": self.name},
        )


def test_selector_prefers_higher_priority() -> None:
    selector = StrategySelector()
    selector.register(StaticStrategy("low", priority=5, score=0.5, confidence=0.4))
    selector.register(StaticStrategy("high", priority=10, score=0.1, confidence=0.9))
    board = chess.Board()
    context = make_context()

    result = selector.select(board, context)
    assert result is not None
    assert result.strategy_name == "high"


def test_selector_breaks_ties_by_score_then_confidence() -> None:
    selector = StrategySelector()
    selector.register(StaticStrategy("primary", priority=5, score=0.5, confidence=0.3))
    selector.register(StaticStrategy("better_score", priority=5, score=0.8, confidence=0.1))
    selector.register(StaticStrategy("higher_confidence", priority=5, score=0.8, confidence=0.9))

    board = chess.Board()
    context = make_context()
    result = selector.select(board, context)
    assert result is not None
    assert result.strategy_name == "higher_confidence"


def test_selector_returns_definitive_result_immediately() -> None:
    selector = StrategySelector()
    selector.register(StaticStrategy("fallback", priority=1, score=0.0, confidence=0.0))
    selector.register(StaticStrategy("definitive", priority=0, score=0.0, confidence=0.0, definitive=True))

    board = chess.Board()
    context = make_context()
    result = selector.select(board, context)
    assert result is not None
    assert result.strategy_name == "definitive"
    assert result.definitive is True


def test_selector_ignores_exceptions() -> None:
    class ExplodingStrategy(MoveStrategy):
        def __init__(self) -> None:
            super().__init__(name="boom", priority=100)

        def is_applicable(self, context: StrategyContext) -> bool:  # type: ignore[override]
            return True

        def generate_move(self, board: chess.Board, context: StrategyContext):  # type: ignore[override]
            raise RuntimeError("strategy failure")

    selector = StrategySelector()
    selector.register(ExplodingStrategy())
    selector.register(StaticStrategy("safe", priority=1, score=0.0, confidence=0.0))

    board = chess.Board()
    context = make_context()
    result = selector.select(board, context)
    assert result is not None
    assert result.strategy_name == "safe"

