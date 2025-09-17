"""Search algorithms and evaluation abstractions for the chess engine."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Tuple

import chess

from .features import DEFAULT_PIECE_VALUES, FeatureBundle, extract_features


class Evaluator(Protocol):
    """Protocol describing objects that can score a chess position."""

    def evaluate(self, board: chess.Board) -> float:
        """Return a score (in centipawns) where positive means an advantage for white."""


class HeuristicEvaluator:
    """Classical handcrafted evaluation based on material and mobility heuristics."""

    _CHECKMATE_VALUE = 100_000.0

    def __init__(
        self,
        *,
        piece_values: Optional[Mapping[chess.PieceType, float]] = None,
        mobility_weight: float = 10.0,
    ) -> None:
        self.piece_values = dict(piece_values or DEFAULT_PIECE_VALUES)
        self.mobility_weight = mobility_weight
        self._piece_square_tables = self._build_piece_square_tables()

    @staticmethod
    def _build_piece_square_tables() -> Dict[chess.PieceType, Tuple[int, ...]]:
        # Piece-square tables adapted from classical chess programming resources.
        # The values are intentionally coarse; they primarily serve as tie breakers
        # and improve play compared to using pure material counts.
        return {
            chess.PAWN: (
                0, 0, 0, 0, 0, 0, 0, 0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5, 5, 10, 25, 25, 10, 5, 5,
                0, 0, 0, 20, 20, 0, 0, 0,
                5, -5, -10, 0, 0, -10, -5, 5,
                5, 10, 10, -20, -20, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0,
            ),
            chess.KNIGHT: (
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50,
            ),
            chess.BISHOP: (
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -20, -10, -10, -10, -10, -10, -10, -20,
            ),
            chess.ROOK: (
                0, 0, 5, 10, 10, 5, 0, 0,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                5, 10, 10, 10, 10, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0,
            ),
            chess.QUEEN: (
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 5, 5, 5, 0, -10,
                -5, 0, 5, 5, 5, 5, 0, -5,
                0, 0, 5, 5, 5, 5, 0, -5,
                -10, 5, 5, 5, 5, 5, 0, -10,
                -10, 0, 5, 0, 0, 0, 0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20,
            ),
            chess.KING: (
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                20, 20, 0, 0, 0, 0, 20, 20,
                20, 30, 10, 0, 0, 10, 30, 20,
            ),
        }

    def evaluate(self, board: chess.Board) -> float:
        if board.is_checkmate():
            # Negative score because the side to move is checkmated.
            return -self._CHECKMATE_VALUE
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        features = extract_features(board)
        material_score = self._material_score(features)
        positional_score = self._piece_square_score(board)
        mobility_score = self._mobility_score(board)

        return material_score + positional_score + mobility_score

    def _material_score(self, features: FeatureBundle) -> float:
        score = 0.0
        for piece_type, value in self.piece_values.items():
            white_bb = features.bitboards[(chess.WHITE, piece_type)]
            black_bb = features.bitboards[(chess.BLACK, piece_type)]
            score += (white_bb.bit_count() - black_bb.bit_count()) * value
        return score

    def _piece_square_score(self, board: chess.Board) -> float:
        score = 0.0
        for square, piece in board.piece_map().items():
            table = self._piece_square_tables.get(piece.piece_type)
            if not table:
                continue
            if piece.color == chess.WHITE:
                score += table[square]
            else:
                score -= table[chess.square_mirror(square)]
        return score

    def _mobility_score(self, board: chess.Board) -> float:
        if board.is_game_over():
            return 0.0

        own_moves = board.legal_moves.count()
        try:
            board.push(chess.Move.null())
        except chess.IllegalMoveError:
            return 0.0
        try:
            opp_moves = board.legal_moves.count()
        finally:
            board.pop()

        return self.mobility_weight * float(own_moves - opp_moves)


class AlphaBetaSearcher:
    """Negamax alpha-beta search that relies on an :class:`Evaluator`."""

    CHECKMATE_VALUE = 1_000_000.0

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        max_depth: int = 3,
        time_limit: Optional[float] = None,
    ) -> None:
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit
        self._start_time: float = 0.0

    def search(self, board: chess.Board) -> Optional[chess.Move]:
        self._start_time = time.time()
        best_score = -math.inf
        best_move: Optional[chess.Move] = None

        for move in board.legal_moves:
            if self._time_exceeded():
                break

            board.push(move)
            score = -self._alphabeta(board, self.max_depth - 1, -math.inf, math.inf)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _alphabeta(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        if depth <= 0 or board.is_game_over() or self._time_exceeded():
            return self._evaluate(board, depth)

        value = -math.inf
        for move in board.legal_moves:
            board.push(move)
            score = -self._alphabeta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > value:
                value = score
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

            if self._time_exceeded():
                break

        return value

    def _evaluate(self, board: chess.Board, depth: int) -> float:
        if board.is_checkmate():
            return -self.CHECKMATE_VALUE + depth
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        value = self.evaluator.evaluate(board)
        return value if board.turn == chess.WHITE else -value

    def _time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.time() - self._start_time) >= self.time_limit


InferenceBackend = Callable[[FeatureBundle], float]


class ExternalInferenceEvaluator(Evaluator):
    """Adapter that delegates scoring to an external (potentially async) backend."""

    def __init__(
        self,
        backend: InferenceBackend,
        fallback: Optional[Evaluator] = None,
        *,
        executor: Optional[ThreadPoolExecutor] = None,
        inference_timeout: float = 0.0,
    ) -> None:
        self._backend = backend
        self._fallback = fallback or HeuristicEvaluator()
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._timeout = inference_timeout
        self._lock = threading.Lock()
        self._inflight: MutableMapping[str, Tuple[Future[float], FeatureBundle]] = {}

    def evaluate(self, board: chess.Board) -> float:
        fen = board.fen()

        with self._lock:
            entry = self._inflight.get(fen)
            if entry is None:
                features = extract_features(board)
                future = self._executor.submit(self._backend, features)
                self._inflight[fen] = (future, features)
                entry = (future, features)

        future, features = entry
        if future.done():
            try:
                value = future.result(timeout=self._timeout)
            except Exception:
                value = self._fallback.evaluate(board)
            else:
                with self._lock:
                    self._inflight.pop(fen, None)
                return value

        # Fall back to the heuristic evaluator without blocking.
        return self._fallback.evaluate(board)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


def heuristic_backend_from_scalars(features: FeatureBundle) -> float:
    """Small helper backend that approximates evaluation from scalar summaries."""

    return features.scalars.get("material_balance", 0.0)


__all__ = [
    "Evaluator",
    "HeuristicEvaluator",
    "AlphaBetaSearcher",
    "ExternalInferenceEvaluator",
    "heuristic_backend_from_scalars",
]

