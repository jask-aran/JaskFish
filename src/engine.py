import sys
import io
import random
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import chess


class _SearchTimeout(Exception):
    """Internal exception used to abort search when the time budget is exhausted."""

    pass


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

EVAL_PIECE_VALUES = {piece: value * 100 for piece, value in PIECE_VALUES.items()}

PIECE_SQUARE_TABLES = {
    chess.PAWN: [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        0,
        0,
        0,
        20,
        20,
        0,
        0,
        0,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        5,
        10,
        10,
        -20,
        -20,
        10,
        10,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ],
    chess.KNIGHT: [
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
        -40,
        -20,
        0,
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -30,
        5,
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -40,
        -20,
        0,
        5,
        5,
        0,
        -20,
        -40,
        -50,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -50,
    ],
    chess.BISHOP: [
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        10,
        10,
        5,
        0,
        -10,
        -10,
        5,
        5,
        10,
        10,
        5,
        5,
        -10,
        -10,
        0,
        10,
        10,
        10,
        10,
        0,
        -10,
        -10,
        10,
        10,
        10,
        10,
        10,
        10,
        -10,
        -10,
        5,
        0,
        0,
        0,
        0,
        5,
        -10,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -20,
    ],
    chess.ROOK: [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        5,
        10,
        10,
        10,
        10,
        10,
        10,
        5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        -5,
        0,
        0,
        0,
        0,
        0,
        0,
        -5,
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
    ],
    chess.QUEEN: [
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -5,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        0,
        0,
        5,
        5,
        5,
        5,
        0,
        -5,
        -10,
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        0,
        0,
        0,
        0,
        0,
        -10,
        -20,
        -10,
        -10,
        -5,
        -5,
        -10,
        -10,
        -20,
    ],
    chess.KING: [
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -30,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        20,
        20,
        0,
        0,
        0,
        0,
        20,
        20,
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
    ],
}

# Ensure stdout is line-buffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


@dataclass
class StrategyContext:
    """Snapshot of board metrics used by move strategies."""

    fullmove_number: int
    halfmove_clock: int
    piece_count: int
    material_imbalance: int
    turn: bool = True
    fen: str = ""
    repetition_info: Dict[str, bool] = field(default_factory=dict)
    legal_moves_count: int = 0
    time_controls: Optional[Dict[str, int]] = None


@dataclass
class StrategyResult:
    """Container describing the outcome of a strategy evaluation."""

    move: Optional[str]
    strategy_name: str
    score: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class TranspositionFlag(IntEnum):
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2


@dataclass
class TranspositionEntry:
    depth: int
    value: float
    flag: TranspositionFlag
    move: Optional[chess.Move]


class MoveStrategy(ABC):
    """Base class for all move selection strategies."""

    def __init__(self, name: Optional[str] = None, priority: int = 0, confidence: Optional[float] = None):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.confidence = confidence

    @abstractmethod
    def is_applicable(self, context: StrategyContext) -> bool:
        """Return whether the strategy should be considered in the given context."""

    @abstractmethod
    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        """Produce a move suggestion when applicable."""


class StrategySelector:
    """Manages strategy registration and selection for the engine."""

    def __init__(
        self,
        strategies: Optional[Iterable[MoveStrategy]] = None,
        selection_policy: Optional[Callable[..., Optional[StrategyResult]]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._strategies: List[MoveStrategy] = []
        self._logger = logger or (lambda message: None)
        self._selection_policy = selection_policy or self._default_selection_policy
        self._uses_default_policy = selection_policy is None
        if strategies:
            for strategy in strategies:
                self.register_strategy(strategy)

    def register_strategy(self, strategy: MoveStrategy) -> None:
        """Register a strategy and maintain priority ordering."""

        self._strategies.append(strategy)
        self._strategies.sort(key=lambda item: item.priority, reverse=True)
        self._logger(f"strategy registered: {strategy.name} (priority={strategy.priority})")

    def clear_strategies(self) -> None:
        self._strategies.clear()

    def get_strategies(self) -> Tuple[MoveStrategy, ...]:
        return tuple(self._strategies)

    def set_selection_policy(
        self,
        policy: Optional[Callable[..., Optional[StrategyResult]]],
    ) -> None:
        if policy is None:
            self._selection_policy = self._default_selection_policy
            self._uses_default_policy = True
        else:
            self._selection_policy = policy
            self._uses_default_policy = False
        self._logger("strategy selection policy updated")

    def select_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        """Evaluate registered strategies and choose a result using the selection policy."""

        strategy_results: List[Tuple[MoveStrategy, StrategyResult]] = []
        for strategy in self._strategies:
            if not strategy.is_applicable(context):
                self._logger(f"strategy skipped (not applicable): {strategy.name}")
                continue

            self._logger(f"strategy evaluating: {strategy.name}")
            try:
                result = strategy.generate_move(board, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger(f"strategy error in {strategy.name}: {exc}")
                continue

            if result is None:
                self._logger(f"strategy produced no result: {strategy.name}")
                continue

            strategy_results.append((strategy, result))
            if result.move and self._uses_default_policy:
                break

        if not strategy_results:
            self._logger("no strategies produced a move suggestion")

        return self._selection_policy(strategy_results, board=board, context=context)

    @staticmethod
    def _default_selection_policy(
        strategy_results: List[Tuple[MoveStrategy, StrategyResult]],
        **_: Any,
    ) -> Optional[StrategyResult]:
        for _, result in strategy_results:
            if result.move:
                return result
        return None


class OpeningBookStrategy(MoveStrategy):
    def __init__(self, opening_book: Optional[Dict[str, str]] = None, max_fullmove: int = 12, **kwargs):
        super().__init__(priority=90, **kwargs)
        self.opening_book = opening_book or {}
        self.max_fullmove = max_fullmove

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.fullmove_number <= self.max_fullmove and bool(self.opening_book)

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self.opening_book.get(board.fen())
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence or 1.0,
            metadata={"source": "opening_book"},
        )


class EndgameTableStrategy(MoveStrategy):
    def __init__(self, table: Optional[Dict[str, str]] = None, piece_threshold: int = 6, **kwargs):
        super().__init__(priority=80, **kwargs)
        self.table = table or {}
        self.piece_threshold = piece_threshold

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.piece_count <= self.piece_threshold

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self.table.get(board.fen())
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence,
            metadata={"source": "endgame_table"},
        )


class TacticalResponseStrategy(MoveStrategy):
    def __init__(
        self,
        min_tactical_score: float = 150.0,
        check_bonus: float = 150.0,
        promotion_bonus: float = 250.0,
        capture_weight: float = 1.0,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(priority=75, **kwargs)
        self.min_tactical_score = min_tactical_score
        self.check_bonus = check_bonus
        self.promotion_bonus = promotion_bonus
        self.capture_weight = capture_weight
        self._logger = logger or (lambda *_: None)

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        mover = board.turn
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")

        for move in board.legal_moves:
            score = 0.0
            if board.gives_check(move):
                score += self.check_bonus
            if move.promotion:
                score += self.promotion_bonus
                score += EVAL_PIECE_VALUES.get(move.promotion, 0)

            material_delta = 0.0
            if board.is_capture(move) or move.promotion:
                material_delta = self._material_delta(board, move, mover)
            if material_delta:
                score += material_delta * self.capture_weight

            if score > best_score:
                best_score = score
                best_move = move

        if best_move and best_score >= self.min_tactical_score:
            uci_move = best_move.uci()
            self._logger(
                f"tactical strategy chose {uci_move} (score={best_score:.1f})"
            )
            return StrategyResult(
                move=uci_move,
                strategy_name=self.name,
                score=best_score,
                confidence=self.confidence or 0.9,
                metadata={
                    "pattern": "forcing",
                    "tactical_score": best_score,
                },
            )

        return None

    def _material_delta(self, board: chess.Board, move: chess.Move, perspective: bool) -> float:
        before = self._material_balance(board, perspective)
        board.push(move)
        after = self._material_balance(board, perspective)
        board.pop()
        return after - before

    @staticmethod
    def _material_balance(board: chess.Board, perspective: bool) -> float:
        balance = 0.0
        for piece in board.piece_map().values():
            value = EVAL_PIECE_VALUES.get(piece.piece_type, 0)
            if piece.color == chess.WHITE:
                balance += value
            else:
                balance -= value
        return balance if perspective == chess.WHITE else -balance


class HeuristicSearchStrategy(MoveStrategy):
    def __init__(
        self,
        fallback: Optional[MoveStrategy] = None,
        search_depth: int = 4,
        quiescence_depth: int = 6,
        mobility_weight: float = 4.0,
        king_safety_weight: float = 12.0,
        pawn_structure_weight: float = 10.0,
        rook_activity_weight: float = 6.0,
        bishop_pair_bonus: float = 30.0,
        base_time_limit: float = 2.0,
        max_time_limit: float = 5.0,
        min_time_limit: float = 0.05,
        time_allocation_factor: float = 0.03,
        transposition_table_size: int = 200000,
        **kwargs,
    ):
        super().__init__(priority=70, **kwargs)
        self._fallback = fallback
        self.search_depth = max(1, search_depth)
        self.quiescence_depth = max(0, quiescence_depth)
        self.mobility_weight = mobility_weight
        self.king_safety_weight = king_safety_weight
        self.pawn_structure_weight = pawn_structure_weight
        self.rook_activity_weight = rook_activity_weight
        self.bishop_pair_bonus = bishop_pair_bonus
        self.base_time_limit = base_time_limit
        self.max_time_limit = max_time_limit
        self.min_time_limit = min_time_limit
        self.time_allocation_factor = time_allocation_factor
        self._mate_score = 100000
        self._transposition_table: Dict[int, TranspositionEntry] = {}
        self._transposition_table_limit = max(1000, transposition_table_size)
        self._history_scores: Dict[Tuple[bool, int, int], float] = defaultdict(float)
        self._killer_slots = 2

        # Search state (initialised per search invocation)
        self._search_deadline: Optional[float] = None
        self._search_start_time: float = 0.0
        self._nodes_visited: int = 0
        self._killer_moves: List[List[Optional[chess.Move]]] = []

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(
        self, board: chess.Board, context: StrategyContext
    ) -> Optional[StrategyResult]:
        if context.legal_moves_count == 0:
            if board.is_checkmate():
                return StrategyResult(
                    move=None,
                    strategy_name=self.name,
                    score=-float(self._mate_score),
                    metadata={"status": "checkmate"},
                )
            if board.is_stalemate():
                return StrategyResult(
                    move=None,
                    strategy_name=self.name,
                    score=0.0,
                    metadata={"status": "stalemate"},
                )
            if self._fallback:
                return self._fallback.generate_move(board, context)
            return None

        depth_limit = self._resolve_depth_limit(context)
        time_budget = self._determine_time_budget(context)

        self._search_start_time = time.perf_counter()
        self._search_deadline = (
            None
            if time_budget is None
            else self._search_start_time + max(time_budget, self.min_time_limit)
        )
        self._nodes_visited = 0
        self._killer_moves = [list() for _ in range(depth_limit + self.quiescence_depth + 4)]

        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        completed_depth = 0

        try:
            for current_depth in range(1, depth_limit + 1):
                alpha = -float("inf")
                beta = float("inf")
                iteration_best_move: Optional[chess.Move] = None
                iteration_best_score = -float("inf")

                tt_entry = self._transposition_table.get(board.zobrist_hash())
                tt_move = tt_entry.move if tt_entry else None
                moves = self._order_moves(board, 0, tt_move, best_move)

                for move in moves:
                    if self._time_exceeded():
                        raise _SearchTimeout()

                    color = board.turn
                    board.push(move)
                    score = -self._alpha_beta(
                        board,
                        depth=current_depth - 1,
                        alpha=-beta,
                        beta=-alpha,
                        ply=1,
                    )
                    board.pop()

                    if score > iteration_best_score:
                        iteration_best_score = score
                        iteration_best_move = move

                    alpha = max(alpha, score)
                    if alpha >= beta:
                        self._record_history(color, move, current_depth)
                        break

                if iteration_best_move is not None:
                    best_move = iteration_best_move
                    best_score = iteration_best_score
                    completed_depth = current_depth

                if self._time_exceeded():
                    break

        except _SearchTimeout:
            pass

        search_time = time.perf_counter() - self._search_start_time

        if best_move is None and self._fallback:
            return self._fallback.generate_move(board, context)

        if best_move is None:
            return None

        metadata = {
            "depth": completed_depth,
            "nodes": self._nodes_visited,
            "time": search_time,
            "principal_move": best_move.uci(),
        }
        return StrategyResult(
            move=best_move.uci(),
            strategy_name=self.name,
            score=best_score,
            confidence=self.confidence or 0.85,
            metadata=metadata,
        )

    def _resolve_depth_limit(self, context: StrategyContext) -> int:
        depth_limit = self.search_depth
        if context.time_controls:
            depth_override = context.time_controls.get("depth")
            if depth_override:
                depth_limit = max(1, int(depth_override))
        return max(1, depth_limit)

    def _determine_time_budget(self, context: StrategyContext) -> Optional[float]:
        if context.time_controls:
            tc = context.time_controls
            if tc.get("infinite"):
                return None

            if tc.get("movetime"):
                return min(
                    self.max_time_limit,
                    max(self.min_time_limit, tc["movetime"] / 1000.0),
                )

            turn_key = "wtime" if context.turn == chess.WHITE else "btime"
            inc_key = "winc" if context.turn == chess.WHITE else "binc"
            time_left = tc.get(turn_key)
            if time_left is not None:
                increment = tc.get(inc_key, 0)
                moves_to_go = tc.get("movestogo")
                if moves_to_go:
                    allocation = time_left / max(1, moves_to_go)
                else:
                    allocation = time_left * self.time_allocation_factor
                allocation += increment
                seconds = allocation / 1000.0
                return min(self.max_time_limit, max(self.min_time_limit, seconds))

        return self.base_time_limit

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self._nodes_visited += 1

        if self._time_exceeded():
            raise _SearchTimeout()

        if board.is_checkmate():
            return -float(self._mate_score) + ply
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0

        if depth == 0:
            return self._quiescence(board, alpha, beta, self.quiescence_depth, ply)

        key = board.zobrist_hash()
        entry = self._transposition_table.get(key)
        if entry and entry.depth >= depth:
            if entry.flag == TranspositionFlag.EXACT:
                return entry.value
            if entry.flag == TranspositionFlag.LOWERBOUND:
                alpha = max(alpha, entry.value)
            elif entry.flag == TranspositionFlag.UPPERBOUND:
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        tt_move = entry.move if entry else None
        best_value = -float("inf")
        best_move: Optional[chess.Move] = None
        alpha_original = alpha
        beta_original = beta

        moves = self._order_moves(board, ply, tt_move)
        if not moves:
            return self._evaluate_board(board)

        for move in moves:
            if self._time_exceeded():
                raise _SearchTimeout()

            color = board.turn
            board.push(move)
            try:
                score = -self._alpha_beta(board, depth - 1, -beta, -alpha, ply + 1)
            except _SearchTimeout:
                board.pop()
                raise
            board.pop()

            if score > best_value:
                best_value = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if not board.is_capture(move):
                    self._record_killer(ply, move)
                    self._record_history(color, move, depth)
                self._store_transposition_entry(
                    key,
                    depth,
                    best_value,
                    TranspositionFlag.LOWERBOUND,
                    best_move,
                )
                return best_value

        flag = TranspositionFlag.EXACT
        if best_value <= alpha_original:
            flag = TranspositionFlag.UPPERBOUND

        self._store_transposition_entry(key, depth, best_value, flag, best_move)
        return best_value

    def _quiescence(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int,
        ply: int,
    ) -> float:
        self._nodes_visited += 1
        if self._time_exceeded():
            raise _SearchTimeout()

        stand_pat = self._evaluate_board(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if depth <= 0:
            return stand_pat

        moves = self._generate_quiescence_moves(board)
        for move in moves:
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, depth - 1, ply + 1)
            except _SearchTimeout:
                board.pop()
                raise
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def _generate_quiescence_moves(self, board: chess.Board) -> List[chess.Move]:
        captures = []
        for move in board.legal_moves:
            if board.is_capture(move) or move.promotion or board.gives_check(move):
                captures.append(move)
        captures.sort(
            key=lambda mv: self._capture_score(board, mv) + (500 if board.gives_check(mv) else 0),
            reverse=True,
        )
        return captures

    def _capture_score(self, board: chess.Board, move: chess.Move) -> float:
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None and board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        captured_value = (
            EVAL_PIECE_VALUES.get(captured_piece.piece_type, 0)
            if captured_piece
            else 0
        )
        moving_piece = board.piece_at(move.from_square)
        moving_value = (
            EVAL_PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
        )
        return captured_value - moving_value

    def _order_moves(
        self,
        board: chess.Board,
        ply: int,
        tt_move: Optional[chess.Move] = None,
        principal_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        moves = list(board.legal_moves)
        if not moves:
            return moves

        killer_moves = self._killer_moves[ply] if ply < len(self._killer_moves) else []

        def score_move(move: chess.Move) -> float:
            if tt_move and move == tt_move:
                return 1_000_000
            if principal_move and move == principal_move:
                return 900_000

            score = 0.0
            if board.is_capture(move):
                score += 500_000 + self._capture_score(board, move)
            if move in killer_moves:
                score += 80_000
            history_key = (board.turn, move.from_square, move.to_square)
            score += self._history_scores.get(history_key, 0.0)
            if move.promotion:
                score += 60_000 + EVAL_PIECE_VALUES.get(move.promotion, 0)
            if board.gives_check(move):
                score += 40_000
            return score

        moves.sort(key=score_move, reverse=True)
        return moves

    def _record_killer(self, ply: int, move: chess.Move) -> None:
        if ply >= len(self._killer_moves):
            return
        killers = self._killer_moves[ply]
        if move in killers:
            return
        killers.insert(0, move)
        while len(killers) > self._killer_slots:
            killers.pop()

    def _record_history(self, color: bool, move: chess.Move, depth: int) -> None:
        key = (color, move.from_square, move.to_square)
        self._history_scores[key] += depth * depth
        if self._history_scores[key] > 500000:
            self._history_scores[key] *= 0.5

    def _store_transposition_entry(
        self,
        key: int,
        depth: int,
        value: float,
        flag: TranspositionFlag,
        move: Optional[chess.Move],
    ) -> None:
        existing = self._transposition_table.get(key)
        if existing and existing.depth > depth:
            return
        self._transposition_table[key] = TranspositionEntry(depth, value, flag, move)
        if len(self._transposition_table) > self._transposition_table_limit:
            self._transposition_table.pop(next(iter(self._transposition_table)))

    def _time_exceeded(self) -> bool:
        if self._search_deadline is None:
            return False
        return time.perf_counter() >= self._search_deadline

    def _evaluate_board(self, board: chess.Board) -> float:
        material = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        piece_square = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        pawn_files = {
            chess.WHITE: [0] * 8,
            chess.BLACK: [0] * 8,
        }
        bishop_counts = {
            chess.WHITE: len(board.pieces(chess.BISHOP, chess.WHITE)),
            chess.BLACK: len(board.pieces(chess.BISHOP, chess.BLACK)),
        }

        for square, piece in board.piece_map().items():
            value = EVAL_PIECE_VALUES.get(piece.piece_type, 0)
            table = PIECE_SQUARE_TABLES.get(piece.piece_type)
            color = piece.color
            material[color] += value
            if table:
                pst_index = square if color == chess.WHITE else chess.square_mirror(square)
                piece_square[color] += table[pst_index]
            if piece.piece_type == chess.PAWN:
                pawn_files[color][chess.square_file(square)] += 1

        material_score = material[chess.WHITE] - material[chess.BLACK]
        piece_square_score = piece_square[chess.WHITE] - piece_square[chess.BLACK]

        pawn_structure_score = self._evaluate_pawn_structure(board, pawn_files)
        rook_activity_score = self._evaluate_rook_activity(board)
        bishop_pair_score = 0.0
        if bishop_counts[chess.WHITE] >= 2:
            bishop_pair_score += self.bishop_pair_bonus
        if bishop_counts[chess.BLACK] >= 2:
            bishop_pair_score -= self.bishop_pair_bonus

        mobility_score = self._mobility_score(board)
        king_safety_score = self._king_safety_score(board)

        phase = self._game_phase(board)
        opening_weight = phase
        endgame_weight = 1.0 - phase

        positional = (
            piece_square_score
            + self.pawn_structure_weight * pawn_structure_score
            + self.rook_activity_weight * rook_activity_score
            + bishop_pair_score
        )

        dynamic = (
            self.mobility_weight * mobility_score * opening_weight
            + self.king_safety_weight * king_safety_score * opening_weight
        )

        endgame_terms = king_safety_score * endgame_weight * 0.5

        total = material_score + positional + dynamic + endgame_terms
        return total if board.turn == chess.WHITE else -total

    def _mobility_score(self, board: chess.Board) -> float:
        current_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        opponent_mobility = board.legal_moves.count()
        board.pop()
        return (current_mobility - opponent_mobility)

    def _king_safety_score(self, board: chess.Board) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            king_square = board.king(color)
            if king_square is None:
                continue
            opposing_attackers = len(board.attackers(not color, king_square))
            friendly_cover = 0
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            for file_delta in (-1, 0, 1):
                for rank_delta in (-1, 0, 1):
                    nf = king_file + file_delta
                    nr = king_rank + rank_delta
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        neighbour = chess.square(nf, nr)
                        piece = board.piece_at(neighbour)
                        if piece and piece.color == color:
                            friendly_cover += 1
            penalty = opposing_attackers * 10 - friendly_cover * 2
            if color == chess.WHITE:
                score -= penalty
            else:
                score += penalty
        return score

    def _evaluate_pawn_structure(
        self, board: chess.Board, pawn_files: Dict[bool, List[int]]
    ) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            pawns = board.pieces(chess.PAWN, color)
            opponent_pawns = board.pieces(chess.PAWN, not color)
            for file_index, count in enumerate(pawn_files[color]):
                if count > 1:
                    penalty = 8 * (count - 1)
                    score -= penalty if color == chess.WHITE else -penalty
                if count == 0:
                    continue
                adjacent_counts = 0
                if file_index > 0:
                    adjacent_counts += pawn_files[color][file_index - 1]
                if file_index < 7:
                    adjacent_counts += pawn_files[color][file_index + 1]
                if adjacent_counts == 0:
                    penalty = 12
                    score -= penalty if color == chess.WHITE else -penalty

            for pawn_square in pawns:
                file_index = chess.square_file(pawn_square)
                rank_range = (
                    range(chess.square_rank(pawn_square) + 1, 8)
                    if color == chess.WHITE
                    else range(chess.square_rank(pawn_square) - 1, -1, -1)
                )
                blocked = False
                for rank in rank_range:
                    sq = chess.square(file_index, rank)
                    if sq in opponent_pawns:
                        blocked = True
                        break
                if not blocked:
                    bonus = 15
                    score += bonus if color == chess.WHITE else -bonus

        return score / 100.0

    def _evaluate_rook_activity(self, board: chess.Board) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            friendly_pawns = board.pieces(chess.PAWN, color)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            friendly_files = {chess.square_file(sq) for sq in friendly_pawns}
            enemy_files = {chess.square_file(sq) for sq in enemy_pawns}
            for rook_square in board.pieces(chess.ROOK, color):
                file_index = chess.square_file(rook_square)
                friendly_blockers = file_index in friendly_files
                enemy_blockers = file_index in enemy_files
                if not friendly_blockers and not enemy_blockers:
                    bonus = 20
                elif not friendly_blockers and enemy_blockers:
                    bonus = 10
                else:
                    bonus = 0
                score += bonus if color == chess.WHITE else -bonus
        return score / 100.0

    def _game_phase(self, board: chess.Board) -> float:
        phase_values = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK: 2,
            chess.QUEEN: 4,
        }
        initial_counts = {
            chess.PAWN: 16,
            chess.KNIGHT: 4,
            chess.BISHOP: 4,
            chess.ROOK: 4,
            chess.QUEEN: 2,
        }
        max_phase = sum(phase_values[piece] * initial_counts[piece] for piece in phase_values)
        current_phase = 0
        for piece_type, value in phase_values.items():
            count = len(board.pieces(piece_type, chess.WHITE)) + len(
                board.pieces(piece_type, chess.BLACK)
            )
            current_phase += value * count
        if max_phase == 0:
            return 0.0
        phase = current_phase / max_phase
        return min(1.0, max(0.0, phase))


class FallbackRandomStrategy(MoveStrategy):
    def __init__(self, random_move_provider: Callable[[chess.Board], Optional[str]], **kwargs):
        super().__init__(priority=0, **kwargs)
        self._random_move_provider = random_move_provider

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self._random_move_provider(board)
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence,
            metadata={"source": "random_fallback"},
        )


class ChessEngine:
    """
    A simple chess engine that communicates using a custom UCI-like protocol.
    It can process commands to set up positions, calculate moves, and manage engine state.
    """

    def __init__(self):
        """Initialize the chess engine with default settings and state."""
        # Engine identification
        self.engine_name = 'JaskFish'
        self.engine_author = 'Jaskaran Singh'

        # Engine state
        self.board = chess.Board()
        self.debug = False
        self.move_calculating = False
        self.running = True

        # Lock to manage concurrent access to engine state
        self.state_lock = threading.Lock()

        # Strategy management
        self.strategy_selector = StrategySelector(logger=self._log_debug)
        self._register_default_strategies()

        # Dispatch table mapping commands to handler methods
        self.dispatch_table = {
            'quit': self.handle_quit,
            'debug': self.handle_debug,
            'isready': self.handle_isready,
            'position': self.handle_position,
            'boardpos': self.handle_boardpos,
            'go': self.handle_go,
            'ucinewgame': self.handle_ucinewgame,
            'uci': self.handle_uci
        }

    def _log_debug(self, message: str) -> None:
        if self.debug:
            print(f"info string {message}")

    def _register_default_strategies(self) -> None:
        opening_strategy = OpeningBookStrategy(
            opening_book=self._create_default_opening_book(),
            name="OpeningBookStrategy",
        )
        endgame_strategy = EndgameTableStrategy(name="EndgameTableStrategy")
        tactical_strategy = TacticalResponseStrategy(
            name="TacticalResponseStrategy",
            logger=self._log_debug,
        )
        fallback_strategy = FallbackRandomStrategy(self.random_move, name="FallbackRandomStrategy")
        heuristic_strategy = HeuristicSearchStrategy(
            fallback=fallback_strategy,
            name="HeuristicSearchStrategy",
            search_depth=3,
        )

        for strategy in (
            opening_strategy,
            endgame_strategy,
            tactical_strategy,
            heuristic_strategy,
            fallback_strategy,
        ):
            self.strategy_selector.register_strategy(strategy)

    def _create_default_opening_book(self) -> Dict[str, str]:
        start_board = chess.Board()
        return {
            start_board.fen(): "e2e4",
        }

    def create_strategy_context(
        self, board: chess.Board, time_controls: Optional[Dict[str, int]] = None
    ) -> StrategyContext:
        piece_map = board.piece_map()
        piece_count = len(piece_map)
        material_imbalance = self.compute_material_imbalance(board)
        repetition_info = {
            "is_threefold_repetition": board.is_repetition(),
            "can_claim_threefold": board.can_claim_threefold_repetition(),
            "is_fivefold_repetition": board.is_fivefold_repetition(),
        }
        legal_moves_count = self.get_legal_moves_count(board)

        return StrategyContext(
            fullmove_number=board.fullmove_number,
            halfmove_clock=board.halfmove_clock,
            piece_count=piece_count,
            material_imbalance=material_imbalance,
            turn=board.turn,
            fen=board.fen(),
            repetition_info=repetition_info,
            legal_moves_count=legal_moves_count,
            time_controls=dict(time_controls) if time_controls else None,
        )

    def compute_material_imbalance(self, board: chess.Board) -> int:
        material_balance = 0
        for piece in board.piece_map().values():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            material_balance += value if piece.color == chess.WHITE else -value
        return material_balance

    def get_legal_moves_count(self, board: chess.Board) -> int:
        return board.legal_moves.count()

    def register_strategy(self, strategy: MoveStrategy) -> None:
        self.strategy_selector.register_strategy(strategy)

    def set_selection_policy(
        self,
        policy: Optional[Callable[..., Optional[StrategyResult]]],
    ) -> None:
        self.strategy_selector.set_selection_policy(policy)

    def get_strategy_selector(self) -> StrategySelector:
        return self.strategy_selector

    def get_strategy_context(self) -> StrategyContext:
        with self.state_lock:
            board_snapshot = self.board.copy(stack=True)
        return self.create_strategy_context(board_snapshot)

    def start(self):
        self.handle_uci()
        self.command_processor()
        
    def handle_uci(self, args=None):
        print(f'id name {self.engine_name}')
        print(f'id author {self.engine_author}')
        print('uciok')

    def command_processor(self):
        """
        Continuously read and process commands from stdin.
        Commands are dispatched to appropriate handler methods based on the dispatch table.
        """
        while self.running:
            try:
                command = sys.stdin.readline()
                if not command:
                    break  # EOF reached
                command = command.strip()
                if not command:
                    continue  # Ignore empty lines

                # Split the command into parts for dispatching
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ''

                # Dispatch the command to the appropriate handler
                handler = self.dispatch_table.get(cmd, self.handle_unknown)
                handler(args)
            except Exception as e:
                print(f'info string Error processing command: {e}')
            finally:
                sys.stdout.flush()


    def handle_unknown(self, args):
        print(f"unknown command received: '{args}'")

    def handle_quit(self, args):
        print('info string Engine shutting down')
        self.running = False

    def handle_debug(self, args):
        setting = args.strip().lower()
        if setting == "on":
            self.debug = True
        elif setting == "off":
            self.debug = False
        else:
            print("info string Invalid debug setting. Use 'on' or 'off'.")
            return
        print(f"info string Debug:{self.debug}")

    def handle_isready(self, args):
        with self.state_lock:
            if not self.move_calculating:
                print("readyok")
            else:
                print("info string Engine is busy processing a move")

    def handle_position(self, args):
        with self.state_lock:
            if args.startswith("startpos"):
                self.board.reset()
                if self.debug:
                    print(f"info string Set to start position: {self.board.fen()}")
            elif args.startswith("fen"):
                fen = args[4:].strip()
                try:
                    self.board.set_fen(fen)
                    if self.debug:
                        print(f"info string setpos {self.board.fen()}")
                except ValueError:
                    print("info string Invalid FEN string provided.")
            else:
                print("info string Unknown position command.")

    def handle_boardpos(self, args):
        with self.state_lock:
            print(f"info string Position: {self.board.fen()}" if self.board else "info string Board state not set")

    def _parse_go_args(self, args: str) -> Dict[str, int]:
        tokens = args.split()
        if not tokens:
            return {}

        parsed: Dict[str, int] = {}
        iterator = iter(tokens)
        for token in iterator:
            key = token.lower()
            if key in {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth"}:
                try:
                    value_token = next(iterator)
                    parsed[key] = int(value_token)
                except (StopIteration, ValueError):
                    continue
            elif key == "infinite":
                parsed[key] = True
            elif key == "ponder":
                parsed[key] = True
        return parsed

    def handle_go(self, args):
        time_controls = self._parse_go_args(args)
        with self.state_lock:
            if self.move_calculating:
                print('info string Please wait for computer move')
                return
            self.move_calculating = True

        # Start the move calculation in a separate thread
        move_thread = threading.Thread(target=self.process_go_command, args=(time_controls,))
        move_thread.start()

    def handle_ucinewgame(self, args):
        with self.state_lock:
            self.board.reset()
            if self.debug:
                print("info string New game started, board reset to initial position")
            print("info string New game initialized")

    def random_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        selected_move = random.choice(legal_moves)
        return selected_move.uci()

    def process_go_command(self, time_controls: Optional[Dict[str, int]] = None):
        try:
            with self.state_lock:
                board_snapshot = self.board.copy(stack=True)
                context = self.create_strategy_context(board_snapshot, time_controls=time_controls)

            if self.debug:
                self._log_debug(
                    "context prepared: "
                    f"fullmove={context.fullmove_number}, "
                    f"halfmove={context.halfmove_clock}, "
                    f"pieces={context.piece_count}, "
                    f"material={context.material_imbalance}"
                )

            result = self.strategy_selector.select_move(board_snapshot, context) if self.strategy_selector else None

            if result and result.move:
                print(f"info string strategy {result.strategy_name} selected move {result.move}")
            elif self.debug:
                self._log_debug("no strategy produced a move")

            move = result.move if result else None

            with self.state_lock:
                if move:
                    print(f"bestmove {move}")
                else:
                    print("bestmove (none)")
                print("readyok")
                self.move_calculating = False
        except Exception as exc:
            print(f"info string Error generating move: {exc}")
            with self.state_lock:
                print("bestmove (none)")
                print("readyok")
                self.move_calculating = False



engine = ChessEngine()
engine.start()
