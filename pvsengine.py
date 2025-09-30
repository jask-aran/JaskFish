"""Refactored chess engine implementing the JaskFish strategy stack.

This module keeps protocol compatibility with ``engine.py`` but restructures the
engine into composable units so that the search logic, configuration handling,
and UCI façade are easier to reason about and test in isolation.
"""

from __future__ import annotations

import io
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import chess
from chess import polyglot


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _ensure_line_buffered_stdout() -> None:
    stdout = sys.stdout
    if isinstance(stdout, io.TextIOBase) and getattr(stdout, "line_buffering", False):
        return
    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        return
    sys.stdout = io.TextIOWrapper(buffer, line_buffering=True)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -30, -30, -40, -40, -30, -30, -30,
        -20, -20, -20, -20, -20, -20, -20, -20,
        -10, -10, -10, -10, -10, -10, -10, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ]
}


NSEC_PER_SEC = 1_000_000_000


# ---------------------------------------------------------------------------
# Strategy interfaces and context models
# ---------------------------------------------------------------------------


@dataclass
class StrategyContext:
    fullmove_number: int
    halfmove_clock: int
    piece_count: int
    material_imbalance: int
    turn: bool
    fen: str
    repetition_info: Dict[str, bool]
    legal_moves_count: int
    time_controls: Optional[Dict[str, int]]
    in_check: bool
    opponent_mate_in_one_threat: bool


@dataclass
class StrategyResult:
    move: Optional[str]
    strategy_name: str
    score: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    definitive: bool = False


class MoveStrategy(ABC):
    def __init__(self, *, name: Optional[str] = None, priority: int = 0, confidence: Optional[float] = None):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.confidence = confidence

    @abstractmethod
    def is_applicable(self, context: StrategyContext) -> bool:
        ...

    @abstractmethod
    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        ...

    def apply_config(self, config: Any) -> None:
        pass


class StrategySelector:
    def __init__(self, *, logger: Optional[Callable[[str], None]] = None):
        self._strategies: List[MoveStrategy] = []
        self._logger = logger or (lambda *_: None)

    def register(self, strategy: MoveStrategy) -> None:
        self._strategies.append(strategy)
        self._strategies.sort(key=lambda s: s.priority, reverse=True)
        self._logger(f"strategy registered: {strategy.name} (priority={strategy.priority})")

    def strategies(self) -> Tuple[MoveStrategy, ...]:
        return tuple(self._strategies)

    def select(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        best_result: Optional[StrategyResult] = None
        best_key = (-float("inf"), -float("inf"), -float("inf"))

        for strategy in self._strategies:
            if not strategy.is_applicable(context):
                continue
            try:
                result = strategy.generate_move(board, context)
            except Exception as exc:
                self._logger(f"strategy {strategy.name} error: {exc}")
                continue
            if not result or not result.move:
                continue
            if result.definitive:
                return result

            score = float(result.score) if result.score is not None else 0.0
            confidence = float(result.confidence) if result.confidence is not None else 0.0
            key = (float(strategy.priority), score, confidence)
            if key > best_key:
                best_key = key
                best_result = result

        if best_result is None:
            self._logger("no strategies produced a move suggestion")
        return best_result


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class StrategyToggle(Enum):
    MATE_IN_ONE = "mate_in_one"
    HEURISTIC = "heuristic"


@dataclass(slots=True, frozen=True)
class MetaParams:
    strength: float = 0.6
    speed_bias: float = 0.4
    risk: float = 0.2
    stability: float = 0.5
    tt_budget_mb: int = 64
    style_tactical: float = 0.5
    endgame_focus: float = 0.4
    avoid_repetition: bool = True
    repetition_penalty: float = 45.0
    repetition_strong_penalty: float = 90.0

    def clamp(self) -> "MetaParams":
        return replace(
            self,
            strength=_clamp(self.strength, 0.0, 1.0),
            speed_bias=_clamp(self.speed_bias, 0.0, 1.0),
            risk=_clamp(self.risk, 0.0, 1.0),
            stability=_clamp(self.stability, 0.0, 1.0),
            style_tactical=_clamp(self.style_tactical, 0.0, 1.0),
            endgame_focus=_clamp(self.endgame_focus, 0.0, 1.0),
            tt_budget_mb=max(1, int(self.tt_budget_mb)),
        )


class MetaRegistry:
    PRESETS: Dict[str, MetaParams] = {
        "balanced": MetaParams(),
        "fastblitz": MetaParams(
            strength=0.45,
            speed_bias=0.8,
            risk=0.35,
            stability=0.6,
            tt_budget_mb=32,
        ),
        "tournament": MetaParams(
            strength=0.9,
            speed_bias=0.2,
            risk=0.15,
            stability=0.75,
            tt_budget_mb=256,
            style_tactical=0.55,
            endgame_focus=0.6,
        ),
    }

    @classmethod
    def resolve(cls, preset: str) -> MetaParams:
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown meta preset '{preset}'")
        return cls.PRESETS[preset].clamp()


class SearchLimits:
    def __init__(self, *, min_time: float, max_time: float, base_time: float, time_factor: float) -> None:
        self.min_time = min_time
        self.max_time = max_time
        self.base_time = base_time
        self.time_factor = time_factor

    def resolve_budget(self, context: StrategyContext) -> Optional[float]:
        tc = context.time_controls or {}
        if tc.get("infinite"):
            return None

        if "movetime" in tc:
            seconds = tc["movetime"] / 1000.0
            return self._clamp(seconds)

        turn_key = "wtime" if context.turn == chess.WHITE else "btime"
        inc_key = "winc" if context.turn == chess.WHITE else "binc"
        if turn_key in tc:
            time_left = max(tc.get(turn_key, 0), 0)
            increment = max(tc.get(inc_key, 0), 0)
            moves_to_go = tc.get("movestogo")
            if moves_to_go:
                budget = time_left / max(1, moves_to_go)
            else:
                budget = time_left * self.time_factor
            budget += increment
            return self._clamp(budget / 1000.0)

        complexity = context.legal_moves_count
        phase = context.piece_count

        phase_factor = 0.3 + min(1.0, max(0.0, (phase - 2) / 30.0)) * 0.7
        if complexity <= 10:
            complexity_factor = 0.25
        elif complexity <= 20:
            complexity_factor = 0.5
        elif complexity <= 35:
            complexity_factor = 0.9
        elif complexity <= 60:
            complexity_factor = 1.4
        else:
            complexity_factor = 1.8 + min(0.6, (complexity - 60) * 0.01)

        tension_factor = 1.0
        if context.in_check or context.opponent_mate_in_one_threat:
            tension_factor = max(tension_factor, 1.35)

        return self._clamp(self.base_time * complexity_factor * phase_factor * tension_factor)

    def _clamp(self, seconds: float) -> float:
        return max(self.min_time, min(self.max_time, seconds))


class SearchReporter:
    def __init__(self, *, logger: Callable[[str], None]):
        self._log = logger

    def trace(self, message: str) -> None:
        self._log(message)

    def perf_summary(
        self,
        label: str,
        depth: int,
        nodes: int,
        time_spent: float,
        select_time: float,
        timing_breakdown: Optional[Dict[str, float]],
    ) -> None:
        nps = int(nodes / time_spent) if time_spent else 0
        segments = [
            f"info string perf {label} depth={depth if depth else '-'}",
            f"nodes={nodes}",
            f"time={time_spent:.3f}s",
            f"nps={nps}",
            f"select={select_time:.3f}s",
        ]
        if timing_breakdown:
            parts = [f"{key}={value:.3f}s" for key, value in timing_breakdown.items()]
            if parts:
                segments.append("timers=" + ",".join(parts))
        print(" ".join(segments))


@dataclass
class SearchTuning:
    search_depth: int
    quiescence_depth: int
    base_time_limit: float
    min_time_limit: float
    max_time_limit: float
    time_allocation_factor: float
    aspiration_window: float
    aspiration_growth: float
    aspiration_fail_reset: int
    depth_stop_ratio: float
    null_min_depth: int
    null_depth_scale: int
    null_base_reduction: int
    lmr_min_depth: int
    lmr_min_index: int
    futility_depth_limit: int
    futility_margin: float
    razoring_depth_limit: int
    razoring_margin: float
    qsearch_see_threshold: float
    killer_slots: int
    history_decay: float
    bishop_pair_bonus: float
    mobility_unit: float
    king_safety_opening_penalty: float
    king_safety_endgame_bonus: float
    passed_pawn_base_bonus: float
    passed_pawn_rank_bonus: float

def build_search_tuning(meta: MetaParams) -> Tuple[SearchTuning, SearchLimits]:
    strength = meta.strength
    speed_bias = meta.speed_bias
    risk = meta.risk
    stability = meta.stability
    style_tactical = meta.style_tactical
    endgame_focus = meta.endgame_focus

    search_depth = 3 + int(6 * strength * (1.0 - 0.5 * speed_bias))
    quiescence_depth = 3 + int(6 * strength * (0.5 + 0.5 * stability))
    base_time = 0.2 + 6.0 * strength * (1.0 - 0.7 * speed_bias)
    time_alloc = 0.06 + 0.16 * strength * (1.0 - speed_bias)
    min_time = 0.05 + 0.35 * (1.0 - speed_bias)
    max_time = 1.0 + 30.0 * strength
    asp_window = 30 + int(120 * (1.0 - stability))
    asp_growth = 1.5 + 1.0 * (1.0 - stability)
    asp_reset = 1 if stability >= 0.7 else 2
    null_min_depth = 2 + int(3 * strength)
    null_scale = 3 + int(3 * speed_bias)
    lmr_min_depth = 3 + int(3 * strength * speed_bias)
    lmr_min_index = 3 + int(6 * speed_bias)
    fut_depth = 1 + int(2 * speed_bias)
    fut_margin = 80 + int(140 * (1.0 - risk))
    razor_depth = 1 + int(2 * speed_bias)
    razor_margin = 220 + int(240 * (1.0 - risk))
    depth_stop_ratio = 0.65 + 0.25 * stability
    q_threshold = -1.5 + risk
    killer_slots = 1 if speed_bias > 0.7 else 2
    history_decay = 0.85 + 0.1 * stability
    bishop_pair_bonus = 20 + 60 * style_tactical
    mobility_unit = 1.0 + 3.0 * style_tactical
    ks_open_pen = 8 + 16 * (1.0 - risk)
    ks_end_bonus = 2 + 10 * endgame_focus
    passed_base = 10 + 30 * style_tactical
    passed_rank = 4 + 10 * style_tactical

    tuning = SearchTuning(
        search_depth=search_depth,
        quiescence_depth=quiescence_depth,
        base_time_limit=base_time,
        min_time_limit=min_time,
        max_time_limit=max_time,
        time_allocation_factor=time_alloc,
        aspiration_window=float(asp_window),
        aspiration_growth=float(asp_growth),
        aspiration_fail_reset=int(asp_reset),
        depth_stop_ratio=float(depth_stop_ratio),
        null_min_depth=int(null_min_depth),
        null_depth_scale=int(null_scale),
        null_base_reduction=1,
        lmr_min_depth=int(lmr_min_depth),
        lmr_min_index=int(lmr_min_index),
        futility_depth_limit=int(fut_depth),
        futility_margin=float(fut_margin),
        razoring_depth_limit=int(razor_depth),
        razoring_margin=float(razor_margin),
        qsearch_see_threshold=float(q_threshold),
        killer_slots=int(killer_slots),
        history_decay=float(history_decay),
        bishop_pair_bonus=float(bishop_pair_bonus),
        mobility_unit=float(mobility_unit),
        king_safety_opening_penalty=float(ks_open_pen),
        king_safety_endgame_bonus=float(ks_end_bonus),
        passed_pawn_base_bonus=float(passed_base),
        passed_pawn_rank_bonus=float(passed_rank),
    )

    limits = SearchLimits(
        min_time=min_time,
        max_time=max_time,
        base_time=base_time,
        time_factor=time_alloc,
    )

    return tuning, limits


@dataclass(slots=True)
class SearchOutcome:
    move: Optional[chess.Move]
    score: float
    completed_depth: int
    nodes: int
    time_spent: float
    principal_variation: Tuple[chess.Move, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchBackend:
    def __init__(self, *, max_threads: int = 1) -> None:
        self.max_threads = max(1, int(max_threads))

    def configure(self, tuning: SearchTuning, meta: MetaParams) -> None:
        raise NotImplementedError

    def search(
        self,
        board: chess.Board,
        context: StrategyContext,
        limits: SearchLimits,
        reporter: SearchReporter,
        budget_seconds: Optional[float],
    ) -> SearchOutcome:
        raise NotImplementedError


@dataclass(slots=True)
class _TTEntry:
    key: int
    depth: int
    value: float
    flag: int
    move: Optional[chess.Move]
    static_eval: Optional[float]


class PackedTranspositionTable:
    def __init__(self, size: int) -> None:
        power = max(10, int(size).bit_length())
        self.size = 1 << power
        self.mask = self.size - 1
        self._table: List[Optional[_TTEntry]] = [None] * self.size
        self.hits = 0
        self.probes = 0

    def resize(self, size: int) -> None:
        power = max(10, int(size).bit_length())
        new_size = 1 << power
        if new_size == self.size:
            return
        old_entries = [entry for entry in self._table if entry is not None]
        self.size = new_size
        self.mask = new_size - 1
        self._table = [None] * new_size
        for entry in old_entries:
            self.store(entry.key, entry.depth, entry.value, entry.flag, entry.move, entry.static_eval)

    def probe(self, key: int) -> Optional[_TTEntry]:
        self.probes += 1
        slot = key & self.mask
        entry = self._table[slot]
        if entry is not None and entry.key == key:
            self.hits += 1
            return entry
        return None

    def store(
        self,
        key: int,
        depth: int,
        value: float,
        flag: int,
        move: Optional[chess.Move],
        static_eval: Optional[float],
    ) -> None:
        slot = key & self.mask
        existing = self._table[slot]
        if existing is not None and existing.key == key and existing.depth > depth:
            return
        self._table[slot] = _TTEntry(key, depth, value, flag, move, static_eval)


class HistoryTable:
    __slots__ = ("decay", "values")

    def __init__(self, decay: float) -> None:
        self.decay = decay
        self.values: Dict[Tuple[bool, int, int], float] = {}

    def score(self, move: chess.Move, color: bool) -> float:
        return self.values.get((color, move.from_square, move.to_square), 0.0)

    def record(self, move: chess.Move, color: bool, depth: int) -> None:
        key = (color, move.from_square, move.to_square)
        self.values[key] = self.values.get(key, 0.0) + depth * depth
        if self.values[key] > 500_000:
            self.values[key] *= 0.5

    def decay_all(self) -> None:
        if not self.values:
            return
        for key, value in list(self.values.items()):
            new_value = value * self.decay
            if new_value < 1.0:
                self.values.pop(key, None)
            else:
                self.values[key] = new_value


class KillerTable:
    __slots__ = ("slots", "table")

    def __init__(self, max_depth: int, slots: int) -> None:
        self.slots = max(1, slots)
        self.table: List[List[Optional[chess.Move]]] = [[None] * self.slots for _ in range(max_depth + 5)]

    def reset(self, max_depth: int) -> None:
        self.table = [[None] * self.slots for _ in range(max_depth + 5)]

    def record(self, ply: int, move: chess.Move) -> None:
        if ply >= len(self.table):
            return
        killers = self.table[ply]
        if move in killers:
            return
        killers.insert(0, move)
        del killers[self.slots:]

    def is_killer(self, ply: int, move: chess.Move) -> bool:
        return ply < len(self.table) and move in self.table[ply]


_MATE_SCORE = 100000


class _SearchState:
    __slots__ = (
        "board",
        "tuning",
        "meta",
        "limits",
        "reporter",
        "context",
        "deadline",
        "tt",
        "killers",
        "history",
        "nodes",
        "principal_variation",
        "eval_cache",
        "avoid_repetition",
        "repetition_penalty",
        "repetition_strong_penalty",
        "budget",
    )

    def __init__(
        self,
        *,
        board: chess.Board,
        tuning: SearchTuning,
        meta: MetaParams,
        limits: SearchLimits,
        reporter: SearchReporter,
        context: StrategyContext,
        budget: Optional[float],
        deadline: Optional[float] = None,
    ) -> None:
        self.board = board
        self.tuning = tuning
        self.meta = meta
        self.limits = limits
        self.reporter = reporter
        self.context = context
        self.budget = budget
        base_deadline = None if budget is None else time.perf_counter() + max(budget, tuning.min_time_limit)
        self.deadline = deadline if deadline is not None else base_deadline
        tt_size = int((meta.tt_budget_mb * 1024 * 1024) / 40)
        self.tt = PackedTranspositionTable(max(1024, tt_size))
        self.killers = KillerTable(tuning.search_depth + tuning.quiescence_depth, tuning.killer_slots)
        self.history = HistoryTable(tuning.history_decay)
        self.nodes = 0
        self.principal_variation: List[chess.Move] = []
        self.eval_cache: Dict[int, float] = {}
        self.avoid_repetition = meta.avoid_repetition
        self.repetition_penalty = meta.repetition_penalty
        self.repetition_strong_penalty = meta.repetition_strong_penalty

    def time_exceeded(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def record_killer(self, ply: int, move: chess.Move) -> None:
        self.killers.record(ply, move)

    def repetition_penalty_for(self, board: chess.Board) -> float:
        if not self.avoid_repetition:
            return 0.0
        if board.is_fivefold_repetition():
            return float(_MATE_SCORE)
        penalty = 0.0
        if board.can_claim_fifty_moves():
            penalty = max(penalty, self.repetition_strong_penalty)
        try:
            if board.is_repetition():
                penalty = max(penalty, self.repetition_strong_penalty)
            elif board.can_claim_threefold_repetition():
                penalty = max(penalty, self.repetition_strong_penalty * 0.8)
        except ValueError:
            penalty = max(penalty, self.repetition_strong_penalty * 0.8)
        try:
            if board.is_repetition(2):
                penalty = max(penalty, self.repetition_penalty)
        except ValueError:
            penalty = max(penalty, self.repetition_penalty)
        return penalty


class PVSearchBackend(SearchBackend):
    def __init__(self, *, max_threads: int = 1) -> None:
        super().__init__(max_threads=max_threads)
        self.tuning: Optional[SearchTuning] = None
        self.meta: Optional[MetaParams] = None
        self.limits: Optional[SearchLimits] = None
        # Persistent executor to avoid per-iteration overhead
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=max_threads) if max_threads > 1 else None
        )

    def shutdown(self) -> None:
        """Shutdown the persistent thread pool executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

    def configure(self, tuning: SearchTuning, meta: MetaParams) -> None:
        self.tuning = tuning
        self.meta = meta
        self.limits = SearchLimits(
            min_time=tuning.min_time_limit,
            max_time=tuning.max_time_limit,
            base_time=tuning.base_time_limit,
            time_factor=tuning.time_allocation_factor,
        )

    def search(
        self,
        board: chess.Board,
        context: StrategyContext,
        limits: SearchLimits,
        reporter: SearchReporter,
        budget_seconds: Optional[float],
    ) -> SearchOutcome:
        if self.tuning is None or self.meta is None:
            raise RuntimeError("PVSearchBackend not configured")

        state = _SearchState(
            board=board,
            tuning=self.tuning,
            meta=self.meta,
            limits=limits,
            reporter=reporter,
            context=context,
            budget=budget_seconds,
        )

        start = time.perf_counter()
        result = self._iterative_deepening(state)
        elapsed = max(1e-9, time.perf_counter() - start)

        metadata = {
            "nodes": state.nodes,
            "time": elapsed,
            "depth": result.completed_depth,
            "pv": [mv.uci() for mv in result.principal_variation],
            "tt_hit_rate": state.tt.hits / max(1, state.tt.probes),
        }

        reporter.perf_summary(
            label="move",
            depth=result.completed_depth,
            nodes=state.nodes,
            time_spent=elapsed,
            select_time=0.0,
            timing_breakdown=None,
        )

        return SearchOutcome(
            move=result.move,
            score=result.score,
            completed_depth=result.completed_depth,
            nodes=state.nodes,
            time_spent=elapsed,
            principal_variation=tuple(result.principal_variation),
            metadata=metadata,
        )

    def _iterative_deepening(self, state: _SearchState) -> SearchOutcome:
        tuning = state.tuning
        board = state.board
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        completed_depth = 0
        last_score = None
        aspiration = tuning.aspiration_window
        pv: List[chess.Move] = []

        for depth in range(1, tuning.search_depth + 1):
            if state.time_exceeded():
                state.reporter.trace(f"PVS: timeout before depth={depth}")
                break
            
            # Track depth iteration metrics
            depth_start_time = time.perf_counter()
            depth_start_nodes = state.nodes
            
            state.history.decay_all()
            alpha = -float(_MATE_SCORE)
            beta = float(_MATE_SCORE)
            if last_score is not None:
                alpha = max(alpha, last_score - aspiration)
                beta = min(beta, last_score + aspiration)

            iteration_best: Optional[chess.Move] = None
            iteration_score = -float("inf")

            while True:
                alpha_window = alpha
                beta_window = beta
                iteration_best = None
                iteration_score = -float("inf")

                fail_low = False
                fail_high = False
                ordered_moves = self._order_root(state, board, pv[0] if pv else best_move)
                if self.max_threads > 1 and len(ordered_moves) > 1:
                    results = self._evaluate_root_parallel(state, board, ordered_moves, depth, alpha_window, beta_window)
                    if state.time_exceeded():
                        break
                    if not results:
                        break
                    iteration_best, iteration_score = max(results, key=lambda item: item[1])
                    if iteration_score > alpha_window:
                        alpha_window = iteration_score
                    if iteration_score <= alpha and alpha > -float(_MATE_SCORE):
                        fail_low = True
                    if any(score >= beta_window for _, score in results) and beta < float(_MATE_SCORE):
                        fail_high = True
                        for move, score in results:
                            if score >= beta_window and not board.is_capture(move):
                                state.history.record(move, board.turn, depth)
                                state.record_killer(0, move)
                else:
                    for index, move in enumerate(ordered_moves):
                        if state.time_exceeded():
                            break
                        score = self._search_root_move(state, board, move, depth, alpha_window, beta_window, index == 0)
                        if state.time_exceeded():
                            break
                        if score > iteration_score:
                            iteration_score = score
                            iteration_best = move
                        if score > alpha_window:
                            alpha_window = score
                        if alpha_window >= beta_window:
                            state.history.record(move, board.turn, depth)
                            state.record_killer(0, move)
                            break
                    if state.time_exceeded():
                        break
                    if iteration_best is None:
                        break
                    if iteration_score <= alpha and alpha > -float(_MATE_SCORE):
                        fail_low = True
                    if iteration_score >= beta and beta < float(_MATE_SCORE):
                        fail_high = True
                    alpha = max(-float(_MATE_SCORE), iteration_score - aspiration)
                    beta = min(float(_MATE_SCORE), iteration_score + aspiration)
                    continue
                if iteration_score >= beta and beta < float(_MATE_SCORE):
                    aspiration *= tuning.aspiration_growth
                    alpha = max(-float(_MATE_SCORE), iteration_score - aspiration)
                    beta = min(float(_MATE_SCORE), iteration_score + aspiration)
                    continue

                best_move = iteration_best
                best_score = iteration_score
                completed_depth = depth
                last_score = iteration_score
                pv = self._extract_pv(state, board, depth)
                if not pv and iteration_best is not None:
                    pv = [iteration_best]
                
                # Log depth completion
                depth_elapsed = time.perf_counter() - depth_start_time
                depth_nodes = state.nodes - depth_start_nodes
                depth_nps = int(depth_nodes / depth_elapsed) if depth_elapsed > 1e-6 else 0
                pv_text = " ".join(mv.uci() for mv in pv[:6])
                state.reporter.trace(
                    f"PVS: depth={depth} score={iteration_score:.1f} "
                    f"nodes={state.nodes}(+{depth_nodes}) time={depth_elapsed:.2f}s "
                    f"nps={depth_nps} pv={pv_text}"
                )
                
                # Check if we should stop iterating due to budget consumption
                if state.budget is not None and depth_elapsed >= state.budget * tuning.depth_stop_ratio:
                    usage_pct = depth_elapsed / state.budget * 100
                    state.reporter.trace(
                        f"PVS: depth={depth} consumed {usage_pct:.0f}% of budget; halting deeper search"
                    )
                    break
                
                break

            if state.time_exceeded():
                # Log timeout during depth iteration
                depth_elapsed = time.perf_counter() - depth_start_time
                state.reporter.trace(
                    f"PVS: depth={depth} timeout after {depth_elapsed:.2f}s "
                    f"nodes={state.nodes - depth_start_nodes} visited"
                )
                break

        if best_move is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = legal_moves[0]
                best_score = 0.0
                pv = [best_move]

        return SearchOutcome(
            move=best_move,
            score=best_score,
            completed_depth=completed_depth,
            nodes=state.nodes,
            time_spent=0.0,
            principal_variation=tuple(pv),
        )

    def _order_root(
        self,
        state: _SearchState,
        board: chess.Board,
        principal_move: Optional[chess.Move],
    ) -> List[chess.Move]:
        moves = list(board.legal_moves)
        tt_entry = state.tt.probe(polyglot.zobrist_hash(board))
        tt_move = tt_entry.move if tt_entry else None

        def score(move: chess.Move) -> int:
            if tt_move and move == tt_move:
                return 12_000_000
            total = 0
            if principal_move and move == principal_move:
                total += 11_000_000
            if board.is_capture(move):
                total += 8_000_000 + int(self._see(board, move))
            if move.promotion:
                total += 6_000_000 + EVAL_PIECE_VALUES.get(move.promotion, 0)
            if board.gives_check(move):
                total += 500_000
            total += int(state.history.score(move, board.turn))
            return total

        moves.sort(key=score, reverse=True)
        return moves

    def _search_root_move(
        self,
        state: _SearchState,
        board: chess.Board,
        move: chess.Move,
        depth: int,
        alpha: float,
        beta: float,
        is_pv: bool,
    ) -> float:
        board.push(move)
        try:
            score = -self._alpha_beta(state, board, depth - 1, -beta, -alpha, ply=1, is_pv=is_pv, allow_null=True)
            penalty = 0.0
            if state.avoid_repetition:
                penalty = state.repetition_penalty_for(board)
            score -= penalty
        finally:
            board.pop()
        return score

    def _evaluate_root_parallel(
        self,
        state: _SearchState,
        board: chess.Board,
        moves: Sequence[chess.Move],
        depth: int,
        alpha: float,
        beta: float,
    ) -> List[Tuple[chess.Move, float]]:
        # Use persistent executor if available, otherwise fall back to sequential
        if self._executor is None:
            return self._evaluate_root_sequential(state, board, moves, depth, alpha, beta)
        
        tasks: List[Tuple[int, chess.Move, float]] = []
        futures = []
        for index, move in enumerate(moves):
            futures.append(
                (
                    index,
                    self._executor.submit(
                        self._evaluate_root_worker,
                        state,
                        board,
                        move,
                        depth,
                        alpha,
                        beta,
                    ),
                )
            )
        for index, future in futures:
            move, score, nodes = future.result()
            tasks.append((index, move, score))
            state.nodes += nodes
        tasks.sort(key=lambda item: item[0])
        return [(move, score) for _, move, score in tasks]

    def _evaluate_root_sequential(
        self,
        state: _SearchState,
        board: chess.Board,
        moves: Sequence[chess.Move],
        depth: int,
        alpha: float,
        beta: float,
    ) -> List[Tuple[chess.Move, float]]:
        """Sequential fallback for single-threaded execution."""
        results: List[Tuple[chess.Move, float]] = []
        for move in moves:
            if state.time_exceeded():
                break
            score = self._search_root_move(state, board, move, depth, alpha, beta, False)
            results.append((move, score))
        return results

    def _evaluate_root_worker(
        self,
        parent_state: _SearchState,
        board: chess.Board,
        move: chess.Move,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[chess.Move, float, int]:
        # Use stack=False for ~40% faster copying; root-level moves rarely repeat
        local_board = board.copy(stack=False)
        local_state = _SearchState(
            board=local_board,
            tuning=parent_state.tuning,
            meta=parent_state.meta,
            limits=parent_state.limits,
            reporter=parent_state.reporter,
            context=parent_state.context,
            budget=parent_state.budget,
            deadline=parent_state.deadline,
        )
        score = self._search_root_move(local_state, local_board, move, depth, alpha, beta, False)
        return move, score, local_state.nodes

    def _alpha_beta(
        self,
        state: _SearchState,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        *,
        ply: int,
        is_pv: bool,
        allow_null: bool,
    ) -> float:
        if state.time_exceeded():
            return alpha

        state.nodes += 1
        if board.is_checkmate():
            return -float(_MATE_SCORE) + ply
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        if depth <= 0:
            return self._quiescence(state, board, alpha, beta, state.tuning.quiescence_depth, ply)

        key = polyglot.zobrist_hash(board)
        tt_entry = state.tt.probe(key)
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == 0:
                return tt_entry.value
            if tt_entry.flag == 1:
                alpha = max(alpha, tt_entry.value)
            else:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                return tt_entry.value

        in_check = board.is_check()
        static_eval = tt_entry.static_eval if tt_entry else None

        if not in_check and not is_pv and allow_null and depth >= state.tuning.null_min_depth and self._has_material(board):
            board.push(chess.Move.null())
            try:
                reduction = state.tuning.null_base_reduction + max(0, depth // max(1, state.tuning.null_depth_scale))
                score = -self._alpha_beta(
                    state,
                    board,
                    depth - 1 - reduction,
                    -beta,
                    -beta + 1,
                    ply=ply + 1,
                    is_pv=False,
                    allow_null=False,
                )
            finally:
                board.pop()
            if score >= beta:
                state.tt.store(key, depth, score, 1, None, static_eval)
                return score

        tt_move = tt_entry.move if tt_entry else None
        moves = self._order_moves(state, board, ply, tt_move)
        if not moves:
            return self._evaluate(state, board)

        best_value = -float("inf")
        best_move = None
        alpha_orig = alpha

        for index, move in enumerate(moves):
            if state.time_exceeded():
                break
            is_capture = board.is_capture(move)
            gives_check = board.gives_check(move)
            promotion = move.promotion is not None
            color = board.turn

            board.push(move)
            try:
                child_is_pv = is_pv and index == 0
                new_allow_null = allow_null and not is_capture
                next_depth = depth - 1
                if depth >= state.tuning.lmr_min_depth and index >= state.tuning.lmr_min_index and not child_is_pv and not is_capture and not gives_check and not promotion:
                    reduction = max(1, int((math.log(depth + 1, 2) * math.log(index + 1, 2)) / 1.5))
                    next_depth = max(0, depth - 1 - reduction)
                if child_is_pv:
                    score = -self._alpha_beta(
                        state,
                        board,
                        next_depth,
                        -beta,
                        -alpha,
                        ply=ply + 1,
                        is_pv=True,
                        allow_null=new_allow_null,
                    )
                else:
                    score = -self._alpha_beta(
                        state,
                        board,
                        next_depth,
                        -alpha - 1,
                        -alpha,
                        ply=ply + 1,
                        is_pv=False,
                        allow_null=new_allow_null,
                    )
                    if score > alpha:
                        score = -self._alpha_beta(
                            state,
                            board,
                            depth - 1,
                            -beta,
                            -alpha,
                            ply=ply + 1,
                            is_pv=True,
                            allow_null=new_allow_null,
                        )
            finally:
                board.pop()

            if score > best_value:
                best_value = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_capture:
                    state.record_killer(ply, move)
                    state.history.record(move, color, depth)
                state.tt.store(key, depth, best_value, 1, best_move, static_eval)
                return best_value

        flag = 0 if best_value > alpha_orig else 2
        if static_eval is None and not in_check:
            static_eval = self._evaluate(state, board)
        state.tt.store(key, depth, best_value, flag, best_move, static_eval)
        return best_value

    def _order_moves(self, state: _SearchState, board: chess.Board, ply: int, tt_move: Optional[chess.Move]) -> List[chess.Move]:
        moves = list(board.legal_moves)

        def score(move: chess.Move) -> int:
            total = 0
            if tt_move and move == tt_move:
                total += 12_000_000
            if state.killers.is_killer(ply, move):
                total += 9_500_000
            if board.is_capture(move):
                total += 8_000_000 + int(self._see(board, move))
            if move.promotion:
                total += 6_000_000 + EVAL_PIECE_VALUES.get(move.promotion, 0)
            if board.gives_check(move):
                total += 500_000
            total += int(state.history.score(move, board.turn))
            return total

        moves.sort(key=score, reverse=True)
        return moves

    def _quiescence(self, state: _SearchState, board: chess.Board, alpha: float, beta: float, depth: int, ply: int) -> float:
        if state.time_exceeded():
            return alpha
        state.nodes += 1
        stand_pat = self._evaluate(state, board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if depth <= 0:
            return stand_pat

        for move in self._generate_qmoves(board):
            if state.time_exceeded():
                break
            board.push(move)
            try:
                score = -self._quiescence(state, board, -beta, -alpha, depth - 1, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _evaluate(self, state: _SearchState, board: chess.Board) -> float:
        key = polyglot.zobrist_hash(board)
        cached = state.eval_cache.get(key)
        if cached is not None:
            return cached

        material = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        pst = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        bishops = {chess.WHITE: 0, chess.BLACK: 0}

        for square, piece in board.piece_map().items():
            value = EVAL_PIECE_VALUES.get(piece.piece_type, 0)
            material[piece.color] += value
            table = PIECE_SQUARE_TABLES.get(piece.piece_type)
            if table:
                idx = square if piece.color == chess.WHITE else chess.square_mirror(square)
                pst[piece.color] += table[idx]
            if piece.piece_type == chess.BISHOP:
                bishops[piece.color] += 1

        score = (material[chess.WHITE] - material[chess.BLACK]) + (pst[chess.WHITE] - pst[chess.BLACK])
        if bishops[chess.WHITE] >= 2:
            score += state.tuning.bishop_pair_bonus
        if bishops[chess.BLACK] >= 2:
            score -= state.tuning.bishop_pair_bonus

        score += self._passed_pawn_bonus(state, board)
        score += self._mobility_term(state, board)
        score += self._king_safety(state, board)

        result = score if board.turn == chess.WHITE else -score
        state.eval_cache[key] = result
        return result

    def _passed_pawn_bonus(self, state: _SearchState, board: chess.Board) -> float:
        total = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            enemy_pawns = set(board.pieces(chess.PAWN, not color))
            for square in board.pieces(chess.PAWN, color):
                file_idx = chess.square_file(square)
                rank = chess.square_rank(square)
                direction = 1 if color == chess.WHITE else -1
                passed = True
                for df in (-1, 0, 1):
                    nf = file_idx + df
                    if not 0 <= nf < 8:
                        continue
                    r = rank + direction
                    while 0 <= r < 8:
                        sq = chess.square(nf, r)
                        if sq in enemy_pawns:
                            passed = False
                            break
                        r += direction
                    if not passed:
                        break
                if passed:
                    advancement = rank - 1 if color == chess.WHITE else 6 - rank
                    bonus = state.tuning.passed_pawn_base_bonus + state.tuning.passed_pawn_rank_bonus * max(0, advancement)
                    total += sign * bonus
        return total

    def _mobility_term(self, state: _SearchState, board: chess.Board) -> float:
        mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        try:
            opponent = board.legal_moves.count()
        finally:
            board.pop()
        return (mobility - opponent) * state.tuning.mobility_unit

    def _king_safety(self, state: _SearchState, board: chess.Board) -> float:
        phase = self._game_phase(board)
        opening = phase
        endgame = 1.0 - phase
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is None:
                continue
            attackers = len(board.attackers(not color, king_sq))
            ring = 0
            file = chess.square_file(king_sq)
            rank = chess.square_rank(king_sq)
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if not df and not dr:
                        continue
                    nf = file + df
                    nr = rank + dr
                    if 0 <= nf < 8 and 0 <= nr < 8:
                        piece = board.piece_at(chess.square(nf, nr))
                        if piece and piece.color == color:
                            ring += 1
            opening_term = opening * (-state.tuning.king_safety_opening_penalty * attackers + 2.0 * ring)
            central_distance = abs(file - 3.5) + abs(rank - 3.5)
            endgame_term = endgame * state.tuning.king_safety_endgame_bonus * (3.5 - central_distance)
            score += sign * (opening_term + endgame_term)
        return score

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
        max_phase = sum(phase_values[p] * initial_counts[p] for p in phase_values)
        current = 0
        for piece_type, value in phase_values.items():
            count = len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            current += value * count
        if not max_phase:
            return 0.0
        return min(1.0, max(0.0, current / max_phase))

    def _extract_pv(self, state: _SearchState, board: chess.Board, depth: int) -> List[chess.Move]:
        pv: List[chess.Move] = []
        probe = board.copy(stack=False)
        for _ in range(depth):
            entry = state.tt.probe(polyglot.zobrist_hash(probe))
            if entry is None or entry.move is None:
                break
            pv.append(entry.move)
            probe.push(entry.move)
        return pv

    def _generate_qmoves(self, board: chess.Board) -> Sequence[chess.Move]:
        in_check = board.is_check()
        moves: List[chess.Move] = []
        for move in board.legal_moves:
            if in_check:
                moves.append(move)
                continue
            if board.is_capture(move):
                if self._see(board, move) >= -50:
                    moves.append(move)
            elif move.promotion or board.gives_check(move):
                moves.append(move)
        moves.sort(key=lambda mv: self._see(board, mv) + (500 if board.gives_check(mv) else 0), reverse=True)
        return moves

    def _see(self, board: chess.Board, move: chess.Move) -> float:
        if hasattr(board, "see"):
            try:
                return float(board.see(move))
            except (ValueError, AttributeError):
                pass
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)
        captured_val = EVAL_PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        mover = board.piece_at(move.from_square)
        mover_val = EVAL_PIECE_VALUES.get(mover.piece_type, 0) if mover else 0
        return captured_val - mover_val

    def _has_material(self, board: chess.Board) -> bool:
        minor_or_rook = False
        pawns = 0
        queens = 0
        for piece in board.piece_map().values():
            if piece.piece_type in (chess.BISHOP, chess.KNIGHT, chess.ROOK):
                minor_or_rook = True
            elif piece.piece_type == chess.PAWN:
                pawns += 1
            elif piece.piece_type == chess.QUEEN:
                queens += 1
        if minor_or_rook:
            return True
        if queens:
            return queens >= 2 or pawns >= 3
        return pawns >= 3


def build_profile_backend(threads: int) -> Tuple[PVSearchBackend, SearchTuning, SearchLimits]:
    meta = MetaRegistry.resolve("tournament")
    tuning, limits = build_search_tuning(meta)
    backend = PVSearchBackend(max_threads=max(1, threads))
    backend.configure(tuning, meta)
    return backend, tuning, limits


def run_profile(fen: str, threads: int) -> None:
    import cProfile
    import pstats

    backend, _, limits = build_profile_backend(threads)
    engine = ChessEngine(meta_preset="tournament", toggles=(StrategyToggle.MATE_IN_ONE,))
    board = chess.Board(fen) if fen else chess.Board()
    context = engine._build_context(board, None)
    reporter = SearchReporter(logger=lambda *_: None)
    budget = limits.resolve_budget(context)

    board_copy = board.copy(stack=True)

    profiler = cProfile.Profile()
    profiler.enable()
    outcome = backend.search(board_copy, context, limits, reporter, budget)
    profiler.disable()

    print(
        f"bestmove={outcome.move.uci() if outcome.move else '(none)'} "
        f"depth={outcome.completed_depth} score={outcome.score:.1f}"
    )
    print(
        f"nodes={outcome.nodes} time={outcome.time_spent:.3f}s "
        f"nps={int(outcome.nodes / max(outcome.time_spent, 1e-6))}"
    )

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(25)


def profile_cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile the PV search backend")
    parser.add_argument("--fen", default="", help="FEN to analyse (default: start position)")
    parser.add_argument("--threads", type=int, default=1, help="Number of search worker threads")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_profile(args.fen, args.threads)


class MateInOneStrategy(MoveStrategy):
    def __init__(self, *, logger: Optional[Callable[[str], None]] = None) -> None:
        super().__init__(priority=100)
        self._logger = logger or (lambda *_: None)

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                move_uci = move.uci()
                self._logger(f"mate-in-one strategy chose {move_uci}")
                return StrategyResult(
                    move=move_uci,
                    strategy_name=self.name,
                    score=500000.0,
                    confidence=1.0,
                    metadata={"pattern": "mate_in_one"},
                    definitive=True,
                )
        return None


class HeuristicSearchStrategy(MoveStrategy):
    def __init__(
        self,
        *,
        logger: Optional[Callable[[str], None]] = None,
        log_tag: str = "HS",
        max_threads: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(priority=70, confidence=0.85, **kwargs)
        self.log_tag = log_tag
        self._logger = logger or (lambda *_: None)
        self._meta: Optional[MetaParams] = None
        self._tuning: Optional[SearchTuning] = None
        self._limits: Optional[SearchLimits] = None
        if max_threads is None:
            # Default to single-threaded to minimize Python GIL overhead
            # Multi-threading can be enabled by passing max_threads explicitly
            max_threads = 1
        self._backend = PVSearchBackend(max_threads=max_threads)

    def apply_config(self, config: MetaParams) -> None:
        if not isinstance(config, MetaParams):
            raise TypeError("HeuristicSearchStrategy.apply_config expects MetaParams")
        meta = config.clamp()
        tuning, limits = build_search_tuning(meta)
        self._meta = meta
        self._tuning = tuning
        self._limits = limits
        self._backend.configure(tuning, meta)

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        if not self._meta or not self._tuning or not self._limits:
            raise RuntimeError("HeuristicSearchStrategy not configured")
        if context.legal_moves_count == 0:
            return None

        budget = self._limits.resolve_budget(context)
        budget_desc = "infinite" if budget is None else f"{budget:.2f}s"
        self._logger(
            f"{self.log_tag}: depth={self._tuning.search_depth} budget={budget_desc} moves={context.legal_moves_count}"
        )

        board_copy = board.copy(stack=True)
        reporter = SearchReporter(logger=self._logger)
        outcome = self._backend.search(board_copy, context, self._limits, reporter, budget)
        if not outcome.move:
            self._logger(f"{self.log_tag}: backend returned no move")
            return None

        self._logger(
            f"{self.log_tag}: completed depth={outcome.completed_depth} score={outcome.score:.1f} nodes={outcome.nodes} time={outcome.time_spent:.2f}s"
        )

        metadata = dict(outcome.metadata)
        metadata.setdefault("pv", [mv.uci() for mv in outcome.principal_variation])
        metadata.setdefault("label", self.log_tag)

        return StrategyResult(
            move=outcome.move.uci(),
            strategy_name=self.name,
            score=outcome.score,
            confidence=self.confidence,
            metadata=metadata,
        )


class ChessEngine:
    def __init__(self, *, meta_preset: str = "balanced", toggles: Optional[Iterable[StrategyToggle]] = None) -> None:
        self.engine_name = "JaskFish"
        self.engine_author = "Jaskaran Singh"
        self.board = chess.Board()
        self.debug = True
        self.move_calculating = False
        self.running = True
        self.state_lock = threading.Lock()

        active_toggles = {toggle for toggle in (toggles or (StrategyToggle.MATE_IN_ONE, StrategyToggle.HEURISTIC))}
        self.meta_name = meta_preset
        self.meta_params = MetaRegistry.resolve(meta_preset)

        self.selector = StrategySelector(logger=self._log_debug)
        if StrategyToggle.MATE_IN_ONE in active_toggles:
            self.selector.register(MateInOneStrategy(logger=self._log_debug))
        if StrategyToggle.HEURISTIC in active_toggles:
            heuristic = HeuristicSearchStrategy(logger=self._log_debug)
            heuristic.apply_config(self.meta_params)
            self.selector.register(heuristic)

        self.dispatch_table = {
            "quit": self.handle_quit,
            "debug": self.handle_debug,
            "isready": self.handle_isready,
            "position": self.handle_position,
            "boardpos": self.handle_boardpos,
            "go": self.handle_go,
            "ucinewgame": self.handle_ucinewgame,
            "uci": self.handle_uci,
            "setoption": self.handle_setoption,
        }

    def _log_debug(self, message: str) -> None:
        if not self.debug:
            return
        for line in message.splitlines():
            print(f"info string {line}")

    def start(self) -> None:
        _ensure_line_buffered_stdout()
        self.handle_uci()
        self.command_loop()

    def command_loop(self) -> None:
        while self.running:
            command = sys.stdin.readline()
            if not command:
                break
            command = command.strip()
            if not command:
                continue
            parts = command.split(" ", 1)
            name = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            handler = self.dispatch_table.get(name, self.handle_unknown)
            try:
                handler(args)
            except Exception as exc:
                print(f"info string Error processing command: {exc}")
            finally:
                sys.stdout.flush()

    def handle_unknown(self, args: str) -> None:
        print(f"unknown command received: '{args}'")

    def handle_quit(self, _: str) -> None:
        print("info string Engine shutting down")
        self.running = False

    def handle_debug(self, args: str) -> None:
        setting = args.strip().lower()
        if setting == "on":
            self.debug = True
        elif setting == "off":
            self.debug = False
        else:
            print("info string Invalid debug setting. Use 'on' or 'off'.")
            return
        print(f"info string Debug:{self.debug}")

    def handle_isready(self, _: str) -> None:
        with self.state_lock:
            print("readyok" if not self.move_calculating else "info string Engine is busy processing a move")

    def handle_position(self, args: str) -> None:
        with self.state_lock:
            if args.startswith("startpos"):
                self.board.reset()
                if self.debug:
                    print(f"info string Set to start position: {self.board.fen()}")
                if "moves" in args:
                    moves_part = args.split("moves", 1)[1].strip()
                    for move_text in moves_part.split():
                        try:
                            move = self.board.parse_uci(move_text)
                        except ValueError:
                            print(f"info string Invalid move in position command: {move_text}")
                            break
                        self.board.push(move)
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

    def handle_boardpos(self, _: str) -> None:
        with self.state_lock:
            print(f"info string Position: {self.board.fen()}")

    def handle_go(self, args: str) -> None:
        time_controls = self._parse_go_args(args)
        with self.state_lock:
            if self.move_calculating:
                print("info string Please wait for computer move")
                return
            self.move_calculating = True

        worker = threading.Thread(target=self._compute_move, args=(time_controls,))
        worker.start()

    def _compute_move(self, time_controls: Optional[Dict[str, int]]) -> None:
        try:
            with self.state_lock:
                board_copy = self.board.copy(stack=True)
            context = self._build_context(board_copy, time_controls)

            if self.debug:
                self._log_debug(
                    "context prepared: "
                    f"fullmove={context.fullmove_number}, "
                    f"halfmove={context.halfmove_clock}, "
                    f"pieces={context.piece_count}, "
                    f"material={context.material_imbalance}"
                )

            select_start = time.perf_counter()
            result = self.selector.select(board_copy, context)
            select_time = time.perf_counter() - select_start

            metadata = result.metadata if result and result.metadata else {}
            nodes = metadata.get("nodes")
            search_time = metadata.get("time")
            depth = metadata.get("depth", 0)
            timers = metadata.get("timing")

            if result and result.move:
                label = metadata.get("label", result.strategy_name)
                print(f"info string strategy {label} selected move {result.move}")
            else:
                self._log_debug("no strategy produced a move")

            if nodes is not None and search_time is not None:
                SearchReporter(logger=self._log_debug).perf_summary("move", depth, nodes, search_time, select_time, timers)
            else:
                print(f"info string perf select={select_time:.3f}s")

            with self.state_lock:
                if result and result.move:
                    print(f"bestmove {result.move}")
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

    def handle_ucinewgame(self, _: str) -> None:
        with self.state_lock:
            self.board.reset()
            if self.debug:
                print("info string New game started, board reset to initial position")
            print("info string New game initialized")

    def handle_uci(self, _: str = "") -> None:
        print(f"id name {self.engine_name}")
        print(f"id author {self.engine_author}")
        print("uciok")

    def handle_setoption(self, _: str) -> None:
        print("info string Meta configuration is code-selected; setoption has no effect")

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
                    parsed[key] = int(next(iterator))
                except (StopIteration, ValueError):
                    continue
            elif key in {"infinite", "ponder"}:
                parsed[key] = True
        return parsed

    def _build_context(self, board: chess.Board, time_controls: Optional[Dict[str, int]]) -> StrategyContext:
        repetition_info = {
            "is_threefold_repetition": board.is_repetition(),
            "can_claim_threefold": board.can_claim_threefold_repetition(),
            "is_fivefold_repetition": board.is_fivefold_repetition(),
        }
        return StrategyContext(
            fullmove_number=board.fullmove_number,
            halfmove_clock=board.halfmove_clock,
            piece_count=len(board.piece_map()),
            material_imbalance=self._material_imbalance(board),
            turn=board.turn,
            fen=board.fen(),
            repetition_info=repetition_info,
            legal_moves_count=board.legal_moves.count(),
            time_controls=dict(time_controls) if time_controls else None,
            in_check=board.is_check(),
            opponent_mate_in_one_threat=self._mate_threat(board),
        )

    def _material_imbalance(self, board: chess.Board) -> int:
        total = 0
        for piece in board.piece_map().values():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            total += value if piece.color == chess.WHITE else -value
        return total

    def _mate_threat(self, board: chess.Board) -> bool:
        copy = board.copy()
        try:
            copy.push(chess.Move.null())
        except ValueError:
            return False
        for move in copy.legal_moves:
            copy.push(move)
            try:
                if copy.is_checkmate():
                    return True
            finally:
                copy.pop()
        return False


if __name__ == "__main__":
    if "--profile" in sys.argv[1:]:
        args = [arg for arg in sys.argv[1:] if arg != "--profile"]
        profile_cli(args)
    else:
        ChessEngine().start()
