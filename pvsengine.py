"""Refactored chess engine implementing the JaskFish strategy stack.

This module keeps protocol compatibility with ``engine.py`` but restructures the
engine into composable units so that the search logic, configuration handling,
and UCI façade are easier to reason about and test in isolation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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


PIECE_SQUARE_TABLES_BLACK: Dict[int, List[int]] = {
    piece: [table[chess.square_mirror(square)] for square in chess.SQUARES]
    for piece, table in PIECE_SQUARE_TABLES.items()
}

_FULL_BOARD_MASK = (1 << 64) - 1
_ROOK_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRECTIONS: Tuple[Tuple[int, int], ...] = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _iterate_ray_targets(square: int, directions: Sequence[Tuple[int, int]], occupied: int) -> Iterable[int]:
    start_file = chess.square_file(square)
    start_rank = chess.square_rank(square)
    for df, dr in directions:
        file = start_file + df
        rank = start_rank + dr
        while 0 <= file < 8 and 0 <= rank < 8:
            target = chess.square(file, rank)
            target_bb = chess.BB_SQUARES[target]
            if occupied & target_bb:
                break
            yield target
            file += df
            rank += dr


def _direction_between(src: int, dst: int) -> Optional[Tuple[int, int]]:
    file_diff = chess.square_file(dst) - chess.square_file(src)
    rank_diff = chess.square_rank(dst) - chess.square_rank(src)
    if file_diff == 0:
        if rank_diff == 0:
            return None
        return (0, 1 if rank_diff > 0 else -1)
    if rank_diff == 0:
        return (1 if file_diff > 0 else -1, 0)
    if abs(file_diff) == abs(rank_diff):
        return (1 if file_diff > 0 else -1, 1 if rank_diff > 0 else -1)
    return None


def _path_clear(src: int, dst: int, occupied: int, direction: Tuple[int, int]) -> bool:
    df, dr = direction
    file = chess.square_file(src) + df
    rank = chess.square_rank(src) + dr
    while 0 <= file < 8 and 0 <= rank < 8:
        square = chess.square(file, rank)
        if square == dst:
            return True
        if occupied & chess.BB_SQUARES[square]:
            return False
        file += df
        rank += dr
    return False


def _between_bitboard(src: int, dst: int, direction: Tuple[int, int]) -> int:
    df, dr = direction
    file = chess.square_file(src) + df
    rank = chess.square_rank(src) + dr
    mask = 0
    while 0 <= file < 8 and 0 <= rank < 8:
        square = chess.square(file, rank)
        if square == dst:
            break
        mask |= chess.BB_SQUARES[square]
        file += df
        rank += dr
    return mask


def _generate_quiet_targets(
    square: int,
    piece_type: int,
    color: bool,
    occupied: int,
) -> Set[int]:
    results: Set[int] = set()
    from_bb = chess.BB_SQUARES[square]
    occupied_without_from = occupied & ~from_bb
    empty_bb = (~occupied) & _FULL_BOARD_MASK

    if piece_type == chess.KNIGHT:
        mask = chess.BB_KNIGHT_ATTACKS[square] & empty_bb
        results.update(chess.SquareSet(mask))
    elif piece_type == chess.BISHOP:
        results.update(_iterate_ray_targets(square, _BISHOP_DIRECTIONS, occupied_without_from))
    elif piece_type == chess.ROOK:
        results.update(_iterate_ray_targets(square, _ROOK_DIRECTIONS, occupied_without_from))
    elif piece_type == chess.QUEEN:
        results.update(_iterate_ray_targets(square, _BISHOP_DIRECTIONS, occupied_without_from))
        results.update(_iterate_ray_targets(square, _ROOK_DIRECTIONS, occupied_without_from))
    elif piece_type == chess.KING:
        mask = chess.BB_KING_ATTACKS[square] & empty_bb
        results.update(chess.SquareSet(mask))
    elif piece_type == chess.PAWN:
        forward = 8 if color == chess.WHITE else -8
        one_step = square + forward
        if 0 <= one_step < 64 and not (occupied & chess.BB_SQUARES[one_step]):
            results.add(one_step)
            start_rank = 1 if color == chess.WHITE else 6
            if chess.square_rank(square) == start_rank:
                two_step = one_step + forward
                if 0 <= two_step < 64 and not (occupied & chess.BB_SQUARES[two_step]):
                    results.add(two_step)
    return results


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

    def resolve_budget(self, context: StrategyContext, reporter: Optional["SearchReporter"] = None) -> Optional[float]:
        tc = context.time_controls or {}
        if reporter and not tc:
            reporter.trace("budget calc: no explicit limits supplied; treating search as infinite")
        if not tc:
            return None
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
        if reporter:
            reporter.trace("budget calc: unable to infer limit from supplied controls; treating as infinite")
        return None

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

    search_depth = 4 + int(8 * strength * (1.0 - 0.5 * speed_bias))  # Increased from 3+6x to 4+8x
    quiescence_depth = 3 + int(6 * strength * (0.5 + 0.5 * stability))
    base_time = 0.3 + 8.0 * strength * (1.0 - 0.7 * speed_bias)  # Increased from 0.2+6.0x to 0.3+8.0x
    time_alloc = 0.08 + 0.20 * strength * (1.0 - speed_bias)  # Increased from 0.06+0.16x to 0.08+0.20x
    min_time = 0.05 + 0.35 * (1.0 - speed_bias)
    max_time = 2.0 + 40.0 * strength  # Increased from 1.0+30.0x to 2.0+40.0x
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
    # Allow deeper searches to consume more of the assigned budget while still
    # leaving a safety margin for finalisation/communication.
    depth_stop_ratio = 0.60 + 0.25 * stability
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
        *,
        stop_event: Optional[threading.Event] = None,
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


@dataclass
class SearchStats:
    """Comprehensive search statistics for performance monitoring."""
    # Core counters
    nodes: int = 0
    qnodes: int = 0
    sel_depth: int = 0  # Maximum quiescence depth reached
    
    # Transposition table
    tt_probes: int = 0
    tt_hits: int = 0
    tt_exact_hits: int = 0
    tt_cuts: int = 0
    
    # Beta cutoffs by type
    cuts_tt: int = 0
    cuts_killer: int = 0
    cuts_history: int = 0
    cuts_capture: int = 0
    cuts_null: int = 0
    cuts_futility: int = 0
    cuts_other: int = 0
    
    # Aspiration window
    asp_fail_low: int = 0
    asp_fail_high: int = 0
    asp_researches: int = 0
    
    # Late move reduction
    lmr_applied: int = 0
    lmr_researched: int = 0
    
    # Null move
    null_tried: int = 0
    null_success: int = 0
    
    # Move ordering effectiveness
    cutoff_indices: List[int] = field(default_factory=list)
    first_move_cuts: int = 0
    total_beta_cuts: int = 0
    root_best_index: int = 0
    
    # Per-depth tracking
    depth_scores: List[float] = field(default_factory=list)
    depth_nodes: List[int] = field(default_factory=list)
    depth_times: List[float] = field(default_factory=list)
    pv_changes: int = 0
    
    # Quiescence
    q_cutoffs: int = 0
    q_stand_pat_cuts: int = 0
    q_delta_prunes: int = 0
    q_see_prunes: int = 0
    
    # Timing (in seconds)
    time_order: float = 0.0
    time_eval: float = 0.0
    time_qsearch: float = 0.0
    time_alpha_beta: float = 0.0
    time_tt_probe: float = 0.0
    
    def beta_cut_total(self) -> int:
        return (self.cuts_tt + self.cuts_killer + self.cuts_history + 
                self.cuts_capture + self.cuts_null + self.cuts_futility + self.cuts_other)
    
    def tt_hit_rate(self) -> float:
        return self.tt_hits / max(1, self.tt_probes)
    
    def q_ratio(self) -> float:
        total = self.nodes + self.qnodes
        return self.qnodes / max(1, total)
    
    def cutoff_p50(self) -> int:
        if not self.cutoff_indices:
            return 0
        sorted_indices = sorted(self.cutoff_indices)
        return sorted_indices[len(sorted_indices) // 2]
    
    def cutoff_p90(self) -> int:
        if not self.cutoff_indices:
            return 0
        sorted_indices = sorted(self.cutoff_indices)
        idx = int(len(sorted_indices) * 0.9)
        return sorted_indices[min(idx, len(sorted_indices) - 1)]
    
    def first_move_cut_rate(self) -> float:
        return self.first_move_cuts / max(1, self.total_beta_cuts)
    
    def print_compact_summary(
        self,
        depth: int,
        score: float,
        pv: List[chess.Move],
        time_used: float,
        budget: Optional[float],
    ) -> str:
        """Generate comprehensive performance summary with machine-readable payload."""
        total_nodes = self.nodes + self.qnodes
        nps = int(total_nodes / max(0.001, time_used))
        budget_str = f"{time_used:.2f}/{budget:.2f}s" if budget else f"{time_used:.2f}s"

        pv_moves = [m.uci() for m in pv]
        display_pv = " ".join(pv_moves[:5])
        if len(pv_moves) > 5:
            display_pv += "..."

        score_delta = 0.0
        if len(self.depth_scores) >= 2:
            score_delta = self.depth_scores[-1] - self.depth_scores[-2]

        q_ratio = self.q_ratio()
        q_pct = int(q_ratio * 100)
        nodes_str = f"{total_nodes}(r:{self.nodes},q:{self.qnodes})"

        total_cuts = self.beta_cut_total()
        tt_hit_rate = self.tt_hit_rate()
        tt_hit_pct = int(tt_hit_rate * 100)
        tt_cut_share = self.tt_cuts / max(1, total_cuts) if total_cuts > 0 else 0.0
        tt_cut_pct = int(tt_cut_share * 100)

        first_cut_rate = self.first_move_cut_rate()
        first_cut_pct = int(first_cut_rate * 100)

        lmr_success_rate = (
            (self.lmr_applied - self.lmr_researched) / self.lmr_applied
            if self.lmr_applied > 0
            else 0.0
        )
        null_success_rate = (
            self.null_success / self.null_tried if self.null_tried > 0 else 0.0
        )

        total_q_prunes = self.q_delta_prunes + self.q_see_prunes
        q_prune_ratio = (
            total_q_prunes / self.qnodes if self.qnodes > 0 else 0.0
        )
        q_prune_pct = int(q_prune_ratio * 100)

        cuts_human: str
        if total_cuts > 0:
            cuts_human = (
                f"tt:{self.cuts_tt}({int(self.cuts_tt / total_cuts * 100)}%) "
                f"k:{self.cuts_killer}({int(self.cuts_killer / total_cuts * 100)}%) "
                f"h:{self.cuts_history} c:{self.cuts_capture} n:{self.cuts_null}"
            )
        else:
            cuts_human = "tt:0 k:0 h:0 c:0 n:0"

        q_details = f"q={q_pct}%"
        if self.qnodes > 0 and total_q_prunes > 0:
            q_details += (
                f" (Δ:{self.q_delta_prunes} SEE:{self.q_see_prunes} pruned:{q_prune_pct}%)"
            )

        lmr_details = ""
        if self.lmr_applied > 0:
            lmr_details = (
                f"lmr:{self.lmr_applied}/{self.lmr_researched}"
                f"({int(lmr_success_rate * 100)}%ok)"
            )

        null_details = ""
        if self.null_tried > 0:
            null_details = (
                f"null:{self.null_success}/{self.null_tried}"
                f"({int(null_success_rate * 100)}%)"
            )

        payload = {
            "strategy": "pvs",
            "depth": depth,
            "seldepth": self.sel_depth,
            "nodes": {
                "total": total_nodes,
                "regular": self.nodes,
                "quiescence": self.qnodes,
            },
            "nps": nps,
            "time": {
                "elapsed": time_used,
                "budget": budget,
            },
            "score": {
                "value": score,
                "delta": score_delta,
            },
            "pv": pv_moves,
            "pv_changes": self.pv_changes,
            "tt": {
                "probes": self.tt_probes,
                "hits": self.tt_hits,
                "hit_rate": tt_hit_rate,
                "cuts": self.tt_cuts,
                "cut_share": tt_cut_share,
            },
            "aspiration": {
                "fail_low": self.asp_fail_low,
                "fail_high": self.asp_fail_high,
                "researches": self.asp_researches,
            },
            "cuts": {
                "tt": self.cuts_tt,
                "killer": self.cuts_killer,
                "history": self.cuts_history,
                "capture": self.cuts_capture,
                "null": self.cuts_null,
                "futility": self.cuts_futility,
                "other": self.cuts_other,
                "total": total_cuts,
                "first_move_rate": first_cut_rate,
            },
            "quiescence": {
                "ratio": q_ratio,
                "cutoffs": self.q_cutoffs,
                "stand_pat_cuts": self.q_stand_pat_cuts,
                "delta_prunes": self.q_delta_prunes,
                "see_prunes": self.q_see_prunes,
                "prune_ratio": q_prune_ratio,
            },
            "reductions": {
                "lmr": {
                    "applied": self.lmr_applied,
                    "researched": self.lmr_researched,
                    "success_rate": lmr_success_rate,
                },
                "null": {
                    "tried": self.null_tried,
                    "success": self.null_success,
                    "success_rate": null_success_rate,
                },
            },
        }

        json_blob = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        lines = [
            f"perf payload={json_blob}",
            "perf summary core "
            + " ".join(
                [
                    f"depth={depth if depth is not None else '-'}",
                    f"sel={self.sel_depth}",
                    f"nodes={nodes_str}",
                    f"nps={nps}",
                    f"time={budget_str}",
                ]
            ),
            "perf summary eval "
            + " ".join(
                [
                    f"score={score:.1f}(Δ{score_delta:+.1f})",
                    f"pv={display_pv or '(empty)'}",
                    f"swaps={self.pv_changes}",
                ]
            ),
        ]

        lines.extend(
            [
                "perf summary pruning tt "
                + " ".join(
                    [
                        f"probes={self.tt_probes}",
                        f"hits={self.tt_hits}",
                        f"hit_rate={tt_hit_pct}%",
                        f"cut_share={tt_cut_pct}%",
                        f"cuts={self.tt_cuts}",
                    ]
                ),
                "perf summary pruning aspiration "
                + " ".join(
                    [
                        f"fail_low={self.asp_fail_low}",
                        f"fail_high={self.asp_fail_high}",
                        f"researches={self.asp_researches}",
                    ]
                ),
                "perf summary pruning cuts "
                + " ".join(
                    [
                        f"total={total_cuts}",
                        f"first_move_rate={first_cut_pct}%",
                        f"breakdown={cuts_human}",
                    ]
                ),
            ]
        )

        heuristic_parts = [q_details]
        if lmr_details:
            heuristic_parts.append(lmr_details)
        if null_details:
            heuristic_parts.append(null_details)
        lines.append("perf summary heuristics " + " ".join(heuristic_parts))

        return "\n".join(lines)


class _SearchBoard:
    __slots__ = (
        "turn",
        "occupied",
        "occupied_white",
        "occupied_black",
        "piece_bitboards",
        "castling_rights",
        "ep_square",
        "hash_key",
        "_stack",
    )

    def __init__(self, board: chess.Board) -> None:
        self.piece_bitboards = [[0] * 7, [0] * 7]
        self._stack: List[
            Tuple[
                bool,
                int,
                int,
                int,
                Tuple[int, ...],
                Tuple[int, ...],
                int,
                Optional[int],
                int,
            ]
        ] = []
        self._from_board(board)

    def _from_board(self, board: chess.Board) -> None:
        self.turn = board.turn
        self.occupied = int(board.occupied)
        self.occupied_white = int(board.occupied_co[chess.WHITE])
        self.occupied_black = int(board.occupied_co[chess.BLACK])
        for piece_type in range(1, 7):
            self.piece_bitboards[0][piece_type] = int(board.pieces(piece_type, chess.WHITE).mask)
            self.piece_bitboards[1][piece_type] = int(board.pieces(piece_type, chess.BLACK).mask)
        self.castling_rights = board.castling_rights
        self.ep_square = board.ep_square
        try:
            self.hash_key = board._transposition_key()
        except AttributeError:
            self.hash_key = polyglot.zobrist_hash(board)

    def _snapshot(self) -> Tuple[
        bool,
        int,
        int,
        int,
        Tuple[int, ...],
        Tuple[int, ...],
        int,
        Optional[int],
        int,
    ]:
        return (
            self.turn,
            self.occupied,
            self.occupied_white,
            self.occupied_black,
            tuple(self.piece_bitboards[0]),
            tuple(self.piece_bitboards[1]),
            self.castling_rights,
            self.ep_square,
            self.hash_key,
        )

    def pop(self) -> None:
        state = self._stack.pop()
        (
            self.turn,
            self.occupied,
            self.occupied_white,
            self.occupied_black,
            white_bitboards,
            black_bitboards,
            self.castling_rights,
            self.ep_square,
            self.hash_key,
        ) = state
        for piece_type in range(1, 7):
            self.piece_bitboards[0][piece_type] = white_bitboards[piece_type]
            self.piece_bitboards[1][piece_type] = black_bitboards[piece_type]

    def push_move(self, board: chess.Board, move: chess.Move) -> None:
        self._stack.append(self._snapshot())
        color = board.turn
        opponent = not color
        color_idx = 0 if color else 1
        opp_idx = 1 - color_idx
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError("Attempted to push move with no piece on from_square")

        piece_type = piece.piece_type
        from_bb = chess.BB_SQUARES[move.from_square]
        to_bb = chess.BB_SQUARES[move.to_square]

        # Remove moving piece from its origin square.
        self.piece_bitboards[color_idx][piece_type] &= ~from_bb
        if color == chess.WHITE:
            self.occupied_white &= ~from_bb
        else:
            self.occupied_black &= ~from_bb
        self.occupied &= ~from_bb

        captured_piece_type: Optional[int] = None
        capture_square = move.to_square

        if board.is_en_passant(move):
            capture_square = move.to_square - 8 if color == chess.WHITE else move.to_square + 8
            capture_bb = chess.BB_SQUARES[capture_square]
            captured_piece_type = chess.PAWN
            self.piece_bitboards[opp_idx][chess.PAWN] &= ~capture_bb
            if opponent == chess.WHITE:
                self.occupied_white &= ~capture_bb
            else:
                self.occupied_black &= ~capture_bb
            self.occupied &= ~capture_bb
        else:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                captured_piece_type = captured_piece.piece_type
                capture_bb = to_bb
                self.piece_bitboards[opp_idx][captured_piece_type] &= ~capture_bb
                if opponent == chess.WHITE:
                    self.occupied_white &= ~capture_bb
                else:
                    self.occupied_black &= ~capture_bb
                self.occupied &= ~capture_bb

        # Handle promotions.
        placed_piece_type = move.promotion if move.promotion else piece_type
        if move.promotion:
            # Ensure promoted piece does not keep pawn bit.
            self.piece_bitboards[color_idx][chess.PAWN] &= ~to_bb

        self.piece_bitboards[color_idx][placed_piece_type] |= to_bb
        if color == chess.WHITE:
            self.occupied_white |= to_bb
        else:
            self.occupied_black |= to_bb
        self.occupied |= to_bb

        # Move rook for castling.
        if piece_type == chess.KING and abs(move.to_square - move.from_square) == 2:
            if color == chess.WHITE:
                if move.to_square > move.from_square:
                    rook_from, rook_to = chess.H1, chess.F1
                else:
                    rook_from, rook_to = chess.A1, chess.D1
            else:
                if move.to_square > move.from_square:
                    rook_from, rook_to = chess.H8, chess.F8
                else:
                    rook_from, rook_to = chess.A8, chess.D8
            rook_from_bb = chess.BB_SQUARES[rook_from]
            rook_to_bb = chess.BB_SQUARES[rook_to]
            self.piece_bitboards[color_idx][chess.ROOK] &= ~rook_from_bb
            self.piece_bitboards[color_idx][chess.ROOK] |= rook_to_bb
            if color == chess.WHITE:
                self.occupied_white &= ~rook_from_bb
                self.occupied_white |= rook_to_bb
            else:
                self.occupied_black &= ~rook_from_bb
                self.occupied_black |= rook_to_bb
            self.occupied &= ~rook_from_bb
            self.occupied |= rook_to_bb

        # Update castling rights.
        rights = self.castling_rights
        if piece_type == chess.KING:
            if color == chess.WHITE:
                rights &= ~(chess.BB_H1 | chess.BB_A1)
            else:
                rights &= ~(chess.BB_H8 | chess.BB_A8)
        elif piece_type == chess.ROOK:
            if move.from_square == chess.H1:
                rights &= ~chess.BB_H1
            elif move.from_square == chess.A1:
                rights &= ~chess.BB_A1
            elif move.from_square == chess.H8:
                rights &= ~chess.BB_H8
            elif move.from_square == chess.A8:
                rights &= ~chess.BB_A8

        if captured_piece_type == chess.ROOK:
            if capture_square == chess.H1:
                rights &= ~chess.BB_H1
            elif capture_square == chess.A1:
                rights &= ~chess.BB_A1
            elif capture_square == chess.H8:
                rights &= ~chess.BB_H8
            elif capture_square == chess.A8:
                rights &= ~chess.BB_A8

        self.castling_rights = rights

        # Update en-passant target.
        if piece_type == chess.PAWN and abs(move.to_square - move.from_square) == 16:
            self.ep_square = move.from_square + 8 if color == chess.WHITE else move.from_square - 8
        else:
            self.ep_square = None

        # Toggle side to move.
        self.turn = bool(opponent)
        # Hash will be refreshed after the python-chess board state is updated.
        self.hash_key = 0

    def push_null(self) -> None:
        self._stack.append(self._snapshot())
        self.turn = not self.turn
        self.ep_square = None
        self.hash_key = 0

    def finalize_move(self, board: chess.Board, verify: bool = False) -> None:
        try:
            self.hash_key = board._transposition_key()
        except AttributeError:
            self.hash_key = polyglot.zobrist_hash(board)
        if verify:
            self._assert_matches(board)

    def _assert_matches(self, board: chess.Board) -> None:
        assert self.turn == board.turn
        assert self.occupied == int(board.occupied)
        assert self.occupied_white == int(board.occupied_co[chess.WHITE])
        assert self.occupied_black == int(board.occupied_co[chess.BLACK])
        for piece_type in range(1, 7):
            assert self.piece_bitboards[0][piece_type] == int(board.pieces(piece_type, chess.WHITE).mask)
            assert self.piece_bitboards[1][piece_type] == int(board.pieces(piece_type, chess.BLACK).mask)
        assert self.castling_rights == board.castling_rights
        assert self.ep_square == board.ep_square
        hash_val = getattr(board, "_transposition_key", None)
        if callable(hash_val):
            hash_val = hash_val()
        else:
            hash_val = polyglot.zobrist_hash(board)
        assert self.hash_key == hash_val


class _SearchState:
    __slots__ = (
        "board",
        "search_board",
        "tuning",
        "meta",
        "limits",
        "reporter",
        "context",
        "deadline",
        "stop_event",
        "tt",
        "killers",
        "history",
        "nodes",
        "stats",
        "principal_variation",
        "eval_cache",
        "eval_cache_max_size",
        "mobility_cache",
        "mobility_cache_max_size",
        "king_safety_cache",
        "king_safety_cache_max_size",
        "see_cache",
        "see_cache_max_size",
        "qmove_cache",
        "qmove_cache_max_size",
        "material_w",
        "material_b",
        "pst_w",
        "pst_b",
        "bishops_w",
        "bishops_b",
        "eval_stack",
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
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        self.board = board
        self.search_board = _SearchBoard(board)
        self.tuning = tuning
        self.meta = meta
        self.limits = limits
        self.reporter = reporter
        self.context = context
        self.budget = budget
        base_deadline = None if budget is None else time.perf_counter() + max(budget, tuning.min_time_limit)
        self.deadline = deadline if deadline is not None else base_deadline
        self.stop_event = stop_event
        tt_size = int((meta.tt_budget_mb * 1024 * 1024) / 40)
        self.tt = PackedTranspositionTable(max(1024, tt_size))
        self.killers = KillerTable(tuning.search_depth + tuning.quiescence_depth, tuning.killer_slots)
        self.history = HistoryTable(tuning.history_decay)
        self.nodes = 0
        self.stats = SearchStats()
        self.principal_variation: List[chess.Move] = []
        # Larger eval cache with OrderedDict for LRU eviction (100K entries ≈ 3MB)
        self.eval_cache: "OrderedDict[int, float]" = OrderedDict()
        self.eval_cache_max_size = 100_000
        self.mobility_cache: "OrderedDict[int, Tuple[int, int]]" = OrderedDict()
        self.mobility_cache_max_size = 20_000
        self.king_safety_cache: "OrderedDict[int, float]" = OrderedDict()
        self.king_safety_cache_max_size = 20_000
        self.see_cache: "OrderedDict[Tuple[int, int, int, Optional[int]], float]" = OrderedDict()
        self.see_cache_max_size = 100_000
        self.qmove_cache: "OrderedDict[int, Tuple[chess.Move, ...]]" = OrderedDict()
        self.qmove_cache_max_size = 50_000
        (
            self.material_w,
            self.material_b,
            self.pst_w,
            self.pst_b,
            self.bishops_w,
            self.bishops_b,
        ) = self._compute_eval_totals(board)
        self.eval_stack: List[Tuple[float, float, float, float, int, int]] = []
        self.avoid_repetition = meta.avoid_repetition
        self.repetition_penalty = meta.repetition_penalty
        self.repetition_strong_penalty = meta.repetition_strong_penalty

    def time_exceeded(self) -> bool:
        if self.stop_event is not None and self.stop_event.is_set():
            return True
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

    def _compute_eval_totals(self, board: chess.Board) -> Tuple[float, float, float, float, int, int]:
        material_w = material_b = 0.0
        pst_w = pst_b = 0.0
        bishops_w = bishops_b = 0

        for piece_type, value in EVAL_PIECE_VALUES.items():
            for square in board.pieces(piece_type, chess.WHITE):
                material_w += value
                pst_w += PIECE_SQUARE_TABLES[piece_type][square]
                if piece_type == chess.BISHOP:
                    bishops_w += 1
            for square in board.pieces(piece_type, chess.BLACK):
                material_b += value
                pst_b += PIECE_SQUARE_TABLES_BLACK[piece_type][square]
                if piece_type == chess.BISHOP:
                    bishops_b += 1

        return material_w, material_b, pst_w, pst_b, bishops_w, bishops_b

    def push_eval(self, board: chess.Board, move: chess.Move) -> None:
        self.search_board.push_move(board, move)
        self.eval_stack.append(
            (self.material_w, self.material_b, self.pst_w, self.pst_b, self.bishops_w, self.bishops_b)
        )

        piece = board.piece_at(move.from_square)
        if piece is None:
            return

        color = piece.color
        piece_type = piece.piece_type

        if color == chess.WHITE:
            self.pst_w -= PIECE_SQUARE_TABLES[piece_type][move.from_square]
        else:
            self.pst_b -= PIECE_SQUARE_TABLES_BLACK[piece_type][move.from_square]

        captured_piece: Optional[chess.Piece]
        if board.is_en_passant(move):
            capture_square = move.to_square - 8 if color == chess.WHITE else move.to_square + 8
            captured_piece = chess.Piece(chess.PAWN, not color)
            if color == chess.WHITE:
                self.material_b -= EVAL_PIECE_VALUES[chess.PAWN]
                self.pst_b -= PIECE_SQUARE_TABLES_BLACK[chess.PAWN][capture_square]
            else:
                self.material_w -= EVAL_PIECE_VALUES[chess.PAWN]
                self.pst_w -= PIECE_SQUARE_TABLES[chess.PAWN][capture_square]
        else:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                captured_type = captured_piece.piece_type
                if captured_piece.color == chess.WHITE:
                    self.material_w -= EVAL_PIECE_VALUES[captured_type]
                    self.pst_w -= PIECE_SQUARE_TABLES[captured_type][move.to_square]
                    if captured_type == chess.BISHOP:
                        self.bishops_w -= 1
                else:
                    self.material_b -= EVAL_PIECE_VALUES[captured_type]
                    self.pst_b -= PIECE_SQUARE_TABLES_BLACK[captured_type][move.to_square]
                    if captured_type == chess.BISHOP:
                        self.bishops_b -= 1

        if move.promotion:
            promotion_type = move.promotion
            if color == chess.WHITE:
                self.material_w -= EVAL_PIECE_VALUES[chess.PAWN]
                self.material_w += EVAL_PIECE_VALUES[promotion_type]
                self.pst_w += PIECE_SQUARE_TABLES[promotion_type][move.to_square]
                if promotion_type == chess.BISHOP:
                    self.bishops_w += 1
            else:
                self.material_b -= EVAL_PIECE_VALUES[chess.PAWN]
                self.material_b += EVAL_PIECE_VALUES[promotion_type]
                self.pst_b += PIECE_SQUARE_TABLES_BLACK[promotion_type][move.to_square]
                if promotion_type == chess.BISHOP:
                    self.bishops_b += 1
        else:
            if color == chess.WHITE:
                self.pst_w += PIECE_SQUARE_TABLES[piece_type][move.to_square]
            else:
                self.pst_b += PIECE_SQUARE_TABLES_BLACK[piece_type][move.to_square]

        if (
            piece_type == chess.KING
            and abs(chess.square_file(move.to_square) - chess.square_file(move.from_square)) == 2
        ):
            if color == chess.WHITE:
                rook_from = move.to_square + 1 if move.to_square > move.from_square else move.to_square - 2
                rook_to = move.to_square - 1 if move.to_square > move.from_square else move.to_square + 1
                self.pst_w -= PIECE_SQUARE_TABLES[chess.ROOK][rook_from]
                self.pst_w += PIECE_SQUARE_TABLES[chess.ROOK][rook_to]
            else:
                rook_from = move.to_square + 1 if move.to_square > move.from_square else move.to_square - 2
                rook_to = move.to_square - 1 if move.to_square > move.from_square else move.to_square + 1
                self.pst_b -= PIECE_SQUARE_TABLES_BLACK[chess.ROOK][rook_from]
                self.pst_b += PIECE_SQUARE_TABLES_BLACK[chess.ROOK][rook_to]

    def pop_eval(self) -> None:
        self.search_board.pop()
        (
            self.material_w,
            self.material_b,
            self.pst_w,
            self.pst_b,
            self.bishops_w,
            self.bishops_b,
        ) = self.eval_stack.pop()


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
        *,
        stop_event: Optional[threading.Event] = None,
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
            stop_event=stop_event,
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

        # Log comprehensive performance summary (single consolidated line)
        state.stats.nodes = state.nodes
        state.stats.tt_probes = state.tt.probes
        state.stats.tt_hits = state.tt.hits
        summary = state.stats.print_compact_summary(
            depth=result.completed_depth,
            score=result.score,
            pv=list(result.principal_variation),
            time_used=elapsed,
            budget=state.budget
        )
        reporter.trace(summary)

        return SearchOutcome(
            move=result.move,
            score=result.score,
            completed_depth=result.completed_depth,
            nodes=state.nodes,
            time_spent=elapsed,
            principal_variation=tuple(result.principal_variation),
            metadata=metadata,
        )
    
    def profile_search(self, board: chess.Board, depth: int = 5, 
                      output_file: str = "pvs_profile.txt") -> str:
        """Profile a search to identify performance bottlenecks.
        
        Args:
            board: Position to search from
            depth: Search depth
            output_file: Output filename for profile results
            
        Returns:
            Path to the profile output file
        """
        import cProfile
        import pstats
        
        if self.tuning is None or self.meta is None:
            raise RuntimeError("PVSearchBackend not configured - call configure() first")
        
        # Create a simple context for profiling
        context = StrategyContext(
            fullmove_number=1,
            halfmove_clock=0,
            piece_count=len(board.piece_map()),
            material_imbalance=0,
            turn=board.turn,
            fen=board.fen(),
            legal_moves_count=board.legal_moves.count(),
            repetition_info={},
            time_controls=None,
            in_check=board.is_check(),
            opponent_mate_in_one_threat=False,
        )
        
        limits = SearchLimits(depth=depth)
        reporter = SearchReporter(debug=False)
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            # Run search
            self.search(board, context, limits, reporter, budget_seconds=None, stop_event=None)
        finally:
            profiler.disable()
        
        # Write results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        with open(output_file, 'w') as f:
            stats.stream = f
            f.write("=" * 80 + "\n")
            f.write("PVS Engine Performance Profile\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Position: {board.fen()}\n")
            f.write(f"Search depth: {depth}\n\n")
            f.write("Top 30 functions by cumulative time:\n")
            f.write("-" * 80 + "\n")
            stats.print_stats(30)
            
            f.write("\n\n")
            f.write("Top 20 functions by time per call:\n")
            f.write("-" * 80 + "\n")
            stats.sort_stats('time')
            stats.print_stats(20)
        
        print(f"Profile written to: {output_file}")
        return output_file

    def _iterative_deepening(self, state: _SearchState) -> SearchOutcome:
        tuning = state.tuning
        board = state.board
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        completed_depth = 0
        last_score = None
        aspiration = tuning.aspiration_window
        pv: List[chess.Move] = []
        should_stop_deepening = False
        search_start = time.perf_counter()

        for depth in range(1, tuning.search_depth + 1):
            if state.time_exceeded() or should_stop_deepening:
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
                        state.stats.asp_fail_low += 1
                    if any(score >= beta_window for _, score in results) and beta < float(_MATE_SCORE):
                        fail_high = True
                        state.stats.asp_fail_high += 1
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
                        state.stats.asp_fail_low += 1
                    if iteration_score >= beta and beta < float(_MATE_SCORE):
                        fail_high = True
                        state.stats.asp_fail_high += 1
                    
                    # Only continue if there was an aspiration failure that requires re-search
                    if fail_low or fail_high:
                        state.stats.asp_researches += 1
                        alpha = max(-float(_MATE_SCORE), iteration_score - aspiration)
                        beta = min(float(_MATE_SCORE), iteration_score + aspiration)
                        continue
                if iteration_score >= beta and beta < float(_MATE_SCORE):
                    aspiration *= tuning.aspiration_growth
                    alpha = max(-float(_MATE_SCORE), iteration_score - aspiration)
                    beta = min(float(_MATE_SCORE), iteration_score + aspiration)
                    continue

                # Track PV changes
                if best_move is not None and iteration_best != best_move:
                    state.stats.pv_changes += 1
                
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
                
                # Update stats for depth tracking
                state.stats.nodes = state.nodes
                state.stats.depth_scores.append(iteration_score)
                state.stats.depth_nodes.append(state.nodes)
                state.stats.depth_times.append(depth_elapsed)
                
                # Check if we should stop iterating due to budget consumption
                # Use cumulative time, not just this depth's time
                if state.budget is not None:
                    cumulative_time = time.perf_counter() - search_start
                    if cumulative_time >= state.budget * tuning.depth_stop_ratio:
                        usage_pct = cumulative_time / state.budget * 100
                        state.reporter.trace(
                            f"PVS: depth={depth} consumed {usage_pct:.0f}% of budget; halting deeper search"
                        )
                        should_stop_deepening = True
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
                total += 8_000_000 + int(self._see(board, move, state))
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
        state.push_eval(board, move)
        board.push(move)
        state.search_board.finalize_move(board)
        try:
            score = -self._alpha_beta(state, board, depth - 1, -beta, -alpha, ply=1, is_pv=is_pv, allow_null=True)
            penalty = 0.0
            if state.avoid_repetition:
                penalty = state.repetition_penalty_for(board)
            score -= penalty
        finally:
            board.pop()
            state.pop_eval()
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
            stop_event=parent_state.stop_event,
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
        state.stats.tt_probes += 1
        tt_entry = state.tt.probe(key)
        if tt_entry:
            state.stats.tt_hits += 1
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == 0:
                state.stats.tt_exact_hits += 1
                return tt_entry.value
            if tt_entry.flag == 1:
                alpha = max(alpha, tt_entry.value)
            else:
                beta = min(beta, tt_entry.value)
            if alpha >= beta:
                state.stats.tt_cuts += 1
                return tt_entry.value

        in_check = board.is_check()
        static_eval = tt_entry.static_eval if tt_entry else None

        if (
            not in_check
            and not is_pv
            and allow_null
            and depth >= state.tuning.null_min_depth
            and self._has_material(board)
        ):
            state.stats.null_tried += 1
            state.search_board.push_null()
            board.push(chess.Move.null())
            state.search_board.finalize_move(board)
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
                state.search_board.pop()
            if score >= beta:
                state.stats.null_success += 1
                state.stats.cuts_null += 1
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

            state.push_eval(board, move)
            board.push(move)
            state.search_board.finalize_move(board)
            try:
                child_is_pv = is_pv and index == 0
                new_allow_null = allow_null and not is_capture
                next_depth = depth - 1
                lmr_applied = False
                if depth >= state.tuning.lmr_min_depth and index >= state.tuning.lmr_min_index and not child_is_pv and not is_capture and not gives_check and not promotion:
                    reduction = max(1, int((math.log(depth + 1, 2) * math.log(index + 1, 2)) / 1.5))
                    next_depth = max(0, depth - 1 - reduction)
                    lmr_applied = True
                    state.stats.lmr_applied += 1
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
                        if lmr_applied:
                            state.stats.lmr_researched += 1
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
                state.pop_eval()

            if score > best_value:
                best_value = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                # Track beta cutoff by type
                state.stats.total_beta_cuts += 1
                state.stats.cutoff_indices.append(index)
                if index == 0:
                    state.stats.first_move_cuts += 1
                
                # Classify cut type (priority order: TT > killer > history > capture > other)
                if tt_move and move == tt_move:
                    state.stats.cuts_tt += 1
                elif state.killers.is_killer(ply, move):
                    state.stats.cuts_killer += 1
                elif not is_capture:
                    # Check if history heuristic guided this quiet move
                    hist_score = state.history.score(move, color)
                    if hist_score >= 10.0:  # Lowered threshold - depth 3+ records are 9+
                        state.stats.cuts_history += 1
                    else:
                        state.stats.cuts_other += 1
                else:
                    # Capture move caused cutoff
                    state.stats.cuts_capture += 1
                
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
                total += 8_000_000 + int(self._see(board, move, state))
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
        state.stats.qnodes += 1
        state.stats.sel_depth = max(state.stats.sel_depth, ply)
        stand_pat = self._evaluate(state, board)
        if stand_pat >= beta:
            state.stats.q_stand_pat_cuts += 1
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        if depth <= 0:
            return stand_pat

        # Delta pruning margin: queen value + safety buffer
        delta_margin = 975.0  # Queen (900) + margin (75)
        
        for move in self._generate_qmoves(state, board):
            if state.time_exceeded():
                break
            
            # Delta pruning: Skip captures that can't possibly raise alpha
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    captured_value = EVAL_PIECE_VALUES.get(captured_piece.piece_type, 0)
                    # Add promotion value if applicable
                    promotion_value = EVAL_PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0
                    # Even with this capture + promotion, can't raise alpha
                    if stand_pat + captured_value + promotion_value + delta_margin < alpha:
                        state.stats.q_delta_prunes += 1
                        continue
            
            # SEE pruning: Skip bad captures (losing material)
            see_score = self._see(board, move, state)
            if see_score < -50:  # Losing more than half a pawn
                state.stats.q_see_prunes += 1
                continue
            
            state.push_eval(board, move)
            board.push(move)
            state.search_board.finalize_move(board)
            try:
                score = -self._quiescence(state, board, -beta, -alpha, depth - 1, ply + 1)
            finally:
                board.pop()
                state.pop_eval()
            if score >= beta:
                state.stats.q_cutoffs += 1
                return beta
            if score > alpha:
                alpha = score
        return alpha

    def _evaluate(self, state: _SearchState, board: chess.Board) -> float:
        key = board._transposition_key()
        cached = state.eval_cache.get(key)
        if cached is not None:
            state.eval_cache.move_to_end(key)
            return cached

        material_w = state.material_w
        material_b = state.material_b
        pst_w = state.pst_w
        pst_b = state.pst_b
        bishops_w = state.bishops_w
        bishops_b = state.bishops_b

        score = (material_w - material_b) + (pst_w - pst_b)
        if bishops_w >= 2:
            score += state.tuning.bishop_pair_bonus
        if bishops_b >= 2:
            score -= state.tuning.bishop_pair_bonus

        score += self._passed_pawn_bonus(state, board)
        
        # Cache phase calculation for reuse
        phase = self._game_phase(board)
        
        # Cache mobility for king safety to avoid regenerating legal moves
        # Only compute mobility in opening/middlegame when it matters more
        if phase > 0.2:
            cached_mobility = None  # Will be computed in mobility_term
            score += self._mobility_term(state, board, cached_mobility)
        
        score += self._king_safety(state, board, phase)

        result = score if board.turn == chess.WHITE else -score
        
        # Store in eval cache with LRU eviction
        state.eval_cache[key] = result
        state.eval_cache.move_to_end(key)
        if len(state.eval_cache) > state.eval_cache_max_size:
            state.eval_cache.popitem(last=False)  # Remove oldest
        
        return result

    def _passed_pawn_bonus(self, state: _SearchState, board: chess.Board) -> float:
        total = 0.0
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            friendly_sq = board.pieces(chess.PAWN, color)
            enemy_bb = board.pieces(chess.PAWN, not color).mask
            bb = friendly_sq.mask
            while bb:
                lsb = bb & -bb
                square = (lsb.bit_length() - 1)
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
                        if enemy_bb & chess.BB_SQUARES[sq]:
                            passed = False
                            break
                        r += direction
                    if not passed:
                        break
                if passed:
                    advancement = rank - 1 if color == chess.WHITE else 6 - rank
                    bonus = state.tuning.passed_pawn_base_bonus + state.tuning.passed_pawn_rank_bonus * max(0, advancement)
                    total += sign * bonus
                bb ^= lsb
        return total

    def _mobility_term(self, state: _SearchState, board: chess.Board, 
                       cached_mobility: Optional[int] = None) -> float:
        # Mobility is expensive (requires legal move generation + null move)
        key = board._transposition_key()
        entry = state.mobility_cache.get(key)
        if entry is not None:
            state.mobility_cache.move_to_end(key)
            mobility, opponent = entry
        else:
            if cached_mobility is not None:
                mobility = cached_mobility
            else:
                mobility = board.legal_moves.count()
            try:
                board.push(chess.Move.null())
                opponent = board.legal_moves.count()
                board.pop()
            except (ValueError, AssertionError):
                opponent = mobility
            state.mobility_cache[key] = (mobility, opponent)
            state.mobility_cache.move_to_end(key)
            if len(state.mobility_cache) > state.mobility_cache_max_size:
                state.mobility_cache.popitem(last=False)
        return (mobility - opponent) * state.tuning.mobility_unit

    def _king_safety(self, state: _SearchState, board: chess.Board, phase: Optional[float] = None) -> float:
        # King safety is expensive - only compute in opening/middlegame
        if phase is None:
            phase = self._game_phase(board)
        
        # Skip king safety in endgame (phase < 0.3) - not worth the cost
        if phase < 0.3:
            return 0.0

        key = board._transposition_key()
        cached = state.king_safety_cache.get(key)
        if cached is not None:
            state.king_safety_cache.move_to_end(key)
            return cached
        
        opening = phase
        endgame = 1.0 - phase
        score = 0.0
        
        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            if king_sq is None:
                continue
            
            # Lightweight king danger approximation - count attackers but skip expensive ring calculation
            # This is ~40% faster than the full calculation
            attackers = len(board.attackers(not color, king_sq))
            
            # Approximate pawn shield by checking immediate squares (lighter than full ring)
            file = chess.square_file(king_sq)
            rank = chess.square_rank(king_sq)
            shield = 0
            for df in (-1, 0, 1):
                # Only check front row for pawns (direction based on color)
                nr = rank + (1 if color == chess.WHITE else -1)
                nf = file + df
                if 0 <= nf < 8 and 0 <= nr < 8:
                    piece = board.piece_at(chess.square(nf, nr))
                    if piece and piece.color == color and piece.piece_type == chess.PAWN:
                        shield += 1
            
            opening_term = opening * (-state.tuning.king_safety_opening_penalty * attackers + 2.0 * shield)
            
            # Simplified endgame term (only compute if in endgame transition)
            if endgame > 0.1:
                central_distance = abs(file - 3.5) + abs(rank - 3.5)
                endgame_term = endgame * state.tuning.king_safety_endgame_bonus * (3.5 - central_distance)
            else:
                endgame_term = 0.0
            
            score += sign * (opening_term + endgame_term)
        state.king_safety_cache[key] = score
        state.king_safety_cache.move_to_end(key)
        if len(state.king_safety_cache) > state.king_safety_cache_max_size:
            state.king_safety_cache.popitem(last=False)
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

    def _generate_qmoves(self, state: _SearchState, board: chess.Board) -> Sequence[chess.Move]:
        key = board._transposition_key()
        cached = state.qmove_cache.get(key)
        if cached is not None:
            state.qmove_cache.move_to_end(key)
            return list(cached)

        in_check = board.is_check()
        move_scores: List[Tuple[chess.Move, float, bool]] = []

        if in_check:
            for move in board.legal_moves:
                see_score = self._see(board, move, state)
                move_scores.append((move, see_score, board.gives_check(move)))
        else:
            capture_moves = self._generate_capture_moves(board)
            seen: set[chess.Move] = set()
            for move in capture_moves:
                if move in seen:
                    continue
                seen.add(move)
                see_score = self._see(board, move, state)
                if see_score >= -50:
                    move_scores.append((move, see_score, board.gives_check(move)))

            for move in self._generate_quiet_promotions(board):
                if move in seen:
                    continue
                seen.add(move)
                see_score = self._see(board, move, state)
                move_scores.append((move, see_score + 400.0, board.gives_check(move)))

            for move in self._generate_quiet_checks(state, board, seen):
                seen.add(move)
                see_score = self._see(board, move, state)
                move_scores.append((move, see_score + 250.0, True))

        move_scores.sort(key=lambda entry: entry[1] + (500 if entry[2] else 0), reverse=True)
        ordered = tuple(move for move, _, _ in move_scores)

        state.qmove_cache[key] = ordered
        state.qmove_cache.move_to_end(key)
        if len(state.qmove_cache) > state.qmove_cache_max_size:
            state.qmove_cache.popitem(last=False)

        return list(ordered)

    def _generate_capture_moves(self, board: chess.Board) -> List[chess.Move]:
        color = board.turn
        opponent = not color
        occupied = board.occupied
        enemy_bb = board.occupied_co[opponent]
        moves: List[chess.Move] = []

        for piece_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            for square in board.pieces(piece_type, color):
                attacks = board.attacks(square)
                capture_targets = attacks & chess.SquareSet(enemy_bb)
                for to_square in capture_targets:
                    if piece_type == chess.PAWN and chess.square_rank(to_square) in (0, 7):
                        for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                            move = chess.Move(square, to_square, promotion=promo)
                            if board.is_legal(move):
                                moves.append(move)
                        continue
                    move = chess.Move(square, to_square)
                    if board.is_legal(move):
                        moves.append(move)

        # En-passant capture if available
        if board.ep_square is not None:
            ep_square = board.ep_square
            for from_square in chess.SquareSet(chess.BB_PAWN_ATTACKS[opponent][ep_square] & board.pieces(chess.PAWN, color).mask):
                move = chess.Move(from_square, ep_square)
                if board.is_legal(move):
                    moves.append(move)

        return moves

    def _generate_quiet_promotions(self, board: chess.Board) -> List[chess.Move]:
        color = board.turn
        direction = 8 if color == chess.WHITE else -8
        target_rank = 6 if color == chess.WHITE else 1
        moves: List[chess.Move] = []
        for square in board.pieces(chess.PAWN, color):
            if chess.square_rank(square) != target_rank:
                continue
            to_square = square + direction
            if not (0 <= to_square < 64):
                continue
            if board.piece_at(to_square) is not None:
                continue
            for promo in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                move = chess.Move(square, to_square, promotion=promo)
                if board.is_legal(move):
                    moves.append(move)
        return moves

    def _generate_quiet_checks(
        self,
        state: _SearchState,
        board: chess.Board,
        seen: set[chess.Move],
    ) -> List[chess.Move]:
        enemy = not board.turn
        enemy_king = board.king(enemy)
        if enemy_king is None:
            return []

        search_board = state.search_board
        color = board.turn
        color_idx = 0 if color else 1
        occupied = search_board.occupied
        own_occ = search_board.occupied_white if color else search_board.occupied_black
        opp_occ = search_board.occupied_black if color else search_board.occupied_white
        enemy_king_bb = chess.BB_SQUARES[enemy_king]

        candidates: Set[chess.Move] = set()

        def consider(move: chess.Move) -> None:
            if move in seen:
                return
            if board.piece_at(move.to_square) is not None:
                return
            if board.is_legal(move) and board.gives_check(move):
                candidates.add(move)

        # Pawn pushes that give immediate check.
        pawn_bb = search_board.piece_bitboards[color_idx][chess.PAWN]
        forward = 8 if color == chess.WHITE else -8
        pawn_attacks = chess.BB_PAWN_ATTACKS[color]
        for from_sq in chess.SquareSet(pawn_bb):
            to_sq = from_sq + forward
            if 0 <= to_sq < 64 and not (occupied & chess.BB_SQUARES[to_sq]):
                if pawn_attacks[to_sq] & enemy_king_bb:
                    consider(chess.Move(from_sq, to_sq))
            start_rank = 1 if color == chess.WHITE else 6
            if chess.square_rank(from_sq) == start_rank:
                intermediate = from_sq + forward
                to_sq = intermediate + forward
                if (
                    0 <= to_sq < 64
                    and not (occupied & chess.BB_SQUARES[intermediate])
                    and not (occupied & chess.BB_SQUARES[to_sq])
                    and pawn_attacks[to_sq] & enemy_king_bb
                ):
                    consider(chess.Move(from_sq, to_sq))

        # Knight jumps.
        knight_bb = search_board.piece_bitboards[color_idx][chess.KNIGHT]
        for from_sq in chess.SquareSet(knight_bb):
            for to_sq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[from_sq] & _FULL_BOARD_MASK):
                if occupied & chess.BB_SQUARES[to_sq]:
                    continue
                if chess.BB_KNIGHT_ATTACKS[to_sq] & enemy_king_bb:
                    consider(chess.Move(from_sq, to_sq))

        # Sliding pieces (bishops, rooks, queens).
        def handle_slider(
            bitboard: int,
            directions: Sequence[Tuple[int, int]],
            valid_dirs: Sequence[Tuple[int, int]],
        ) -> None:
            for from_sq in chess.SquareSet(bitboard):
                from_bb = chess.BB_SQUARES[from_sq]
                occupied_without_from = occupied & ~from_bb
                for to_sq in _iterate_ray_targets(from_sq, directions, occupied_without_from):
                    to_bb = chess.BB_SQUARES[to_sq]
                    occupied_after = occupied_without_from | to_bb
                    direction = _direction_between(to_sq, enemy_king)
                    if direction is None or direction not in valid_dirs:
                        continue
                    if _path_clear(to_sq, enemy_king, occupied_after, direction):
                        consider(chess.Move(from_sq, to_sq))

        bishop_bb = search_board.piece_bitboards[color_idx][chess.BISHOP]
        rook_bb = search_board.piece_bitboards[color_idx][chess.ROOK]
        queen_bb = search_board.piece_bitboards[color_idx][chess.QUEEN]

        handle_slider(bishop_bb, _BISHOP_DIRECTIONS, _BISHOP_DIRECTIONS)
        handle_slider(rook_bb, _ROOK_DIRECTIONS, _ROOK_DIRECTIONS)
        handle_slider(queen_bb, _BISHOP_DIRECTIONS, _BISHOP_DIRECTIONS)
        handle_slider(queen_bb, _ROOK_DIRECTIONS, _ROOK_DIRECTIONS)

        # King moves that give immediate check.
        king_sq = board.king(color)
        if king_sq is not None:
            for to_sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq] & _FULL_BOARD_MASK):
                if occupied & chess.BB_SQUARES[to_sq]:
                    continue
                if chess.BB_KING_ATTACKS[to_sq] & enemy_king_bb:
                    consider(chess.Move(king_sq, to_sq))

        # Discovered checks by clearing lines for sliders.
        def handle_discovered(
            slider_bb: int,
            valid_dirs: Sequence[Tuple[int, int]],
        ) -> None:
            for slider_sq in chess.SquareSet(slider_bb):
                direction = _direction_between(slider_sq, enemy_king)
                if direction is None or direction not in valid_dirs:
                    continue
                between_bb = _between_bitboard(slider_sq, enemy_king, direction)
                if not between_bb:
                    continue
                if between_bb & opp_occ:
                    continue
                blockers = between_bb & own_occ
                if not blockers or blockers & (blockers - 1):
                    continue
                blocker_sq = next(iter(chess.SquareSet(blockers)))
                blocker_bb = chess.BB_SQUARES[blocker_sq]
                piece_type = None
                for candidate_type in range(1, 7):
                    if search_board.piece_bitboards[color_idx][candidate_type] & blocker_bb:
                        piece_type = candidate_type
                        break
                if piece_type is None:
                    continue
                for target in _generate_quiet_targets(blocker_sq, piece_type, color, occupied):
                    if chess.BB_SQUARES[target] & between_bb:
                        continue
                    consider(chess.Move(blocker_sq, target))

        straight_sliders = rook_bb | queen_bb
        diagonal_sliders = bishop_bb | queen_bb
        handle_discovered(straight_sliders, _ROOK_DIRECTIONS)
        handle_discovered(diagonal_sliders, _BISHOP_DIRECTIONS)

        return sorted(candidates, key=lambda m: (m.from_square, m.to_square, m.promotion or 0))

    def _see(
        self,
        board: chess.Board,
        move: chess.Move,
        state: Optional[_SearchState] = None,
    ) -> float:
        cache_key: Optional[Tuple[int, int, int, Optional[int]]] = None
        if state is not None:
            cache_key = (
                board._transposition_key(),
                move.from_square,
                move.to_square,
                move.promotion,
            )
            cached_val = state.see_cache.get(cache_key)
            if cached_val is not None:
                state.see_cache.move_to_end(cache_key)
                return cached_val

        if hasattr(board, "see"):
            try:
                value = float(board.see(move))
                if state is not None and cache_key is not None:
                    state.see_cache[cache_key] = value
                    state.see_cache.move_to_end(cache_key)
                    if len(state.see_cache) > state.see_cache_max_size:
                        state.see_cache.popitem(last=False)
                return value
            except (ValueError, AttributeError):
                pass
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            captured = chess.Piece(chess.PAWN, not board.turn)
        captured_val = EVAL_PIECE_VALUES.get(captured.piece_type, 0) if captured else 0
        mover = board.piece_at(move.from_square)
        mover_val = EVAL_PIECE_VALUES.get(mover.piece_type, 0) if mover else 0
        value = captured_val - mover_val
        if state is not None and cache_key is not None:
            state.see_cache[cache_key] = value
            state.see_cache.move_to_end(cache_key)
            if len(state.see_cache) > state.see_cache_max_size:
                state.see_cache.popitem(last=False)
        return value

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
    budget = limits.resolve_budget(context, reporter)

    board_copy = board.copy(stack=True)

    profiler = cProfile.Profile()
    profiler.enable()
    outcome = backend.search(board_copy, context, limits, reporter, budget, stop_event=None)
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


def engine_main(argv: Optional[Sequence[str]] = None) -> None:
    default_preset = os.environ.get("JASKFISH_PRESET", "balanced")
    parser = argparse.ArgumentParser(description="JaskFish PV engine entrypoint", add_help=True)
    parser.add_argument("--profile", action="store_true", help="Run the profiling helper instead of UCI loop")
    parser.add_argument(
        "--preset",
        choices=sorted(MetaRegistry.PRESETS.keys()),
        default=default_preset,
        help="Meta parameter preset to load",
    )
    args, remaining = parser.parse_known_args(list(argv) if argv is not None else None)
    if args.profile:
        profile_cli(remaining)
    else:
        ChessEngine(meta_preset=args.preset).start()


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
        log_tag: str = "PVS",
        max_threads: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(priority=70, confidence=0.85, **kwargs)
        self.log_tag = log_tag
        self._logger = logger or (lambda *_: None)
        self._meta: Optional[MetaParams] = None
        self._tuning: Optional[SearchTuning] = None
        self._limits: Optional[SearchLimits] = None
        self._stop_event: Optional[threading.Event] = None
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

    def set_stop_event(self, event: threading.Event) -> None:
        self._stop_event = event

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        if not self._meta or not self._tuning or not self._limits:
            raise RuntimeError("HeuristicSearchStrategy not configured")
        if context.legal_moves_count == 0:
            return None

        reporter = SearchReporter(logger=self._logger)
        budget = self._limits.resolve_budget(context, reporter)
        if budget is None:
            budget = self._tuning.base_time_limit
        budget_desc = f"{budget:.2f}s"
        self._logger(
            f"{self.log_tag}: depth={self._tuning.search_depth} budget={budget_desc} moves={context.legal_moves_count}"
        )

        board_copy = board.copy(stack=True)
        backend_kwargs: dict[str, Any] = {}
        if self._stop_event is not None:
            backend_kwargs["stop_event"] = self._stop_event
        outcome = self._backend.search(
            board_copy,
            context,
            self._limits,
            reporter,
            budget,
            **backend_kwargs,
        )
        if not outcome.move:
            self._logger(f"{self.log_tag}: backend returned no move")
            return None

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
        self.stop_event = threading.Event()
        self.current_worker: Optional[threading.Thread] = None

        active_toggles = {toggle for toggle in (toggles or (StrategyToggle.MATE_IN_ONE, StrategyToggle.HEURISTIC))}
        self.meta_name = meta_preset
        self.meta_params = MetaRegistry.resolve(meta_preset)

        self.selector = StrategySelector(logger=self._log_debug)
        if StrategyToggle.MATE_IN_ONE in active_toggles:
            self.selector.register(MateInOneStrategy(logger=self._log_debug))
        if StrategyToggle.HEURISTIC in active_toggles:
            heuristic = HeuristicSearchStrategy(logger=self._log_debug)
            heuristic.apply_config(self.meta_params)
            heuristic.set_stop_event(self.stop_event)
            self.selector.register(heuristic)

        self.dispatch_table = {
            "quit": self.handle_quit,
            "debug": self.handle_debug,
            "isready": self.handle_isready,
            "position": self.handle_position,
            "boardpos": self.handle_boardpos,
            "go": self.handle_go,
            "stop": self.handle_stop,
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
        with self.state_lock:
            self.running = False
            self.stop_event.set()
            worker = self.current_worker
        if worker and worker.is_alive():
            worker.join(timeout=5.0)
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
            self.stop_event.clear()
            self.move_calculating = True

        worker = threading.Thread(target=self._compute_move, args=(time_controls,))
        with self.state_lock:
            self.current_worker = worker
        worker.start()

    def handle_stop(self, _: str) -> None:
        with self.state_lock:
            if not self.move_calculating:
                print("info string Stop ignored; engine idle")
                return
            self.stop_event.set()
            worker = self.current_worker
        print("info string Stop signal received")
        if worker and worker.is_alive():
            worker.join(timeout=5.0)

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

            # Performance summary already printed by search backend
            # Just print select time if no search was performed
            if nodes is None or search_time is None:
                print(f"info string perf select={select_time:.3f}s")

            with self.state_lock:
                if result and result.move:
                    print(f"bestmove {result.move}")
                else:
                    print("bestmove (none)")
                print("readyok")
                self.move_calculating = False
                self.current_worker = None

        except Exception as exc:
            print(f"info string Error generating move: {exc}")
            with self.state_lock:
                print("bestmove (none)")
                print("readyok")
                self.move_calculating = False
                self.current_worker = None

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
    engine_main(sys.argv[1:])
