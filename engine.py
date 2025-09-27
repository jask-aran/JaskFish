import sys
import io
import random
import time
import threading
import math
import os
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import chess
from chess import polyglot

NSEC_PER_SEC = 1_000_000_000


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


def _ensure_line_buffered_stdout() -> None:
    """Wrap ``sys.stdout`` with line buffering if possible."""

    stdout = sys.stdout
    if isinstance(stdout, io.TextIOBase) and getattr(stdout, "line_buffering", False):
        return
    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        return
    sys.stdout = io.TextIOWrapper(buffer, line_buffering=True)


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
    in_check: bool = False
    opponent_mate_in_one_threat: bool = False


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
    # Optional cached static evaluation for this node (side-to-move aware)
    static_eval: Optional[float] = None


class MoveStrategy(ABC):
    """Base class for all move selection strategies."""

    def __init__(
        self,
        name: Optional[str] = None,
        priority: int = 0,
        confidence: Optional[float] = None,
        short_circuit: bool = True,
    ):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.confidence = confidence
        self.short_circuit = short_circuit

    @abstractmethod
    def is_applicable(self, context: StrategyContext) -> bool:
        """Return whether the strategy should be considered in the given context."""

    @abstractmethod
    def generate_move(
        self, board: chess.Board, context: StrategyContext
    ) -> Optional[StrategyResult]:
        """Produce a move suggestion when applicable."""


# Toggle individual strategies by flipping these booleans. The engine will
# always fall back to a random legal move if no strategies produce a result.
STRATEGY_ENABLE_FLAGS = {
    "mate_in_one": True,
    "repetition_avoidance": True,
    "heuristic": True,
    "fallback_random": False,
}


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
        display_name = getattr(strategy, "log_tag", strategy.name)
        self._logger(
            f"strategy registered: {display_name} (priority={strategy.priority})"
        )

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

    def select_move(
        self, board: chess.Board, context: StrategyContext
    ) -> Optional[StrategyResult]:
        """Evaluate registered strategies and choose a result using the selection policy."""

        strategy_results: List[Tuple[MoveStrategy, StrategyResult]] = []
        for strategy in self._strategies:
            display_name = getattr(strategy, "log_tag", strategy.name)
            if not strategy.is_applicable(context):
                self._logger(f"strategy skipped (not applicable): {display_name}")
                continue

            self._logger(f"strategy evaluating: {display_name}")
            try:
                result = strategy.generate_move(board, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger(f"strategy error in {display_name}: {exc}")
                continue

            if result is None:
                self._logger(f"strategy produced no result: {display_name}")
                continue

            strategy_results.append((strategy, result))
            if result.move:
                if strategy.short_circuit:
                    if self._uses_default_policy:
                        return result
                elif self._uses_default_policy:
                    return result

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

    @staticmethod
    def priority_score_selection_policy(
        strategy_results: List[Tuple[MoveStrategy, StrategyResult]],
        **_: Any,
    ) -> Optional[StrategyResult]:
        """Pick the result with the highest (priority, score, confidence) tuple."""

        best_result: Optional[StrategyResult] = None
        best_key: Optional[Tuple[float, float, float]] = None
        for strategy, result in strategy_results:
            if not result.move:
                continue
            score = float(result.score) if result.score is not None else 0.0
            confidence = (
                float(result.confidence) if result.confidence is not None else 0.0
            )
            key = (float(strategy.priority), score, confidence)
            if best_key is None or key > best_key:
                best_key = key
                best_result = result
        return best_result


class MateInOneStrategy(MoveStrategy):
    def __init__(
        self,
        mate_score: float = 500000.0,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(priority=100, short_circuit=True, **kwargs)
        self.mate_score = mate_score
        self._logger = logger or (lambda *_: None)

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(
        self, board: chess.Board, context: StrategyContext
    ) -> Optional[StrategyResult]:
        for move in board.legal_moves:
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                uci_move = move.uci()
                self._logger(f"mate-in-one strategy chose {uci_move}")
                return StrategyResult(
                    move=uci_move,
                    strategy_name=self.name,
                    score=self.mate_score,
                    confidence=self.confidence or 1.0,
                    metadata={"pattern": "mate_in_one"},
                )
        return None


class HeuristicSearchStrategy(MoveStrategy):
    def __init__(
        self,
        search_depth: int = 6,
        quiescence_depth: int = 6,
        base_time_limit: float = 6.0,
        max_time_limit: float = 30.0,
        min_time_limit: float = 0.25,
        time_allocation_factor: float = 0.10,
        transposition_table_size: int = 2000000,
        avoid_repetition: bool = False,
        repetition_penalty: float = 45.0,
        repetition_strong_penalty: float = 90.0,
        logger: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        self._log_tag = kwargs.pop("log_tag", "HS")
        self.log_tag = self._log_tag
        super().__init__(priority=70, **kwargs)
        self.search_depth = max(1, search_depth)
        self.quiescence_depth = max(0, quiescence_depth)
        self.base_time_limit = base_time_limit
        self.min_time_limit = min_time_limit
        self.max_time_limit = max(max_time_limit, self.min_time_limit)
        self.time_allocation_factor = time_allocation_factor
        self._mate_score = 100000
        self._transposition_table: "OrderedDict[str, TranspositionEntry]" = OrderedDict()
        self._transposition_table_limit = max(1000, transposition_table_size)
        self._history_scores: Dict[Tuple[bool, int, int], float] = defaultdict(float)
        self._killer_slots = 2
        self._aspiration_window = 50.0
        self._aspiration_growth = 2.0
        self._aspiration_fail_reset = 2
        self._null_move_min_depth = 3
        self._null_move_reduction_base = 1
        self._null_move_depth_scale = 4
        self._lmr_min_depth = 4
        self._lmr_min_move_index = 4
        self._futility_depth_limit = 2
        self._futility_base_margin = 120.0
        self._razoring_depth_limit = 2
        self._razoring_margin = 325.0
        self._depth_iteration_stop_ratio = 0.7
        self._avoid_repetition = avoid_repetition
        self._repetition_penalty = repetition_penalty
        self._repetition_strong_penalty = repetition_strong_penalty

        # Search state (initialised per search invocation)
        self._search_deadline: Optional[float] = None
        self._search_start_time: float = 0.0
        self._nodes_visited: int = 0
        self._killer_moves: List[List[Optional[chess.Move]]] = []
        self._principal_variation: List[chess.Move] = []
        # Lightweight per-search cache for expensive static evaluations
        self._eval_cache: Dict[str, float] = {}
        # Quiescence pruning: drop clearly losing captures unless they check or promote
        self._qsearch_see_prune_threshold: float = -0.5
        self._logger = logger or (lambda *_: None)
        # Lazily populated timing buckets for telemetry (ns units)
        self._timing_totals: Optional[Dict[str, int]] = None
        # Per-search heuristic instrumentation
        self._decision_stats: Optional[Dict[str, Any]] = None

        # Evaluation constants (centipawns)
        self.bishop_pair_bonus = 40.0
        self._passed_pawn_base_bonus = 20.0
        self._passed_pawn_rank_bonus = 8.0
        self._mobility_unit = 2.0
        self._king_safety_opening_penalty = 12.0
        self._king_safety_endgame_bonus = 6.0
        # Time checking configuration
        self._time_check_interval = 512
        self._time_check_counter = 0
        self._deadline_reached = False

    def _position_key(self, board: chess.Board) -> int:
        """Return a Zobrist hash for the given board state."""

        return polyglot.zobrist_hash(board)

    def _initialise_decision_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "tt_probes": 0,
            "tt_exact": 0,
            "tt_lower": 0,
            "tt_upper": 0,
            "tt_miss": 0,
            "null_tried": 0,
            "null_success": 0,
            "futility_prunes": 0,
            "razor_attempts": 0,
            "razor_prunes": 0,
            "lmr_applied": 0,
            "lmr_research": 0,
            "beta_cutoffs": {
                "tt": 0,
                "capture": 0,
                "killer": 0,
                "history": 0,
                "check": 0,
                "promo": 0,
                "other": 0,
            },
            "asp_fail_low": 0,
            "asp_fail_high": 0,
            "asp_resets": 0,
            "root_scores": [],
            "root_moves": [],
            "root_best_indices": [],
            "pv_swaps": 0,
            "eval_cache_hits": 0,
            "eval_cache_misses": 0,
            "eval_cache_store": 0,
            "q_nodes": 0,
            "q_cutoffs": 0,
            "q_moves_considered": 0,
            "history_updates": 0,
            "killer_updates": 0,
        }
        self._decision_stats = stats
        return stats

    @staticmethod
    def _percentile(values: Sequence[int], percentile: float) -> Optional[int]:
        if not values:
            return None
        percentile = min(max(percentile, 0.0), 1.0)
        if len(values) == 1:
            return values[0]
        sorted_vals = sorted(values)
        index = int(round(percentile * (len(sorted_vals) - 1)))
        index = min(max(index, 0), len(sorted_vals) - 1)
        return sorted_vals[index]

    def _build_analysis_summary(
        self,
        stats: Optional[Dict[str, Any]],
        completed_depth: int,
        best_score: float,
        search_time: float,
        search_interrupted: bool,
        capped_by_budget: bool,
        timing_totals_sec: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Any]]:
        if stats is None:
            return None

        total_nodes = self._nodes_visited
        nps = int(total_nodes / search_time) if search_time > 0 else 0

        root_scores: List[float] = stats["root_scores"]
        score_delta = 0.0
        score_jitter = 0.0
        last_delta = 0.0
        if len(root_scores) >= 2:
            diffs = [root_scores[i] - root_scores[i - 1] for i in range(1, len(root_scores))]
            score_delta = root_scores[-1] - root_scores[0]
            score_jitter = max(abs(diff) for diff in diffs)
            last_delta = diffs[-1]

        order_indices: List[int] = stats["root_best_indices"]
        order_p25 = self._percentile(order_indices, 0.25)
        order_p50 = self._percentile(order_indices, 0.50)
        order_p90 = self._percentile(order_indices, 0.90)
        best_index_last = order_indices[-1] if order_indices else None

        tt_probes = stats["tt_probes"]
        tt_hits = stats["tt_exact"] + stats["tt_lower"] + stats["tt_upper"]
        tt_hit_pct = (tt_hits / tt_probes * 100.0) if tt_probes else 0.0

        null_tried = stats["null_tried"]
        null_success = stats["null_success"]

        prune_futility = stats["futility_prunes"]
        prune_razor = stats["razor_prunes"]
        prune_total = prune_futility + prune_razor + null_success
        prune_rate = (prune_total / total_nodes * 100.0) if total_nodes else 0.0

        q_nodes = stats["q_nodes"]
        q_cutoffs = stats["q_cutoffs"]
        q_ratio = (q_nodes / total_nodes) if total_nodes else 0.0

        cache_hits = stats["eval_cache_hits"]
        cache_misses = stats["eval_cache_misses"]
        cache_store = stats["eval_cache_store"]

        summary = {
            "label": self._log_tag,
            "depth": completed_depth,
            "score": best_score,
            "nodes": total_nodes,
            "nps": nps,
            "search_time": search_time,
            "score_delta": score_delta,
            "score_jitter": score_jitter,
            "last_delta": last_delta,
            "pv_swaps": stats["pv_swaps"],
            "pv_head": [move.uci() for move in self._principal_variation[:6]],
            "tt": {
                "probes": tt_probes,
                "hits": tt_hits,
                "hit_pct": tt_hit_pct,
                "exact": stats["tt_exact"],
                "lower": stats["tt_lower"],
                "upper": stats["tt_upper"],
                "miss": stats["tt_miss"],
            },
            "null": {
                "tried": null_tried,
                "success": null_success,
            },
            "lmr": {
                "applied": stats["lmr_applied"],
                "research": stats["lmr_research"],
            },
            "asp": {
                "fail_low": stats["asp_fail_low"],
                "fail_high": stats["asp_fail_high"],
                "resets": stats["asp_resets"],
            },
            "prune": {
                "futility": prune_futility,
                "razor": prune_razor,
                "null": null_success,
                "total": prune_total,
                "rate": prune_rate,
            },
            "cutoffs": stats["beta_cutoffs"].copy(),
            "qsearch": {
                "nodes": q_nodes,
                "ratio": q_ratio,
                "cutoffs": q_cutoffs,
                "moves": stats["q_moves_considered"],
            },
            "cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "stored": cache_store,
            },
            "updates": {
                "killer": stats["killer_updates"],
                "history": stats["history_updates"],
            },
            "order": {
                "p25": order_p25,
                "p50": order_p50,
                "p90": order_p90,
                "last": best_index_last,
                "count": len(order_indices),
            },
            "status": {
                "timeout": search_interrupted,
                "capped": capped_by_budget,
            },
            "timers": timing_totals_sec or {},
            "root_scores": root_scores,
        }
        return summary

    def _emit_analysis_summary(self, summary: Dict[str, Any]) -> None:
        pv_text = " ".join(summary.get("pv_head", []))
        if len(pv_text) > 60:
            pv_text = pv_text[:57] + "..."

        status = summary.get("status", {})
        status_bits: List[str] = []
        if status.get("timeout"):
            status_bits.append("timeout")
        if status.get("capped"):
            status_bits.append("capped")
        status_text = ",".join(status_bits) if status_bits else "ok"

        line_root = (
            f"info string analysis root depth={summary.get('depth', '-')} "
            f"score={summary.get('score', 0.0):.1f} nodes={summary.get('nodes', 0)} "
            f"nps={summary.get('nps', 0)} time={summary.get('search_time', 0.0):.2f}s "
            f"delta={summary.get('score_delta', 0.0):+.1f} jitter={summary.get('score_jitter', 0.0):.1f} "
            f"pv_swaps={summary.get('pv_swaps', 0)} status={status_text}"
        )
        if pv_text:
            line_root += f" pv={pv_text}"
        print(line_root)

        tt = summary.get("tt", {})
        null = summary.get("null", {})
        lmr = summary.get("lmr", {})
        asp = summary.get("asp", {})
        prune = summary.get("prune", {})
        cut = summary.get("cutoffs", {})

        tt_hits = tt.get("hits", 0)
        tt_probes = tt.get("probes", 0)
        null_tried_raw = null.get("tried", 0)
        null_success = null.get("success", 0)
        null_pct = (
            null_success / null_tried_raw * 100.0 if null_tried_raw else 0.0
        )
        prune_rate = prune.get("rate", 0.0)

        cut_summary_items = sorted(
            ((label, value) for label, value in cut.items() if value),
            key=lambda item: item[1],
            reverse=True,
        )
        cut_summary_text = ",".join(f"{label}:{value}" for label, value in cut_summary_items[:4])
        if not cut_summary_text:
            cut_summary_text = "-"

        line_heur = (
            f"info string analysis heur tt={tt.get('hit_pct', 0.0):.0f}%({tt_hits}/{tt_probes}) "
            f"null={null_success}/{null_tried_raw}({null_pct:.0f}%) "
            f"lmr={lmr.get('applied', 0)}/{lmr.get('research', 0)} "
            f"asp=FL{asp.get('fail_low', 0)} FH{asp.get('fail_high', 0)} R{asp.get('resets', 0)} "
            f"prune={prune_rate:.1f}% cut={cut_summary_text}"
        )
        print(line_heur)

        order = summary.get("order", {})
        cache = summary.get("cache", {})
        updates = summary.get("updates", {})
        qsearch = summary.get("qsearch", {})

        def _fmt(value: Optional[int]) -> str:
            return str(value) if value is not None else "-"

        cache_hits = cache.get("hits", 0)
        cache_total = cache_hits + cache.get("misses", 0)
        cache_rate = (cache_hits / cache_total * 100.0) if cache_total else 0.0
        q_ratio_pct = qsearch.get("ratio", 0.0) * 100.0

        line_quality = (
            f"info string analysis quality order_p50={_fmt(order.get('p50'))} "
            f"p90={_fmt(order.get('p90'))} last={_fmt(order.get('last'))} "
            f"cache={cache_hits}/{cache_total}({cache_rate:.0f}%) add={cache.get('stored', 0)} "
            f"updates=K{updates.get('killer', 0)} H{updates.get('history', 0)} "
            f"q={qsearch.get('nodes', 0)}({q_ratio_pct:.0f}%) cut={qsearch.get('cutoffs', 0)}"
        )
        print(line_quality)

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
            return None

        depth_limit = self._resolve_depth_limit(context)
        time_budget = self._determine_time_budget(context)

        self._logger(
            f"{self._log_tag}: start search depth_limit={depth_limit} "
            f"time_budget={'infinite' if time_budget is None else f'{time_budget:.2f}s'} "
            f"legal_moves={context.legal_moves_count}"
        )

        self._search_start_time = time.perf_counter()
        self._search_deadline = (
            None
            if time_budget is None
            else self._search_start_time + max(time_budget, self.min_time_limit)
        )
        self._nodes_visited = 0
        self._time_check_counter = 0
        self._deadline_reached = False
        self._killer_moves = [
            list() for _ in range(depth_limit + self.quiescence_depth + 4)
        ]
        self._eval_cache.clear()
        timing_totals = {
            "alpha_beta": 0,
            "order_moves": 0,
            "quiescence": 0,
            "evaluate": 0,
        }
        self._timing_totals = timing_totals
        decision_stats = self._initialise_decision_stats()

        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        completed_depth = 0
        last_score: Optional[float] = None
        self._principal_variation = []

        stats = decision_stats

        search_interrupted = False
        capped_by_budget = False
        current_depth = 0
        try:
            for current_depth in range(1, depth_limit + 1):
                if current_depth > 1:
                    self._decay_history_scores()
                aspiration = self._aspiration_window
                alpha_window = -float(self._mate_score)
                beta_window = float(self._mate_score)
                if last_score is not None and current_depth >= 2:
                    alpha_window = max(
                        last_score - aspiration, -float(self._mate_score)
                    )
                    beta_window = min(
                        last_score + aspiration, float(self._mate_score)
                    )

                depth_start_nodes = self._nodes_visited
                depth_start_time = time.perf_counter()
                alpha_start_ns = timing_totals["alpha_beta"]
                order_start_ns = timing_totals["order_moves"]
                eval_start_ns = timing_totals["evaluate"]
                q_start_ns = timing_totals["quiescence"]

                failure_count = 0

                while True:
                    search_alpha = alpha_window
                    search_beta = beta_window
                    alpha = search_alpha
                    beta = search_beta
                    iteration_best_move: Optional[chess.Move] = None
                    iteration_best_score = -float("inf")
                    iteration_best_index = -1
                    fail_low = False
                    fail_high = False

                    root_key = self._position_key(board)
                    tt_entry = self._transposition_table.get(root_key)
                    tt_move = tt_entry.move if tt_entry else None
                    principal_move = (
                        self._principal_variation[0]
                        if self._principal_variation
                        else best_move
                    )
                    moves = self._order_moves(board, 0, tt_move, principal_move)

                    for move_index, move in enumerate(moves):
                        if self._time_exceeded():
                            raise _SearchTimeout()

                        color = board.turn
                        penalty = 0.0
                        board.push(move)
                        try:
                            if move_index == 0:
                                score = -self._alpha_beta(
                                    board,
                                    depth=current_depth - 1,
                                    alpha=-beta,
                                    beta=-alpha,
                                    ply=1,
                                    is_pv=True,
                                )
                            else:
                                score = -self._alpha_beta(
                                    board,
                                    depth=current_depth - 1,
                                    alpha=-alpha - 1,
                                    beta=-alpha,
                                    ply=1,
                                    is_pv=False,
                                )
                                if score > alpha:
                                    score = -self._alpha_beta(
                                        board,
                                        depth=current_depth - 1,
                                        alpha=-beta,
                                        beta=-alpha,
                                        ply=1,
                                        is_pv=True,
                                    )

                            if self._avoid_repetition:
                                try:
                                    penalty = self._repetition_penalty_value(board)
                                except Exception as exc:  # pragma: no cover - defensive
                                    self._logger(
                                        f"{self._log_tag}: repetition penalty evaluation failed: {exc}"
                                    )
                                    penalty = 0.0
                        finally:
                            board.pop()

                        if penalty:
                            score -= penalty

                        if score > iteration_best_score:
                            iteration_best_score = score
                            iteration_best_move = move
                            iteration_best_index = move_index

                        if score > alpha:
                            alpha = score

                        if alpha >= beta:
                            fail_high = True
                            if not board.is_capture(move):
                                self._record_history(color, move, current_depth)
                            break

                    if self._time_exceeded():
                        raise _SearchTimeout()

                    if iteration_best_move is None:
                        break

                    if iteration_best_score <= search_alpha and search_alpha > -float(
                        self._mate_score
                    ):
                        fail_low = True

                    if iteration_best_score >= search_beta and search_beta < float(
                        self._mate_score
                    ):
                        fail_high = True

                    if fail_low or fail_high:
                        if fail_low:
                            stats["asp_fail_low"] += 1
                        if fail_high:
                            stats["asp_fail_high"] += 1
                        failure_count += 1
                        cause = "fail-high" if fail_high else "fail-low"
                        aspiration *= self._aspiration_growth
                        alpha_window = max(
                            -float(self._mate_score),
                            iteration_best_score - aspiration,
                        )
                        beta_window = min(
                            float(self._mate_score),
                            iteration_best_score + aspiration,
                        )

                        self._logger(
                            f"{self._log_tag}: depth={current_depth} {cause} "
                            f"aspiration -> [{alpha_window:.1f}, {beta_window:.1f}]"
                        )

                        if failure_count >= self._aspiration_fail_reset:
                            alpha_window = -float(self._mate_score)
                            beta_window = float(self._mate_score)
                            aspiration = float(self._mate_score)
                            failure_count = 0
                            stats["asp_resets"] += 1
                            self._logger(
                                f"{self._log_tag}: depth={current_depth} switching to full-window search"
                            )

                        if (
                            alpha_window <= -float(self._mate_score)
                            and beta_window >= float(self._mate_score)
                        ):
                            break
                        continue

                    best_move = iteration_best_move
                    best_score = iteration_best_score
                    last_score = iteration_best_score
                    completed_depth = current_depth
                    self._principal_variation = self._extract_principal_variation(
                        board, current_depth
                    )

                    if iteration_best_move is not None:
                        move_uci = iteration_best_move.uci()
                        previous_moves = stats["root_moves"]
                        if previous_moves and move_uci != previous_moves[-1]:
                            stats["pv_swaps"] += 1
                        previous_moves.append(move_uci)
                        stats["root_scores"].append(iteration_best_score)
                        if iteration_best_index >= 0:
                            stats["root_best_indices"].append(iteration_best_index)

                    depth_time = time.perf_counter() - depth_start_time
                    depth_nodes = self._nodes_visited - depth_start_nodes
                    pv_line = " ".join(
                        move.uci() for move in self._principal_variation
                    )
                    nps = int(depth_nodes / depth_time) if depth_time > 0 else 0
                    alpha_delta_ms = (
                        timing_totals["alpha_beta"] - alpha_start_ns
                    ) / 1_000_000
                    order_delta_ms = (
                        timing_totals["order_moves"] - order_start_ns
                    ) / 1_000_000
                    eval_delta_ms = (
                        timing_totals["evaluate"] - eval_start_ns
                    ) / 1_000_000
                    q_delta_ms = (
                        timing_totals["quiescence"] - q_start_ns
                    ) / 1_000_000
                    perf_line = (
                        f"perf nps={nps} best_index={iteration_best_index} "
                        f"alpha_ms={alpha_delta_ms:.3f} order_ms={order_delta_ms:.3f} "
                        f"eval_ms={eval_delta_ms:.3f} q_ms={q_delta_ms:.3f}"
                    )
                    self._logger(
                        f"{self._log_tag}: depth={current_depth} "
                        f"score={iteration_best_score:.1f} nodes={self._nodes_visited} "
                        f"(+{depth_nodes}) time={depth_time:.2f}s "
                        f"pv={pv_line or iteration_best_move.uci()}\n{perf_line}"
                    )

                    if self._search_deadline is not None:
                        total_budget = self._search_deadline - self._search_start_time
                        if total_budget > 0 and depth_time >= total_budget * self._depth_iteration_stop_ratio:
                            usage = depth_time / total_budget
                            capped_by_budget = True
                            self._logger(
                                f"{self._log_tag}: depth={current_depth} consumed {usage:.0%} of budget; halting deeper search"
                            )
                            break
                    break

                if self._time_exceeded():
                    break

                if capped_by_budget:
                    break

        except _SearchTimeout:
            search_interrupted = True
            elapsed = time.perf_counter() - self._search_start_time
            self._logger(
                f"{self._log_tag}: search timeout at depth={current_depth} "
                f"nodes={self._nodes_visited} time={elapsed:.2f}s"
            )

        search_time = time.perf_counter() - self._search_start_time

        if best_move is None:
            self._logger(
                f"{self._log_tag}: no move found (interrupted={search_interrupted or capped_by_budget})"
            )
            self._timing_totals = None
            self._decision_stats = None
            return None

        self._logger(
            f"{self._log_tag}: completed depth={completed_depth} score={best_score:.1f} "
            f"best={best_move.uci()} nodes={self._nodes_visited} time={search_time:.2f}s "
            f"interrupted={search_interrupted} capped={capped_by_budget}"
        )

        timing_totals_sec: Optional[Dict[str, float]] = None
        analysis_summary: Optional[Dict[str, Any]] = None
        if self._timing_totals:
            timing_totals_sec = {
                key: value / NSEC_PER_SEC for key, value in self._timing_totals.items()
            }
            nps_total = int(self._nodes_visited / search_time) if search_time else 0
            summary_segments = [
                f"info string perf summary depth={completed_depth}",
                f"nodes={self._nodes_visited}",
                f"time={search_time:.3f}s",
                f"nps={nps_total}",
            ]
            timer_parts = [
                f"{key.split('_')[0]}={timing_totals_sec[key]:.3f}s"
                for key in ("alpha_beta", "order_moves", "evaluate", "quiescence")
                if key in timing_totals_sec
            ]
            if timer_parts:
                summary_segments.append("timers=" + ",".join(timer_parts))
            print(" ".join(summary_segments))
            analysis_summary = self._build_analysis_summary(
                stats,
                completed_depth,
                best_score,
                search_time,
                search_interrupted,
                capped_by_budget,
                timing_totals_sec,
            )
            if analysis_summary is not None:
                self._emit_analysis_summary(analysis_summary)

        repetition_penalty = 0.0
        if self._avoid_repetition and best_move is not None:
            board.push(best_move)
            try:
                repetition_penalty = self._repetition_penalty_value(board)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger(
                    f"{self._log_tag}: repetition penalty evaluation failed: {exc}"
                )
                repetition_penalty = 0.0
            finally:
                board.pop()

        metadata = {
            "depth": completed_depth,
            "nodes": self._nodes_visited,
            "time": search_time,
            "principal_move": best_move.uci(),
            "pv": [move.uci() for move in self._principal_variation],
        }
        if self._avoid_repetition:
            metadata["avoid_repetition"] = True
            if repetition_penalty:
                metadata["repetition_penalty"] = repetition_penalty
        if timing_totals_sec is not None:
            metadata["timing"] = timing_totals_sec
        if analysis_summary is not None:
            metadata_analysis = dict(analysis_summary)
            metadata_analysis.pop("root_scores", None)
            metadata["analysis"] = metadata_analysis
        metadata["label"] = self._log_tag
        self._timing_totals = None
        self._decision_stats = None
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
        tc = context.time_controls or {}
        if tc.get("infinite"):
            return None

        movetime = tc.get("movetime")
        if movetime is not None:
            seconds = movetime / 1000.0
            return float(
                max(self.min_time_limit, min(self.max_time_limit, seconds))
            )

        turn_key = "wtime" if context.turn == chess.WHITE else "btime"
        inc_key = "winc" if context.turn == chess.WHITE else "binc"
        if turn_key in tc:
            time_left = max(0, tc.get(turn_key, 0))
            increment = max(0, tc.get(inc_key, 0))
            moves_to_go = tc.get("movestogo")
            if moves_to_go:
                budget_ms = time_left / max(1, moves_to_go)
            else:
                budget_ms = time_left * self.time_allocation_factor
            budget_ms += increment
            seconds = budget_ms / 1000.0
            return float(
                max(self.min_time_limit, min(self.max_time_limit, seconds))
            )

        fallback = max(self.min_time_limit, min(self.max_time_limit, self.base_time_limit))

        if context is not None:
            complexity = context.legal_moves_count
            phase = context.piece_count
            # Normalise piece count to an opening/endgame scale (0.3 - 1.0)
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

            dynamic_budget = fallback * complexity_factor * phase_factor * tension_factor
            fallback = max(self.min_time_limit, min(self.max_time_limit, dynamic_budget))

        return float(fallback)

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        is_pv: bool,
        allow_null: bool = True,
    ) -> float:
        timing = self._timing_totals
        start_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            self._nodes_visited += 1
            stats = self._decision_stats

            if self._time_exceeded():
                raise _SearchTimeout()

            if board.is_checkmate():
                return -float(self._mate_score) + ply
            if (
                board.is_stalemate()
                or board.is_insufficient_material()
                or board.can_claim_draw()
            ):
                return 0.0

            if depth <= 0:
                return self._quiescence(board, alpha, beta, self.quiescence_depth, ply)

            in_check = board.is_check()
            key = self._position_key(board)
            entry = self._transposition_table.get(key)

            if stats is not None:
                stats["tt_probes"] += 1
                if entry is None:
                    stats["tt_miss"] += 1
                else:
                    if entry.flag == TranspositionFlag.EXACT:
                        stats["tt_exact"] += 1
                    elif entry.flag == TranspositionFlag.LOWERBOUND:
                        stats["tt_lower"] += 1
                    else:
                        stats["tt_upper"] += 1

            if entry and entry.depth >= depth:
                if entry.flag == TranspositionFlag.EXACT:
                    return entry.value
                if entry.flag == TranspositionFlag.LOWERBOUND:
                    alpha = max(alpha, entry.value)
                else:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value

            tt_move = entry.move if entry else None
            static_eval: Optional[float] = entry.static_eval if entry else None

            if not in_check and not is_pv:
                if depth <= self._razoring_depth_limit:
                    if stats is not None:
                        stats["razor_attempts"] += 1
                    static_eval = static_eval or self._evaluate_board(board)
                    if static_eval + self._razoring_margin <= alpha:
                        reduced = self._alpha_beta(
                            board,
                            depth - 1,
                            alpha,
                            beta,
                            ply,
                            is_pv,
                            allow_null,
                        )
                        if reduced <= alpha:
                            if stats is not None:
                                stats["razor_prunes"] += 1
                            return reduced

                if depth <= self._futility_depth_limit:
                    static_eval = static_eval or self._evaluate_board(board)
                    margin = self._futility_margin(depth)
                    if static_eval + margin <= alpha:
                        if stats is not None:
                            stats["futility_prunes"] += 1
                        return static_eval + margin

            if (
                allow_null
                and not is_pv
                and depth >= self._null_move_min_depth
                and not in_check
                and self._has_sufficient_material(board)
            ):
                reduction = self._null_move_reduction(depth)
                board.push(chess.Move.null())
                try:
                    if stats is not None:
                        stats["null_tried"] += 1
                    null_score = -self._alpha_beta(
                        board,
                        depth=depth - 1 - reduction,
                        alpha=-beta,
                        beta=-beta + 1,
                        ply=ply + 1,
                        is_pv=False,
                        allow_null=False,
                    )
                finally:
                    board.pop()
                if null_score >= beta:
                    if stats is not None:
                        stats["null_success"] += 1
                    return beta

            moves = self._order_moves(board, ply, tt_move)
            if not moves:
                return self._evaluate_board(board)

            best_value = -float("inf")
            best_move: Optional[chess.Move] = None
            alpha_original = alpha
            killer_moves = self._killer_moves[ply] if ply < len(self._killer_moves) else []

            for move_index, move in enumerate(moves):
                if self._time_exceeded():
                    raise _SearchTimeout()

                is_capture = board.is_capture(move)
                gives_check = board.gives_check(move)
                promotion = move.promotion is not None
                color = board.turn
                history_key = (color, move.from_square, move.to_square)
                history_score = self._history_scores.get(history_key, 0.0)
                was_killer = move in killer_moves
                was_tt_move = tt_move is not None and move == tt_move

                reduction = 0
                child_is_pv = is_pv and move_index == 0
                search_depth = depth - 1
                new_allow_null = allow_null and not is_capture

                if (
                    depth >= self._lmr_min_depth
                    and move_index >= self._lmr_min_move_index
                    and not child_is_pv
                    and not is_capture
                    and not gives_check
                    and not promotion
                ):
                    reduction = self._late_move_reduction(depth, move_index)
                    search_depth = max(0, depth - 1 - reduction)
                    if stats is not None:
                        stats["lmr_applied"] += 1

                board.push(move)
                try:
                    if child_is_pv:
                        score = -self._alpha_beta(
                            board,
                            search_depth,
                            alpha=-beta,
                            beta=-alpha,
                            ply=ply + 1,
                            is_pv=True,
                            allow_null=new_allow_null,
                        )
                    else:
                        score = -self._alpha_beta(
                            board,
                            search_depth,
                            alpha=-alpha - 1,
                            beta=-alpha,
                            ply=ply + 1,
                            is_pv=False,
                            allow_null=new_allow_null,
                        )
                        if reduction and score > alpha:
                            if stats is not None:
                                stats["lmr_research"] += 1
                            score = -self._alpha_beta(
                                board,
                                depth - 1,
                                alpha=-beta,
                                beta=-alpha,
                                ply=ply + 1,
                                is_pv=True,
                                allow_null=new_allow_null,
                            )
                        elif score > alpha:
                            score = -self._alpha_beta(
                                board,
                                depth - 1,
                                alpha=-beta,
                                beta=-alpha,
                                ply=ply + 1,
                                is_pv=True,
                                allow_null=new_allow_null,
                            )
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
                    if not is_capture:
                        self._record_killer(ply, move)
                        self._record_history(color, move, depth)
                    if stats is not None:
                        if was_tt_move:
                            category = "tt"
                        elif is_capture:
                            category = "capture"
                        elif was_killer:
                            category = "killer"
                        elif history_score > 0:
                            category = "history"
                        elif gives_check:
                            category = "check"
                        elif promotion:
                            category = "promo"
                        else:
                            category = "other"
                        stats["beta_cutoffs"][category] += 1
                    self._store_transposition_entry(
                        key,
                        depth,
                        best_value,
                        TranspositionFlag.LOWERBOUND,
                        best_move,
                        static_eval,
                    )
                    return best_value

            flag = TranspositionFlag.EXACT
            if best_value <= alpha_original:
                flag = TranspositionFlag.UPPERBOUND

            if static_eval is None and not in_check:
                static_eval = self._evaluate_board(board)

            self._store_transposition_entry(
                key,
                depth,
                best_value,
                flag,
                best_move,
                static_eval,
            )
            return best_value
        finally:
            if timing is not None:
                timing["alpha_beta"] += time.perf_counter_ns() - start_ns

    def _quiescence(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        depth: int,
        ply: int,
    ) -> float:
        timing = self._timing_totals
        start_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            self._nodes_visited += 1
            stats = self._decision_stats
            if stats is not None:
                stats["q_nodes"] += 1
            if self._time_exceeded():
                raise _SearchTimeout()

            stand_pat = self._evaluate_board(board)
            if stand_pat >= beta:
                if stats is not None:
                    stats["q_cutoffs"] += 1
                return beta
            if alpha < stand_pat:
                alpha = stand_pat

            if depth <= 0:
                return stand_pat

            moves = self._generate_quiescence_moves(board)
            if stats is not None:
                stats["q_moves_considered"] += len(moves)
            for move in moves:
                board.push(move)
                try:
                    score = -self._quiescence(board, -beta, -alpha, depth - 1, ply + 1)
                except _SearchTimeout:
                    board.pop()
                    raise
                board.pop()

                if score >= beta:
                    if stats is not None:
                        stats["q_cutoffs"] += 1
                    return beta
                if score > alpha:
                    alpha = score

            return alpha
        finally:
            if timing is not None:
                timing["quiescence"] += time.perf_counter_ns() - start_ns

    def _generate_quiescence_moves(self, board: chess.Board) -> List[chess.Move]:
        moves: List[chess.Move] = []
        is_capture = board.is_capture
        gives_check = board.gives_check
        append = moves.append
        see = self._static_exchange_score
        in_check = board.is_check()
        threshold = self._qsearch_see_prune_threshold

        for move in board.legal_moves:
            if in_check:
                append(move)
                continue
            if is_capture(move):
                # Prune clearly losing captures to keep qsearch lean unless they are the
                # only way to address check (handled above).
                if see(board, move) >= threshold:
                    append(move)
            elif move.promotion or gives_check(move):
                append(move)

        moves.sort(
            key=lambda mv: see(board, mv) + (500 if gives_check(mv) else 0),
            reverse=True,
        )
        return moves

    def _capture_score(self, board: chess.Board, move: chess.Move) -> float:
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None and board.is_en_passant(move):
            captured_piece = chess.Piece(chess.PAWN, not board.turn)
        captured_value = (
            EVAL_PIECE_VALUES.get(captured_piece.piece_type, 0) if captured_piece else 0
        )
        moving_piece = board.piece_at(move.from_square)
        moving_value = (
            EVAL_PIECE_VALUES.get(moving_piece.piece_type, 0) if moving_piece else 0
        )
        return captured_value - moving_value

    def _static_exchange_score(self, board: chess.Board, move: chess.Move) -> float:
        if hasattr(board, "see"):
            try:
                return float(board.see(move))
            except (ValueError, AttributeError):
                pass
        return self._capture_score(board, move)

    def _order_moves(
        self,
        board: chess.Board,
        ply: int,
        tt_move: Optional[chess.Move] = None,
        principal_move: Optional[chess.Move] = None,
    ) -> List[chess.Move]:
        timing = self._timing_totals
        start_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            moves = list(board.legal_moves)
            if not moves:
                return moves

            killer_moves = (
                self._killer_moves[ply] if ply < len(self._killer_moves) else []
            )
            killer_ranks = {
                km: idx for idx, km in enumerate(killer_moves) if km is not None
            }
            history_scores = self._history_scores
            is_capture = board.is_capture
            gives_check = board.gives_check

            def score_move(move: chess.Move) -> int:
                # Highest priority: exact TT move.
                if tt_move and move == tt_move:
                    return 12_000_000

                score = 0
                if principal_move and move == principal_move:
                    score += 11_000_000

                killer_idx = killer_ranks.get(move)
                if killer_idx is not None:
                    score += 9_500_000 - killer_idx * 1_000

                if is_capture(move):
                    score += 8_000_000
                    score += int(self._capture_score(board, move))

                if move.promotion:
                    score += 6_000_000 + EVAL_PIECE_VALUES.get(move.promotion, 0)

                if not is_capture(move) and gives_check(move):
                    score += 500_000

                history_key = (board.turn, move.from_square, move.to_square)
                history_value = history_scores.get(history_key, 0.0)
                if history_value:
                    score += int(history_value)

                return score

            moves.sort(key=score_move, reverse=True)
            return moves
        finally:
            if timing is not None:
                timing["order_moves"] += time.perf_counter_ns() - start_ns

    def _record_killer(self, ply: int, move: chess.Move) -> None:
        if ply >= len(self._killer_moves):
            return
        killers = self._killer_moves[ply]
        if move in killers:
            return
        killers.insert(0, move)
        while len(killers) > self._killer_slots:
            killers.pop()
        stats = self._decision_stats
        if stats is not None:
            stats["killer_updates"] += 1

    def _record_history(self, color: bool, move: chess.Move, depth: int) -> None:
        key = (color, move.from_square, move.to_square)
        self._history_scores[key] += depth * depth
        if self._history_scores[key] > 500000:
            self._history_scores[key] *= 0.5
        stats = self._decision_stats
        if stats is not None:
            stats["history_updates"] += 1

    def _decay_history_scores(self, factor: float = 0.9) -> None:
        if not self._history_scores:
            return
        keys = list(self._history_scores.keys())
        for key in keys:
            new_value = self._history_scores[key] * factor
            if new_value < 1.0:
                del self._history_scores[key]
            else:
                self._history_scores[key] = new_value

    def _store_transposition_entry(
        self,
        key: str,
        depth: int,
        value: float,
        flag: TranspositionFlag,
        move: Optional[chess.Move],
        static_eval: Optional[float] = None,
    ) -> None:
        existing = self._transposition_table.get(key)
        if existing and existing.depth > depth:
            return
        if static_eval is None and existing:
            static_eval = existing.static_eval
        self._transposition_table[key] = TranspositionEntry(
            depth,
            value,
            flag,
            move,
            static_eval,
        )
        self._transposition_table.move_to_end(key, last=True)
        while len(self._transposition_table) > self._transposition_table_limit:
            self._transposition_table.popitem(last=False)

    def _futility_margin(self, depth: int) -> float:
        return self._futility_base_margin * max(1, depth)

    def _null_move_reduction(self, depth: int) -> int:
        return self._null_move_reduction_base + max(0, depth // max(1, self._null_move_depth_scale))

    def _has_sufficient_material(self, board: chess.Board) -> bool:
        minor_or_rook_present = False
        pawn_counts = {chess.WHITE: 0, chess.BLACK: 0}
        queen_counts = {chess.WHITE: 0, chess.BLACK: 0}

        for piece in board.piece_map().values():
            if piece.piece_type in (chess.BISHOP, chess.KNIGHT, chess.ROOK):
                minor_or_rook_present = True
            elif piece.piece_type == chess.PAWN:
                pawn_counts[piece.color] += 1
            elif piece.piece_type == chess.QUEEN:
                queen_counts[piece.color] += 1

        if minor_or_rook_present:
            return True

        total_pawns = pawn_counts[chess.WHITE] + pawn_counts[chess.BLACK]
        total_queens = queen_counts[chess.WHITE] + queen_counts[chess.BLACK]

        if total_queens:
            if queen_counts[chess.WHITE] != queen_counts[chess.BLACK]:
                return True
            return total_pawns >= 3

        return total_pawns >= 3

    def _late_move_reduction(self, depth: int, move_index: int) -> int:
        depth_term = math.log(max(depth, 2), 2)
        move_term = math.log(move_index + 1, 2)
        return max(1, int((depth_term * move_term) / 1.5))

    def _extract_principal_variation(
        self, board: chess.Board, depth: int
    ) -> List[chess.Move]:
        pv: List[chess.Move] = []
        if depth <= 0:
            return pv
        probe_board = board.copy(stack=False)
        for _ in range(depth):
            probe_key = self._position_key(probe_board)
            entry = self._transposition_table.get(probe_key)
            if not entry or entry.move is None:
                break
            pv.append(entry.move)
            probe_board.push(entry.move)
        return pv

    def _time_exceeded(self) -> bool:
        if self._search_deadline is None:
            return False
        if self._deadline_reached:
            return True
        self._time_check_counter += 1
        if self._time_check_counter < self._time_check_interval:
            return False
        self._time_check_counter = 0
        if time.perf_counter() >= self._search_deadline:
            self._deadline_reached = True
            return True
        return False

    def _repetition_penalty_value(self, board: chess.Board) -> float:
        if not self._avoid_repetition:
            return 0.0

        penalty = 0.0

        if board.is_fivefold_repetition():
            return float(self._mate_score)

        if board.can_claim_fifty_moves():
            penalty = max(penalty, self._repetition_strong_penalty)

        try:
            if board.is_repetition():
                penalty = max(penalty, self._repetition_strong_penalty)
            elif board.can_claim_threefold_repetition():
                penalty = max(penalty, self._repetition_strong_penalty * 0.8)
        except ValueError:
            penalty = max(penalty, self._repetition_strong_penalty * 0.8)

        try:
            if board.is_repetition(2):
                penalty = max(penalty, self._repetition_penalty)
        except ValueError:
            penalty = max(penalty, self._repetition_penalty)

        return penalty

    def _evaluate_board(self, board: chess.Board) -> float:
        timing = self._timing_totals
        start_ns = time.perf_counter_ns() if timing is not None else 0
        try:
            key = self._position_key(board)
            cached = self._eval_cache.get(key)
            if cached is not None:
                stats = self._decision_stats
                if stats is not None:
                    stats["eval_cache_hits"] += 1
                return cached
            stats = self._decision_stats
            if stats is not None:
                stats["eval_cache_misses"] += 1

            material = {chess.WHITE: 0.0, chess.BLACK: 0.0}
            pst = {chess.WHITE: 0.0, chess.BLACK: 0.0}
            bishops = {chess.WHITE: 0, chess.BLACK: 0}

            for square, piece in board.piece_map().items():
                value = EVAL_PIECE_VALUES.get(piece.piece_type, 0)
                table = PIECE_SQUARE_TABLES.get(piece.piece_type)
                color = piece.color
                material[color] += value
                if table:
                    index = square if color == chess.WHITE else chess.square_mirror(square)
                    pst[color] += table[index]
                if piece.piece_type == chess.BISHOP:
                    bishops[color] += 1

            score = (material[chess.WHITE] - material[chess.BLACK]) + (
                pst[chess.WHITE] - pst[chess.BLACK]
            )

            if bishops[chess.WHITE] >= 2:
                score += self.bishop_pair_bonus
            if bishops[chess.BLACK] >= 2:
                score -= self.bishop_pair_bonus

            # Passed pawns
            for color in (chess.WHITE, chess.BLACK):
                sign = 1 if color == chess.WHITE else -1
                enemy_pawns = set(board.pieces(chess.PAWN, not color))
                for pawn_square in board.pieces(chess.PAWN, color):
                    file_index = chess.square_file(pawn_square)
                    rank = chess.square_rank(pawn_square)
                    direction = 1 if color == chess.WHITE else -1
                    passed = True
                    for file_delta in (-1, 0, 1):
                        nf = file_index + file_delta
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
                        bonus = self._passed_pawn_base_bonus + self._passed_pawn_rank_bonus * max(0, advancement)
                        score += sign * bonus

            # Mobility
            current_mobility = board.legal_moves.count()
            try:
                board.push(chess.Move.null())
                opponent_mobility = board.legal_moves.count()
                board.pop()
            except ValueError:
                opponent_mobility = current_mobility
            score += (current_mobility - opponent_mobility) * self._mobility_unit

            # King safety
            phase = self._game_phase(board)
            opening_weight = phase
            endgame_weight = 1.0 - phase
            for color in (chess.WHITE, chess.BLACK):
                sign = 1 if color == chess.WHITE else -1
                king_square = board.king(color)
                if king_square is None:
                    continue
                attackers = len(board.attackers(not color, king_square))
                ring = 0
                king_file = chess.square_file(king_square)
                king_rank = chess.square_rank(king_square)
                for file_delta in (-1, 0, 1):
                    for rank_delta in (-1, 0, 1):
                        if file_delta == 0 and rank_delta == 0:
                            continue
                        nf = king_file + file_delta
                        nr = king_rank + rank_delta
                        if 0 <= nf < 8 and 0 <= nr < 8:
                            neighbour = chess.square(nf, nr)
                            piece = board.piece_at(neighbour)
                            if piece and piece.color == color:
                                ring += 1
                opening_term = opening_weight * (
                    -self._king_safety_opening_penalty * attackers + 2.0 * ring
                )
                central_distance = abs(king_file - 3.5) + abs(king_rank - 3.5)
                activity = self._king_safety_endgame_bonus * (3.5 - central_distance)
                endgame_term = endgame_weight * activity
                score += sign * (opening_term + endgame_term)

            result = score if board.turn == chess.WHITE else -score
            self._eval_cache[key] = result
            if stats is not None:
                stats["eval_cache_store"] += 1
            return result
        finally:
            if timing is not None:
                timing["evaluate"] += time.perf_counter_ns() - start_ns

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
        max_phase = sum(
            phase_values[piece] * initial_counts[piece] for piece in phase_values
        )
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
    def __init__(
        self, random_move_provider: Callable[[chess.Board], Optional[str]], **kwargs
    ):
        super().__init__(priority=0, **kwargs)
        self._random_move_provider = random_move_provider

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(
        self, board: chess.Board, context: StrategyContext
    ) -> Optional[StrategyResult]:
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

    def __init__(self) -> None:
        """Initialize the chess engine with default settings and state."""
        # Engine identification
        self.engine_name = "JaskFish"
        self.engine_author = "Jaskaran Singh"

        # Engine state
        self.board = chess.Board()
        self.debug = True
        self.move_calculating = False
        self.running = True

        # Lock to manage concurrent access to engine state
        self.state_lock = threading.Lock()

        # Strategy management
        self.strategy_selector = StrategySelector(
            logger=self._log_debug,
            selection_policy=StrategySelector.priority_score_selection_policy,
        )
        self._register_default_strategies()

        # Dispatch table mapping commands to handler methods
        self.dispatch_table = {
            "quit": self.handle_quit,
            "debug": self.handle_debug,
            "isready": self.handle_isready,
            "position": self.handle_position,
            "boardpos": self.handle_boardpos,
            "go": self.handle_go,
            "ucinewgame": self.handle_ucinewgame,
            "uci": self.handle_uci,
        }

    def _log_debug(self, message: str) -> None:
        if not self.debug:
            return
        for line in message.splitlines():
            print(f"info string {line}")

    def _register_default_strategies(self) -> None:
        strategies_to_register: List[MoveStrategy] = []

        if STRATEGY_ENABLE_FLAGS.get("mate_in_one", True):
            strategies_to_register.append(
                MateInOneStrategy(
                    name="MateInOneStrategy",
                    logger=self._log_debug,
                )
            )

        repetition_enabled = STRATEGY_ENABLE_FLAGS.get("repetition_avoidance", False)

        if STRATEGY_ENABLE_FLAGS.get("heuristic", True):
            strategies_to_register.append(
                HeuristicSearchStrategy(
                    name="HeuristicSearchStrategy",
                    search_depth=5,
                    logger=self._log_debug,
                    avoid_repetition=repetition_enabled,
                )
            )

        if STRATEGY_ENABLE_FLAGS.get("fallback_random", False):
            strategies_to_register.append(
                FallbackRandomStrategy(
                    self.random_move, name="FallbackRandomStrategy"
                )
            )

        for strategy in strategies_to_register:
            self.strategy_selector.register_strategy(strategy)

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
        in_check = board.is_check()
        mate_threat = self.detect_opponent_mate_in_one_threat(board)

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
            in_check=in_check,
            opponent_mate_in_one_threat=mate_threat,
        )

    def detect_opponent_mate_in_one_threat(self, board: chess.Board) -> bool:
        board_copy = board.copy()
        try:
            board_copy.push(chess.Move.null())
        except ValueError:
            return False

        for move in board_copy.legal_moves:
            board_copy.push(move)
            try:
                if board_copy.is_checkmate():
                    return True
            finally:
                board_copy.pop()
        return False

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
        _ensure_line_buffered_stdout()
        self.handle_uci()
        self.command_processor()

    def handle_uci(self, args=None):
        print(f"id name {self.engine_name}")
        print(f"id author {self.engine_author}")
        print("uciok")

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
                parts = command.split(" ", 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                # Dispatch the command to the appropriate handler
                handler = self.dispatch_table.get(cmd, self.handle_unknown)
                handler(args)
            except Exception as e:
                print(f"info string Error processing command: {e}")
            finally:
                sys.stdout.flush()

    def handle_unknown(self, args):
        print(f"unknown command received: '{args}'")

    def handle_quit(self, args):
        print("info string Engine shutting down")
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
            print(
                f"info string Position: {self.board.fen()}"
                if self.board
                else "info string Board state not set"
            )

    def _parse_go_args(self, args: str) -> Dict[str, int]:
        tokens = args.split()
        if not tokens:
            return {}

        parsed: Dict[str, int] = {}
        iterator = iter(tokens)
        for token in iterator:
            key = token.lower()
            if key in {
                "wtime",
                "btime",
                "winc",
                "binc",
                "movestogo",
                "movetime",
                "depth",
            }:
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
                print("info string Please wait for computer move")
                return
            self.move_calculating = True

        # Start the move calculation in a separate thread
        move_thread = threading.Thread(
            target=self.process_go_command, args=(time_controls,)
        )
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
                context = self.create_strategy_context(
                    board_snapshot, time_controls=time_controls
                )

            if self.debug:
                self._log_debug(
                    "context prepared: "
                    f"fullmove={context.fullmove_number}, "
                    f"halfmove={context.halfmove_clock}, "
                    f"pieces={context.piece_count}, "
                    f"material={context.material_imbalance}"
                )

            selection_elapsed = 0.0
            selection_start = time.perf_counter()
            result = (
                self.strategy_selector.select_move(board_snapshot, context)
                if self.strategy_selector
                else None
            )
            selection_elapsed = time.perf_counter() - selection_start

            metadata: Dict[str, Any] = {}
            if result and result.metadata:
                metadata = result.metadata

            move = None
            if result and result.move:
                move = result.move
                display_name = metadata.get("label") or result.strategy_name
                print(
                    f"info string strategy {display_name} selected move {result.move}"
                )
            else:
                if self.debug:
                    self._log_debug("no strategy produced a move")

            nodes = metadata.get("nodes") if metadata else None
            search_time = metadata.get("time") if metadata else None
            depth = metadata.get("depth") if metadata else None
            timing_breakdown = metadata.get("timing") if metadata else None

            if search_time is not None and nodes is not None:
                nps_total = int(nodes / search_time) if search_time else 0
                depth_display = depth if depth is not None else "-"
                perf_segments = [
                    f"info string perf move depth={depth_display}",
                    f"nodes={nodes}",
                    f"time={search_time:.3f}s",
                    f"nps={nps_total}",
                    f"select={selection_elapsed:.3f}s",
                ]
                if isinstance(timing_breakdown, dict):
                    ordered_keys = [
                        "alpha_beta",
                        "order_moves",
                        "evaluate",
                        "quiescence",
                    ]
                    timer_parts = [
                        f"{key.split('_')[0]}={timing_breakdown[key]:.3f}s"
                        for key in ordered_keys
                        if key in timing_breakdown
                    ]
                    if timer_parts:
                        perf_segments.append("timers=" + ",".join(timer_parts))
                print(" ".join(perf_segments))
            else:
                print(f"info string perf select={selection_elapsed:.3f}s")

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


if __name__ == "__main__":
    engine = ChessEngine()
    engine.start()
