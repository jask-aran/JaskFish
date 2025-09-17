"""Search algorithms for the JaskFish engine.

This module provides an :class:`AlphaBetaSearcher` that implements a
fairly traditional alpha-beta search with enhancements such as
transposition tables, killer move heuristics, history heuristics and
recently added forward pruning and extension techniques.

The implementation is purposely self contained so that it can be used by
unit tests without needing the remainder of the engine infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess


MATE_VALUE = 100_000


@dataclass
class OrderedMove:
    """Container representing an ordered move and its metadata."""

    move: chess.Move
    is_capture: bool
    is_killer: bool
    is_promotion: bool
    score: int

    @property
    def is_quiet(self) -> bool:
        """Return ``True`` if the move is a quiet move."""

        return not self.is_capture and not self.is_promotion


@dataclass
class TTEntry:
    value: int
    depth: int
    flag: int
    move: Optional[chess.Move]


TT_EXACT = 0
TT_ALPHA = 1
TT_BETA = 2


class AlphaBetaSearcher:
    """Alpha-beta searcher supporting a number of common optimisations.

    Parameters
    ----------
    board:
        Optional initial :class:`chess.Board`. If omitted a fresh board is
        created.
    check_extension:
        Additional depth (in plies) granted to checking moves. This plays
        together with the late-move reduction logic by extending the depth
        *before* any reduction is applied.
    null_move_reduction:
        Number of plies to reduce when performing a null move search.
    null_move_min_depth:
        Minimum depth (plies) required before a null move is attempted.
    lmr_min_depth:
        Minimum depth at which late-move reductions may trigger.
    """

    PIECE_VALUES: Dict[int, int] = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20_000,
    }

    def __init__(
        self,
        board: Optional[chess.Board] = None,
        *,
        check_extension: int = 1,
        null_move_reduction: int = 2,
        null_move_min_depth: int = 3,
        lmr_min_depth: int = 3,
    ) -> None:
        self.board = board or chess.Board()
        self.check_extension = check_extension
        self.null_move_reduction = null_move_reduction
        self.null_move_min_depth = null_move_min_depth
        self.lmr_min_depth = lmr_min_depth

        self.transposition_table: Dict[int, TTEntry] = {}
        self.killer_moves: Dict[int, List[Optional[chess.Move]]] = {}
        self.history: Dict[Tuple[bool, int, int], int] = {}
        self.nodes = 0
        self.max_depth = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def search(self, depth: int) -> Tuple[int, Optional[chess.Move]]:
        """Iterative deepening driver returning the best move and score."""

        self.nodes = 0
        self.max_depth = max(depth, 0)
        best_move: Optional[chess.Move] = None
        best_score = -MATE_VALUE

        for current_depth in range(1, depth + 1):
            score, move = self._search_root(current_depth)
            if move is not None:
                best_move = move
                best_score = score
        return best_score, best_move

    # ------------------------------------------------------------------
    # Core search methods
    # ------------------------------------------------------------------
    def _search_root(self, depth: int) -> Tuple[int, Optional[chess.Move]]:
        alpha = -MATE_VALUE
        beta = MATE_VALUE
        best_move: Optional[chess.Move] = None
        best_score = -MATE_VALUE
        ply = self.board.ply()

        tt_entry = self._probe_transposition()
        tt_move = tt_entry.move if tt_entry is not None else None

        for index, ordered in enumerate(self._order_moves(depth, ply, tt_move), 1):
            move = ordered.move
            self.board.push(move)
            extension = self._check_extension()
            new_depth = depth - 1 + extension
            score = -self._alphabeta(new_depth, -beta, -alpha, is_pv_node=(index == 1))
            self.board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

        if best_move is not None:
            self._store_transposition(depth, best_score, TT_EXACT, best_move, alpha, beta)
        return best_score, best_move

    def _alphabeta(
        self,
        depth: int,
        alpha: int,
        beta: int,
        *,
        is_pv_node: bool,
        allow_null: bool = True,
    ) -> int:
        """Perform an alpha-beta search with various pruning techniques."""

        self.nodes += 1
        alpha_original = alpha

        # Terminal conditions -------------------------------------------------
        if self.board.is_repetition(3) or self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0

        if self.board.is_checkmate():
            # Negative score because the side to move has been checkmated.
            return -MATE_VALUE + self.board.ply()

        if depth <= 0:
            return self._quiescence(alpha, beta)

        in_check = self.board.is_check()

        # Transposition table lookup -----------------------------------------
        tt_entry = self._probe_transposition()
        tt_move = None
        if tt_entry is not None and tt_entry.depth >= depth:
            if tt_entry.flag == TT_EXACT:
                return tt_entry.value
            if tt_entry.flag == TT_ALPHA and tt_entry.value <= alpha:
                return tt_entry.value
            if tt_entry.flag == TT_BETA and tt_entry.value >= beta:
                return tt_entry.value
            tt_move = tt_entry.move

        # Null move pruning ---------------------------------------------------
        if (
            allow_null
            and not is_pv_node
            and not in_check
            and depth >= self.null_move_min_depth
            and self.board.legal_moves.count() > 0
        ):
            reduction = self.null_move_reduction
            null_depth = depth - 1 - reduction
            if null_depth > 0:
                self.board.push(chess.Move.null())
                score = -self._alphabeta(null_depth, -beta, -beta + 1, is_pv_node=False, allow_null=False)
                self.board.pop()
                if score >= beta:
                    return score

        best_move: Optional[chess.Move] = None
        best_score = -MATE_VALUE
        move_made = False
        ply = self.board.ply()

        ordered_moves = self._order_moves(depth, ply, tt_move)
        for move_index, ordered in enumerate(ordered_moves, 1):
            move = ordered.move
            color_to_move = self.board.turn
            move_made = True

            self.board.push(move)
            gives_check = self.board.is_check()
            extension = self._check_extension(gives_check=gives_check)
            new_depth = depth - 1 + extension
            if new_depth < 0:
                new_depth = 0

            next_is_pv = is_pv_node and move_index == 1

            apply_reduction = (
                new_depth > 0
                and depth >= self.lmr_min_depth
                and move_index > 1
                and ordered.is_quiet
                and not ordered.is_killer
                and not gives_check
                and not in_check
                and not next_is_pv
            )

            if apply_reduction:
                reduction = self._calculate_lmr(depth, move_index)
                reduced_depth = max(0, new_depth - reduction)
                score = -self._alphabeta(
                    reduced_depth,
                    -alpha - 1,
                    -alpha,
                    is_pv_node=False,
                )
                if score > alpha:
                    score = -self._alphabeta(
                        new_depth,
                        -beta,
                        -alpha,
                        is_pv_node=next_is_pv,
                    )
            else:
                score = -self._alphabeta(
                    new_depth,
                    -beta,
                    -alpha,
                    is_pv_node=next_is_pv,
                )

            self.board.pop()

            if score >= beta:
                if ordered.is_quiet:
                    self._store_killer(ply, move)
                    self._update_history(color_to_move, move, depth)
                self._store_transposition(depth, score, TT_BETA, move, alpha_original, beta)
                return score

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score
                if ordered.is_quiet:
                    self._update_history(color_to_move, move, depth)

        if not move_made:
            # No legal moves. This was already covered by checkmate earlier.
            return 0

        flag = TT_EXACT
        if best_score <= alpha_original:
            flag = TT_ALPHA
        elif best_score >= beta:
            flag = TT_BETA
        if best_move is not None:
            self._store_transposition(depth, best_score, flag, best_move, alpha_original, beta)
        return best_score

    # ------------------------------------------------------------------
    # Move ordering helpers
    # ------------------------------------------------------------------
    def _order_moves(
        self, depth: int, ply: int, tt_move: Optional[chess.Move]
    ) -> List[OrderedMove]:
        """Return ordered moves enriched with metadata.

        The caller will use the ordering index to decide when late move
        reductions should apply. The killer and history heuristics only
        influence quiet moves.
        """

        killers = self.killer_moves.get(ply, [None, None])
        ordered: List[OrderedMove] = []
        legal_moves = list(self.board.legal_moves)
        for move in legal_moves:
            is_capture = self.board.is_capture(move)
            is_promotion = move.promotion is not None
            is_killer = not is_capture and move in killers if killers else False

            score = 0
            if tt_move is not None and move == tt_move:
                score += 1_000_000
            if is_capture:
                captured = self.board.piece_type_at(move.to_square)
                attacker = self.board.piece_type_at(move.from_square)
                score += 500_000
                if captured is not None:
                    score += self.PIECE_VALUES.get(captured, 0)
                if attacker is not None:
                    score -= self.PIECE_VALUES.get(attacker, 0) // 10
            elif is_killer:
                score += 400_000
            elif is_promotion:
                score += 300_000 + self.PIECE_VALUES.get(move.promotion, 0)
            else:
                history_score = self.history.get((self.board.turn, move.from_square, move.to_square), 0)
                score += history_score

            ordered.append(OrderedMove(move, is_capture, bool(is_killer), is_promotion, score))

        ordered.sort(key=lambda item: item.score, reverse=True)
        return ordered

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def _evaluate(self) -> int:
        score = 0
        for piece_type, value in self.PIECE_VALUES.items():
            score += value * len(self.board.pieces(piece_type, chess.WHITE))
            score -= value * len(self.board.pieces(piece_type, chess.BLACK))

        if self.board.turn == chess.WHITE:
            return score
        return -score

    def _quiescence(self, alpha: int, beta: int) -> int:
        stand_pat = self._evaluate()
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        for move in self.board.legal_moves:
            if not self.board.is_capture(move):
                continue
            self.board.push(move)
            score = -self._quiescence(-beta, -alpha)
            self.board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _check_extension(self, gives_check: Optional[bool] = None) -> int:
        if self.check_extension <= 0:
            return 0
        if gives_check is None:
            gives_check = self.board.is_check()
        return self.check_extension if gives_check else 0

    def _calculate_lmr(self, depth: int, move_index: int) -> int:
        reduction = 1
        if depth > self.lmr_min_depth + 1 and move_index > 6:
            reduction += 1
        return reduction

    def _update_history(self, color: bool, move: chess.Move, depth: int) -> None:
        key = (color, move.from_square, move.to_square)
        self.history[key] = self.history.get(key, 0) + depth * depth

    def _store_killer(self, ply: int, move: chess.Move) -> None:
        killers = self.killer_moves.setdefault(ply, [None, None])
        if move == killers[0]:
            return
        killers[1] = killers[0]
        killers[0] = move

    def _probe_transposition(self) -> Optional[TTEntry]:
        return self.transposition_table.get(self.board.zobrist_hash())

    def _store_transposition(
        self,
        depth: int,
        value: int,
        flag: int,
        move: Optional[chess.Move],
        alpha: int,
        beta: int,
    ) -> None:
        key = self.board.zobrist_hash()
        entry = TTEntry(value=value, depth=depth, flag=flag, move=move)
        self.transposition_table[key] = entry


__all__ = ["AlphaBetaSearcher", "OrderedMove"]
