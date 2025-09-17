"""Search and evaluation utilities for the JaskFish engine."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import chess


@dataclass
class SearchInfo:
    """Container describing the outcome of a search iteration."""

    move: Optional[chess.Move]
    score: float
    depth: int
    nodes: int
    pv: List[chess.Move]
    completed: bool


@dataclass
class TranspositionEntry:
    depth: int
    score: float
    flag: str  # "exact", "lower", "upper"
    move: Optional[chess.Move]


class Evaluation:
    """Static evaluation heuristics blending material and positional knowledge."""

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    # Piece-square tables roughly inspired by simplified evaluations. Values are
    # encoded from White's perspective; lookups are mirrored for Black.
    PAWN_TABLE = [
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
        -20,
        -20,
        10,
        10,
        5,
        5,
        -5,
        -10,
        0,
        0,
        -10,
        -5,
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
        5,
        10,
        25,
        25,
        10,
        5,
        5,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    KNIGHT_TABLE = [
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
        5,
        5,
        0,
        -20,
        -40,
        -30,
        5,
        10,
        15,
        15,
        10,
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
        15,
        20,
        20,
        15,
        5,
        -30,
        -30,
        0,
        10,
        15,
        15,
        10,
        0,
        -30,
        -40,
        -20,
        0,
        0,
        0,
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
    ]

    BISHOP_TABLE = [
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
    ]

    ROOK_TABLE = [
        0,
        0,
        5,
        10,
        10,
        5,
        0,
        0,
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
        5,
        10,
        10,
        10,
        10,
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
    ]

    QUEEN_TABLE = [
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
        5,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        5,
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
    ]

    KING_TABLE_MID = [
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
        -40,
        -40,
        -50,
        -50,
        -40,
        -40,
        -30,
        -30,
        -35,
        -35,
        -40,
        -40,
        -35,
        -35,
        -30,
        -20,
        -30,
        -30,
        -35,
        -35,
        -30,
        -30,
        -20,
        -10,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
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
    ]

    KING_TABLE_END = [
        -50,
        -40,
        -30,
        -20,
        -20,
        -30,
        -40,
        -50,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -30,
        -10,
        20,
        30,
        30,
        20,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
        -30,
        -10,
        40,
        50,
        50,
        40,
        -10,
        -30,
        -30,
        -10,
        30,
        40,
        40,
        30,
        -10,
        -30,
        -30,
        -20,
        -10,
        0,
        0,
        -10,
        -20,
        -30,
        -50,
        -40,
        -30,
        -20,
        -20,
        -30,
        -40,
        -50,
    ]

    PST = {
        chess.PAWN: PAWN_TABLE,
        chess.KNIGHT: KNIGHT_TABLE,
        chess.BISHOP: BISHOP_TABLE,
        chess.ROOK: ROOK_TABLE,
        chess.QUEEN: QUEEN_TABLE,
    }

    def __init__(self) -> None:
        self.cached_board_fen: Optional[str] = None
        self.cached_score: Optional[int] = None

    @staticmethod
    def _mirror_index(index: int) -> int:
        rank = index // 8
        file = index % 8
        mirrored_rank = 7 - rank
        return mirrored_rank * 8 + file

    def _piece_square_bonus(self, piece: chess.Piece, square: int) -> int:
        table = self.PST.get(piece.piece_type)
        if not table:
            return 0
        index = square
        if piece.color == chess.BLACK:
            index = self._mirror_index(index)
        return table[index]

    def _king_table(self, board: chess.Board) -> List[int]:
        # Rough heuristic: treat endgames as positions with little material.
        minor_count = sum(
            len(board.pieces(piece_type, chess.WHITE)) + len(board.pieces(piece_type, chess.BLACK))
            for piece_type in (chess.BISHOP, chess.KNIGHT)
        )
        rook_count = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        queen_count = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        total_non_pawn = minor_count + rook_count + queen_count
        return self.KING_TABLE_END if total_non_pawn <= 4 else self.KING_TABLE_MID

    def evaluate(self, board: chess.Board) -> int:
        """Return a centipawn score (positive for White)."""
        fen = board.board_fen() + (" w" if board.turn == chess.WHITE else " b")
        if fen == self.cached_board_fen and self.cached_score is not None:
            return self.cached_score

        score = 0
        king_table = self._king_table(board)

        for square, piece in board.piece_map().items():
            value = self.PIECE_VALUES[piece.piece_type]
            square_bonus = 0
            if piece.piece_type == chess.KING:
                index = square if piece.color == chess.WHITE else self._mirror_index(square)
                square_bonus = king_table[index]
            else:
                square_bonus = self._piece_square_bonus(piece, square)
            contribution = value + square_bonus
            if piece.color == chess.WHITE:
                score += contribution
            else:
                score -= contribution

        # Mobility bonus: encourage side with more legal moves.
        board_turn = board.turn
        board.turn = chess.WHITE
        white_mobility = board.legal_moves.count()
        board.turn = chess.BLACK
        black_mobility = board.legal_moves.count()
        board.turn = board_turn
        score += 5 * (white_mobility - black_mobility)

        # Pawn structure: simple doubled/isolated penalties.
        score += self._pawn_structure_score(board)

        self.cached_board_fen = fen
        self.cached_score = score
        return score

    def _pawn_structure_score(self, board: chess.Board) -> int:
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            pawns = board.pieces(chess.PAWN, color)
            files = {}
            for square in pawns:
                file_index = chess.square_file(square)
                files.setdefault(file_index, []).append(square)
            for file_squares in files.values():
                if len(file_squares) > 1:
                    penalty = 15 * (len(file_squares) - 1)
                    score += penalty if color == chess.BLACK else -penalty
            for square in pawns:
                file_index = chess.square_file(square)
                adjacent_files = {file_index - 1, file_index + 1}
                has_support = any(
                    chess.square_file(pawn_square) in adjacent_files for pawn_square in pawns if pawn_square != square
                )
                if not has_support:
                    penalty = 10
                    score += penalty if color == chess.BLACK else -penalty
        return score


class AlphaBetaSearcher:
    """Iterative deepening alpha-beta search with basic heuristics."""

    MATE_VALUE = 100_000

    def __init__(
        self,
        board: chess.Board,
        max_depth: int = 4,
        time_limit: Optional[float] = None,
        evaluator: Optional[Evaluation] = None,
        debug: bool = False,
    ) -> None:
        self.board = board.copy(stack=True)
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.evaluator = evaluator or Evaluation()
        self.debug = debug
        self.root_color = self.board.turn
        self.nodes = 0
        self.best_move: Optional[chess.Move] = None
        self.best_score: float = -math.inf
        self.best_pv: List[chess.Move] = []
        self.completed_depth = 0
        self.start_time = time.time()
        self.stop = False
        self.tt: Dict[int, TranspositionEntry] = {}
        self.killer_moves: Dict[int, List[chess.Move]] = {}
        self.history: Dict[Tuple[bool, int, int], int] = {}

    # Public API ---------------------------------------------------------
    def search(self) -> SearchInfo:
        self.start_time = time.time()
        self.stop = False
        for depth in range(1, self.max_depth + 1):
            if self._should_stop():
                break
            score, pv, completed = self._search_depth(depth)
            if not completed:
                break
            if pv:
                self.best_move = pv[0]
            self.best_score = score
            self.best_pv = pv
            self.completed_depth = depth
            if self.debug:
                line = " ".join(move.uci() for move in pv)
                print(f"info string depth {depth} score {score} nodes {self.nodes} pv {line}")
        if not self.best_move:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                self.best_move = legal_moves[0]
                self.best_pv = [self.best_move]
                self.best_score = self._evaluate_terminal()
        return SearchInfo(
            move=self.best_move,
            score=self.best_score,
            depth=self.completed_depth,
            nodes=self.nodes,
            pv=self.best_pv,
            completed=self.completed_depth > 0,
        )

    def stop_search(self) -> None:
        self.stop = True

    # Internal helpers ---------------------------------------------------
    def _search_depth(self, depth: int) -> Tuple[float, List[chess.Move], bool]:
        try:
            score, pv = self._alphabeta(depth, -math.inf, math.inf, True, 0)
            return score, pv, not self.stop
        except TimeoutError:
            return self.best_score, self.best_pv, False

    def _alphabeta(
        self,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        ply: int,
    ) -> Tuple[float, List[chess.Move]]:
        if self._should_stop():
            raise TimeoutError()

        self.nodes += 1

        if self.board.is_repetition(3) or self.board.is_fifty_moves():
            return 0, []

        if depth == 0:
            score = self._quiescence(alpha, beta, maximizing, ply)
            return score, []

        zobrist = self.board.zobrist_hash()
        tt_entry = self.tt.get(zobrist)
        alpha_orig, beta_orig = alpha, beta
        tt_move = tt_entry.move if tt_entry else None
        if tt_entry and tt_entry.depth >= depth:
            if tt_entry.flag == "exact":
                if tt_entry.move is not None:
                    return tt_entry.score, [tt_entry.move]
                return tt_entry.score, []
            if tt_entry.flag == "lower":
                alpha = max(alpha, tt_entry.score)
            elif tt_entry.flag == "upper":
                beta = min(beta, tt_entry.score)
            if alpha >= beta:
                return tt_entry.score, [tt_entry.move] if tt_entry.move else []

        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            if self.board.is_check():
                mate_score = self._mate_score(ply)
                return (mate_score if not maximizing else -mate_score), []
            return 0, []

        best_move: Optional[chess.Move] = None
        best_score = -math.inf if maximizing else math.inf
        best_pv: List[chess.Move] = []

        ordered_moves = self._order_moves(legal_moves, tt_move, ply)

        for move in ordered_moves:
            self.board.push(move)
            score, child_pv = self._alphabeta(depth - 1, alpha, beta, not maximizing, ply + 1)
            self.board.pop()

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_pv = [move] + child_pv
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    self._record_killer(move, ply)
                    break
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                    best_pv = [move] + child_pv
                beta = min(beta, best_score)
                if beta <= alpha:
                    self._record_killer(move, ply)
                    break

        if best_move is None:
            # All moves lead to timeout or something unexpected; fall back.
            return self._evaluate_terminal(), []

        flag = "exact"
        if best_score <= alpha_orig:
            flag = "upper"
        elif best_score >= beta_orig:
            flag = "lower"
        self.tt[zobrist] = TranspositionEntry(depth, best_score, flag, best_move)

        return best_score, best_pv

    def _quiescence(self, alpha: float, beta: float, maximizing: bool, ply: int) -> float:
        if self._should_stop():
            raise TimeoutError()

        stand_pat = self._evaluate_terminal()
        if maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat

        capture_moves = [move for move in self.board.legal_moves if self.board.is_capture(move)]
        capture_moves = self._order_moves(capture_moves, None, ply)

        for move in capture_moves:
            self.board.push(move)
            score = self._quiescence(alpha, beta, not maximizing, ply + 1)
            self.board.pop()

            if maximizing:
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    return alpha
            else:
                if score < beta:
                    beta = score
                if beta <= alpha:
                    return beta

        return alpha if maximizing else beta

    def _order_moves(
        self,
        moves: List[chess.Move],
        tt_move: Optional[chess.Move],
        ply: int,
    ) -> List[chess.Move]:
        killer_moves = self.killer_moves.get(ply, [])

        def score(move: chess.Move) -> int:
            if tt_move and move == tt_move:
                return 10_000
            s = 0
            if move in killer_moves:
                s += 9_000
            if self.board.is_capture(move):
                captured_piece = self.board.piece_at(move.to_square)
                mover = self.board.piece_at(move.from_square)
                captured_value = 0
                if captured_piece:
                    captured_value = self.evaluator.PIECE_VALUES[captured_piece.piece_type]
                elif self.board.is_en_passant(move):
                    captured_value = self.evaluator.PIECE_VALUES[chess.PAWN]
                mover_value = self.evaluator.PIECE_VALUES[mover.piece_type] if mover else 0
                s += 5_000 + 10 * captured_value - mover_value
            if move.promotion:
                s += 4_000 + self.evaluator.PIECE_VALUES.get(move.promotion, 0)
            history_key = (self.board.turn, move.from_square, move.to_square)
            s += self.history.get(history_key, 0)
            return s

        return sorted(moves, key=score, reverse=True)

    def _record_killer(self, move: chess.Move, ply: int) -> None:
        if self.board.is_capture(move):
            return
        killers = self.killer_moves.setdefault(ply, [])
        if move in killers:
            return
        killers.insert(0, move)
        if len(killers) > 2:
            killers.pop()
        history_key = (self.board.turn, move.from_square, move.to_square)
        self.history[history_key] = self.history.get(history_key, 0) + 1

    def _evaluate_terminal(self) -> float:
        if self.board.is_checkmate():
            ply = len(self.board.move_stack)
            mate_score = self._mate_score(ply)
            if self.board.turn == self.root_color:
                return -mate_score
            return mate_score
        if self.board.is_stalemate() or self.board.is_insufficient_material() or self.board.is_seventyfive_moves():
            return 0
        score = self.evaluator.evaluate(self.board)
        return score if self.root_color == chess.WHITE else -score

    def _mate_score(self, ply: int) -> float:
        return self.MATE_VALUE - ply

    def _should_stop(self) -> bool:
        if self.stop:
            return True
        if self.time_limit is not None:
            if time.time() - self.start_time >= self.time_limit:
                self.stop = True
                return True
        return False


def select_opening_move(board: chess.Board) -> Optional[chess.Move]:
    """Very small hard-coded opening book used before search begins."""
    key = (board.board_fen(), board.turn, board.fullmove_number)
    candidates = _OPENING_BOOK.get(key)
    if not candidates:
        return None
    moves, weights = zip(*candidates)
    choice = random.choices(moves, weights=weights, k=1)[0]
    try:
        return board.parse_uci(choice)
    except ValueError:
        return None


# A deliberately tiny set of hand-crafted opening continuations.
_OPENING_BOOK: Dict[Tuple[str, bool, int], List[Tuple[str, int]]] = {
    (chess.STARTING_BOARD_FEN, True, 1): [("e2e4", 4), ("d2d4", 3), ("c2c4", 1), ("g1f3", 2)],
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR", False, 1): [("c7c5", 4), ("e7e5", 4), ("e7e6", 2)],
    ("rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR", True, 2): [("d2d4", 5), ("g1f3", 3)],
}
