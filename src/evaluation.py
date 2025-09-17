"""Evaluation heuristics for the JaskFish chess engine.

This module provides a tapered-evaluation implementation that scores a
``python-chess`` board.  The heuristic tries to stay deliberately small yet
expressive enough for the unit tests that focus on classical chess knowledge
such as passed pawns, bishop pairs, rook placement and king safety.
"""

from __future__ import annotations

import chess


class Evaluation:
    """Feature-based tapered evaluation of a chess position.

    The evaluator returns a score in centipawns from White's perspective.  The
    score is a blend of middle-game and end-game terms with a linear tapering
    factor based on the remaining non-pawn material.
    """

    CHECKMATE_VALUE = 100_000

    PIECE_VALUES_MG = {
        chess.PAWN: 82,
        chess.KNIGHT: 337,
        chess.BISHOP: 365,
        chess.ROOK: 477,
        chess.QUEEN: 1025,
        chess.KING: 0,
    }

    PIECE_VALUES_EG = {
        chess.PAWN: 94,
        chess.KNIGHT: 281,
        chess.BISHOP: 297,
        chess.ROOK: 512,
        chess.QUEEN: 936,
        chess.KING: 0,
    }

    # Piece-square tables adapted from simple, publicly available heuristics.
    # They are intentionally modest – the tests target the additional features
    # added in this task rather than finely tuned PSTs.
    _PST_PAWN_MG = (
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0,
    )

    _PST_PAWN_EG = (
        0, 0, 0, 0, 0, 0, 0, 0,
        10, 10, 10, 10, 10, 10, 10, 10,
        5, 5, 10, 20, 20, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        40, 40, 40, 40, 40, 40, 40, 40,
        0, 0, 0, 0, 0, 0, 0, 0,
    )

    _PST_KNIGHT_MG = (
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    )

    _PST_KNIGHT_EG = (
        -40, -20, -10, -10, -10, -10, -20, -40,
        -20, 0, 10, 10, 10, 10, 0, -20,
        -10, 10, 15, 20, 20, 15, 10, -10,
        -10, 10, 20, 25, 25, 20, 10, -10,
        -10, 10, 20, 25, 25, 20, 10, -10,
        -10, 10, 15, 20, 20, 15, 10, -10,
        -20, 0, 10, 10, 10, 10, 0, -20,
        -40, -20, -10, -10, -10, -10, -20, -40,
    )

    _PST_BISHOP_MG = (
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    )

    _PST_BISHOP_EG = (
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    )

    _PST_ROOK_MG = (
        0, 0, 5, 10, 10, 5, 0, 0,
        -5, 0, 0, 5, 5, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    )

    _PST_ROOK_EG = (
        0, 0, 5, 10, 10, 5, 0, 0,
        -5, 0, 0, 5, 5, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    )

    _PST_QUEEN_MG = (
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    )

    _PST_QUEEN_EG = (
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    )

    _PST_KING_MG = (
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    )

    _PST_KING_EG = (
        -50, -30, -30, -30, -30, -30, -30, -50,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50,
    )

    PST_MG = {
        chess.PAWN: _PST_PAWN_MG,
        chess.KNIGHT: _PST_KNIGHT_MG,
        chess.BISHOP: _PST_BISHOP_MG,
        chess.ROOK: _PST_ROOK_MG,
        chess.QUEEN: _PST_QUEEN_MG,
        chess.KING: _PST_KING_MG,
    }

    PST_EG = {
        chess.PAWN: _PST_PAWN_EG,
        chess.KNIGHT: _PST_KNIGHT_EG,
        chess.BISHOP: _PST_BISHOP_EG,
        chess.ROOK: _PST_ROOK_EG,
        chess.QUEEN: _PST_QUEEN_EG,
        chess.KING: _PST_KING_EG,
    }

    PHASE_WEIGHTS = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
        chess.KING: 0,
    }

    MAX_PHASE = (
        PHASE_WEIGHTS[chess.KNIGHT] * 4
        + PHASE_WEIGHTS[chess.BISHOP] * 4
        + PHASE_WEIGHTS[chess.ROOK] * 4
        + PHASE_WEIGHTS[chess.QUEEN] * 2
    )

    BISHOP_PAIR_MG = 30
    BISHOP_PAIR_EG = 50

    ROOK_OPEN_FILE_MG = 20
    ROOK_OPEN_FILE_EG = 10
    ROOK_SEMI_OPEN_FILE_MG = 12
    ROOK_SEMI_OPEN_FILE_EG = 6
    ROOK_SEVENTH_RANK_MG = 18
    ROOK_SEVENTH_RANK_EG = 24

    PASSED_PAWN_MG = [0, 5, 15, 30, 50, 80, 120, 0]
    PASSED_PAWN_EG = [0, 10, 25, 45, 70, 110, 160, 0]
    PASSED_PAWN_SUPPORT_MG = 12
    PASSED_PAWN_SUPPORT_EG = 8
    CONNECTED_PASSED_MG = [0, 6, 14, 24, 40, 60, 90, 0]
    CONNECTED_PASSED_EG = [0, 12, 24, 42, 70, 100, 150, 0]

    KING_PAWN_SHIELD_MG = 12
    KING_PAWN_SHIELD_EG = 6
    KING_RING_ATTACK_MG = 6
    KING_RING_ATTACK_EG = 3
    KING_TROPISM_WEIGHTS = {
        chess.QUEEN: 4,
        chess.ROOK: 2,
        chess.BISHOP: 1,
        chess.KNIGHT: 2,
    }

    def evaluate(self, board: chess.Board) -> int:
        """Return a tapered centipawn score for ``board``.

        Positive scores favour White.  Checkmate values ignore the side to move
        because the method is predominantly used for static evaluation and not
        to drive the search directly in this kata-style project.
        """

        if board.is_checkmate():
            return -self.CHECKMATE_VALUE if board.turn == chess.WHITE else self.CHECKMATE_VALUE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        mg_score = 0
        eg_score = 0
        phase = 0

        pawn_files = {
            chess.WHITE: [0] * 8,
            chess.BLACK: [0] * 8,
        }
        for color in (chess.WHITE, chess.BLACK):
            for sq in chess.SquareSet(board.pieces(chess.PAWN, color)):
                pawn_files[color][chess.square_file(sq)] += 1

        bishop_counts = {chess.WHITE: 0, chess.BLACK: 0}
        passed_pawns: dict[bool, list[tuple[int, int]]] = {
            chess.WHITE: [],
            chess.BLACK: [],
        }
        pieces_by_type: dict[bool, dict[int, list[int]]] = {
            chess.WHITE: {pt: [] for pt in range(1, 7)},
            chess.BLACK: {pt: [] for pt in range(1, 7)},
        }

        enemy_pawn_masks = {
            chess.WHITE: board.pieces(chess.PAWN, chess.BLACK),
            chess.BLACK: board.pieces(chess.PAWN, chess.WHITE),
        }

        attack_cache: dict[bool, dict[int, int]] = {
            chess.WHITE: {},
            chess.BLACK: {},
        }

        for square, piece in board.piece_map().items():
            color = piece.color
            enemy = not color
            sign = 1 if color == chess.WHITE else -1
            piece_type = piece.piece_type

            pieces_by_type[color][piece_type].append(square)

            phase += self.PHASE_WEIGHTS.get(piece_type, 0)

            mg_score += sign * (
                self.PIECE_VALUES_MG[piece_type]
                + self._pst_value(self.PST_MG[piece_type], square, color)
            )
            eg_score += sign * (
                self.PIECE_VALUES_EG[piece_type]
                + self._pst_value(self.PST_EG[piece_type], square, color)
            )

            if piece_type == chess.BISHOP:
                bishop_counts[color] += 1
            elif piece_type == chess.ROOK:
                file_index = chess.square_file(square)
                if pawn_files[color][file_index] == 0:
                    if pawn_files[enemy][file_index] == 0:
                        mg_score += sign * self.ROOK_OPEN_FILE_MG
                        eg_score += sign * self.ROOK_OPEN_FILE_EG
                    else:
                        mg_score += sign * self.ROOK_SEMI_OPEN_FILE_MG
                        eg_score += sign * self.ROOK_SEMI_OPEN_FILE_EG
                if self._relative_rank(color, square) == 6:
                    mg_score += sign * self.ROOK_SEVENTH_RANK_MG
                    eg_score += sign * self.ROOK_SEVENTH_RANK_EG
            elif piece_type == chess.PAWN:
                rel_rank = self._relative_rank(color, square)
                passed, mg_bonus, eg_bonus = self._evaluate_passed_pawn(
                    board,
                    square,
                    color,
                    enemy_pawn_masks[color],
                    attack_cache,
                )
                if passed:
                    passed_pawns[color].append((square, rel_rank))
                    mg_score += sign * mg_bonus
                    eg_score += sign * eg_bonus

        for color in (chess.WHITE, chess.BLACK):
            if bishop_counts[color] >= 2:
                sign = 1 if color == chess.WHITE else -1
                mg_score += sign * self.BISHOP_PAIR_MG
                eg_score += sign * self.BISHOP_PAIR_EG

        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            pawns = passed_pawns[color]
            for idx in range(len(pawns)):
                sq_i, rank_i = pawns[idx]
                file_i = chess.square_file(sq_i)
                for jdx in range(idx + 1, len(pawns)):
                    sq_j, rank_j = pawns[jdx]
                    if abs(chess.square_file(sq_j) - file_i) == 1 and abs(rank_i - rank_j) <= 1:
                        rel_rank = min(rank_i, rank_j)
                        mg_score += sign * self.CONNECTED_PASSED_MG[rel_rank]
                        eg_score += sign * self.CONNECTED_PASSED_EG[rel_rank]

        for color in (chess.WHITE, chess.BLACK):
            sign = 1 if color == chess.WHITE else -1
            kmg, keg = self._king_safety(board, color, pieces_by_type, attack_cache)
            mg_score += sign * kmg
            eg_score += sign * keg

        phase = min(phase, self.MAX_PHASE)
        mg_phase = phase
        eg_phase = self.MAX_PHASE - mg_phase

        if self.MAX_PHASE == 0:
            blended = eg_score
        else:
            blended = (mg_score * mg_phase + eg_score * eg_phase) // self.MAX_PHASE
        return int(blended)

    @staticmethod
    def _pst_value(table: tuple[int, ...], square: int, color: chess.Color) -> int:
        index = square if color == chess.WHITE else chess.square_mirror(square)
        return table[index]

    @staticmethod
    def _relative_rank(color: chess.Color, square: int) -> int:
        rank = chess.square_rank(square)
        return rank if color == chess.WHITE else 7 - rank

    def _evaluate_passed_pawn(
        self,
        board: chess.Board,
        square: int,
        color: chess.Color,
        enemy_pawns: int,
        attack_cache: dict[bool, dict[int, int]],
    ) -> tuple[bool, int, int]:
        """Return whether the pawn is passed and the tapered bonus."""

        front_span = chess.BB_FRONT_SPAN[color][square]
        if not front_span:
            return False, 0, 0

        adjacency = front_span
        adjacency |= chess.shift_left(front_span)
        adjacency |= chess.shift_right(front_span)

        if enemy_pawns & adjacency:
            return False, 0, 0

        rel_rank = self._relative_rank(color, square)
        rel_rank = min(rel_rank, len(self.PASSED_PAWN_MG) - 1)
        mg_bonus = self.PASSED_PAWN_MG[rel_rank]
        eg_bonus = self.PASSED_PAWN_EG[rel_rank]

        supporter_mask = self._attackers(board, attack_cache, color, square) & board.pieces(chess.PAWN, color)
        supporters = chess.popcount(supporter_mask)
        if supporters:
            mg_bonus += supporters * self.PASSED_PAWN_SUPPORT_MG
            eg_bonus += supporters * self.PASSED_PAWN_SUPPORT_EG

        return True, mg_bonus, eg_bonus

    def _king_safety(
        self,
        board: chess.Board,
        color: chess.Color,
        pieces_by_type: dict[bool, dict[int, list[int]]],
        attack_cache: dict[bool, dict[int, int]],
    ) -> tuple[int, int]:
        king_square = board.king(color)
        if king_square is None:
            return 0, 0

        enemy = not color
        pawn_shield_mask = self._pawn_shield_mask(color, king_square)
        shielders = chess.popcount(board.pieces(chess.PAWN, color) & pawn_shield_mask)
        mg_score = shielders * self.KING_PAWN_SHIELD_MG
        eg_score = shielders * self.KING_PAWN_SHIELD_EG

        ring_mask = chess.BB_KING_ATTACKS[king_square]
        ring_squares = chess.SquareSet(ring_mask)
        enemy_attackers = 0
        for sq in ring_squares:
            enemy_attackers += chess.popcount(self._attackers(board, attack_cache, enemy, sq))
        mg_score -= enemy_attackers * self.KING_RING_ATTACK_MG
        eg_score -= enemy_attackers * self.KING_RING_ATTACK_EG

        tropism = 0
        for piece_type, weight in self.KING_TROPISM_WEIGHTS.items():
            for sq in pieces_by_type[enemy][piece_type]:
                distance = chess.square_distance(king_square, sq)
                tropism += max(0, 6 - distance) * weight
        mg_score -= tropism
        eg_score -= tropism // 2

        return mg_score, eg_score

    @staticmethod
    def _pawn_shield_mask(color: chess.Color, king_square: int) -> int:
        file_index = chess.square_file(king_square)
        rank_index = chess.square_rank(king_square)
        mask = 0
        forward = 1 if color == chess.WHITE else -1
        target_rank = rank_index + forward
        if 0 <= target_rank <= 7:
            for df in (-1, 0, 1):
                f = file_index + df
                if 0 <= f <= 7:
                    mask |= 1 << chess.square(f, target_rank)
        target_rank2 = rank_index + 2 * forward
        if 0 <= target_rank2 <= 7:
            for df in (-1, 0, 1):
                f = file_index + df
                if 0 <= f <= 7:
                    mask |= 1 << chess.square(f, target_rank2)
        return mask

    @staticmethod
    def _attackers(
        board: chess.Board,
        cache: dict[bool, dict[int, int]],
        color: chess.Color,
        square: int,
    ) -> int:
        color_cache = cache[color]
        if square not in color_cache:
            color_cache[square] = board.attackers(color, square)
        return color_cache[square]

