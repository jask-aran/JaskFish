"""Simple heuristic-only UCI engine for head-to-head testing."""

import io
import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import chess

# Basic centipawn piece values
PIECE_VALUES: Dict[int, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Heuristics tuning constants (centipawns)
BISHOP_PAIR_BONUS = 35
ROOK_SEMI_OPEN_BONUS = 9
ROOK_OPEN_FILE_BONUS = 16
ROOK_ON_7TH_BONUS = 18
PASSED_PAWN_BONUS = 45
PAWN_ISOLATED_PENALTY = 20
PAWN_DOUBLED_PENALTY = 14
PAWN_BACKWARD_PENALTY = 10
KING_SHIELD_UNIT = 12
KING_EXPOSED_PENALTY = 18
MOBILITY_WEIGHT = 2.2
PIECE_MOBILITY_WEIGHTS = {  # per legal-destination square
    chess.KNIGHT: 1.5,
    chess.BISHOP: 1.8,
    chess.ROOK: 1.6,
    chess.QUEEN: 0.9,
}
CENTER_CONTROL_BONUS = 12
SPACE_WEIGHT = 0.6
THREAT_WEIGHT = 18
HANGING_PENALTY = 22
OUTPOST_BONUS = 18
DEV_PENALTY = 10
EARLY_QUEEN_PENALTY = 6
UNCASLED_PENALTY = 14
ROOK_CONNECT_BONUS = 10

REPETITION_PENALTY = 70            # ↑ stronger than before
PINGPONG_PENALTY = 40              # immediate back-and-forth
RECREATE_POS_PENALTY = 55          # recreates any earlier position seen

# Noise
NOISE_SCALE_BASE = 18.0            # ↑ more noise
NOISE_SCALE_OPENING = 26.0         # opening gets more variety
NOISE_DIRICHLET_ALPHA = 0.6        # move-list spread

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
CENTER_RING = [
    chess.C3, chess.C4, chess.C5, chess.D3, chess.E3, chess.F3, chess.F4,
    chess.F5, chess.D6, chess.E6, chess.C6, chess.F6,
]

# Piece-square tables (light bias)
PIECE_SQUARE_TABLES: Dict[int, List[int]] = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    chess.ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0,
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    # Use tapered king PST via scaling in evaluation
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -30, -30, -40, -40, -30, -30, -30,
        -20, -20, -20, -20, -20, -20, -20, -20,
        -10, -10, -10, -10, -10, -10, -10, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 30, 10, 0, 0, 10, 30, 20,
    ],
}

# Endgame king PST (safer in center)
KING_ENDGAME_PST = [
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 5, 10, 10, 10, 10, 5, -5,
    -5, 10, 15, 20, 20, 15, 10, -5,
    -5, 10, 20, 30, 30, 20, 10, -5,
    -5, 10, 20, 30, 30, 20, 10, -5,
    -5, 10, 15, 20, 20, 15, 10, -5,
    -5, 5, 10, 10, 10, 10, 5, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
]


@dataclass
class CommandHandler:
    command: str
    handler: Callable[[str], None]


class SimpleEngine:
    """Lightweight engine that scores single plies with static heuristics."""

    def __init__(self) -> None:
        self.board = chess.Board()
        self.running = True
        self.debug = False
        self._pending_position: Optional[str] = None
        # Stronger nondeterminism
        self._rng = random.Random(int.from_bytes(os.urandom(16), "big"))
        # Track recent positions to penalise cycles
        self._history: deque[str] = deque(maxlen=64)

        self._handlers: Dict[str, Callable[[str], None]] = {
            "uci": self.handle_uci,
            "isready": self.handle_isready,
            "ucinewgame": self.handle_ucinewgame,
            "position": self.handle_position,
            "go": self.handle_go,
            "debug": self.handle_debug,
            "quit": self.handle_quit,
            "stop": lambda _: None,
        }

    def start(self) -> None:
        _ensure_line_buffered_stdout()
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
            handler = self._handlers.get(name, self.handle_unknown)
            handler(args)
            sys.stdout.flush()

    def handle_uci(self, _: str) -> None:
        print("id name SimpleEngine+")
        print("id author JaskFish Project")
        print("uciok")

    def handle_isready(self, _: str) -> None:
        print("readyok")

    def handle_ucinewgame(self, _: str) -> None:
        self.board.reset()
        self._history.clear()
        self._push_history_key(self.board)

    def handle_position(self, args: str) -> None:
        tokens = args.split()
        if not tokens:
            return

        board = self.board
        move_tokens: List[str] = []

        if tokens[0] == "startpos":
            board.reset()
            self._history.clear()
            if "moves" in tokens:
                move_index = tokens.index("moves")
                move_tokens = tokens[move_index + 1 :]
        elif tokens[0] == "fen":
            try:
                fen_end = tokens.index("moves")
                fen_tokens = tokens[1:fen_end]
                move_tokens = tokens[fen_end + 1 :]
            except ValueError:
                fen_tokens = tokens[1:]
            fen = " ".join(fen_tokens[:6])
            try:
                board.set_fen(fen)
                self._history.clear()
            except ValueError:
                self._log(f"Invalid FEN received: {fen}")
                return
        else:
            self._log(f"Unsupported position command: {args}")
            return

        self._push_history_key(board)
        for move_text in move_tokens:
            try:
                move = board.parse_uci(move_text)
            except ValueError:
                self._log(f"Illegal move in position command: {move_text}")
                break
            board.push(move)
            self._push_history_key(board)

    def handle_go(self, _: str) -> None:
        move = self._select_move()
        if move is None:
            print("bestmove (none)")
            return
        print(f"bestmove {move.uci()}")

    def handle_debug(self, args: str) -> None:
        setting = args.strip().lower()
        if setting == "on":
            self.debug = True
        elif setting == "off":
            self.debug = False
        else:
            print("info string debug expects 'on' or 'off'")
            return
        self._log(f"Debug set to {self.debug}")

    def handle_quit(self, _: str) -> None:
        self.running = False
        print("info string SimpleEngine shutting down")

    def handle_unknown(self, args: str) -> None:
        self._log(f"Unknown command: {args}")

    # ---------- Move selection with stronger noise and anti-repetition ----------

    def _select_move(self) -> Optional[chess.Move]:
        board = self.board
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # Dirichlet-like per-move noise vector for diversity
        dirichlet = [
            self._rng.gammavariate(NOISE_DIRICHLET_ALPHA, 1.0) for _ in legal_moves
        ]
        s = sum(dirichlet) or 1.0
        dirichlet = [x / s for x in dirichlet]

        turn = board.turn
        best_move: Optional[chess.Move] = None
        best_score = -float("inf") if turn == chess.WHITE else float("inf")

        # Phase-adaptive noise
        phase = self._game_phase(board)  # 0..24
        phase_ratio = max(0.0, min(1.0, phase / 24.0))  # 0 endgame, 1 opening-ish inverted below
        noise_scale = NOISE_SCALE_BASE + (NOISE_SCALE_OPENING - NOISE_SCALE_BASE) * (1.0 - phase_ratio)

        for i, move in enumerate(legal_moves):
            score = self._score_move(board, move, turn)

            # Penalise immediate ping-pong with opponent’s last move
            if board.move_stack:
                last = board.peek()
                if move.from_square == last.to_square and move.to_square == last.from_square:
                    score += (-PINGPONG_PENALTY if turn == chess.WHITE else PINGPONG_PENALTY)

            # Penalise recreating any earlier position
            board.push(move)
            try:
                key = self._pos_key(board)
                if key in self._history:
                    score += (-RECREATE_POS_PENALTY if turn == chess.WHITE else RECREATE_POS_PENALTY)
            finally:
                board.pop()

            # Inject zero-mean noise and Dirichlet spread
            score += self._rng.gauss(0.0, noise_scale)
            score += (dirichlet[i] - 1.0 / len(legal_moves)) * noise_scale * 40.0

            if turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move or legal_moves[0]

    # ---------- Move scoring ----------

    def _score_move(self, board: chess.Board, move: chess.Move, mover: chess.Color) -> float:
        attacker_piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        is_capture = board.is_capture(move)
        is_en_passant = board.is_en_passant(move)
        gives_check = board.gives_check(move)
        promotion = move.promotion

        if is_en_passant:
            capture_value = PIECE_VALUES[chess.PAWN]
        elif captured_piece is not None:
            capture_value = PIECE_VALUES.get(captured_piece.piece_type, 0)
        else:
            capture_value = 0

        attacker_value = PIECE_VALUES.get(attacker_piece.piece_type, 0) if attacker_piece else 0

        board.push(move)
        try:
            score = self._evaluate_board(board)
            score += self._repetition_adjustment(board, mover)
            if gives_check:
                score += 24 if mover == chess.WHITE else -24
            if promotion is not None:
                # Prefer promoting to queen; still single-ply so keep flat
                score += 80 if mover == chess.WHITE else -80
        finally:
            board.pop()

        if is_capture:
            trade_score = capture_value - 0.3 * attacker_value
            score += trade_score if mover == chess.WHITE else -trade_score

        return score

    # ---------- Position evaluation ----------

    def _evaluate_board(self, board: chess.Board) -> float:
        phase = self._game_phase(board)  # 0..24 (higher = earlier)
        t = max(0.0, min(1.0, phase / 24.0))   # 1.0 opening -> 0.0 endgame

        score = 0.0
        score += self._material_and_piece_square_score(board, t)
        score += self._mobility_score(board)
        score += self._piece_mobility_detail(board)
        score += self._pawn_structure_score(board)
        score += self._positional_feature_score(board)
        score += self._king_safety_score(board, t)
        score += self._center_control_score(board)
        score += self._space_score(board)
        score += self._threats_and_hanging(board)
        score += self._development_castling(board, t)
        return score

    def _material_and_piece_square_score(self, board: chess.Board, t_opening: float) -> float:
        material = {chess.WHITE: 0, chess.BLACK: 0}
        pst = {chess.WHITE: 0, chess.BLACK: 0}

        for square, piece in board.piece_map().items():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            color = piece.color
            material[color] += value

            if piece.piece_type == chess.KING:
                # Tapered king PST: opening uses original, endgame uses endgame PST
                idx = square if color == chess.WHITE else chess.square_mirror(square)
                mid = PIECE_SQUARE_TABLES[chess.KING][idx]
                end = KING_ENDGAME_PST[idx]
                v = int(mid * t_opening + end * (1.0 - t_opening))
            else:
                table = PIECE_SQUARE_TABLES.get(piece.piece_type)
                v = 0
                if table:
                    index = square if color == chess.WHITE else chess.square_mirror(square)
                    v = table[index]
            pst[color] += v

        return float((material[chess.WHITE] - material[chess.BLACK]) + (pst[chess.WHITE] - pst[chess.BLACK]))

    def _mobility_score(self, board: chess.Board) -> float:
        if board.is_game_over():
            return 0.0
        white_moves = self._count_legal_moves(board, chess.WHITE)
        black_moves = self._count_legal_moves(board, chess.BLACK)
        return MOBILITY_WEIGHT * float(white_moves - black_moves)

    def _piece_mobility_detail(self, board: chess.Board) -> float:
        # Per-piece mobility to encourage development and activity
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            color_score = 0.0
            for ptype, w in PIECE_MOBILITY_WEIGHTS.items():
                for sq in board.pieces(ptype, color):
                    color_score += w * len(board.attacks(sq))
            # rook on 7th
            for sq in board.pieces(chess.ROOK, color):
                rank = chess.square_rank(sq)
                if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                    color_score += ROOK_ON_7TH_BONUS
            score += color_score if color == chess.WHITE else -color_score
        return score

    def _count_legal_moves(self, board: chess.Board, color: chess.Color) -> int:
        if board.turn == color:
            return sum(1 for _ in board.legal_moves)
        mirror = board.copy(stack=False)
        mirror.turn = color
        return sum(1 for _ in mirror.legal_moves)

    def _pawn_structure_score(self, board: chess.Board) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            pawns = board.pieces(chess.PAWN, color)
            if not pawns:
                continue
            opponent_pawns = board.pieces(chess.PAWN, not color)
            pawns_by_file = [0] * 8
            for sq in pawns:
                pawns_by_file[chess.square_file(sq)] += 1

            doubled = sum(max(count - 1, 0) for count in pawns_by_file)
            structure_score = -PAWN_DOUBLED_PENALTY * doubled

            for sq in pawns:
                if self._is_isolated_pawn(sq, pawns_by_file):
                    structure_score -= PAWN_ISOLATED_PENALTY
                if self._is_passed_pawn(sq, color, opponent_pawns):
                    advance = chess.square_rank(sq) if color == chess.WHITE else (7 - chess.square_rank(sq))
                    structure_score += PASSED_PAWN_BONUS + 5 * advance
                if self._is_backward_pawn(board, sq, color):
                    structure_score -= PAWN_BACKWARD_PENALTY

            score += structure_score if color == chess.WHITE else -structure_score
        return float(score)

    def _is_isolated_pawn(self, square: chess.Square, pawns_by_file: List[int]) -> bool:
        file_index = chess.square_file(square)
        left_file = pawns_by_file[file_index - 1] if file_index > 0 else 0
        right_file = pawns_by_file[file_index + 1] if file_index < 7 else 0
        return (left_file == 0) and (right_file == 0)

    def _is_passed_pawn(
        self,
        square: chess.Square,
        color: chess.Color,
        opponent_pawns: chess.SquareSet,
    ) -> bool:
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        for opp_square in opponent_pawns:
            opp_file = chess.square_file(opp_square)
            if abs(opp_file - file_index) > 1:
                continue
            opp_rank = chess.square_rank(opp_square)
            if color == chess.WHITE and opp_rank > rank:
                return False
            if color == chess.BLACK and opp_rank < rank:
                return False
        return True

    def _is_backward_pawn(self, board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
        # crude: no friendly pawn on adjacent files that is ahead of it and can support advance
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        direction = 1 if color == chess.WHITE else -1
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf < 8:
                for nr in range(r + direction, 8 if color == chess.WHITE else -1, direction):
                    psq = chess.square(nf, nr)
                    p = board.piece_at(psq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        return False
        return True

    def _positional_feature_score(self, board: chess.Board) -> float:
        score = 0.0
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= BISHOP_PAIR_BONUS

        for color in (chess.WHITE, chess.BLACK):
            rooks = board.pieces(chess.ROOK, color)
            friendly_pawns = board.pieces(chess.PAWN, color)
            enemy_pawns = board.pieces(chess.PAWN, not color)
            color_score = 0.0
            for rook_sq in rooks:
                file_index = chess.square_file(rook_sq)
                friendly_on_file = any(chess.square_file(p) == file_index for p in friendly_pawns)
                enemy_on_file = any(chess.square_file(p) == file_index for p in enemy_pawns)
                if not friendly_on_file and not enemy_on_file:
                    color_score += ROOK_OPEN_FILE_BONUS
                elif not friendly_on_file and enemy_on_file:
                    color_score += ROOK_SEMI_OPEN_BONUS
            # Knight outposts: protected by pawn and not attackable by enemy pawns
            for n in board.pieces(chess.KNIGHT, color):
                if self._knight_outpost(board, n, color):
                    color_score += OUTPOST_BONUS
            score += color_score if color == chess.WHITE else -color_score
        return score

    def _knight_outpost(self, board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
        # Protected by friendly pawn
        protectors = board.attackers(color, sq)
        if not any(board.piece_at(p) and board.piece_at(p).piece_type == chess.PAWN for p in protectors):
            return False
        # Not attackable by enemy pawn from adjacent files
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        direction = -1 if color == chess.WHITE else 1  # enemy pawn moves this way
        for df in (-1, 1):
            nf = f + df
            nr = r + direction
            if 0 <= nf < 8 and 0 <= nr < 8:
                psq = chess.square(nf, nr)
                piece = board.piece_at(psq)
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
        return True

    def _king_safety_score(self, board: chess.Board, t_opening: float) -> float:
        # Weight safety more in opening/middlegame
        weight = 0.6 + 0.8 * t_opening
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            king_square = board.king(color)
            if king_square is None:
                continue
            shield_strength = self._king_pawn_shield(board, king_square, color)
            hostile_pressure = self._king_attack_pressure(board, king_square, color)
            king_score = weight * (shield_strength * KING_SHIELD_UNIT - hostile_pressure * KING_EXPOSED_PENALTY)
            score += king_score if color == chess.WHITE else -king_score
        return score

    def _king_pawn_shield(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        direction = 1 if color == chess.WHITE else -1
        file_index = chess.square_file(king_square)
        rank_index = chess.square_rank(king_square)
        shield = 0
        for distance in (1, 2):
            target_rank = rank_index + direction * distance
            if not 0 <= target_rank < 8:
                continue
            for offset in (-1, 0, 1):
                target_file = file_index + offset
                if not 0 <= target_file < 8:
                    continue
                target_square = chess.square(target_file, target_rank)
                piece = board.piece_at(target_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    shield += 2 if distance == 1 else 1
        return shield

    def _king_attack_pressure(self, board: chess.Board, king_square: chess.Square, color: chess.Color) -> int:
        opposing = chess.WHITE if color == chess.BLACK else chess.BLACK
        pressure = 0
        for target in self._king_zone(king_square):
            attackers = board.attackers(opposing, target)
            pressure += len(attackers)
        return pressure

    def _king_zone(self, square: chess.Square) -> List[chess.Square]:
        rank_index = chess.square_rank(square)
        file_index = chess.square_file(square)
        zone: List[chess.Square] = []
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr = rank_index + dr
                nf = file_index + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    zone.append(chess.square(nf, nr))
        return zone

    def _center_control_score(self, board: chess.Board) -> float:
        control = 0.0
        for square in CENTER_SQUARES:
            control += (len(board.attackers(chess.WHITE, square)) - len(board.attackers(chess.BLACK, square))) * CENTER_CONTROL_BONUS
        ring_bonus = CENTER_CONTROL_BONUS * 0.5
        for square in CENTER_RING:
            control += (len(board.attackers(chess.WHITE, square)) - len(board.attackers(chess.BLACK, square))) * ring_bonus
        return control

    def _space_score(self, board: chess.Board) -> float:
        # Count attacked empty squares on opponent half
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            half = range(4, 8) if color == chess.WHITE else range(0, 4)
            attacked = 0
            for sq in chess.SQUARES:
                if chess.square_rank(sq) in half and board.piece_at(sq) is None:
                    if board.is_attacked_by(color, sq):
                        attacked += 1
            score += SPACE_WEIGHT * attacked if color == chess.WHITE else -SPACE_WEIGHT * attacked
        return score

    def _threats_and_hanging(self, board: chess.Board) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            opposing = not color
            color_score = 0.0
            for target_sq, target_piece in board.piece_map().items():
                if target_piece.color != opposing:
                    continue
                attackers = board.attackers(color, target_sq)
                if not attackers:
                    continue
                defenders = board.attackers(opposing, target_sq)
                if len(defenders) == 0:
                    color_score += HANGING_PENALTY
                # basic “threat value”: higher if we attack with cheaper piece
                min_attacker_val = min(PIECE_VALUES[board.piece_at(a).piece_type] for a in attackers)
                color_score += THREAT_WEIGHT * (PIECE_VALUES[target_piece.piece_type] - 0.7 * min_attacker_val) / 100.0
            score += color_score if color == chess.WHITE else -color_score
        return score

    def _development_castling(self, board: chess.Board, t_opening: float) -> float:
        if t_opening < 0.2:
            return 0.0
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            color_score = 0.0
            # undeveloped minors penalised if on back rank and original squares
            back_rank = 0 if color == chess.WHITE else 7
            for ptype in (chess.KNIGHT, chess.BISHOP):
                for sq in board.pieces(ptype, color):
                    if chess.square_rank(sq) == back_rank:
                        color_score -= DEV_PENALTY
            # early queen in opening
            for q in board.pieces(chess.QUEEN, color):
                if chess.square_rank(q) not in (0, 7):
                    color_score -= EARLY_QUEEN_PENALTY
            # castling status
            king_sq = board.king(color)
            if king_sq is not None and chess.square_file(king_sq) in (3, 4):  # near middle files before castling
                # new
                if self._has_castling_rights(board, color):
                    color_score -= UNCASLED_PENALTY

            # rook connectivity
            rooks = list(board.pieces(chess.ROOK, color))
            if len(rooks) >= 2:
                if not self._pieces_between_files(board, rooks[0], rooks[1], color):
                    color_score += ROOK_CONNECT_BONUS
            score += color_score if color == chess.WHITE else -color_score
        return score

    def _pieces_between_files(self, board: chess.Board, r1: chess.Square, r2: chess.Square, color: chess.Color) -> bool:
        # True if own minor/queen on back rank between rooks’ files
        rank = 0 if color == chess.WHITE else 7
        f1, f2 = sorted([chess.square_file(r1), chess.square_file(r2)])
        for f in range(f1 + 1, f2):
            sq = chess.square(f, rank)
            p = board.piece_at(sq)
            if p and p.color == color and p.piece_type in (chess.KNIGHT, chess.BISHOP, chess.QUEEN):
                return True
        return False

    def _has_castling_rights(self, board: chess.Board, color: chess.Color) -> bool:
        # Works across python-chess versions
        if hasattr(board, "has_castling_rights"):
            return board.has_castling_rights(color)  # type: ignore[attr-defined]
        return (board.has_kingside_castling_rights(color) or
                board.has_queenside_castling_rights(color))


    # ---------- Repetition handling ----------

    def _repetition_adjustment(self, board: chess.Board, mover: chess.Color) -> float:
        penalty = 0.0
        if board.is_fivefold_repetition():
            penalty = 3.0 * REPETITION_PENALTY
        elif board.is_repetition(3) or board.can_claim_threefold_repetition():
            penalty = 2.0 * REPETITION_PENALTY
        elif board.is_repetition(2):
            penalty = 1.2 * REPETITION_PENALTY

        if penalty:
            return -penalty if mover == chess.WHITE else penalty
        return 0.0

    # ---------- Helpers ----------

    def _game_phase(self, board: chess.Board) -> int:
        # Taper between opening and endgame by counting non-pawn material (common 24 max)
        phase = 0
        phase += 0 * len(board.pieces(chess.PAWN, True))
        phase += 0 * len(board.pieces(chess.PAWN, False))
        phase += 1 * (len(board.pieces(chess.KNIGHT, True)) + len(board.pieces(chess.KNIGHT, False)))
        phase += 1 * (len(board.pieces(chess.BISHOP, True)) + len(board.pieces(chess.BISHOP, False)))
        phase += 2 * (len(board.pieces(chess.ROOK, True)) + len(board.pieces(chess.ROOK, False)))
        phase += 4 * (len(board.pieces(chess.QUEEN, True)) + len(board.pieces(chess.QUEEN, False)))
        # Higher means earlier, clamp to ~24
        return min(24, phase)

    def _pos_key(self, board: chess.Board) -> str:
        # Shredder FEN excludes move clocks, keeps side/castling/EP
        return board.shredder_fen()

    def _push_history_key(self, board: chess.Board) -> None:
        self._history.append(self._pos_key(board))

    def _log(self, message: str) -> None:
        if self.debug:
            print(f"info string {message}")


def _ensure_line_buffered_stdout() -> None:
    stdout = sys.stdout
    if isinstance(stdout, io.TextIOBase) and getattr(stdout, "line_buffering", False):
        return
    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        return
    sys.stdout = io.TextIOWrapper(buffer, line_buffering=True)


if __name__ == "__main__":
    SimpleEngine().start()
