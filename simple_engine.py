"""Simple heuristic-only UCI engine for head-to-head testing."""

import io
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

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

# Light piece-square bias to encourage development and castling
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
        print("id name SimpleEngine")
        print("id author JaskFish Project")
        print("uciok")

    def handle_isready(self, _: str) -> None:
        print("readyok")

    def handle_ucinewgame(self, _: str) -> None:
        self.board.reset()
        self._pending_position = None

    def handle_position(self, args: str) -> None:
        tokens = args.split()
        if not tokens:
            return

        board = self.board
        move_tokens: List[str] = []

        if tokens[0] == "startpos":
            board.reset()
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
            except ValueError:
                self._log(f"Invalid FEN received: {fen}")
                return
        else:
            self._log(f"Unsupported position command: {args}")
            return

        for move_text in move_tokens:
            try:
                move = board.parse_uci(move_text)
            except ValueError:
                self._log(f"Illegal move in position command: {move_text}")
                break
            board.push(move)

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

    def _select_move(self) -> Optional[chess.Move]:
        board = self.board
        if board.is_game_over():
            return None

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        turn = board.turn
        best_move: Optional[chess.Move] = None
        best_score = -float("inf") if turn == chess.WHITE else float("inf")

        for move in legal_moves:
            board.push(move)
            try:
                score = self._evaluate_board(board)
            finally:
                board.pop()

            if turn == chess.WHITE:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        if best_move is None:
            best_move = legal_moves[0]
        return best_move

    def _evaluate_board(self, board: chess.Board) -> float:
        material = {chess.WHITE: 0, chess.BLACK: 0}
        pst = {chess.WHITE: 0, chess.BLACK: 0}

        for square, piece in board.piece_map().items():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            table = PIECE_SQUARE_TABLES.get(piece.piece_type)
            color = piece.color
            material[color] += value
            if table:
                index = square if color == chess.WHITE else chess.square_mirror(square)
                pst[color] += table[index]

        score = (material[chess.WHITE] - material[chess.BLACK]) + (
            pst[chess.WHITE] - pst[chess.BLACK]
        )
        return float(score)

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
