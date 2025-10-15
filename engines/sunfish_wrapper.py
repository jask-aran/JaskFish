"""UCI compatibility wrapper for the bundled Sunfish engine.

This script adapts the original ``sunfish.py`` implementation so it can be used
as a drop-in engine within the existing orchestration (GUI, self-play, tests).
Sunfish itself remains unmodified; we execute its module code directly and
invoke the searcher programmatically while providing a full UCI façade.
"""

from __future__ import annotations

import io
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Tuple

import chess


# --------------------------------------------------------------------------- #
# Sunfish module loader
# --------------------------------------------------------------------------- #

def _load_sunfish() -> ModuleType:
    """Load ``sunfish.py`` while stripping its command-line interface block."""

    source_path = Path(__file__).with_name("sunfish.py")
    source_text = source_path.read_text(encoding="utf-8")

    first_marker = source_text.find("# minifier-hide start")
    second_marker = source_text.find("# minifier-hide start", first_marker + 1) if first_marker != -1 else -1
    if second_marker != -1:
        second_end = source_text.find("# minifier-hide end", second_marker)
        if second_end != -1:
            source_text = source_text[:second_marker] + source_text[second_end + len("# minifier-hide end") :]

    loop_marker = source_text.find("searcher = Searcher()")
    if loop_marker != -1:
        source_text = source_text[:loop_marker]

    module = ModuleType("sunfish_runtime")
    module.__file__ = str(source_path)
    exec(compile(source_text, str(source_path), "exec"), module.__dict__)
    return module


SUNFISH = _load_sunfish()


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _ensure_line_buffered_stdout() -> None:
    stdout = sys.stdout
    if isinstance(stdout, io.TextIOBase) and getattr(stdout, "line_buffering", False):
        return
    buffer = getattr(stdout, "buffer", None)
    if buffer is None:
        return
    sys.stdout = io.TextIOWrapper(buffer, line_buffering=True)


def _square_index(name: str) -> int:
    return SUNFISH.parse(name)


def _render_index(index: int) -> str:
    return SUNFISH.render(index)


def _castling_tuple(board: chess.Board, color: chess.Color) -> Tuple[bool, bool]:
    return (
        board.has_queenside_castling_rights(color),
        board.has_kingside_castling_rights(color),
    )


def _compute_static_score(board: chess.Board) -> float:
    score = 0.0
    pst = SUNFISH.pst
    for square, piece in board.piece_map().items():
        idx = _square_index(chess.square_name(square))
        table = pst[piece.symbol().upper()]
        if piece.color == chess.WHITE:
            score += table[idx]
        else:
            score -= table[119 - idx]
    return score


def _baseline_board_template() -> List[str]:
    template = list(SUNFISH.initial)
    for idx, char in enumerate(template):
        if char not in (" ", "\n"):
            template[idx] = "."
    return template


def _build_sunfish_position(board: chess.Board) -> SUNFISH.Position:
    layout = _baseline_board_template()
    for square, piece in board.piece_map().items():
        idx = _square_index(chess.square_name(square))
        symbol = piece.symbol().upper()
        layout[idx] = symbol if piece.color == chess.WHITE else symbol.lower()

    score = _compute_static_score(board)
    wc = _castling_tuple(board, chess.WHITE)
    bc = _castling_tuple(board, chess.BLACK)
    ep_square = board.ep_square
    ep_index = _square_index(chess.square_name(ep_square)) if ep_square is not None else 0

    position = SUNFISH.Position(
        "".join(layout),
        int(score),
        wc,
        bc,
        ep_index,
        0,
    )
    if board.turn == chess.BLACK:
        position = position.rotate()
    return position


def _translate_move(move: Optional[SUNFISH.Move], turn: chess.Color) -> Optional[str]:
    if move is None:
        return None
    start, end, prom = move
    if turn == chess.BLACK:
        start = 119 - start
        end = 119 - end
    from_sq = _render_index(start)
    to_sq = _render_index(end)
    promotion = prom.lower() if prom else ""
    return f"{from_sq}{to_sq}{promotion}"


def _legal_fallback(board: chess.Board) -> Optional[str]:
    try:
        move = next(iter(board.legal_moves))
    except StopIteration:
        return None
    return move.uci()


def _parse_go_arguments(argument_text: str) -> Dict[str, int | bool]:
    options: Dict[str, int | bool] = {}
    tokens = argument_text.split()
    idx = 0
    numeric_fields = {
        "wtime",
        "btime",
        "winc",
        "binc",
        "movetime",
        "movestogo",
        "depth",
        "nodes",
    }
    flag_fields = {"infinite", "ponder"}

    while idx < len(tokens):
        token = tokens[idx]
        if token in numeric_fields:
            idx += 1
            if idx < len(tokens):
                try:
                    options[token] = int(tokens[idx])
                except ValueError:
                    pass
        elif token in flag_fields:
            options[token] = True
        idx += 1
    return options


def _resolve_time_budget(board: chess.Board, options: Dict[str, int | bool]) -> Tuple[Optional[float], Optional[int]]:
    depth_limit = int(options["depth"]) if "depth" in options else None

    if options.get("infinite"):
        return None, depth_limit
    if "movetime" in options:
        return max(0.01, int(options["movetime"]) / 1000.0), depth_limit

    active_color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK
    time_key = "wtime" if active_color == chess.WHITE else "btime"
    inc_key = "winc" if active_color == chess.WHITE else "binc"

    total_ms = int(options.get(time_key, 0))
    inc_ms = int(options.get(inc_key, 0))

    if time_key in options or inc_key in options:
        total = max(0.0, total_ms / 1000.0)
        increment = max(0.0, inc_ms / 1000.0)
        if total <= 0.0 and increment <= 0.0:
            return None, depth_limit
        if total <= 0.0:
            return max(0.05, increment), depth_limit
        upper_bound = max(0.05, total / 2.0 - 0.5)
        think = min(total / 40.0 + increment, upper_bound)
        think = max(0.05, think)
        return think, depth_limit

    return None, depth_limit


# --------------------------------------------------------------------------- #
# Engine wrapper
# --------------------------------------------------------------------------- #

@dataclass
class SearchState:
    thread: Optional[threading.Thread] = None
    stop_event: Optional[threading.Event] = None
    bestmove: Optional[str] = None


class SunfishWrapper:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.search_state = SearchState()
        self._lock = threading.Lock()

    # -- Command handlers ------------------------------------------------- #

    def handle_uci(self) -> None:
        print("id name Sunfish Wrapper")
        print("id author Sunfish + Wrapper")
        print("uciok")

    def handle_isready(self) -> None:
        if self.search_state.thread and self.search_state.thread.is_alive():
            print("info string Engine is busy")
        print("readyok")

    def handle_ucinewgame(self) -> None:
        with self._lock:
            self.board = chess.Board()
            self._stop_search(wait=True)
        print("info string New game initialized")

    def handle_position(self, argument: str) -> None:
        tokens = argument.strip().split()
        if not tokens:
            return

        with self._lock:
            if tokens[0] == "startpos":
                board = chess.Board()
                move_index = tokens.index("moves") + 1 if "moves" in tokens else len(tokens)
                for move_text in tokens[move_index:]:
                    try:
                        board.push_uci(move_text)
                    except ValueError:
                        print(f"info string Invalid move in position command: {move_text}")
                        break
                self.board = board
            elif tokens[0] == "fen":
                if len(tokens) < 7:
                    print("info string Invalid FEN (missing fields)")
                    return
                fen = " ".join(tokens[1:7])
                moves: Iterable[str] = tokens[7:]
                try:
                    board = chess.Board(fen)
                except ValueError:
                    print(f"info string Invalid FEN supplied: {fen}")
                    return
                for move_text in moves:
                    try:
                        board.push_uci(move_text)
                    except ValueError:
                        print(f"info string Invalid move in position command: {move_text}")
                        break
                self.board = board
            else:
                print(f"info string Unknown position command: {argument}")

    def handle_go(self, argument: str) -> None:
        options = _parse_go_arguments(argument)
        with self._lock:
            if self.search_state.thread and self.search_state.thread.is_alive():
                print("info string Please wait for the current search to finish")
                return

            time_budget, depth_limit = _resolve_time_budget(self.board, options)
            stop_event = threading.Event()
            search_thread = threading.Thread(
                target=self._run_search,
                args=(stop_event, time_budget, depth_limit),
                daemon=True,
            )
            self.search_state = SearchState(
                thread=search_thread,
                stop_event=stop_event,
                bestmove=None,
            )
            search_thread.start()

    def handle_stop(self) -> None:
        self._stop_search(wait=True)

    def handle_quit(self) -> None:
        self._stop_search(wait=False)
        print("info string Engine shutting down")
        sys.exit(0)

    # -- Internal helpers ------------------------------------------------- #

    def _stop_search(self, wait: bool) -> None:
        state = self.search_state
        if state.stop_event:
            state.stop_event.set()
        if wait and state.thread:
            state.thread.join(timeout=1.0)

    def _run_search(
        self,
        stop_event: threading.Event,
        time_budget: Optional[float],
        depth_limit: Optional[int],
    ) -> None:
        start_time = time.perf_counter()
        position = _build_sunfish_position(self.board)
        history = [position]
        searcher = SUNFISH.Searcher()
        bestmove: Optional[str] = None
        bestdepth = 0

        generator = searcher.search(history)

        try:
            for depth, _gamma, score, move in generator:
                if stop_event.is_set():
                    break

                move_uci = _translate_move(move, self.board.turn)
                if move_uci:
                    centipawns = int(score)
                    print(f"info depth {depth} score cp {centipawns} pv {move_uci}")
                    bestmove = move_uci
                    bestdepth = depth

                if depth_limit is not None and depth >= depth_limit:
                    break

                if time_budget is not None:
                    elapsed = time.perf_counter() - start_time
                    if elapsed >= time_budget:
                        break
        finally:
            generator.close()

        if bestmove is None:
            bestmove = _legal_fallback(self.board) or "(none)"
        elif bestmove != "(none)":
            try:
                # Keep board in sync for potential debugging handlers.
                self.board.push_uci(bestmove)
                self.board.pop()
            except ValueError:
                pass

        print(f"bestmove {bestmove}")
        print("readyok")

        with self._lock:
            self.search_state.bestmove = bestmove
            self.search_state.thread = None
            self.search_state.stop_event = None

    # -- Main loop -------------------------------------------------------- #

    def loop(self) -> None:
        _ensure_line_buffered_stdout()
        while True:
            line = sys.stdin.readline()
            if not line:
                break
            command = line.strip()
            if not command:
                continue

            if command == "uci":
                self.handle_uci()
            elif command == "isready":
                self.handle_isready()
            elif command.startswith("position"):
                self.handle_position(command[len("position") :].strip())
            elif command.startswith("go"):
                self.handle_go(command[len("go") :].strip())
            elif command == "ucinewgame":
                self.handle_ucinewgame()
            elif command == "stop":
                self.handle_stop()
            elif command == "quit":
                self.handle_quit()
            elif command == "debug on":
                print("info string Debug mode is not supported")
            elif command == "debug off":
                print("info string Debug mode is not supported")
            elif command == "boardpos":
                print(f"info string {self.board.fen()}")
            else:
                print(f"info string Unknown command: {command}")


def main() -> None:
    SunfishWrapper().loop()


if __name__ == "__main__":
    main()
