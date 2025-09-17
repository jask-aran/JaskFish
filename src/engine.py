import contextlib
import io
import os
import sys
import threading
from typing import Dict, List, Optional

import chess

from search import AlphaBetaSearcher, Evaluation, SearchInfo, select_opening_move

try:
    from chess import syzygy
except ImportError:  # pragma: no cover - python-chess always ships syzygy, but guard just in case
    syzygy = None  # type: ignore

# Ensure stdout is line-buffered so the GUI receives incremental updates promptly.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


class ChessEngine:
    """A chess engine that communicates over a UCI-like text protocol."""

    def __init__(self) -> None:
        self.engine_name = "JaskFish"
        self.engine_author = "Jaskaran Singh"

        self.board = chess.Board()
        self.debug = False
        self.move_calculating = False
        self.running = True

        self.state_lock = threading.Lock()
        self.evaluator = Evaluation()
        self.active_searcher: Optional[AlphaBetaSearcher] = None
        self.tablebase_max_pieces = 0
        self.tablebase = self._initialise_tablebase()

        self.dispatch_table = {
            "quit": self.handle_quit,
            "debug": self.handle_debug,
            "isready": self.handle_isready,
            "position": self.handle_position,
            "boardpos": self.handle_boardpos,
            "go": self.handle_go,
            "ucinewgame": self.handle_ucinewgame,
            "uci": self.handle_uci,
            "stop": self.handle_stop,
        }

    # ------------------------------------------------------------------
    def start(self) -> None:
        self.command_processor()

    def command_processor(self) -> None:
        while self.running:
            try:
                command = sys.stdin.readline()
                if not command:
                    break
                command = command.strip()
                if not command:
                    continue

                parts = command.split(" ", 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                handler = self.dispatch_table.get(cmd, self.handle_unknown)
                handler(args)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"info string Error processing command: {exc}")
            finally:
                sys.stdout.flush()

    # ------------------------------------------------------------------
    def handle_unknown(self, args: str) -> None:
        print(f"info string Unknown command: {args}")

    def handle_quit(self, args: str) -> None:
        self._request_stop()
        print("info string Engine shutting down")
        self.running = False

    def handle_stop(self, args: str) -> None:
        self._request_stop()

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

    def handle_isready(self, args: str) -> None:
        with self.state_lock:
            if not self.move_calculating:
                print("readyok")
            else:
                print("info string Engine is busy processing a move")

    def handle_position(self, args: str) -> None:
        tokens = args.split()
        if not tokens:
            print("info string Empty position command")
            return

        idx = 0
        with self.state_lock:
            try:
                if tokens[idx] == "startpos":
                    self.board.reset()
                    idx += 1
                elif tokens[idx] == "fen":
                    fen_tokens = tokens[idx + 1 : idx + 7]
                    if len(fen_tokens) < 6:
                        raise ValueError("Incomplete FEN string")
                    fen = " ".join(fen_tokens)
                    self.board.set_fen(fen)
                    idx += 7
                else:
                    print("info string Unknown position command")
                    return

                if idx < len(tokens) and tokens[idx] == "moves":
                    idx += 1
                    move_tokens = tokens[idx:]
                    self._apply_moves(move_tokens)

                if self.debug:
                    print(f"info string Position set: {self.board.fen()}")
            except ValueError as exc:
                print(f"info string Invalid position command: {exc}")

    def handle_boardpos(self, args: str) -> None:
        with self.state_lock:
            print(f"info string Position: {self.board.fen()}")

    def handle_go(self, args: str) -> None:
        with self.state_lock:
            if self.move_calculating:
                print("info string Please wait for the current move to finish")
                return
            self.move_calculating = True

        move_thread = threading.Thread(target=self.process_go_command, args=(args,), daemon=True)
        move_thread.start()

    def handle_ucinewgame(self, args: str) -> None:
        with self.state_lock:
            self.board.reset()
            if self.debug:
                print("info string New game started, board reset to initial position")
            print("info string New game initialized")

    def handle_uci(self, args: str) -> None:
        print(f"id name {self.engine_name}")
        print(f"id author {self.engine_author}")
        print("uciok")

    # ------------------------------------------------------------------
    def process_go_command(self, args: str) -> None:
        limits = self._parse_go_arguments(args)
        with self.state_lock:
            board_copy = self.board.copy(stack=True)
            debug_enabled = self.debug
            side_to_move = self.board.turn

        time_budget = self._determine_time_budget(limits, side_to_move)
        max_depth = limits.get("depth", 6)

        try:
            move = self._probe_opening(board_copy)
            info: Optional[SearchInfo] = None
            source = "book" if move else "search"

            if move is None:
                move = self._probe_tablebase(board_copy)
                source = "tablebase" if move else "search"

            if move is None:
                searcher = AlphaBetaSearcher(
                    board_copy,
                    max_depth=max_depth,
                    time_limit=time_budget,
                    evaluator=self.evaluator,
                    debug=False,
                )
                with self.state_lock:
                    self.active_searcher = searcher
                info = searcher.search()
                move = info.move

            if move:
                if info:
                    self._report_search(info)
                elif debug_enabled:
                    print(f"info string {source} move {move.uci()}")
                print(f"bestmove {move.uci()}")
            else:
                print("bestmove (none)")
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"info string Search error: {exc}")
            print("bestmove (none)")
        finally:
            with self.state_lock:
                self.move_calculating = False
                self.active_searcher = None

    # ------------------------------------------------------------------
    def _apply_moves(self, moves: List[str]) -> None:
        for move_str in moves:
            try:
                move = chess.Move.from_uci(move_str)
            except ValueError:
                print(f"info string Ignoring invalid move: {move_str}")
                return
            if move not in self.board.legal_moves:
                print(f"info string Illegal move in position command: {move_str}")
                return
            self.board.push(move)

    def _parse_go_arguments(self, args: str) -> Dict[str, int]:
        tokens = args.split()
        numeric_keys = {"wtime", "btime", "winc", "binc", "movetime", "movestogo", "depth"}
        limits: Dict[str, int] = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in numeric_keys and i + 1 < len(tokens):
                try:
                    limits[token] = int(tokens[i + 1])
                except ValueError:
                    if self.debug:
                        print(f"info string Invalid numeric value for {token}: {tokens[i + 1]}")
                i += 2
            elif token == "infinite":
                limits["infinite"] = 1
                i += 1
            else:
                i += 1
        return limits

    def _determine_time_budget(self, limits: Dict[str, int], side_to_move: bool) -> Optional[float]:
        if limits.get("infinite"):
            return None
        if "movetime" in limits:
            return max(0.01, limits["movetime"] / 1000.0)

        time_key = "wtime" if side_to_move == chess.WHITE else "btime"
        inc_key = "winc" if side_to_move == chess.WHITE else "binc"

        if time_key not in limits:
            return None

        time_remaining = max(0, limits[time_key])
        increment = limits.get(inc_key, 0)
        moves_to_go = limits.get("movestogo", 0)

        if moves_to_go:
            allocation = time_remaining / max(1, moves_to_go)
        else:
            allocation = time_remaining / 30.0

        allocation += increment
        return max(0.01, allocation / 1000.0)

    def _probe_opening(self, board: chess.Board) -> Optional[chess.Move]:
        if len(board.move_stack) >= 12:
            return None
        return select_opening_move(board)

    def _probe_tablebase(self, board: chess.Board) -> Optional[chess.Move]:
        if not self.tablebase:
            return None
        if len(board.piece_map()) > self.tablebase_max_pieces:
            return None
        if syzygy is None:
            return None
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        try:
            for move in board.legal_moves:
                board.push(move)
                try:
                    wdl = -self.tablebase.probe_wdl(board)
                finally:
                    board.pop()
                if wdl > best_score:
                    best_score = wdl
                    best_move = move
            if best_move and self.debug:
                print(f"info string Tablebase move {best_move.uci()} wdl {best_score}")
            return best_move
        except (syzygy.MissingTableError, syzygy.TablebaseError):
            return None

    def _report_search(self, info: SearchInfo) -> None:
        score = int(info.score)
        mate_threshold = AlphaBetaSearcher.MATE_VALUE - 1000
        if abs(score) >= mate_threshold:
            mate_distance = (AlphaBetaSearcher.MATE_VALUE - abs(score)) // 2
            mate_sign = 1 if score > 0 else -1
            score_str = f"mate {mate_sign * max(1, mate_distance)}"
        else:
            score_str = f"cp {score}"
        pv = " ".join(move.uci() for move in info.pv)
        print(
            f"info depth {info.depth} nodes {info.nodes} score {score_str} pv {pv}"
        )

    def _request_stop(self) -> None:
        with self.state_lock:
            if self.active_searcher:
                self.active_searcher.stop_search()

    def _initialise_tablebase(self):
        path = os.environ.get("SYZYGY_PATH")
        if not path or not syzygy:
            return None
        if not os.path.isdir(path):
            if self.debug:
                print(f"info string Syzygy path does not exist: {path}")
            return None
        tablebase = syzygy.Tablebase()
        try:
            tablebase.add_directory(path)
            self.tablebase_max_pieces = tablebase.max_pieces
            if self.debug:
                print(f"info string Syzygy tablebase initialised ({tablebase.max_pieces} pieces)")
            return tablebase
        except OSError as exc:
            print(f"info string Failed to load Syzygy tablebase: {exc}")
            return None

    def close(self) -> None:
        if self.tablebase:
            with contextlib.suppress(Exception):
                self.tablebase.close()
            self.tablebase = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        self.close()


engine = ChessEngine()
engine.start()
