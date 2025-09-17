import sys
import io
import random
import shlex
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List

import chess

from utils import write_search_log_entry

# Ensure stdout is line-buffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


@dataclass
class SearchInfo:
    board_fen: str
    best_move: str
    principal_variation: List[str]
    score: float


class ChessEngine:
    """
    A simple chess engine that communicates using a custom UCI-like protocol.
    It can process commands to set up positions, calculate moves, and manage engine state.
    """

    def __init__(self):
        """Initialize the chess engine with default settings and state."""
        # Engine identification
        self.engine_name = 'JaskFish'
        self.engine_author = 'Jaskaran Singh'

        # Engine state
        self.board = chess.Board()
        self.debug = False
        self.move_calculating = False
        self.running = True
        self.collect_search_data = False
        self.search_log_format = 'json'
        self.search_log_path = Path('logs/search_logs.jsonl')

        # Lock to manage concurrent access to engine state
        self.state_lock = threading.Lock()

        # Dispatch table mapping commands to handler methods
        self.dispatch_table = {
            'quit': self.handle_quit,
            'debug': self.handle_debug,
            'isready': self.handle_isready,
            'position': self.handle_position,
            'boardpos': self.handle_boardpos,
            'go': self.handle_go,
            'ucinewgame': self.handle_ucinewgame,
            'uci': self.handle_uci,
            'collectdata': self.handle_collectdata
        }

    def start(self):
        self.handle_uci()
        self.command_processor()
        
    def handle_uci(self, args=None):
        print(f'id name {self.engine_name}')
        print(f'id author {self.engine_author}')
        print('uciok')

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
                parts = command.split(' ', 1)
                cmd = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ''

                # Dispatch the command to the appropriate handler
                handler = self.dispatch_table.get(cmd, self.handle_unknown)
                handler(args)
            except Exception as e:
                print(f'info string Error processing command: {e}')
            finally:
                sys.stdout.flush()


    def handle_unknown(self, args):
        print(f"unknown command received: '{args}'")

    def handle_quit(self, args):
        print('info string Engine shutting down')
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

    def handle_collectdata(self, args):
        tokens = shlex.split(args)
        if not tokens:
            state = 'enabled' if self.collect_search_data else 'disabled'
            print(f"info string Data collection currently {state} (format={self.search_log_format}, path={self.search_log_path})")
            return

        command = tokens[0].lower()
        if command in {"on", "off"}:
            self.collect_search_data = command == "on"
            state = 'enabled' if self.collect_search_data else 'disabled'
            print(f"info string Data collection {state}")
            return

        if command == "format" and len(tokens) >= 2:
            fmt = tokens[1].lower()
            if fmt not in {"json", "csv"}:
                print("info string Unsupported format. Choose 'json' or 'csv'.")
                return
            self.search_log_format = fmt
            print(f"info string Data collection format set to {fmt}")
            return

        if command == "path" and len(tokens) >= 2:
            path_value = tokens[1]
            self.search_log_path = Path(path_value)
            print(f"info string Data collection path set to {self.search_log_path}")
            return

        print("info string Usage: collectdata on|off | format <json|csv> | path <file>")

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
            print(f"info string Position: {self.board.fen()}" if self.board else "info string Board state not set")

    def handle_go(self, args):
        with self.state_lock:
            if self.move_calculating:
                print('info string Please wait for computer move')
                return
            self.move_calculating = True

        # Start the move calculation in a separate thread
        move_thread = threading.Thread(target=self.process_go_command)
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

    def evaluate_board(self, board):
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        score = 0
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value
        return score
        
    def process_go_command(self):
        with self.state_lock:
            current_fen = self.board.fen()

        if self.debug:
            print(f"info string calc start: {current_fen}")

        time.sleep(2)

        try:
            with self.state_lock:
                move = self.random_move(self.board)
                pv = [move] if move else []

                if move:
                    board_copy = self.board.copy()
                    board_copy.push_uci(move)
                    score = float(self.evaluate_board(board_copy))
                else:
                    score = float(self.evaluate_board(self.board))

                should_log = self.collect_search_data
                search_info = SearchInfo(
                    board_fen=current_fen,
                    best_move=move or "(none)",
                    principal_variation=pv,
                    score=score,
                )

            if should_log:
                try:
                    field_order = [
                        "board_fen",
                        "best_move",
                        "principal_variation",
                        "score",
                    ]
                    write_search_log_entry(
                        search_info,
                        self.search_log_path,
                        self.search_log_format,
                        field_order=field_order,
                    )
                except Exception as exc:
                    print(f"info string Failed to write search log: {exc}")

            if move:
                print(f"bestmove {move}")
                print("readyok")
            else:
                print("bestmove (none)")
        finally:
            with self.state_lock:
                self.move_calculating = False



engine = ChessEngine()
engine.start()
