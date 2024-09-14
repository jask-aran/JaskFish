import sys
import io
import random
import chess
import time
import threading

# Ensure stdout is line-buffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


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
            'uci': self.handle_uci
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
        
    def process_go_command(self):
        if self.debug:
            print(f"info string calc start: {self.board.fen()}")

        # Simulate a long move calculation
        time.sleep(2)

        with self.state_lock:
            move = self.random_move(self.board)
            if move:
                print(f"bestmove {move}")
                print("readyok")
            else:
                print("bestmove (none)")
            self.move_calculating = False



engine = ChessEngine()
engine.start()
