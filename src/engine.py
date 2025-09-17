import sys
import io
import random
import chess
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

# Ensure stdout is line-buffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


class SearchTimeout(Exception):
    """Raised when a search must be aborted because its time budget expired."""


@dataclass
class SearchResult:
    """Container describing the outcome of a search iteration."""

    move: Optional[chess.Move]
    score: int
    depth: int
    nodes: int
    pv: List[chess.Move]
    fail_type: Optional[str] = None


class AlphaBetaSearcher:
    """Iterative deepening alpha-beta searcher with aspiration windows."""

    def __init__(self, aspiration_delta: int = 50, mate_value: int = 100_000) -> None:
        self.aspiration_delta = max(1, aspiration_delta)
        self.mate_value = mate_value
        self.piece_values: Dict[int, int] = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0,
        }
        self.nodes = 0

    def search(
        self,
        board: chess.Board,
        max_depth: int,
        time_limit: Optional[float] = None,
    ) -> SearchResult:
        """Search for the best move up to ``max_depth`` within ``time_limit`` seconds."""

        if max_depth <= 0:
            return SearchResult(move=None, score=0, depth=0, nodes=0, pv=[])

        board_copy = board.copy(stack=False)

        if board_copy.is_game_over():
            terminal_score = self._evaluate(board_copy, 0)
            return SearchResult(move=None, score=terminal_score, depth=0, nodes=0, pv=[])

        start_time = time.perf_counter()
        self.nodes = 0
        best_result = SearchResult(move=None, score=-self.mate_value, depth=0, nodes=0, pv=[])
        previous_score = 0

        for depth in range(1, max_depth + 1):
            if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
                break

            window = self.aspiration_delta
            if depth > 1 and best_result.move is not None:
                alpha = max(-self.mate_value, previous_score - window)
                beta = min(self.mate_value, previous_score + window)
            else:
                alpha = -self.mate_value
                beta = self.mate_value

            while True:
                attempt_alpha = alpha
                attempt_beta = beta
                try:
                    result = self._search_depth(
                        board_copy, depth, attempt_alpha, attempt_beta, start_time, time_limit
                    )
                except SearchTimeout:
                    return best_result

                if (
                    result.fail_type == "fail-low"
                    and attempt_alpha > -self.mate_value
                ):
                    window = max(window * 2, self.aspiration_delta)
                    alpha = max(-self.mate_value, attempt_alpha - window)
                    continue

                if (
                    result.fail_type == "fail-high"
                    and attempt_beta < self.mate_value
                ):
                    window = max(window * 2, self.aspiration_delta)
                    beta = min(self.mate_value, attempt_beta + window)
                    continue

                best_result = result
                previous_score = result.score
                break

        return best_result

    def _search_depth(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        start_time: float,
        time_limit: Optional[float],
    ) -> SearchResult:
        nodes_before = self.nodes
        try:
            score, pv, flag = self._alpha_beta(board, depth, alpha, beta, start_time, time_limit)
        except SearchTimeout:
            raise

        fail_type = flag if flag in {"fail-low", "fail-high"} else None

        if fail_type and (alpha > -self.mate_value or beta < self.mate_value):
            # Re-search with a full window to recover a valid principal variation.
            score, pv, _ = self._alpha_beta(
                board, depth, -self.mate_value, self.mate_value, start_time, time_limit
            )

        nodes = self.nodes - nodes_before
        move = pv[0] if pv else None

        return SearchResult(move=move, score=score, depth=depth, nodes=nodes, pv=pv, fail_type=fail_type)

    def _alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        start_time: float,
        time_limit: Optional[float],
    ) -> tuple[int, List[chess.Move], str]:
        if time_limit is not None and (time.perf_counter() - start_time) >= time_limit:
            raise SearchTimeout

        self.nodes += 1

        if depth == 0 or board.is_game_over():
            return self._evaluate(board, depth), [], "exact"

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._evaluate(board, depth), [], "exact"

        best_move: Optional[chess.Move] = None
        best_line: List[chess.Move] = []
        best_score = -self.mate_value
        initial_alpha = alpha

        for move in legal_moves:
            board.push(move)
            try:
                score, child_pv, _ = self._alpha_beta(
                    board, depth - 1, -beta, -alpha, start_time, time_limit
                )
            except SearchTimeout:
                board.pop()
                raise
            board.pop()

            score = -score

            if score >= beta:
                return score, [move] + child_pv, "fail-high"

            if score > best_score:
                best_score = score
                best_move = move
                best_line = child_pv

            if score > alpha:
                alpha = score

        if best_move is None:
            return best_score, [], "fail-low"

        flag = "exact" if alpha > initial_alpha else "fail-low"
        return alpha, [best_move] + best_line, flag

    def _evaluate(self, board: chess.Board, depth: int) -> int:
        if board.is_checkmate():
            # Prefer quicker mates for the side to move.
            return -self.mate_value + depth

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0

        score = 0
        for piece_type, value in self.piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value

        return score if board.turn == chess.WHITE else -score

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

        # Search and timing configuration
        self.searcher = AlphaBetaSearcher()
        self.default_max_depth = 4
        self.default_time_budget = 1.0  # seconds
        self.move_overhead = 0.05  # seconds reserved for communication/overhead
        self.safety_margin = 0.1  # extra reserve to avoid flagging
        self.default_moves_to_go = 30
        self.increment_blend = 0.5
        self.max_time_allocation = 0.6
        self.min_time_budget = 0.01
        self.panic_time = 1.0
        self.panic_allocation_ratio = 0.5

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
        move_thread = threading.Thread(target=self.process_go_command, args=(args,))
        move_thread.start()

    def handle_ucinewgame(self, args):
        with self.state_lock:
            self.board.reset()
            if self.debug:
                print("info string New game started, board reset to initial position")
            print("info string New game initialized")

    def random_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        return random.choice(legal_moves)
        
    def process_go_command(self, args: str):
        move_uci: Optional[str] = None
        try:
            go_params = self._parse_go_args(args or "")

            with self.state_lock:
                board_copy = self.board.copy(stack=False)

            max_depth = max(1, go_params.get('depth', self.default_max_depth))
            time_budget = self._determine_time_budget(go_params, board_copy)

            search_start = time.perf_counter()
            result = self.searcher.search(
                board_copy, max_depth=max_depth, time_limit=time_budget
            )
            elapsed = time.perf_counter() - search_start

            move_obj = result.move or self.random_move(board_copy)
            if move_obj is not None:
                move_uci = move_obj.uci()

            if self.debug:
                info_parts = [
                    f"depth {result.depth}",
                    f"score {result.score}",
                    f"nodes {result.nodes}",
                    f"elapsed {elapsed:.2f}s",
                ]
                if time_budget is not None:
                    info_parts.append(f"budget {time_budget:.2f}s")
                if result.pv:
                    info_parts.append("pv " + ' '.join(move.uci() for move in result.pv))
                print("info string " + " | ".join(info_parts))

            if move_uci:
                print(f"bestmove {move_uci}")
            else:
                print("bestmove (none)")

            print("readyok")
        except Exception as exc:
            print(f"info string Search error: {exc}")
            print("bestmove (none)")
            print("readyok")
        finally:
            with self.state_lock:
                self.move_calculating = False

    def _parse_go_args(self, args: str) -> Dict[str, int]:
        tokens = args.split()
        params: Dict[str, int] = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "depth", "nodes"}:
                if i + 1 < len(tokens):
                    value = tokens[i + 1]
                    try:
                        params[token] = int(value)
                    except ValueError:
                        pass
                    i += 1
            elif token in {"ponder", "infinite"}:
                params[token] = 1
            i += 1
        return params

    def _determine_time_budget(self, params: Dict[str, int], board: chess.Board) -> Optional[float]:
        if params.get('infinite'):
            return None

        movetime = params.get('movetime')
        if movetime is not None and movetime > 0:
            think_time = max(0.0, movetime / 1000.0 - self.move_overhead)
            return max(self.min_time_budget, think_time)

        color_prefix = 'w' if board.turn == chess.WHITE else 'b'
        time_left_ms = params.get(f'{color_prefix}time')
        increment_ms = params.get(f'{color_prefix}inc', 0)
        moves_to_go = params.get('movestogo')

        if time_left_ms is None:
            return self.default_time_budget

        time_left_s = max(0.0, time_left_ms / 1000.0 - self.move_overhead)
        increment_s = increment_ms / 1000.0 if increment_ms else 0.0
        moves_remaining = moves_to_go if moves_to_go and moves_to_go > 0 else self.default_moves_to_go

        think_available = max(0.0, time_left_s - self.safety_margin)
        if think_available <= 0.0:
            return self.min_time_budget

        allocation = think_available / moves_remaining
        if increment_s:
            allocation += increment_s * self.increment_blend

        max_slice = think_available * self.max_time_allocation
        allocation = min(allocation, max_slice)
        allocation = min(allocation, think_available)

        if time_left_s <= self.panic_time:
            panic_budget = max(time_left_s * self.panic_allocation_ratio, self.min_time_budget)
            allocation = min(allocation, panic_budget)

        return max(self.min_time_budget, allocation)
if __name__ == "__main__":
    engine = ChessEngine()
    engine.start()
