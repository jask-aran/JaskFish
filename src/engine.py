import sys
import io
import random
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import chess


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

EVAL_PIECE_VALUES = {piece: value * 100 for piece, value in PIECE_VALUES.items()}

PIECE_SQUARE_TABLES = {
    chess.PAWN: [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        50,
        10,
        10,
        20,
        30,
        30,
        20,
        10,
        10,
        5,
        5,
        10,
        25,
        25,
        10,
        5,
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
        -5,
        -10,
        0,
        0,
        -10,
        -5,
        5,
        5,
        10,
        10,
        -20,
        -20,
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
    ],
    chess.KNIGHT: [
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
        0,
        0,
        0,
        -20,
        -40,
        -30,
        0,
        10,
        15,
        15,
        10,
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
        15,
        20,
        20,
        15,
        0,
        -30,
        -30,
        5,
        10,
        15,
        15,
        10,
        5,
        -30,
        -40,
        -20,
        0,
        5,
        5,
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
    ],
    chess.BISHOP: [
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
    ],
    chess.ROOK: [
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
        10,
        10,
        10,
        10,
        5,
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
        0,
        0,
        0,
        5,
        5,
        0,
        0,
        0,
    ],
    chess.QUEEN: [
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
        0,
        5,
        5,
        5,
        5,
        0,
        -10,
        -10,
        0,
        0,
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
    ],
    chess.KING: [
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
        -30,
        -30,
        -40,
        -40,
        -30,
        -30,
        -30,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -20,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
        -10,
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
        20,
        30,
        10,
        0,
        0,
        10,
        30,
        20,
    ],
}

# Ensure stdout is line-buffered
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


@dataclass
class StrategyContext:
    """Snapshot of board metrics used by move strategies."""

    fullmove_number: int
    halfmove_clock: int
    piece_count: int
    material_imbalance: int
    turn: bool = True
    fen: str = ""
    repetition_info: Dict[str, bool] = field(default_factory=dict)
    legal_moves_count: int = 0
    time_controls: Optional[Dict[str, int]] = None


@dataclass
class StrategyResult:
    """Container describing the outcome of a strategy evaluation."""

    move: Optional[str]
    strategy_name: str
    score: Optional[float] = None
    confidence: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class MoveStrategy(ABC):
    """Base class for all move selection strategies."""

    def __init__(self, name: Optional[str] = None, priority: int = 0, confidence: Optional[float] = None):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.confidence = confidence

    @abstractmethod
    def is_applicable(self, context: StrategyContext) -> bool:
        """Return whether the strategy should be considered in the given context."""

    @abstractmethod
    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        """Produce a move suggestion when applicable."""


class StrategySelector:
    """Manages strategy registration and selection for the engine."""

    def __init__(
        self,
        strategies: Optional[Iterable[MoveStrategy]] = None,
        selection_policy: Optional[Callable[..., Optional[StrategyResult]]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self._strategies: List[MoveStrategy] = []
        self._logger = logger or (lambda message: None)
        self._selection_policy = selection_policy or self._default_selection_policy
        self._uses_default_policy = selection_policy is None
        if strategies:
            for strategy in strategies:
                self.register_strategy(strategy)

    def register_strategy(self, strategy: MoveStrategy) -> None:
        """Register a strategy and maintain priority ordering."""

        self._strategies.append(strategy)
        self._strategies.sort(key=lambda item: item.priority, reverse=True)
        self._logger(f"strategy registered: {strategy.name} (priority={strategy.priority})")

    def clear_strategies(self) -> None:
        self._strategies.clear()

    def get_strategies(self) -> Tuple[MoveStrategy, ...]:
        return tuple(self._strategies)

    def set_selection_policy(
        self,
        policy: Optional[Callable[..., Optional[StrategyResult]]],
    ) -> None:
        if policy is None:
            self._selection_policy = self._default_selection_policy
            self._uses_default_policy = True
        else:
            self._selection_policy = policy
            self._uses_default_policy = False
        self._logger("strategy selection policy updated")

    def select_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        """Evaluate registered strategies and choose a result using the selection policy."""

        strategy_results: List[Tuple[MoveStrategy, StrategyResult]] = []
        for strategy in self._strategies:
            if not strategy.is_applicable(context):
                self._logger(f"strategy skipped (not applicable): {strategy.name}")
                continue

            self._logger(f"strategy evaluating: {strategy.name}")
            try:
                result = strategy.generate_move(board, context)
            except Exception as exc:  # pragma: no cover - defensive logging
                self._logger(f"strategy error in {strategy.name}: {exc}")
                continue

            if result is None:
                self._logger(f"strategy produced no result: {strategy.name}")
                continue

            strategy_results.append((strategy, result))
            if result.move and self._uses_default_policy:
                break

        if not strategy_results:
            self._logger("no strategies produced a move suggestion")

        return self._selection_policy(strategy_results, board=board, context=context)

    @staticmethod
    def _default_selection_policy(
        strategy_results: List[Tuple[MoveStrategy, StrategyResult]],
        **_: Any,
    ) -> Optional[StrategyResult]:
        for _, result in strategy_results:
            if result.move:
                return result
        return None


class OpeningBookStrategy(MoveStrategy):
    def __init__(self, opening_book: Optional[Dict[str, str]] = None, max_fullmove: int = 12, **kwargs):
        super().__init__(priority=90, **kwargs)
        self.opening_book = opening_book or {}
        self.max_fullmove = max_fullmove

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.fullmove_number <= self.max_fullmove and bool(self.opening_book)

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self.opening_book.get(board.fen())
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence or 1.0,
            metadata={"source": "opening_book"},
        )


class EndgameTableStrategy(MoveStrategy):
    def __init__(self, table: Optional[Dict[str, str]] = None, piece_threshold: int = 6, **kwargs):
        super().__init__(priority=80, **kwargs)
        self.table = table or {}
        self.piece_threshold = piece_threshold

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.piece_count <= self.piece_threshold

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self.table.get(board.fen())
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence,
            metadata={"source": "endgame_table"},
        )


class HeuristicSearchStrategy(MoveStrategy):
    def __init__(
        self,
        fallback: Optional[MoveStrategy] = None,
        search_depth: int = 3,
        mobility_weight: float = 5.0,
        king_safety_weight: float = 10.0,
        **kwargs,
    ):
        super().__init__(priority=70, **kwargs)
        self._fallback = fallback
        self.search_depth = max(1, search_depth)
        self.mobility_weight = mobility_weight
        self.king_safety_weight = king_safety_weight
        self._mate_score = 100000

    def is_applicable(self, context: StrategyContext) -> bool:
        return True

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        if context.legal_moves_count == 0:
            if board.is_checkmate():
                return StrategyResult(
                    move=None,
                    strategy_name=self.name,
                    score=-float(self._mate_score),
                    metadata={"status": "checkmate"},
                )
            if board.is_stalemate():
                return StrategyResult(
                    move=None,
                    strategy_name=self.name,
                    score=0.0,
                    metadata={"status": "stalemate"},
                )
            if self._fallback:
                return self._fallback.generate_move(board, context)
            return None

        ordered_moves = self._order_moves(board)
        best_move: Optional[chess.Move] = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        node_counter = 0

        def alpha_beta(depth: int, alpha_bound: float, beta_bound: float) -> float:
            nonlocal node_counter
            node_counter += 1

            if depth == 0 or board.is_game_over():
                return self._evaluate_board(board)

            value = -float("inf")
            for child_move in self._order_moves(board):
                board.push(child_move)
                score = -alpha_beta(depth - 1, -beta_bound, -alpha_bound)
                board.pop()
                value = max(value, score)
                alpha_bound = max(alpha_bound, score)
                if alpha_bound >= beta_bound:
                    break
            return value

        for move in ordered_moves:
            board.push(move)
            score = -alpha_beta(self.search_depth - 1, -beta, -alpha)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        if best_move is None:
            if self._fallback:
                return self._fallback.generate_move(board, context)
            return None

        move_str = best_move.uci()
        metadata = {
            "depth": self.search_depth,
            "nodes": node_counter,
            "searched_moves": len(ordered_moves),
        }
        return StrategyResult(
            move=move_str,
            strategy_name=self.name,
            score=best_score,
            confidence=self.confidence or 0.75,
            metadata=metadata,
        )

    def _order_moves(self, board: chess.Board) -> List[chess.Move]:
        def move_score(move: chess.Move) -> float:
            score = 0.0
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece is None and board.is_en_passant(move):
                    captured_piece = chess.Piece(chess.PAWN, not board.turn)
                if captured_piece:
                    score += EVAL_PIECE_VALUES.get(captured_piece.piece_type, 0)
                moving_piece = board.piece_at(move.from_square)
                if moving_piece:
                    score += EVAL_PIECE_VALUES.get(moving_piece.piece_type, 0) * 0.1
            if board.gives_check(move):
                score += 50
            if move.promotion:
                score += EVAL_PIECE_VALUES.get(move.promotion, 0)
            return score

        moves = list(board.legal_moves)
        moves.sort(key=move_score, reverse=True)
        return moves

    def _evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -float(self._mate_score)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        material_score = 0.0
        piece_square_score = 0.0
        for square, piece in board.piece_map().items():
            piece_value = EVAL_PIECE_VALUES.get(piece.piece_type, 0)
            table = PIECE_SQUARE_TABLES.get(piece.piece_type, None)
            if piece.color == chess.WHITE:
                material_score += piece_value
                if table:
                    piece_square_score += table[square]
            else:
                material_score -= piece_value
                if table:
                    mirrored_square = chess.square_mirror(square)
                    piece_square_score -= table[mirrored_square]

        mobility_score = self._mobility_score(board)
        king_safety_score = self._king_safety_score(board)

        total_score = material_score + piece_square_score + mobility_score + king_safety_score
        return total_score if board.turn == chess.WHITE else -total_score

    def _mobility_score(self, board: chess.Board) -> float:
        current_mobility = board.legal_moves.count()
        board.push(chess.Move.null())
        opponent_mobility = board.legal_moves.count()
        board.pop()
        mobility_delta = current_mobility - opponent_mobility
        return mobility_delta * self.mobility_weight

    def _king_safety_score(self, board: chess.Board) -> float:
        score = 0.0
        for color in (chess.WHITE, chess.BLACK):
            king_square = board.king(color)
            if king_square is None:
                continue
            attackers = len(board.attackers(not color, king_square))
            penalty = attackers * self.king_safety_weight
            if color == chess.WHITE:
                score -= penalty
            else:
                score += penalty
        return score


class FallbackRandomStrategy(MoveStrategy):
    def __init__(self, random_move_provider: Callable[[chess.Board], Optional[str]], **kwargs):
        super().__init__(priority=0, **kwargs)
        self._random_move_provider = random_move_provider

    def is_applicable(self, context: StrategyContext) -> bool:
        return context.legal_moves_count > 0

    def generate_move(self, board: chess.Board, context: StrategyContext) -> Optional[StrategyResult]:
        move = self._random_move_provider(board)
        if not move:
            return None
        return StrategyResult(
            move=move,
            strategy_name=self.name,
            confidence=self.confidence,
            metadata={"source": "random_fallback"},
        )


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

        # Strategy management
        self.strategy_selector = StrategySelector(logger=self._log_debug)
        self._register_default_strategies()

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

    def _log_debug(self, message: str) -> None:
        if self.debug:
            print(f"info string {message}")

    def _register_default_strategies(self) -> None:
        opening_strategy = OpeningBookStrategy(
            opening_book=self._create_default_opening_book(),
            name="OpeningBookStrategy",
        )
        endgame_strategy = EndgameTableStrategy(name="EndgameTableStrategy")
        fallback_strategy = FallbackRandomStrategy(self.random_move, name="FallbackRandomStrategy")
        heuristic_strategy = HeuristicSearchStrategy(
            fallback=fallback_strategy,
            name="HeuristicSearchStrategy",
            search_depth=3,
        )

        for strategy in (opening_strategy, endgame_strategy, heuristic_strategy, fallback_strategy):
            self.strategy_selector.register_strategy(strategy)

    def _create_default_opening_book(self) -> Dict[str, str]:
        start_board = chess.Board()
        return {
            start_board.fen(): "e2e4",
        }

    def create_strategy_context(self, board: chess.Board) -> StrategyContext:
        piece_map = board.piece_map()
        piece_count = len(piece_map)
        material_imbalance = self.compute_material_imbalance(board)
        repetition_info = {
            "is_threefold_repetition": board.is_repetition(),
            "can_claim_threefold": board.can_claim_threefold_repetition(),
            "is_fivefold_repetition": board.is_fivefold_repetition(),
        }
        legal_moves_count = self.get_legal_moves_count(board)

        return StrategyContext(
            fullmove_number=board.fullmove_number,
            halfmove_clock=board.halfmove_clock,
            piece_count=piece_count,
            material_imbalance=material_imbalance,
            turn=board.turn,
            fen=board.fen(),
            repetition_info=repetition_info,
            legal_moves_count=legal_moves_count,
            time_controls=None,
        )

    def compute_material_imbalance(self, board: chess.Board) -> int:
        material_balance = 0
        for piece in board.piece_map().values():
            value = PIECE_VALUES.get(piece.piece_type, 0)
            material_balance += value if piece.color == chess.WHITE else -value
        return material_balance

    def get_legal_moves_count(self, board: chess.Board) -> int:
        return board.legal_moves.count()

    def register_strategy(self, strategy: MoveStrategy) -> None:
        self.strategy_selector.register_strategy(strategy)

    def set_selection_policy(
        self,
        policy: Optional[Callable[..., Optional[StrategyResult]]],
    ) -> None:
        self.strategy_selector.set_selection_policy(policy)

    def get_strategy_selector(self) -> StrategySelector:
        return self.strategy_selector

    def get_strategy_context(self) -> StrategyContext:
        with self.state_lock:
            board_snapshot = self.board.copy(stack=True)
        return self.create_strategy_context(board_snapshot)

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
        try:
            with self.state_lock:
                board_snapshot = self.board.copy(stack=True)
                context = self.create_strategy_context(board_snapshot)

            if self.debug:
                self._log_debug(
                    "context prepared: "
                    f"fullmove={context.fullmove_number}, "
                    f"halfmove={context.halfmove_clock}, "
                    f"pieces={context.piece_count}, "
                    f"material={context.material_imbalance}"
                )

            # Simulate thinking delay for now
            time.sleep(2)

            result = self.strategy_selector.select_move(board_snapshot, context) if self.strategy_selector else None

            if result and result.move:
                print(f"info string strategy {result.strategy_name} selected move {result.move}")
            elif self.debug:
                self._log_debug("no strategy produced a move")

            move = result.move if result else None

            with self.state_lock:
                if move:
                    print(f"bestmove {move}")
                else:
                    print("bestmove (none)")
                print("readyok")
                self.move_calculating = False
        except Exception as exc:
            print(f"info string Error generating move: {exc}")
            with self.state_lock:
                print("bestmove (none)")
                print("readyok")
                self.move_calculating = False



engine = ChessEngine()
engine.start()
