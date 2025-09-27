import contextlib
from collections import deque
import io
import re
import tempfile
from pathlib import Path
from typing import Optional

import chess
import pytest

import engine as engine_module
from self_play import SelfPlayManager


class DummyStrategy(engine_module.MoveStrategy):
    """Deterministic strategy used to make engine responses predictable in tests."""

    def __init__(self, move: str):
        super().__init__(name="DummyStrategy", priority=100, short_circuit=True)
        self._move = move

    def is_applicable(self, context: engine_module.StrategyContext) -> bool:  # type: ignore[name-defined]
        return True

    def generate_move(
        self, board: chess.Board, context: engine_module.StrategyContext  # type: ignore[name-defined]
    ) -> engine_module.StrategyResult:
        return engine_module.StrategyResult(
            move=self._move,
            strategy_name=self.name,
            confidence=1.0,
        )


@pytest.fixture()
def deterministic_engine():
    engine = engine_module.ChessEngine()
    engine.strategy_selector.clear_strategies()
    engine.register_strategy(DummyStrategy("e2e4"))
    return engine


def test_handle_uci_reports_identity(capsys):
    engine = engine_module.ChessEngine()
    engine.handle_uci()
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert output_lines[-3:] == [
        "id name JaskFish",
        "id author Jaskaran Singh",
        "uciok",
    ]


def test_handle_isready_reflects_engine_state(capsys, deterministic_engine):
    deterministic_engine.handle_isready("")
    ready_output = capsys.readouterr().out.strip().splitlines()
    assert ready_output[-1] == "readyok"

    deterministic_engine.move_calculating = True
    deterministic_engine.handle_isready("")
    busy_response = capsys.readouterr().out.strip()
    assert "Engine is busy" in busy_response


def test_handle_position_supports_startpos_and_fen(deterministic_engine):
    deterministic_engine.board.push_san("e4")
    deterministic_engine.handle_position("startpos")
    assert deterministic_engine.board.fen() == chess.STARTING_FEN

    target_fen = "rnbqkbnr/pp2pppp/2p5/3p4/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 0 4"
    deterministic_engine.handle_position(f"fen {target_fen}")
    assert deterministic_engine.board.fen() == target_fen


def test_handle_position_rejects_invalid_inputs(capsys, deterministic_engine):
    deterministic_engine.board.push_san("e4")
    original_fen = deterministic_engine.board.fen()

    deterministic_engine.handle_position("fen invalid-fen")
    output = capsys.readouterr().out
    assert "Invalid FEN" in output
    assert deterministic_engine.board.fen() == original_fen

    deterministic_engine.handle_position("loadsomething")
    output = capsys.readouterr().out
    assert "Unknown position command" in output
    assert deterministic_engine.board.fen() == original_fen


def test_handle_boardpos_reports_board_state(capsys, deterministic_engine):
    deterministic_engine.board.set_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
    deterministic_engine.handle_boardpos("")
    output = capsys.readouterr().out.strip()
    assert deterministic_engine.board.fen() in output


def test_handle_debug_toggles_engine_state(capsys, deterministic_engine):
    deterministic_engine.handle_debug("on")
    out_on = capsys.readouterr().out.splitlines()
    assert deterministic_engine.debug is True
    assert any("Debug:True" in line for line in out_on)

    deterministic_engine.handle_debug("off")
    out_off = capsys.readouterr().out.splitlines()
    assert deterministic_engine.debug is False
    assert any("Debug:False" in line for line in out_off)

    deterministic_engine.handle_debug("maybe")
    invalid = capsys.readouterr().out
    assert "Invalid debug setting" in invalid


def test_handle_ucinewgame_resets_board(capsys, deterministic_engine):
    deterministic_engine.board.push_san("e4")
    deterministic_engine.debug = True
    deterministic_engine.handle_ucinewgame("")
    output = capsys.readouterr().out
    assert deterministic_engine.board.fen() == chess.STARTING_FEN
    assert "New game initialized" in output


def test_handle_quit_sets_running_false(capsys, deterministic_engine):
    deterministic_engine.handle_quit("")
    output = capsys.readouterr().out
    assert "Engine shutting down" in output
    assert deterministic_engine.running is False


def test_parse_go_args_interprets_time_controls(deterministic_engine):
    parsed = deterministic_engine._parse_go_args(
        "wtime 1000 btime 2000 winc 10 binc 20 movestogo 30 movetime 40 depth 5 infinite ponder"
    )
    assert parsed == {
        "wtime": 1000,
        "btime": 2000,
        "winc": 10,
        "binc": 20,
        "movestogo": 30,
        "movetime": 40,
        "depth": 5,
        "infinite": True,
        "ponder": True,
    }


def test_parse_go_args_handles_incomplete_pairs(deterministic_engine):
    parsed = deterministic_engine._parse_go_args("wtime abc winc 15 depth")
    assert parsed == {"winc": 15}


def test_handle_go_prevents_reentrancy(capsys, deterministic_engine):
    deterministic_engine.move_calculating = True
    deterministic_engine.handle_go("wtime 5000")
    output = capsys.readouterr().out
    assert "Please wait" in output
    assert deterministic_engine.move_calculating is True


def test_create_strategy_context_exposes_board_snapshot(deterministic_engine):
    deterministic_engine.board.set_fen(
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    )
    time_controls = {"wtime": 2500, "btime": 1800}
    context = deterministic_engine.create_strategy_context(
        deterministic_engine.board.copy(stack=True), time_controls=time_controls
    )
    assert context.fen == deterministic_engine.board.fen()
    assert context.legal_moves_count == deterministic_engine.board.legal_moves.count()
    assert context.time_controls == time_controls
    assert context.piece_count == len(deterministic_engine.board.piece_map())


def test_create_strategy_context_captures_checks_and_mate_threats(deterministic_engine):
    repetition_board = chess.Board()
    sequence = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for _ in range(3):
        for move in sequence:
            repetition_board.push_uci(move)

    threat_board = chess.Board()
    threat_board.clear()
    threat_board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    threat_board.set_piece_at(chess.G2, chess.Piece(chess.PAWN, chess.WHITE))
    threat_board.set_piece_at(chess.H2, chess.Piece(chess.PAWN, chess.WHITE))
    threat_board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    threat_board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.BLACK))
    threat_board.turn = chess.WHITE

    deterministic_engine.board = threat_board
    context = deterministic_engine.create_strategy_context(
        threat_board.copy(stack=True)
    )
    assert context.in_check is False
    assert context.opponent_mate_in_one_threat is True

    deterministic_engine.board = repetition_board
    context = deterministic_engine.create_strategy_context(
        repetition_board.copy(stack=True)
    )
    assert context.repetition_info["is_threefold_repetition"] is True
    assert context.repetition_info["can_claim_threefold"] is True
    assert context.repetition_info["is_fivefold_repetition"] is False


def test_process_go_command_uses_registered_strategy(capsys, deterministic_engine):
    deterministic_engine.move_calculating = True
    deterministic_engine.process_go_command({})
    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any(line.startswith("bestmove e2e4") for line in output_lines)
    assert "readyok" in output_lines
    assert deterministic_engine.move_calculating is False


def test_process_go_command_handles_strategy_exception(capsys, deterministic_engine, monkeypatch):
    deterministic_engine.move_calculating = True

    def explode(board, context):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        deterministic_engine.strategy_selector,
        "select_move",
        lambda board, context: explode(board, context),
    )

    deterministic_engine.process_go_command({})
    output = capsys.readouterr().out
    assert "Error generating move" in output
    assert "bestmove (none)" in output
    assert "readyok" in output
    assert deterministic_engine.move_calculating is False


def test_process_go_command_handles_missing_move(capsys, deterministic_engine, monkeypatch):
    monkeypatch.setattr(
        deterministic_engine.strategy_selector,
        "select_move",
        lambda board, context: None,
    )

    def fail_random_move(board):
        raise AssertionError("random fallback should not be invoked")

    monkeypatch.setattr(deterministic_engine, "random_move", fail_random_move)
    deterministic_engine.move_calculating = True
    deterministic_engine.process_go_command({})
    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert not any("fallback RandomFallback" in line for line in output_lines)
    assert "bestmove (none)" in output_lines
    assert "readyok" in output_lines
    assert deterministic_engine.move_calculating is False


def test_random_move_handles_positions_without_legal_moves(deterministic_engine):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert deterministic_engine.random_move(board) is None


def test_register_default_strategies_respects_feature_flags(monkeypatch):
    for key in engine_module.STRATEGY_ENABLE_FLAGS:
        monkeypatch.setitem(engine_module.STRATEGY_ENABLE_FLAGS, key, False)
    monkeypatch.setitem(engine_module.STRATEGY_ENABLE_FLAGS, "heuristic", True)
    monkeypatch.setitem(engine_module.STRATEGY_ENABLE_FLAGS, "fallback_random", True)

    engine = engine_module.ChessEngine()
    strategy_names = [strategy.name for strategy in engine.strategy_selector.get_strategies()]
    assert strategy_names == ["HeuristicSearchStrategy", "FallbackRandomStrategy"]


def test_strategy_selector_priority_and_short_circuit():
    selector = engine_module.StrategySelector()
    board = chess.Board()
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=board.legal_moves.count(),
    )

    triggered = {"second_called": False}

    class First(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="First", priority=10, short_circuit=True)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            return engine_module.StrategyResult("e2e4", strategy_name=self.name)

    class Second(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="Second", priority=1)

        def is_applicable(self, context):  # type: ignore[override]
            triggered["second_called"] = True
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            triggered["second_called"] = True
            return engine_module.StrategyResult("d2d4", strategy_name=self.name)

    selector.register_strategy(Second())
    selector.register_strategy(First())
    strategies = selector.get_strategies()
    assert strategies[0].name == "First"

    result = selector.select_move(board, context)
    assert result is not None and result.move == "e2e4"
    assert triggered["second_called"] is False


def test_strategy_selector_custom_policy():
    captured_logs = []

    def logger(message: str) -> None:
        captured_logs.append(message)

    selector = engine_module.StrategySelector(logger=logger)
    board = chess.Board()
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=board.legal_moves.count(),
    )

    class Low(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="Low", priority=1, short_circuit=False)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            return engine_module.StrategyResult(
                "a2a4", strategy_name=self.name, score=1.0, confidence=0.2
            )

    class High(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="High", priority=2, short_circuit=False)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            return engine_module.StrategyResult(
                "b2b4", strategy_name=self.name, score=0.5, confidence=0.9
            )

    selector.register_strategy(Low())
    selector.register_strategy(High())

    def custom_policy(results, **_):
        return max(results, key=lambda item: item[1].confidence)[1]

    selector.set_selection_policy(custom_policy)
    result = selector.select_move(board, context)
    assert result is not None and result.move == "b2b4"
    assert any("strategy evaluating" in entry for entry in captured_logs)


def test_strategy_selector_ignores_exceptions():
    selector = engine_module.StrategySelector()
    board = chess.Board()
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=board.legal_moves.count(),
    )

    class Explodes(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="Explodes", priority=5)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            raise RuntimeError("boom")

    class Safe(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="Safe", priority=1)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            return engine_module.StrategyResult("h2h4", strategy_name=self.name)

    selector.register_strategy(Explodes())
    selector.register_strategy(Safe())

    result = selector.select_move(board, context)
    assert result is not None and result.move == "h2h4"


def test_detect_opponent_mate_in_one_threat_flags_positions(deterministic_engine):
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.G2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.H2, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.G3, chess.Piece(chess.QUEEN, chess.BLACK))
    board.set_piece_at(chess.E1, chess.Piece(chess.ROOK, chess.BLACK))
    board.turn = chess.WHITE
    assert deterministic_engine.detect_opponent_mate_in_one_threat(board) is True

    board.turn = chess.BLACK
    assert deterministic_engine.detect_opponent_mate_in_one_threat(board) is False


def test_compute_material_imbalance_accounts_for_piece_values(deterministic_engine):
    board = chess.Board()
    board.clear()
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E5, chess.Piece(chess.ROOK, chess.BLACK))
    imbalance = deterministic_engine.compute_material_imbalance(board)
    assert imbalance == 9 - 5


def test_get_legal_moves_count_matches_board_api(deterministic_engine):
    board = chess.Board()
    board.set_fen("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
    expected = board.legal_moves.count()
    assert deterministic_engine.get_legal_moves_count(board) == expected


def test_command_processor_handles_unknown_commands(monkeypatch, capsys):
    engine = engine_module.ChessEngine()
    commands = io.StringIO("foo\nquit\n")
    monkeypatch.setattr(engine_module.sys, "stdin", commands)
    engine.command_processor()
    output = capsys.readouterr().out.splitlines()
    assert "unknown command received: ''" in output
    assert any("Engine shutting down" in line for line in output)


def test_parse_go_args_handles_ponder_flag(deterministic_engine):
    parsed = deterministic_engine._parse_go_args("ponder")
    assert parsed == {"ponder": True}


def test_process_go_command_logs_when_strategies_fail(monkeypatch, deterministic_engine, capsys):
    deterministic_engine.strategy_selector.clear_strategies()

    class NullStrategy(engine_module.MoveStrategy):
        def __init__(self):
            super().__init__(name="Null", priority=99)

        def is_applicable(self, context):  # type: ignore[override]
            return True

        def generate_move(self, board, context):  # type: ignore[override]
            return None

    deterministic_engine.register_strategy(NullStrategy())
    deterministic_engine.debug = True
    deterministic_engine.move_calculating = True
    deterministic_engine.process_go_command({})
    output_lines = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert any("strategy produced no result: Null" in line for line in output_lines)
    assert "bestmove (none)" in output_lines
    assert deterministic_engine.move_calculating is False


def test_heuristic_time_budget_derivation():
    strategy = engine_module.HeuristicSearchStrategy(
        base_time_limit=0.5,
        max_time_limit=5.0,
        min_time_limit=0.1,
        time_allocation_factor=0.05,
    )
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=20,
        time_controls={"wtime": 2000, "btime": 3000, "winc": 100, "movestogo": 10},
    )
    assert pytest.approx(strategy._determine_time_budget(context), 0.0001) == 0.3

    infinite_context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=20,
        time_controls={"infinite": True},
    )
    assert strategy._determine_time_budget(infinite_context) is None

    movetime_context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=20,
        time_controls={"movetime": 30},
    )
    assert strategy._determine_time_budget(movetime_context) == strategy.min_time_limit


def test_heuristic_timeout_returns_none_when_time_exceeded():
    strategy = engine_module.HeuristicSearchStrategy(
        search_depth=2, quiescence_depth=0
    )
    board = chess.Board()
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=board.legal_moves.count(),
    )

    strategy._time_exceeded = lambda: True

    result = strategy.generate_move(board, context)
    assert result is None


def test_quiescence_move_order_prioritizes_captures_and_checks():
    strategy = engine_module.HeuristicSearchStrategy(search_depth=1, quiescence_depth=2)
    board = chess.Board("4k3/4q3/2r5/8/3N4/4Q3/8/4K3 w - - 0 1")
    moves = [move.uci() for move in strategy._generate_quiescence_moves(board)]
    assert moves[:2] == ["e3e7", "d4c6"]


def test_transposition_and_killer_tables_are_bounded():
    strategy = engine_module.HeuristicSearchStrategy(search_depth=1)
    strategy._transposition_table_limit = 5
    for idx in range(10):
        strategy._store_transposition_entry(idx, 1, float(idx), engine_module.TranspositionFlag.EXACT, None)
    assert len(strategy._transposition_table) <= 5

    strategy._killer_moves = [[None] for _ in range(3)]
    move_a = chess.Move.from_uci("e2e4")
    move_b = chess.Move.from_uci("d2d4")
    move_c = chess.Move.from_uci("c2c4")
    strategy._record_killer(0, move_a)
    strategy._record_killer(0, move_b)
    strategy._record_killer(0, move_c)
    assert len(strategy._killer_moves[0]) <= strategy._killer_slots


def test_history_scores_decay_when_exceeding_threshold():
    strategy = engine_module.HeuristicSearchStrategy(search_depth=1)
    move = chess.Move.from_uci("e2e4")
    key = (True, move.from_square, move.to_square)
    for _ in range(1000):
        strategy._record_history(True, move, depth=30)
    assert strategy._history_scores[key] < 500000


def test_history_scores_decay_and_prune_entries():
    strategy = engine_module.HeuristicSearchStrategy(search_depth=1)
    keep_key = (True, chess.E2, chess.E4)
    prune_key = (True, chess.A2, chess.A3)
    strategy._history_scores[keep_key] = 100.0
    strategy._history_scores[prune_key] = 0.5

    strategy._decay_history_scores(factor=0.5)

    assert pytest.approx(strategy._history_scores[keep_key], 0.01) == 50.0
    assert prune_key not in strategy._history_scores


def test_heuristic_generate_move_populates_principal_variation():
    board = chess.Board("8/8/8/3k4/8/8/8/3K4 w - - 0 1")
    strategy = engine_module.HeuristicSearchStrategy(search_depth=2, quiescence_depth=0)
    context = engine_module.StrategyContext(
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=board.fen(),
        legal_moves_count=board.legal_moves.count(),
    )

    result = strategy.generate_move(board, context)
    assert result is not None
    pv_line = result.metadata.get("pv")
    assert isinstance(pv_line, list)
    if pv_line:
        assert pv_line[0] == result.move
    assert result.metadata["depth"] >= 1
    assert result.metadata.get("principal_move") == result.move


def test_static_exchange_score_falls_back_when_missing_see():
    strategy = engine_module.HeuristicSearchStrategy(search_depth=1)
    board = chess.Board()
    move = chess.Move.from_uci("g1f3")
    board.see = lambda _move: (_ for _ in ()).throw(ValueError("missing"))
    assert strategy._static_exchange_score(board, move) == strategy._capture_score(
        board, move
    )


def test_alpha_beta_triggers_null_move_pruning():
    board = chess.Board("4k3/8/8/8/8/8/4PPP1/3K4 w - - 0 1")
    strategy = engine_module.HeuristicSearchStrategy(search_depth=3, quiescence_depth=0)
    strategy._futility_depth_limit = 0
    strategy._razoring_depth_limit = 0
    strategy._null_move_min_depth = 2
    strategy._lmr_min_depth = 10
    null_depths: list[int] = []
    strategy._null_move_reduction = lambda depth: null_depths.append(depth) or 1

    strategy._alpha_beta(board, depth=3, alpha=-1000, beta=1000, ply=0, is_pv=False)

    assert null_depths and all(depth >= 2 for depth in null_depths)


def test_alpha_beta_applies_late_move_reductions():
    board = chess.Board("4k3/8/8/8/8/8/4PPP1/3K4 w - - 0 1")
    strategy = engine_module.HeuristicSearchStrategy(search_depth=3, quiescence_depth=0)
    strategy._null_move_min_depth = 99  # disable null-move pruning to focus on LMR
    strategy._lmr_min_depth = 2
    strategy._lmr_min_move_index = 1
    strategy._futility_depth_limit = 0
    strategy._razoring_depth_limit = 0
    strategy._evaluate_board = lambda _board: 0.0
    reductions: list[tuple[int, int]] = []
    strategy._late_move_reduction = (
        lambda depth, index: reductions.append((depth, index)) or 1
    )

    strategy._alpha_beta(board, depth=2, alpha=-1000, beta=1000, ply=0, is_pv=False)

    assert any(index >= 1 for _, index in reductions)


def test_generate_move_expands_aspiration_window_on_failures():
    board = chess.Board()
    strategy = engine_module.HeuristicSearchStrategy(search_depth=3, quiescence_depth=0)
    strategy._resolve_depth_limit = lambda _context: 2
    root_move = chess.Move.from_uci("h2h3")
    strategy._order_moves = (
        lambda *args, **kwargs: [root_move]
    )

    outputs = iter([-50.0, -500.0, -150.0, -150.0])
    windows: list[tuple[float, float, int]] = []

    def scripted_alpha_beta(board, depth, alpha, beta, ply, is_pv, allow_null=True):
        windows.append((alpha, beta, depth))
        try:
            return next(outputs)
        except StopIteration:
            return -150.0

    strategy._alpha_beta = scripted_alpha_beta

    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=board.fen(),
        legal_moves_count=board.legal_moves.count(),
    )

    result = strategy.generate_move(board, context)

    assert result is not None and result.move == root_move.uci()
    depth_one_windows = {
        (alpha, beta)
        for alpha, beta, depth in windows
        if depth == 1
    }
    assert len(depth_one_windows) >= 2


def test_heuristic_trace_logging_across_positions():
    messages: list[str] = []
    strategy = engine_module.HeuristicSearchStrategy(
        search_depth=2,
        quiescence_depth=0,
        base_time_limit=0.4,
        max_time_limit=0.6,
        min_time_limit=0.05,
        logger=messages.append,
    )

    fens = [
        chess.STARTING_FEN,
        "rnbqkb1r/pppp1ppp/5n2/4p3/1P6/2P5/P2PPPPP/RNBQKBNR w KQkq - 0 3",
        "2r2rk1/1b2bppp/p2p1n2/1p1Pp3/1P2P3/P1N1BN2/1B3PPP/2RR2K1 w - - 0 21",
    ]

    for fen in fens:
        board = chess.Board(fen)
        context = engine_module.StrategyContext(
            fullmove_number=board.fullmove_number,
            halfmove_clock=board.halfmove_clock,
            piece_count=len(board.piece_map()),
            material_imbalance=0,
            turn=board.turn,
            fen=board.fen(),
            legal_moves_count=board.legal_moves.count(),
        )
        result = strategy.generate_move(board, context)
        assert result is not None

    assert any("depth=1" in msg for msg in messages)
    assert any("completed depth" in msg for msg in messages)


def test_heuristic_stops_when_depth_consumes_budget(monkeypatch):
    messages: list[str] = []
    strategy = engine_module.HeuristicSearchStrategy(
        search_depth=3,
        quiescence_depth=0,
        base_time_limit=1.0,
        max_time_limit=1.0,
        min_time_limit=0.1,
        logger=messages.append,
    )
    strategy._depth_iteration_stop_ratio = 0.2

    class FakeTimer:
        def __init__(self, step: float = 0.3):
            self.current = 0.0
            self.step = step

        def perf_counter(self) -> float:
            value = self.current
            self.current += self.step
            return value

    fake_timer = FakeTimer()
    monkeypatch.setattr(engine_module.time, "perf_counter", fake_timer.perf_counter)

    root_move = chess.Move.from_uci("e2e4")
    strategy._order_moves = lambda *args, **kwargs: [root_move]
    strategy._generate_quiescence_moves = lambda board: []

    def scripted_alpha_beta(board, depth, alpha, beta, ply, is_pv, allow_null=True):
        strategy._nodes_visited += 1
        return 0.0

    strategy._alpha_beta = scripted_alpha_beta

    board = chess.Board()
    context = engine_module.StrategyContext(
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=board.fen(),
        legal_moves_count=board.legal_moves.count(),
        time_controls={"movetime": 1000},
    )

    result = strategy.generate_move(board, context)
    assert result is not None
    assert any("consumed" in msg for msg in messages)
    assert not any("depth=2" in msg for msg in messages)


@pytest.fixture
def require_dev_marker(request):
    marker_expression = request.config.getoption("-m")
    if not marker_expression or "dev" not in marker_expression:
        pytest.skip("Run with -m dev to execute dev-only tests")


class HeadlessSelfPlayUI:
    def __init__(self, board: Optional[chess.Board] = None) -> None:
        self.board = board or chess.Board()
        self.self_play_active = False
        self.board_enabled = True
        self.manual_enabled = True
        self.info_message = ""
        self.last_activity: Optional[tuple[str, str]] = None

    def set_self_play_active(self, active: bool) -> None:
        self.self_play_active = active

    def set_board_interaction_enabled(self, enabled: bool) -> None:
        self.board_enabled = enabled

    def set_manual_controls_enabled(self, enabled: bool) -> None:
        self.manual_enabled = enabled

    def set_info_message(self, message: str) -> None:
        self.info_message = message

    def indicate_engine_activity(self, engine_label: str, context: str) -> None:
        self.last_activity = (engine_label, context)
        self.info_message = f"{context}: {engine_label} evaluating…"

    def clear_engine_activity(self, message: Optional[str] = None) -> None:
        self.last_activity = None
        self.info_message = message or "Engines idle"

    def self_play_evaluation_complete(self, engine_label: str) -> None:
        self.info_message = f"Self-play: {engine_label} move received"


class EngineHarness:
    def __init__(self) -> None:
        self.engine = engine_module.ChessEngine()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.engine.handle_debug("on")
        self.bootstrap_output = buffer.getvalue()

    def process_command(self, command: str) -> str:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self._dispatch(command)
        return buffer.getvalue()

    def _dispatch(self, command: str) -> None:
        if command == "ucinewgame":
            self.engine.handle_ucinewgame("")
            return
        if command.startswith("position "):
            self.engine.handle_position(command[len("position "):])
            return
        if command == "go":
            self.engine.move_calculating = True
            self.engine.process_go_command({})
            return
        if command == "isready":
            self.engine.handle_isready("")
            return
        if command.startswith("debug"):
            self.engine.handle_debug(command[5:].strip())
            return
        if command == "stop":
            return
        self.engine.handle_unknown(command)


class SelfPlayTestHarness:
    def __init__(self, ui: Optional[HeadlessSelfPlayUI] = None) -> None:
        self.ui = ui or HeadlessSelfPlayUI()
        self.white_engine = EngineHarness()
        self.black_engine = EngineHarness()
        self._trace_dir_cm = tempfile.TemporaryDirectory(prefix="self_play_traces_")
        trace_dir = Path(self._trace_dir_cm.name)

        assert "Debug:True" in self.white_engine.bootstrap_output
        assert "Debug:True" in self.black_engine.bootstrap_output

        self.pending_go_outputs = {chess.WHITE: deque(), chess.BLACK: deque()}

        engines = {chess.WHITE: self.white_engine, chess.BLACK: self.black_engine}
        self._reverse_lookup = {self.white_engine: chess.WHITE, self.black_engine: chess.BLACK}

        def send_command(engine_handle: EngineHarness, command: str) -> None:
            chunk = engine_handle.process_command(command)
            color = self._reverse_lookup[engine_handle]
            if command == "go":
                for line in chunk.splitlines():
                    stripped = line.strip()
                    if stripped:
                        self.manager.on_engine_output(color, stripped)
                self.pending_go_outputs[color].append(chunk)

        self.manager = SelfPlayManager(
            self.ui,
            engines,
            send_command,
            {
                chess.WHITE: "White - Engine Harness [1]",
                chess.BLACK: "Black - Engine Harness [2]",
            },
            trace_directory=trace_dir,
        )

    def start(self) -> None:
        assert self.manager.start() is True
        assert self.ui.self_play_active is True
        assert self.ui.board_enabled is False
        assert self.ui.manual_enabled is False

    def play_plies(self, plies: int) -> tuple[list[str], list[str]]:
        captured_moves: list[str] = []
        go_chunks: list[str] = []
        for _ in range(plies):
            color = self.manager.current_expected_color()
            assert color is not None
            assert self.pending_go_outputs[color], "Expected pending search output"
            go_chunk = self.pending_go_outputs[color].popleft()
            go_chunks.append(go_chunk)
            match = re.search(r"bestmove\s+(\S+)", go_chunk)
            assert match, "Expected bestmove output from engine"
            move_uci = match.group(1)
            move = chess.Move.from_uci(move_uci)
            assert move in self.ui.board.legal_moves, "Engine produced illegal move"
            self.ui.board.push(move)
            captured_moves.append(move_uci)
            self.manager.on_engine_move(color, move_uci)
        return captured_moves, go_chunks

    def stop(self) -> None:
        assert self.manager.stop("test complete") is True
        assert self.ui.self_play_active is False
        assert self.ui.board_enabled is True
        assert self.ui.manual_enabled is True

    def trace_path(self) -> Optional[Path]:
        return self.manager.last_trace_path

    def trace_contents(self) -> str:
        path = self.manager.last_trace_path
        assert path is not None, "Expected a trace file to be exported"
        return path.read_text(encoding="utf-8")


@pytest.mark.dev
def test_headless_self_play_debug_trace(require_dev_marker):
    harness = SelfPlayTestHarness()
    harness.start()

    plies = 12
    captured_moves, go_chunks = harness.play_plies(plies)
    harness.stop()

    combined_trace = "".join(go_chunks)
    assert combined_trace.count("bestmove") == plies
    assert all("HeuristicSearchStrategy: start search" in chunk for chunk in go_chunks)
    assert any("completed depth=5" in chunk for chunk in go_chunks)
    assert "time_budget=" in combined_trace
    assert captured_moves[:6] == [
        "g1f3",
        "g8f6",
        "f3d4",
        "b8c6",
        "d4c6",
        "b7c6",
    ]
    assert len(captured_moves) == plies
    assert harness.ui.board.fullmove_number >= 7

    trace_path = harness.trace_path()
    assert trace_path is not None
    trace_data = harness.trace_contents()
    assert "HeuristicSearchStrategy: start search" in trace_data
    assert "bestmove" in trace_data
    assert "Initial FEN:" in trace_data


@pytest.mark.dev
def test_headless_self_play_midgame_trace(require_dev_marker):
    ui = HeadlessSelfPlayUI()
    midgame_fen = "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10"
    ui.board.set_fen(midgame_fen)

    harness = SelfPlayTestHarness(ui)
    harness.start()

    plies = 8
    captured_moves, go_chunks = harness.play_plies(plies)
    harness.stop()

    combined_trace = "".join(go_chunks)
    assert combined_trace.count("bestmove") == plies
    assert any("search timeout" in chunk for chunk in go_chunks)
    assert any("completed depth=3" in chunk for chunk in go_chunks)
    expected_line = [
        "d4c5",
        "e7c5",
        "d1e1",
        "c5e3",
        "f2e3",
        "d8e8",
        "f1f2",
        "a8d8",
    ]
    assert captured_moves == expected_line
    assert len(captured_moves) == plies
    assert harness.ui.board.fullmove_number >= 14

    trace_path = harness.trace_path()
    assert trace_path is not None
    trace_data = harness.trace_contents()
    assert midgame_fen in trace_data
    assert "search timeout" in trace_data
    assert "Stop reason:" in trace_data
