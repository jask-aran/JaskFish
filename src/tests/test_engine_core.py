import io

import chess
import pytest

from src import engine as engine_module


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
    engine = engine_module.ChessEngine(opening_book_path=None)
    engine.strategy_selector.clear_strategies()
    engine.register_strategy(DummyStrategy("e2e4"))
    return engine


@pytest.fixture()
def engine_with_opening_book(monkeypatch):
    # Reactivate the opening book strategy for tests even if disabled in defaults.
    monkeypatch.setitem(engine_module.STRATEGY_ENABLE_FLAGS, "opening_book", True)

    engine = engine_module.ChessEngine(opening_book_path=None)
    strategies = engine.strategy_selector.get_strategies()
    opening_strategy = next(
        (
            strategy
            for strategy in strategies
            if isinstance(strategy, engine_module.OpeningBookStrategy)
        ),
        None,
    )
    if opening_strategy is None:
        pytest.skip("OpeningBookStrategy is not available when opening book is disabled")
    return engine, opening_strategy


def test_handle_uci_reports_identity(capsys):
    engine = engine_module.ChessEngine(opening_book_path=None)
    engine.handle_uci()
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert output_lines == [
        "id name JaskFish",
        "id author Jaskaran Singh",
        "uciok",
    ]


def test_handle_isready_reflects_engine_state(capsys, deterministic_engine):
    deterministic_engine.handle_isready("")
    assert capsys.readouterr().out.strip() == "readyok"

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


def test_opening_book_start_position_metadata(engine_with_opening_book):
    engine, opening_strategy = engine_with_opening_book
    board = chess.Board()
    context = engine.create_strategy_context(board.copy(stack=True))

    assert opening_strategy.is_applicable(context)

    result = opening_strategy.generate_move(board, context)
    assert result is not None
    assert result.move == "e2e4"
    assert result.metadata["source"] == "dict_book"
    assert result.metadata["book_moves"] == ["e2e4", "d2d4", "c2c4", "g1f3"]


@pytest.mark.parametrize(
    "moves, expected",
    [
        (["e2e4"], "c7c5"),
        (["e2e4", "c7c5"], "g1f3"),
        (["d2d4"], "g8f6"),
        (["d2d4", "g8f6"], "c2c4"),
        (["c2c4"], "e7e5"),
        (["c2c4", "e7e5"], "b1c3"),
    ],
)
def test_opening_book_varied_responses(engine_with_opening_book, moves, expected):
    engine, opening_strategy = engine_with_opening_book
    board = chess.Board()
    for move in moves:
        board.push_uci(move)

    context = engine.create_strategy_context(board.copy(stack=True))
    assert opening_strategy.is_applicable(context)

    result = opening_strategy.generate_move(board, context)
    assert result is not None
    assert result.move == expected
    assert result.metadata["source"] == "dict_book"


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

    engine = engine_module.ChessEngine(opening_book_path=None)
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
    engine = engine_module.ChessEngine(opening_book_path=None)
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
    setattr(board, "zobrist_hash", lambda: hash(board.fen()))
    context = engine_module.StrategyContext(
        fullmove_number=1,
        halfmove_clock=0,
        piece_count=32,
        material_imbalance=0,
        turn=chess.WHITE,
        legal_moves_count=board.legal_moves.count(),
    )

    strategy._alpha_beta = lambda *args, **kwargs: (_ for _ in ()).throw(engine_module._SearchTimeout())

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
