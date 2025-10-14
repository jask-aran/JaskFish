import chess
import pytest

from main import SelfPlayManager


class DummyGui:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.self_play_active = False
        self.board_enabled = True
        self.manual_enabled = True
        self.info_message = ""
        self.last_activity = None
        self.completed = []

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
        self.info_message = f"{context}: {engine_label} evaluating"

    def clear_engine_activity(self, message: str = "") -> None:
        self.last_activity = None
        self.info_message = message or "Engines idle"

    def self_play_evaluation_complete(self, engine_label: str) -> None:
        self.completed.append(engine_label)
        self.info_message = f"Self-play: {engine_label} move received"


class EngineStub:
    def __init__(self) -> None:
        self.commands = []


def make_manager(gui=None, *, capture_payload: bool = True, trace_dir=None):
    gui = gui or DummyGui()
    white_engine = EngineStub()
    black_engine = EngineStub()

    def send_command(engine, command: str) -> None:
        engine.commands.append(command)

    manager = SelfPlayManager(
        gui,
        {chess.WHITE: white_engine, chess.BLACK: black_engine},
        send_command,
        {
            chess.WHITE: "White Engine",
            chess.BLACK: "Black Engine",
        },
        capture_payload=capture_payload,
        trace_directory=trace_dir,
    )
    return manager, gui, white_engine, black_engine


def test_start_initialises_engines_and_requests_move(tmp_path) -> None:
    manager, gui, white_engine, black_engine = make_manager(trace_dir=tmp_path)
    assert manager.start() is True
    assert gui.self_play_active is True
    assert gui.board_enabled is False
    assert gui.manual_enabled is False
    assert gui.info_message == "Self-play: White Engine evaluating"
    assert white_engine.commands[0] == "ucinewgame"
    assert white_engine.commands[1] == f"position fen {gui.board.fen()}"
    assert white_engine.commands[2].startswith("go ")
    assert "movetime" in white_engine.commands[2]
    assert black_engine.commands == ["ucinewgame"]


def test_start_with_shared_engine_sends_single_ucinewgame(tmp_path) -> None:
    gui = DummyGui()
    shared = EngineStub()

    def send(engine, command: str) -> None:
        engine.commands.append(command)

    manager = SelfPlayManager(
        gui,
        {chess.WHITE: shared, chess.BLACK: shared},
        send,
        {
            chess.WHITE: "Same",
            chess.BLACK: "Same",
        },
        trace_directory=tmp_path,
    )

    assert manager.start() is True
    assert shared.commands.count("ucinewgame") == 1


def test_on_engine_move_switches_to_opponent(tmp_path) -> None:
    manager, gui, white_engine, black_engine = make_manager(trace_dir=tmp_path)
    manager.start()
    move = chess.Move.from_uci("e2e4")
    gui.board.push(move)
    manager.on_engine_move(chess.WHITE, move.uci())

    assert "White Engine" in gui.completed
    assert gui.info_message == "Self-play: Black Engine evaluating"
    assert black_engine.commands[-2] == f"position fen {gui.board.fen()}"
    assert black_engine.commands[-1].startswith("go ")
    assert "movetime" in black_engine.commands[-1]
    assert manager.current_expected_color() == chess.BLACK


def test_stop_sends_stop_and_exports_trace(tmp_path) -> None:
    manager, gui, white_engine, _ = make_manager(trace_dir=tmp_path)
    manager.start()
    manager.on_engine_output(chess.WHITE, "info string depth=1")
    manager.stop("Finished")

    assert white_engine.commands[-1] == "stop"
    assert gui.self_play_active is False
    assert gui.board_enabled is True
    assert gui.manual_enabled is True
    assert "Finished" in gui.info_message
    trace_files = list(tmp_path.glob("*_selfplay.txt"))
    assert trace_files, "Expected trace export"
    assert manager.last_trace_path is not None
    assert manager.last_trace_path.exists()


def test_should_apply_move_respects_pending_ignore(tmp_path) -> None:
    manager, gui, white_engine, _ = make_manager(trace_dir=tmp_path)
    manager.start()
    manager.stop()
    assert manager.should_apply_move(chess.WHITE) is False
    assert manager.should_apply_move(chess.WHITE) is True


def test_update_engines_stops_active_session(tmp_path) -> None:
    manager, gui, white_engine, black_engine = make_manager(trace_dir=tmp_path)
    manager.start()
    replacement = EngineStub()

    manager.update_engines(
        {chess.WHITE: replacement, chess.BLACK: black_engine},
        {
            chess.WHITE: "Replacement",
            chess.BLACK: "Black Engine",
        },
    )

    assert white_engine.commands[-1] == "stop"
    assert manager.active is False
    assert manager.current_expected_color() is None


def test_on_engine_output_respects_capture_flag(tmp_path) -> None:
    manager, gui, white_engine, _ = make_manager(trace_dir=tmp_path, capture_payload=False)
    manager.start()
    manager.on_engine_output(chess.WHITE, "info string perf payload={\"nodes\":1}")
    assert manager._session_traces[chess.WHITE] == []
    manager.on_engine_output(chess.WHITE, "info string depth=2")
    assert manager._session_traces[chess.WHITE] == ["info string depth=2"]
    manager.stop()

    trace_files = list(tmp_path.glob("*_selfplay.txt"))
    assert trace_files
    content = trace_files[0].read_text()
    assert "payload" not in content
    assert "depth=2" in content
