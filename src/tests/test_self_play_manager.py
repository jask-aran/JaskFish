import chess
import pytest

from src.self_play import SelfPlayManager


class DummyGui:
    def __init__(self, board: chess.Board):
        self.board = board
        self.self_play_active = False
        self.board_enabled = True
        self.manual_enabled = True
        self.info_message = ""
        self.last_activity = None

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

    def clear_engine_activity(self, message: str = "") -> None:
        self.last_activity = None
        self.info_message = message or "Engines idle"

    def self_play_evaluation_complete(self, engine_label: str) -> None:
        self.info_message = f"Self-play: {engine_label} move received"


class EngineStub:
    def __init__(self) -> None:
        self.commands = []


@pytest.fixture()
def self_play_setup():
    board = chess.Board()
    gui = DummyGui(board)
    white_engine = EngineStub()
    black_engine = EngineStub()

    def send_command_stub(engine, command: str) -> None:
        engine.commands.append(command)

    manager = SelfPlayManager(
        gui,
        {chess.WHITE: white_engine, chess.BLACK: black_engine},
        send_command_stub,
        {chess.WHITE: "Engine 1 [White]", chess.BLACK: "Engine 2 [Black]"},
    )

    return manager, gui, white_engine, black_engine


def test_self_play_start_initialises_engines(self_play_setup):
    manager, gui, white_engine, black_engine = self_play_setup

    assert manager.start() is True
    assert gui.self_play_active is True
    assert gui.board_enabled is False
    assert gui.manual_enabled is False
    assert gui.info_message == "Self-play: Engine 1 [White] evaluating…"

    assert white_engine.commands[0] == "ucinewgame"
    assert white_engine.commands[1] == f"position fen {gui.board.fen()}"
    assert white_engine.commands[2] == "go"
    assert black_engine.commands == ["ucinewgame"]


def test_self_play_stop_sends_stop_and_resets_ui(self_play_setup):
    manager, gui, white_engine, _ = self_play_setup

    manager.start()
    assert manager.stop() is True

    assert white_engine.commands[-1] == "stop"
    assert gui.self_play_active is False
    assert gui.board_enabled is True
    assert gui.manual_enabled is True
    assert gui.info_message.startswith("Self-play stopped")

    # First engine response after stop should be ignored
    assert manager.should_apply_move(chess.WHITE) is False
    assert manager.should_apply_move(chess.WHITE) is True


def test_self_play_requests_alternate_engine(self_play_setup):
    manager, gui, white_engine, black_engine = self_play_setup

    manager.start()
    move = chess.Move.from_uci("e2e4")
    gui.board.push(move)
    manager.on_engine_move(chess.WHITE, "e2e4")

    assert black_engine.commands[0] == "ucinewgame"
    assert black_engine.commands[1] == f"position fen {gui.board.fen()}"
    assert black_engine.commands[2] == "go"
    assert manager.active is True
    assert gui.info_message == "Self-play: Engine 2 [Black] evaluating…"
