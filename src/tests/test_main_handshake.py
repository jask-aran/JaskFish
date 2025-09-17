import sys

import chess
import pytest
from PySide2.QtWidgets import QApplication

from gui import ChessGUI
from main import engine_output_processor, start_engine_process


class DummyProcess:
    def __init__(self):
        self.started = None
        self.writes = []

    def start(self, program, arguments):
        self.started = (program, arguments)

    def write(self, data):
        self.writes.append(data)

    def waitForBytesWritten(self):
        return True


class OutputProcess(DummyProcess):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = [output.encode() for output in outputs]

    def canReadLine(self):
        return bool(self._outputs)

    def readLine(self):
        return self._outputs.pop(0)


@pytest.fixture(scope="session")
def qt_app():
    existing = QApplication.instance()
    if existing:
        yield existing
        return
    app = QApplication(sys.argv)
    yield app
    app.quit()


def test_start_engine_process_sends_handshake(qt_app):
    gui = ChessGUI(chess.Board(), dev=True)
    proc = DummyProcess()
    engine_state = {"debug_on_handshake": True}

    start_engine_process(proc, ["stockfish"], gui, engine_state)

    assert engine_state["handshake_complete"] is False
    assert not gui.ready_button.isEnabled()
    assert proc.started == ("stockfish", [])
    assert proc.writes and proc.writes[0] == b"uci\n"
    gui.close()


def test_engine_output_processor_enables_controls(qt_app):
    gui = ChessGUI(chess.Board(), dev=True)
    proc = OutputProcess(["uciok\n"])
    engine_state = {"handshake_complete": False, "pending_debug": True, "debug_on_handshake": True}

    engine_output_processor(proc, gui, engine_state)

    assert engine_state["handshake_complete"] is True
    assert gui.ready_button.isEnabled()
    assert b"debug on\n" in proc.writes
    gui.close()
