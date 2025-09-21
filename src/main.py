# MAIN
import sys
import chess
import argparse
import os
from typing import Iterable, Optional

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess

from gui import ChessGUI
from self_play import SelfPlayManager
from utils import cleanup, sending_text, recieved_text, info_text, debug_text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fen", help="Set the initial board state to the given FEN string"
    )
    parser.add_argument("-dev", action="store_true", help="Enable debug mode")
    return parser.parse_args()
    # Example FEN: k7/2Q4P/8/8/8/8/8/K2R4


def send_command(proc, command):
    print(sending_text(command))
    proc.write((command + "\n").encode())
    proc.waitForBytesWritten()


def handle_command_go(proc, fen_string):
    send_command(proc, f"position fen {fen_string}")
    send_command(proc, "go")


def handle_command_readyok(proc):
    send_command(proc, "isready")


def handle_bestmove(proc, bestmove, gui):
    print(recieved_text(bestmove))
    parts = bestmove.strip().split()
    if len(parts) >= 2:
        move_uci = parts[1]
        gui.attempt_engine_move(move_uci)
        return move_uci
    print("info string No best move found.")
    return None


def engine_output_processor(proc, gui, self_play_manager=None, engine_color: Optional[bool] = None):
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if output.startswith("bestmove"):
            if self_play_manager and engine_color is not None:
                if not self_play_manager.should_apply_move(engine_color):
                    continue
            move_uci = handle_bestmove(proc, output, gui)
            if (
                move_uci
                and self_play_manager
                and engine_color is not None
            ):
                self_play_manager.on_engine_move(engine_color, move_uci)
        elif output:
            print(recieved_text(output))


def restart_engines(
    engines: Iterable[QProcess],
    gui,
    engine_path: str,
    dev: bool,
    self_play_manager: Optional[SelfPlayManager],
) -> None:
    print(info_text("Restarting engines..."))

    if self_play_manager:
        self_play_manager.stop(message="Self-play stopped (engines restarting)")

    for engine_process in engines:
        if engine_process.state() == QProcess.Running:
            engine_process.kill()
            engine_process.waitForFinished()

        engine_process.start(sys.executable, [engine_path])
        if not engine_process.waitForStarted(5000):
            print("info string Engine process failed to start within timeout")
            continue
        if dev:
            send_command(engine_process, "debug on")

    gui.set_info_message("Engines restarted")
    print(info_text("Engines restarted"))


def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev

    # Find absolute path to engine.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(script_dir, "engine.py")

    # Construct GUI and engine processes
    app = QApplication(sys.argv)

    engine_process_white = QProcess()
    engine_process_white.setProcessChannelMode(QProcess.MergedChannels)
    engine_process_white.start(sys.executable, [engine_path])
    if not engine_process_white.waitForStarted(5000):
        print("info string White engine failed to start")

    engine_process_black = QProcess()
    engine_process_black.setProcessChannelMode(QProcess.MergedChannels)
    engine_process_black.start(sys.executable, [engine_path])
    if not engine_process_black.waitForStarted(5000):
        print("info string Black engine failed to start")

    if dev:
        if engine_process_white.state() == QProcess.Running:
            send_command(engine_process_white, "debug on")
        if engine_process_black.state() == QProcess.Running:
            send_command(engine_process_black, "debug on")

    gui = ChessGUI(
        board,
        dev=dev,
        go_callback=lambda fen_string: handle_command_go(
            engine_process_white, fen_string
        ),
        ready_callback=lambda: handle_command_readyok(engine_process_white),
        restart_engine_callback=None,
    )

    self_play_manager = SelfPlayManager(
        gui,
        {chess.WHITE: engine_process_white, chess.BLACK: engine_process_black},
        send_command,
    )
    gui.set_self_play_callbacks(
        start_callback=self_play_manager.start,
        stop_callback=self_play_manager.stop,
    )

    gui.restart_engine_callback = lambda: restart_engines(
        (engine_process_white, engine_process_black),
        gui,
        engine_path,
        dev,
        self_play_manager,
    )

    engine_process_white.readyReadStandardOutput.connect(
        lambda: engine_output_processor(
            engine_process_white, gui, self_play_manager, chess.WHITE
        )
    )
    engine_process_black.readyReadStandardOutput.connect(
        lambda: engine_output_processor(
            engine_process_black, gui, self_play_manager, chess.BLACK
        )
    )

    def shutdown():
        self_play_manager.stop()
        for proc, label in (
            (engine_process_white, "white"),
            (engine_process_black, "black"),
        ):
            if proc.state() != QProcess.NotRunning:
                try:
                    send_command(proc, "quit")
                    proc.closeWriteChannel()
                except Exception as exc:
                    if dev:
                        print(debug_text(f"Failed to send quit command to {label} engine: {exc}"))
            cleanup(proc, None, app, dev=dev, quit_app=False)

    app.aboutToQuit.connect(shutdown)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
