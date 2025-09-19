# MAIN
import sys
import chess
import argparse
import os

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QProcess

from gui import ChessGUI
from utils import cleanup, sending_text, recieved_text, info_text


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
    else:
        print("info string No best move found.")


def engine_output_processor(proc, gui):
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if output.startswith("bestmove"):
            handle_bestmove(proc, output, gui)
        elif output:
            print(recieved_text(output))


def restart_engine(engine_process, gui, engine_path):
    print(info_text("Restarting engine..."))
    if engine_process.state() == QProcess.Running:
        engine_process.kill()
        engine_process.waitForFinished()

    engine_process.start(sys.executable, [engine_path])
    send_command(engine_process, "debug on") if gui.dev else None
    print(info_text("Engine restarted"))


def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev

    # Find absolute path to engine.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(script_dir, "engine.py")

    # Construct GUI object
    app = QApplication(sys.argv)
    go_callback = lambda fen_string: handle_command_go(engine_process, fen_string)
    ready_callback = lambda: handle_command_readyok(engine_process)
    restart_engine_callback = lambda: restart_engine(engine_process, gui, engine_path)
    gui = ChessGUI(
        board,
        dev=dev,
        go_callback=go_callback,
        ready_callback=ready_callback,
        restart_engine_callback=restart_engine_callback,
    )

    # Start engine
    engine_process = QProcess()
    engine_process.setProcessChannelMode(QProcess.MergedChannels)
    engine_process.readyReadStandardOutput.connect(
        lambda: engine_output_processor(engine_process, gui)
    )
    engine_process.start(sys.executable, [engine_path])

    # Enable debug mode
    send_command(engine_process, "debug on") if dev else None

    app.aboutToQuit.connect(lambda: cleanup(engine_process, None, app, dev=dev))

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
