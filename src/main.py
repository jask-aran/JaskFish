# MAIN
import sys
import chess
import argparse

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QProcess

from gui import ChessGUI
from utils import cleanup, sending_text, recieved_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    return parser.parse_args()
    # Example FEN: k7/2Q4P/8/8/8/8/8/K2R4

def send_command(proc, command):
    print(sending_text(command))
    proc.write((command + '\n').encode())
    proc.waitForBytesWritten()

def handle_command_go(proc, fen_string):
    send_command(proc, f"position fen {fen_string}")
    send_command(proc, "go")

def handle_bestmove(proc, bestmove, gui):
    print(recieved_text(bestmove))
    parts = bestmove.strip().split()
    if len(parts) >= 2:
        move_uci = parts[1]
        gui.attempt_engine_move(move_uci)
    else:
        print("info string No best move found.")

def handle_command_readyok(proc):
    send_command(proc, "isready")

def engine_output_processor(proc, gui):
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if output.startswith('bestmove'):
            handle_bestmove(proc, output, gui)
        elif output:
            print(recieved_text(output))

def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev

    app = QApplication(sys.argv)
    engine_process = QProcess()
    engine_process.setProcessChannelMode(QProcess.MergedChannels)
    engine_process.readyReadStandardOutput.connect(lambda: engine_output_processor(engine_process, gui))
    engine_process.start("python3", ["./engine.py"])

    send_command(engine_process, 'debug on') if dev else None

    go_callback = lambda fen_string: handle_command_go(engine_process, fen_string)
    ready_callback = lambda: handle_command_readyok(engine_process)
    gui = ChessGUI(board, dev=dev, go_callback=go_callback, ready_callback=ready_callback)

    app.aboutToQuit.connect(lambda: cleanup(engine_process, None, app, dev=dev))

    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
