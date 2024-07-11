import sys
import subprocess
import threading
import queue
import chess
import argparse

from PySide2.QtWidgets import QApplication
from functools import partial

from gui import ChessGUI
from utils import cleanup, color_text, info_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    return parser.parse_args()
    # k7/2Q4P/8/8/8/8/8/K2R4

def engine_output_processor(proc):
    while True:
        output = proc.stdout.readline().strip()
        if output == '' and proc.poll() is not None:
            break
        elif output.startswith('bestmove'):
            print(color_text('BESTMOVE ', '33') + output[9:])
        elif output:
            print(color_text('RECIEVED ', '34') + output)


def send_command(proc, command):
    print(color_text('SENDING  ', '32') + command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    
def handle_command_go(proc, fen_string):
    send_command(proc, f"position fen {fen_string}")
    send_command(proc, "go")

def handle_command_readyok(proc):
    send_command(proc, "isready")

def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev  # CHANGE LATER

    

    engine_process = subprocess.Popen(
        ["python3", "engine.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    engine_thread = threading.Thread(target=engine_output_processor, args=(engine_process,), daemon=True)
    engine_thread.start()

    # Create a partial function for go_command_handler
    go_callback = partial(handle_command_go, engine_process)
    ready_callback = partial(handle_command_readyok, engine_process)
    app = QApplication(sys.argv)
    gui = ChessGUI(board, dev=dev, go_callback=go_callback, ready_callback=ready_callback) # Pass the go_callback to the GUI
    
    
    app.aboutToQuit.connect(lambda: cleanup(engine_process, engine_thread, app, dev=dev))
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
