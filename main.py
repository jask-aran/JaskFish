import sys
import subprocess
import threading
import queue
import chess
from PySide2.QtWidgets import QApplication
from gui import ChessGUI
import argparse
from utils import cleanup, color_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    return parser.parse_args()
    # k7/2Q4P/8/8/8/8/8/K2R4

def engine_output_processor(output_queue, proc):
    while True:
        output = proc.stdout.readline().strip()
        if output == '' and proc.poll() is not None:
            break
        elif output:
            print(color_text('Received  ', '34') + output)
            # output_queue.put(output.strip())
            
# def process_output_queue(output_queue, window):
#     while not output_queue.empty():
#         output = output_queue.get()
#         print(color_text('Received  ', '34') + output)


def send_command(proc, command):
    print(color_text('Sending   ', '32') + command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()

def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev # CHANGE LATER

    app = QApplication(sys.argv)
    gui = ChessGUI(board, dev=dev)
    

    engine_process = subprocess.Popen(
                    ["python3", "engine.py"], 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    bufsize=1,
                    universal_newlines=True)
    
    # engine_input_queue = queue.Queue()
    engine_output_queue = queue.Queue()
    engine_thread = threading.Thread(target=engine_output_processor, args=(engine_output_queue, engine_process), daemon=True)
    engine_thread.start()


    app.aboutToQuit.connect(lambda: cleanup(engine_process, engine_thread, app, dev=dev))
    
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
