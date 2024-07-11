import sys
import subprocess
import threading
import queue
import chess
from PySide2.QtWidgets import QApplication
from gui import ChessGUI
import argparse
import signal

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def engine_output_processor(output_queue, proc, stop_event):
    while not stop_event.is_set():
        try:
            output = proc.stdout.readline().strip()
            if output == '' and proc.poll() is not None:
                break
            elif output == 'quit':
                stop_event.set()
                break
            elif output:
                output_queue.put(output)
        except Exception as e:
            print(f"Error in engine output processor: {e}")
            break

def cleanup(engine_process, engine_thread, stop_event, app):
    print("Cleaning up resources...")
    stop_event.set()
    
    if engine_process.poll() is None:
        try:
            engine_process.communicate("quit\n", timeout=5)
        except subprocess.TimeoutExpired:
            engine_process.kill()
    
    engine_thread.join(timeout=5)
    if engine_thread.is_alive():
        print("Warning: Engine thread did not terminate properly")
    
    app.quit()

def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = args.dev

    app = QApplication(sys.argv)
    gui = ChessGUI(board, dev=dev)
    
    engine_input_queue = queue.Queue()
    engine_output_queue = queue.Queue()
    stop_event = threading.Event()
    
    engine_process = subprocess.Popen(
        ["python3", "engine.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    engine_thread = threading.Thread(
        target=engine_output_processor,
        args=(engine_output_queue, engine_process, stop_event)
    )
    engine_thread.start()

    def signal_handler(sig, frame):
        cleanup(engine_process, engine_thread, stop_event, app)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    app.aboutToQuit.connect(lambda: cleanup(engine_process, engine_thread, stop_event, app))

    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()