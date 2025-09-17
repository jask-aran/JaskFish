# MAIN
import sys
import chess
import argparse
import os

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QProcess

from gui import ChessGUI
from utils import cleanup, sending_text, recieved_text, info_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    parser.add_argument('--collect-data', action='store_true', help='Enable engine search-data logging')
    parser.add_argument('--data-log-path', help='Override the engine search-data log path')
    parser.add_argument('--data-log-format', choices=['json', 'csv'], help='Override the engine search-data log format')
    return parser.parse_args()
    # Example FEN: k7/2Q4P/8/8/8/8/8/K2R4

def send_command(proc, command):
    print(sending_text(command))
    proc.write((command + '\n').encode())
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
        if output.startswith('bestmove'):
            handle_bestmove(proc, output, gui)
        elif output:
            print(recieved_text(output))
            
def restart_engine(engine_process, gui, engine_path):
    print(info_text("Restarting engine..."))
    if engine_process.state() == QProcess.Running:
        engine_process.kill()
        engine_process.waitForFinished()

    engine_process.start("python3", [engine_path])
    configure_data_logging(
        engine_process,
        path=getattr(gui, 'data_log_path', None),
        fmt=getattr(gui, 'data_log_format', None),
    )
    if getattr(gui, 'collect_data_enabled', False):
        handle_collectdata(engine_process, True)
    send_command(engine_process, 'debug on') if gui.dev else None
    print(info_text("Engine restarted"))


def handle_collectdata(proc, enabled):
    state = 'on' if enabled else 'off'
    send_command(proc, f"collectdata {state}")


def configure_data_logging(proc, path=None, fmt=None):
    if path:
        send_command(proc, f"collectdata path {path}")
    if fmt:
        send_command(proc, f"collectdata format {fmt}")

def main():
    args = parse_args()
    if args.data_log_format and not args.data_log_path:
        default_extension = 'csv' if args.data_log_format == 'csv' else 'jsonl'
        args.data_log_path = os.path.join('logs', f'search_logs.{default_extension}')
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
    collect_callback = lambda enabled: handle_collectdata(engine_process, enabled)
    gui = ChessGUI(
        board,
        dev=dev,
        go_callback=go_callback,
        ready_callback=ready_callback,
        restart_engine_callback=restart_engine_callback,
        collect_callback=collect_callback,
        collect_data_enabled=args.collect_data,
    )
    gui.data_log_path = args.data_log_path
    gui.data_log_format = args.data_log_format
    
    # Start engine
    engine_process = QProcess()
    engine_process.setProcessChannelMode(QProcess.MergedChannels)
    engine_process.readyReadStandardOutput.connect(lambda: engine_output_processor(engine_process, gui))
    engine_process.start("python3", [engine_path])
    
    configure_data_logging(
        engine_process,
        path=args.data_log_path,
        fmt=args.data_log_format,
    )

    if args.collect_data:
        handle_collectdata(engine_process, True)

    # Enable debug mode
    send_command(engine_process, 'debug on') if dev else None


    app.aboutToQuit.connect(lambda: cleanup(engine_process, None, app, dev=dev))

    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
