# MAIN
import sys
import argparse
import os
import shlex

import chess

from PySide2.QtWidgets import QApplication
from PySide2.QtCore import QProcess

from gui import ChessGUI
from utils import cleanup, sending_text, recieved_text, info_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    parser.add_argument('-dev', action='store_true', help='Enable debug mode')
    parser.add_argument('--engine-cmd', help='Override the default engine launch command')
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
        gui.set_info_message("Engine move played")
    else:
        print("info string No best move found.")

def engine_output_processor(proc, gui, engine_state):
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if not output:
            continue
        if output.startswith('bestmove'):
            handle_bestmove(proc, output, gui)
            continue

        print(recieved_text(output))

        if output == 'uciok':
            engine_state['handshake_complete'] = True
            gui.set_engine_controls_enabled(True)
            gui.set_info_message("Engine ready")
            if engine_state.get('pending_debug'):
                send_command(proc, 'debug on')
                engine_state['pending_debug'] = False
        elif output == 'readyok':
            gui.set_info_message("Engine ready")


def resolve_engine_command(engine_path, override_cmd):
    if override_cmd:
        command = shlex.split(override_cmd)
        if not command:
            raise ValueError("Engine command override was provided but empty")
        return command
    return ["python3", engine_path]


def start_engine_process(engine_process, engine_cmd, gui, engine_state):
    gui.set_engine_controls_enabled(False)
    gui.set_info_message("Waiting for engine handshake...")
    engine_state['handshake_complete'] = False
    engine_state['pending_debug'] = engine_state.get('debug_on_handshake', False)

    program, *arguments = engine_cmd
    engine_process.start(program, arguments)
    send_command(engine_process, 'uci')


def restart_engine(engine_process, gui, engine_cmd, engine_state):
    print(info_text("Restarting engine..."))
    if engine_process.state() == QProcess.Running:
        engine_process.kill()
        engine_process.waitForFinished()
    start_engine_process(engine_process, engine_cmd, gui, engine_state)
    print(info_text("Engine restarted"))

def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev

    # Find absolute path to engine.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    engine_path = os.path.join(script_dir, "engine.py")
    try:
        engine_cmd = resolve_engine_command(engine_path, args.engine_cmd)
    except ValueError as exc:
        print(info_text(str(exc)))
        return

    # Construct GUI object
    app = QApplication(sys.argv)
    engine_process = QProcess()
    engine_process.setProcessChannelMode(QProcess.MergedChannels)

    engine_state = {
        'handshake_complete': False,
        'debug_on_handshake': dev,
        'pending_debug': dev,
    }

    def go_callback(fen_string):
        if not engine_state['handshake_complete']:
            gui.set_info_message("Engine not ready yet")
            return
        handle_command_go(engine_process, fen_string)

    def ready_callback():
        if not engine_state['handshake_complete']:
            gui.set_info_message("Engine not ready yet")
            return
        handle_command_readyok(engine_process)

    def restart_engine_callback():
        restart_engine(engine_process, gui, engine_cmd, engine_state)

    gui = ChessGUI(board, dev=dev, go_callback=go_callback, ready_callback=ready_callback, restart_engine_callback=restart_engine_callback)
    gui.set_engine_controls_enabled(False)
    gui.set_info_message("Starting engine...")

    # Start engine
    engine_process.readyReadStandardOutput.connect(lambda: engine_output_processor(engine_process, gui, engine_state))
    start_engine_process(engine_process, engine_cmd, gui, engine_state)


    app.aboutToQuit.connect(lambda: cleanup(engine_process, None, app, dev=dev))

    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
