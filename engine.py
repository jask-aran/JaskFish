import sys
import io

import chess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

print('id name JaskFish')
print('id author Jaskaran Singh')
print('uciok')


def handle_go(fenstring):
    print(f"Calculating for position: {fenstring}")
    print(":)")


def engine_command_processor():
    boardstate = ""
    while True:
        command = sys.stdin.readline().strip()
        if command == "quit":
            print('info string Engine shutting down')
            break
        elif command.startswith("position startpos"):
            boardstate = chess.Board().fen()
        elif command.startswith("position fen "):
            boardstate = command[13:]
        elif command == "boardpos":
            if boardstate:
                print(f"info string Position: {boardstate}")
            else:
                print("info string Board state not found")
        
        elif command == "go":
            if boardstate:
                handle_go(boardstate)
            else:
                print("info string Board state not found")
        elif command:
            print(f"{command} received from GUI")
        sys.stdout.flush()

engine_command_processor()

