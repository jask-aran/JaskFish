import sys
import io
import random
import chess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

print('id name JaskFish')
print('id author Jaskaran Singh')
print('uciok')

go_command_in_progress = False

def random_move(board):
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None
    random_move = random.choice(legal_moves)
    return random_move.uci()

def handle_go(fenstring, debug):
    global go_command_in_progress
    
    if not go_command_in_progress:
        go_command_in_progress = True
    print(f"info string calc pos {fenstring}") if debug else None
    move = random_move(chess.Board(fenstring))
    print(f"bestmove {move}")


def engine_command_processor():
    boardstate = ""
    debug = False
    while True:
        command = sys.stdin.readline().strip()
        if command == "quit":
            print('info string Engine shutting down')
            break
        
        elif command.startswith("debug "):
            setting = command.split(' ', 1)[1] if ' ' in command else ""
            debug = True if setting == "on" else False
            print(f"info string Debug:{debug}")
        
        elif command == "isready":
            print("readyok")
        
        elif command.startswith("position startpos"):
            boardstate = chess.Board().fen()
            
        elif command.startswith("position fen "):
            boardstate = command[13:]
            
        elif command == "boardpos":
            print(f"info string Position: {boardstate}") if boardstate else print("info string Board state not found")
        
        elif command == "go":
            handle_go(boardstate, debug) if boardstate else print("info string Board state not found")

        elif command:
            print(f"unknown command: '{command}' received from GUI")
        sys.stdout.flush()

engine_command_processor()

