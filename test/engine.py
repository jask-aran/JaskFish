import sys
import io

import chess

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

print('id name JaskFish')
print('id author Jaskaran Singh')
print('uciok')

fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def engine_command_processor():
    global fen
    while True:
        command = sys.stdin.readline().strip()
        if command == "quit":
            print('info string Engine shutting down')
            break
        elif command.startswith("position fen"):
            fen = command.split(' ', 1)[2] if ' ' in command else ""
        elif command == "boardpos":
            print(fen)
        
        elif command.startswith("go"):
            print(f"{command} received from GUI (GO COMMAND)")
        elif command:
            print(f"{command} received from GUI")
        sys.stdout.flush()

def main():
    engine_command_processor()
    
main()

