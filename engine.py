import chess
import random
import sys
import time
import io

import threading
import queue

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

go_command_in_progress = False

def handle_isready():
    print("readyok")
    
def get_random_legal_move(fen):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None
    random_move = random.choice(legal_moves)
    return random_move.uci()

def handle_go(fenstring):
    # Placeholder for go logic
    # Implement the logic to start the engine calculation based on the current board state
    
    global go_command_in_progress
    
    print("info string Calculating move")
    time.sleep(0.1)  # Simulate a long calculation
    
    move = get_random_legal_move(fenstring)
    print('bestmove ' + move)
    
    go_command_in_progress = False

def engine_command_processor(command_queue):
    """
    Processes commands sent from the GUI to the chess engine.
    """
    global go_command_in_progress
    
    while True:
        if not command_queue.empty():
            command = command_queue.get()

            if command == "quit":
                print('info string Engine shutting down')
                break
            
            elif command == "isready":
                handle_isready()
                
            elif command.startswith("go"):
                # Start a new thread for each 'go' command to allow simultaneous processing
                if not go_command_in_progress:
                    go_command_in_progress = True
                    fenstring = command.split(' ', 1)[1] if ' ' in command else ""
                    threading.Thread(target=handle_go, args=(fenstring,)).start()
                else:
                    print("info string A 'go' command is already being processed.")
                    
            else:
                print("info string Unknown command:", command)

def read_stdin(command_queue):
    """
    Continuously read from stdin and put commands into the queue.
    """
    for line in sys.stdin:
        command_queue.put(line.strip())
        if line.strip() == "quit":
            break

def main():
    global go_command_in_progress
    go_command_in_progress = False
    
    command_queue = queue.Queue()

    # Start the thread to process engine commands
    engine_thread = threading.Thread(target=engine_command_processor, args=(command_queue,))
    engine_thread.start()
    

    # Start the stdin reading in the main thread
    
    print("info string init jaskfish")
    read_stdin(command_queue)
    # time.sleep(1)
    # Wait for the engine thread to finish (if needed)
    
    engine_thread.join()
    
    

if __name__ == "__main__":
    main()