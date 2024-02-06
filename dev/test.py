import subprocess
import chess

# Generate a fresh chess board
board = chess.Board()

# Convert the board to FEN string
fen_string = board.fen()
command = 'isready'  # Ensure newline is added so that it mimics pressing 'Enter' in a terminal

print(f"Sending command to engine: {command}")

# Execute engine.py and pass the command through stdin
process = subprocess.Popen(["python3", "engine.py"], 
                           stdin=subprocess.PIPE, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE, 
                           text=True)

stdout, stderr = process.communicate(input=command)

# Error handling
if stderr:
    print("Error:", stderr)

if stdout:
    print("Output from engine:")
    print(stdout.strip())
else:
    print("No output received from engine.")
