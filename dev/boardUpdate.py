import subprocess
import chess


# Generate a fresh chess board
board = chess.Board()

# Convert the board to FEN string
fen_string = board.fen()
command = fen_string
print(command)


# Execute randMove.py and pass the fen string through stdin
process = subprocess.Popen(["python3", "randMove.py"], 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True)

stdout, stderr = process.communicate(input=command)

# Error handling
if stderr:
    print("Error:", stderr)



# Update the board state with the move returned in stdout
move_uci = stdout.strip()
print(move_uci)
if move_uci:
    move = chess.Move.from_uci(move_uci)
    board.push(move)

# Print the updated board state
print("Updated board state:")
print(board)

