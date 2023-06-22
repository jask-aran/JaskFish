import chess, chess.svg
# import chess.svg
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
import cairosvg
import random
import ctypes as ct
from tkinter import simpledialog

# Initialize the chess board.
board = chess.Board()

# def draw_board():
#     svg_data = chess.svg.board(board=board, lastmove=board.peek() if board.move_stack else None)
#     png_image = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
#     image = Image.open(BytesIO(png_image))
#     photo_image = ImageTk.PhotoImage(image)
#     label.config(image=photo_image)
#     label.image = photo_image

def draw_board():
    # Convert the SVG to a PNG.
    png_image = cairosvg.svg2png(
        bytestring=bytes(chess.svg.board(board=board, lastmove=board.peek() if board.move_stack else None), encoding="utf8")
    )

    # Create an image from the PNG.

    # Create a PhotoImage from the PIL image.
    photo_image = ImageTk.PhotoImage(Image.open(BytesIO(png_image)))

    # Update the label with the new image.
    label.config(image=photo_image)
    label.image = photo_image


def make_random_move():
    # Get a list of all legal moves.
    legal_moves = list(board.legal_moves)

    # Choose a random move from the list of legal moves.
    move = random.choice(legal_moves)

    # Make the chosen move.
    board.push(move)

    # Redraw the board.
    draw_board()

def make_move():
    move_uci = simpledialog.askstring("Make a move", "Enter your move in UCI format:")
    if move_uci is not None:
        try:
            # Create a chess.Move object from UCI format input from user
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves and board.is_legal(move):
                board.push(move)
            else:
                print("Invalid Move")
        except Exception as e:
            print(f"Invalid Move: {e}")
    
    draw_board()
    
def reset_board():
    board.reset()
    draw_board()


# Create the main Tkinter window.
window = tk.Tk()
window.title("My Chess Game")
# Create a label to hold the image.
label = tk.Label(window)
label.pack()

# Create a frame at the bottom of the window.
button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

# Create a "Make Move" button and pack it into the frame.
button_make_move = tk.Button(button_frame, text="Make Move", command=make_move)
button_make_move.pack(side=tk.LEFT)

button_make_random_move = tk.Button(button_frame, text="Make Random Move", command=make_random_move)
button_make_random_move.pack(side=tk.LEFT)

# Create a "Reset Board" button and pack it into the frame.
button_reset_board = tk.Button(button_frame, text="Reset Board", command=reset_board)
button_reset_board.pack(side=tk.LEFT)


# Draw the initial board state.
draw_board()

# Start the Tkinter event loop.
window.mainloop()
