import chess
import chess.svg
import tkinter as tk
from PIL import Image, ImageTk
from io import BytesIO
import cairosvg
import random

# Initialize the chess board.
board = chess.Board()


def draw_board():
    # Convert the SVG to a PNG.
    png_image = cairosvg.svg2png(
        bytestring=bytes(chess.svg.board(board=board), encoding="utf8")
    )

    # Create an image from the PNG.
    image = Image.open(BytesIO(png_image))

    # Create a PhotoImage from the PIL image.
    photo_image = ImageTk.PhotoImage(image)

    # Update the label with the new image.
    label.config(image=photo_image)
    label.image = photo_image


def make_move():
    # Get a list of all legal moves.
    legal_moves = list(board.legal_moves)

    # Choose a random move from the list of legal moves.
    move = random.choice(legal_moves)

    # Make the chosen move.
    board.push(move)

    # Redraw the board.
    draw_board()


# Create the main Tkinter window.
window = tk.Tk()

# Create a label to hold the image.
label = tk.Label(window)
label.pack()

# Create a button to make a move.
button = tk.Button(window, text="Make Move", command=make_move)
button.pack()

# Draw the initial board state.
draw_board()

# Start the Tkinter event loop.
window.mainloop()
