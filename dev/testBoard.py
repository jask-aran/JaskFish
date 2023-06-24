import chess, chess.svg
import tkinter as tk
import cairosvg
import random
import tkinter.messagebox as messagebox

from PIL import Image, ImageTk
from io import BytesIO

board = chess.Board()
board_size = 500  # Set a board svg size

def svg_to_photo_image(svg_string):
    png_image = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
    
    image = Image.open(BytesIO(png_image))
    
    return ImageTk.PhotoImage(image)

def draw_board(squares):
    if squares:
        squares = {chess.SQUARES[chess.parse_square(squares)]: 'green'}
    
    # Get SVG of board state with last made move if available
    svg_data = chess.svg.board(board=board, lastmove=board.peek() if board.move_stack else None, size=board_size, squares=squares)
    
    # Convert SVG to PNG

    photo_image = svg_to_photo_image(svg_data)

    label.config(image=photo_image)
    label.image = photo_image
    
    turn_label.config(text="White Move Now" if board.turn == chess.WHITE else "Black Move Now")

def get_square_from_pixel(x, y):
    border_size = 18  # Width of the border in pixels
    square_size = (board_size - 2 * border_size) / 8  # Subtract the border width from the board size before calculating square size

    # Check if the click was on the border, and return None if it was
    if x < border_size or y < border_size or x >= board_size - (border_size) or y >= board_size - (border_size):
        return 

    # Adjust the coordinates to account for the border, then calculate the row and column
    row = 7 - int((y - border_size) / square_size)
    col = int((x - border_size) / square_size)

    # Convert to square number
    square_number = chess.square(col, row)

    # Convert to algebraic notation (e.g. e4, d5)
    square = chess.square_name(square_number)

    return square


def on_click(event):
    square = get_square_from_pixel(event.x, event.y)
    print(square)  # For now, just print the square. You might want to do something else with it
    draw_board(square)
    
def reset_board():
    board.reset()
    draw_board()
    
def make_random_move():
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
    draw_board(None)


def make_uci_move(move_entry):
    move_uci = move_entry.get()
    move_entry.delete(0, tk.END)  # Clear the text input box
    if move_uci:
        try:
            move = chess.Move.from_uci(move_uci)
            if move in board.legal_moves and board.is_legal(move):
                board.push(move)
            else:
                messagebox.showwarning("Invalid Move", "The move entered is not valid.")
        except Exception as e:
            messagebox.showwarning("Invalid Move", f"An error occurred while making the move:\n{e}")
    
    draw_board(None)

def create_move_entry():
    move_entry = tk.Entry(button_frame)
    move_entry.pack(side=tk.LEFT)
    return move_entry

def on_enter_press(event):
    # Execute the "Make Move" button command when Enter key is pressed
    button_make_move.invoke()

# Tkinter window
window = tk.Tk()
window.title("Chess")

# Turn label
turn_label = tk.Label(window, text="White's Turn", font=("Times", 16))
turn_label.pack(pady=10)

# label to hold the image
label = tk.Label(window)
label.pack()

# Frame at the bottom of the window to hold buttons
button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

move_entry = tk.Entry(button_frame)
move_entry.pack(side=tk.LEFT)

button_make_move = tk.Button(button_frame, text="Make Move", command=lambda: make_uci_move(move_entry))
button_make_move.pack(side=tk.LEFT)

button_make_random_move = tk.Button(button_frame, text="Make Random Move", command=make_random_move)
button_make_random_move.pack(side=tk.LEFT)

button_reset_board = tk.Button(button_frame, text="Reset Board", command=reset_board)
button_reset_board.pack(side=tk.LEFT)

move_entry.bind('<Return>', on_enter_press)
label.bind("<Button-1>", on_click)

draw_board(None)
window.mainloop()
