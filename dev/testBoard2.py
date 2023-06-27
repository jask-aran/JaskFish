import chess, chess.svg
import tkinter as tk
import cairosvg
import random
import tkinter.messagebox as messagebox
from tkinter import simpledialog

from PIL import Image, ImageTk
from io import BytesIO

board = chess.Board('k7/7P/8/8/8/8/8/K7')
# board = chess.Board()
board_size = 500  # Set a board svg size
selected_piece_integer = None
    
    
def get_square_from_pixel(x, y):
    border_size = 18
    square_size = (board_size - 2 * border_size) / 8
    
    if x < border_size or y < border_size or x >= board_size - \
        (border_size) or y >= board_size - (border_size):
        return []
    
    # Adjust the coordinates to account for the border, then calculate the row and column
    row = 7 - int((y - border_size) / square_size)
    col = int((x - border_size) / square_size)

    return chess.square(col, row)

def handle_piece_selection(square_integer):
    global selected_piece_integer
    piece = board.piece_at(square_integer)
    if piece and piece.color == board.turn:
        selected_piece_integer = square_integer
    else:
        selected_piece_integer = None
    draw_board(square_integer)
    
def get_promotion_piece():
    answer = simpledialog.askstring("Promotion", "Promote pawn to (q,r,b,n):", parent=window)
    if answer in ['q', 'r', 'b', 'n']:
        return {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}[answer]
    else:
        messagebox.showerror("Error", "Invalid promotion piece. Please enter 'q', 'r', 'b', or 'n'.")
        return None


def move_piece(from_square, to_square, promotion=None):
    move = chess.Move(from_square, to_square, promotion=promotion)
    if move in board.legal_moves:
        board.push(move)
        return True
    return False


def handle_move_attempt(from_square, to_square):
    global selected_piece_integer
    piece = board.piece_at(from_square)
    if piece.piece_type == chess.PAWN and chess.square_rank(to_square) in [0, 7]:
        promotion = get_promotion_piece()
        if promotion is None:
            return
    else:
        promotion = None

    if move_piece(from_square, to_square, promotion):
        selected_piece_integer = None

        if board.is_check():
            messagebox.showinfo("Check", "The " + ("white" if board.turn else "black") + " king is in check!")

        draw_board()


def on_click(event):
    square_integer = get_square_from_pixel(event.x, event.y)
    if not square_integer:
        return

    if selected_piece_integer:
        handle_move_attempt(selected_piece_integer, square_integer)
    else:
        handle_piece_selection(square_integer)



def draw_board(square=[]):
    check_game_status()
    

    
    available_moves = [] # Available moves for user to select (once a piece is selected)
    if square:
        legal_moves = [move for move in board.legal_moves if move.from_square == square]
        # Parse piece symbol at selected square integer, and then square name of selected square integer
        print(str(board.piece_at(square)) + ' ' + str(chess.square_name(square)))
        print("Legal Moves:" + str(legal_moves))

        # print([chess.square_name(move.to_square) for move in legal_moves])
        available_moves = chess.SquareSet([move.to_square for move in legal_moves])
    
    svg_data = chess.svg.board(board=board, \
        lastmove=board.peek() if board.move_stack else None, size=board_size, squares=available_moves)

    # Convert the generated SVG into a png that TK inter can use
    png_image = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(BytesIO(png_image))
    photo_image = ImageTk.PhotoImage(image)
    
    # Load the Chessboard image into the board label
    board_label.config(image=photo_image)
    board_label.image = photo_image
    
    # Change turn label as neccessary
    turn_label.config(text="White Move Now" if board.turn == chess.WHITE else "Black Move Now")


def reset_board():
    board.reset()
    draw_board(None)
    
    
def make_random_move():
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
    draw_board()    
    
    
def check_game_status():
    if board.is_checkmate():
        messagebox.showinfo("Game Over", "Checkmate!")
    elif board.is_stalemate():
        messagebox.showinfo("Game Over", "Stalemate!")
    elif board.is_insufficient_material():
        messagebox.showinfo("Game Over", "Draw due to insufficient material!")
    elif board.is_seventyfive_moves():
        messagebox.showinfo("Game Over", "Draw due to 75 moves rule!")
    elif board.is_fivefold_repetition():
        messagebox.showinfo("Game Over", "Draw due to fivefold repetition!")
    elif board.is_variant_draw():
        messagebox.showinfo("Game Over", "Draw due to variant-specific rules!")



# Main Window
window = tk.Tk()
window.title("Chess")

# Turn label
turn_label = tk.Label(window, text="White's Turn", font=("Times", 16))
turn_label.pack(pady=10)

# Label to hold the board
board_label = tk.Label(window)
board_label.pack()

# Frame at the bottom of the window to hold buttons
button_frame = tk.Frame(window)
button_frame.pack(side=tk.BOTTOM)

# Make Random Move Button
button_make_random_move = tk.Button(button_frame, text="Make Random Move", command=make_random_move)
button_make_random_move.pack(side=tk.LEFT)

# Reset Board Button
button_reset_board = tk.Button(button_frame, text="Reset Board", command=reset_board)
button_reset_board.pack(side=tk.RIGHT)

# Bind a left click event to on_click()
board_label.bind("<Button-1>", on_click)

draw_board()
window.mainloop()


