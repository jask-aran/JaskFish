import chess, chess.svg
import tkinter as tk
import cairosvg
import random
import tkinter.messagebox as messagebox
from tkinter import simpledialog
import subprocess
import os

from PIL import Image, ImageTk
from io import BytesIO

# board = chess.Board('k7/7P/8/8/8/8/8/K7')
board = chess.Board()
board_size = 500  # Set a board svg size
selected_piece = None

def svg_to_photo_image(svg_string):
    png_image = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
    image = Image.open(BytesIO(png_image))
    return ImageTk.PhotoImage(image)

def draw_board(square):
    check_game_status()
    moves = []
    if square:
        piece = chess.parse_square(square)
        print(piece)
        legal_moves = [move for move in board.legal_moves if move.from_square == piece]
        print("Legal Moves:" + str(legal_moves))
        moves = chess.SquareSet([move.to_square for move in legal_moves])
    
    # Get SVG of board state with last made move if available
    svg_data = chess.svg.board(board=board, lastmove=board.peek() if \
        board.move_stack else None, size=board_size, squares=moves)
    
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

    # Convert to algebraic notation (e.g. e4, d5)
    return chess.square_name(chess.square(col, row))

def promote_pawn(move):
    while True:
        promotion_piece = simpledialog.askstring("Pawn Promotion", "Promote pawn to Q/R/B/N?")
        if promotion_piece.lower() in ["q", "r", "b", "n"]:
            promotion = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[promotion_piece.lower()]
            move.promotion = promotion
            break
        else:
            messagebox.showwarning("Invalid Input", "Please enter Q, R, B, or N.")

def on_click(event):
    global selected_piece
    square = get_square_from_pixel(event.x, event.y)
    if square is None:
        return

    if selected_piece:
        print(selected_piece)
        if board.piece_at(chess.parse_square(selected_piece)).piece_type == chess.PAWN and chess.square_rank(chess.parse_square(square)) in [0, 7]:
            for promotion_piece in ["q", "r", "b", "n"]:
                try:
                    move = chess.Move.from_uci(selected_piece + square + promotion_piece)
                    if move in board.legal_moves:
                        promote_pawn(move)
                        board.push(move)
                        selected_piece = None
                        draw_board(None)
                        return
                except:
                    pass
        else:
            try:
                move = chess.Move.from_uci(selected_piece + square)
                if move in board.legal_moves:
                    board.push(move)
                    selected_piece = None
                    draw_board(None)
                    return
            except:
                pass

    piece = board.piece_at(chess.parse_square(square))
    if piece and piece.color == board.turn:
        selected_piece = square
        draw_board(square)
    else:
        selected_piece = None
        draw_board(None)

    
def reset_board():
    board.reset()
    draw_board(None)
    
def make_random_move():
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
    draw_board(None)

def make_engine_move():
    fen_string = board.fen()
    print(fen_string)
    
    process = subprocess.Popen(["python3", "randMove.py"], 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True)

    stdout, stderr = process.communicate(input=fen_string)
    
    if stderr:
        print("Error:", stderr)
        
    move_uci = stdout.strip()
    if move_uci:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
    
    draw_board(None)

    
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

button_make_random_move = tk.Button(button_frame, text="Make Random Move", command=make_random_move)
button_make_random_move.pack(side=tk.LEFT)

button_make_engine_move = tk.Button(button_frame, text="Make Engine Move", command=make_engine_move)
button_make_engine_move.pack(side=tk.LEFT)

button_reset_board = tk.Button(button_frame, text="Reset Board", command=reset_board)
button_reset_board.pack(side=tk.LEFT)

label.bind("<Button-1>", on_click)


draw_board(None)
window.mainloop()