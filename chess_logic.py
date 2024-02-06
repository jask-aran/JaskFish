import chess
from tkinter import messagebox, simpledialog
import random

from engine_comm import send_command
from utils import *



board = chess.Board()
board = chess.Board('1k6/6P1/8/8/8/8/8/1K6')
board_size = 500
selected_square = None
player_color = "White"


def check_game_status():
    game_over_messages = [
        (board.is_checkmate(), "Game Over", "Checkmate!"),
        (board.is_stalemate(), "Game Over", "Stalemate!"),
        (board.is_insufficient_material(), "Game Over", "Draw due to insufficient material!"),
        (board.is_seventyfive_moves(), "Game Over", "Draw due to 75 moves rule!"),
        (board.is_fivefold_repetition(), "Game Over", "Draw due to fivefold repetition!"),
        (board.is_variant_draw(), "Game Over", "Draw due to variant-specific rules!"),
    ]
    for condition, title, message in game_over_messages:
        if condition:
            messagebox.showinfo(title, message)
            break
        
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

    
def choose_promotion_piece():
    """
    Prompts the user to choose a piece for pawn promotion.
    Returns the chosen piece as a character ('q', 'r', 'b', 'n').
    """
    # This is a placeholder implementation.
    # You can replace this with a GUI dialog or another method to select a piece.
    promotion_pieces = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
    choice = simpledialog.askstring("Pawn Promotion", "Choose a piece for promotion (q, r, b, n):")
    return promotion_pieces.get(choice, chess.QUEEN)  # Default to queen if input is invalid 


def is_ready(proc):
    send_command(proc, 'isready')
    
def go_command(proc):
    fen_string = board.fen()
    send_command(proc, f'go {fen_string}')
    