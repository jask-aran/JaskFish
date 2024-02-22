import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import queue
import chess, chess.svg
import random
from PIL import Image, ImageTk
from tkinter import messagebox, simpledialog


from utils import color_text, debug_text, svg_to_photo_image


board = chess.Board()
board = chess.Board('1k6/6P1/8/8/8/8/8/1K6')
board_size = 500
selected_square = None
player_color = "White"


def engine_output_processor(output_queue, proc):
    while True:
        output = proc.stdout.readline()
        if output == '' and proc.poll() is not None:
            break
        if output:
            output_queue.put(output.strip())
    
def send_command(proc, command):
    print(color_text('Sending   ', '32') + command)
    proc.stdin.write(command + "\n")
    proc.stdin.flush()
    

def process_output_queue(output_queue, window):
    while not output_queue.empty():
        output = output_queue.get()
        parse_engine_output(output)
        print(color_text('Received  ', '34') + output)
    window.after(200, process_output_queue, output_queue, window)      
        
        
def parse_engine_output(output):
    global engine_label, board  # Ensure 'board' is accessible globally
    
    if output.startswith('readyok'):
        engine_label.config(text="readyok")
    elif output.startswith("info string"):
        engine_label.config(text=output[11:])
    elif output.startswith("bestmove"):
        bestmove_str = output.split(" ")[1]  # Assuming the format is "bestmove <move>"
        if bestmove_str != "(none)":  # Check if a valid move is provided
            # Handle promotion in the move, if any
            if len(bestmove_str) > 4:
                from_square = bestmove_str[:2]
                to_square = bestmove_str[2:4]
                promotion_piece = bestmove_str[4]  # The promotion piece is indicated by the fifth character
                move = chess.Move.from_uci(from_square + to_square + promotion_piece)
            else:
                move = chess.Move.from_uci(bestmove_str)
            
            if move in board.legal_moves:
                board.push(move)  # Make the move on the board
                draw_board()  # Redraw the board to reflect the move
                engine_label.config(text="Best move: " + bestmove_str)  # Update UI with the best move
            else:
                print("Invalid move received from engine:", bestmove_str)


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
            board.reset()
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

def choose_player_color():
    global player_color
    colors = {'w': 'White', 'b': 'Black'}
    choice = simpledialog.askstring("Start Game", "Choose a player colour ('w' or 'b')")
    player_color = colors.get(choice, 'White')

def make_random_move():
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
    draw_board()

def reset_board():
    board.reset()
    draw_board()
    
def is_ready(proc):
    send_command(proc, 'isready')
    
def go_command(proc):
    fen_string = board.fen()
    send_command(proc, f'go {fen_string}')
    

def draw_board(selected_square=None, best_move=None):
    global board_label, turn_label
    
    check_game_status()
    available_moves = []
    best_move_arrow = []
        
    if best_move:
            # Check if the move indicates a promotion and extract the move part
            move_str = best_move[:4]  # This extracts the move part without promotion
            best_move_arrow = [chess.svg.Arrow(chess.parse_square(move_str[:2]), chess.parse_square(move_str[2:]))]

    if selected_square:
        piece = chess.parse_square(selected_square)
        legal_moves = [move for move in board.legal_moves if move.from_square == piece]
        print(debug_text('Debug') + "     " + str(selected_square) + str([move.uci() for move in legal_moves]))
        available_moves = chess.SquareSet([move.to_square for move in legal_moves])
    
    # Get SVG of board state with last made move if available
    svg_data = chess.svg.board(board=board, lastmove=board.peek() if \
        board.move_stack else None, size=board_size, squares=available_moves, arrows=best_move_arrow)
    
    # Convert SVG to PNG
    photo_image = svg_to_photo_image(svg_data)

    board_label.config(image=photo_image)
    board_label.image = photo_image
    
    turn_label.config(text="White Move Now" if board.turn == chess.WHITE else "Black Move Now")

def on_click(event):
    global selected_square

    clicked_square = get_square_from_pixel(event.x, event.y)

    if selected_square:
        if clicked_square == selected_square:
            # Clear the selection if the same square is clicked
            selected_square = None
            draw_board()
            return

        piece = chess.parse_square(selected_square)
        legal_moves = [move for move in board.legal_moves if move.from_square == piece]

        if clicked_square:
            move_to_square = chess.parse_square(clicked_square)

            # Find if there is a legal promotion move to the clicked square
            is_promotion = False
            for move in legal_moves:
                if move.to_square == move_to_square and move.promotion is not None:
                    is_promotion = True
                    break

            if is_promotion:
                # If it's a legal promotion move, prompt for piece selection
                promotion_piece = choose_promotion_piece()
                move = chess.Move(piece, move_to_square, promotion=promotion_piece)
            else:
                try:
                    move = chess.Move.from_uci(selected_square + clicked_square)
                except chess.InvalidMoveError:
                    # Handle invalid move
                    selected_square = None
                    draw_board()
                    return

            if move in legal_moves:
                board.push(move)
                selected_square = None
                draw_board()
            else:
                handle_non_move_click(clicked_square)
    else:
        handle_non_move_click(clicked_square)

def handle_non_move_click(clicked_square):
    global selected_square
    move_to_square = chess.parse_square(clicked_square)
    if board.piece_at(move_to_square) and board.color_at(move_to_square) == board.turn:
        selected_square = clicked_square
        draw_board(selected_square=clicked_square)
    else:
        draw_board()



def setup_labels(window):
    # Frame for text labels
    text_frame = tk.Frame(window)
    text_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=15, pady=10)

    global turn_label, engine_label
    turn_label = tk.Label(text_frame, text="White's Turn", font=("Times", 22))
    turn_label.pack(pady=6, padx=(75,0))

    engine_label = tk.Label(text_frame, text="Waiting", font=("Times", 13))
    engine_label.pack(pady=2, padx=(75,0))

    # Frame for the image
    status_frame = ttk.Frame(window)
    status_frame.pack(side=tk.RIGHT, padx=10, pady=10)

    static_image = ImageTk.PhotoImage(Image.open("resources/static_image3.png"))
    image_label = ttk.Label(status_frame, image=static_image)
    image_label.pack()
    # Keep a reference to prevent garbage collection
    image_label.image = static_image
    
def setup_board(window):
    global board_label
    board_label = tk.Label(window)
    board_label.pack(side=tk.BOTTOM)
    
    board_label.bind("<Button-1>", on_click)

def setup_buttons(window, process):
    button_frame = tk.Frame(window)
    button_frame.pack(side=tk.BOTTOM)
    
    button_make_random_move = tk.Button(button_frame, text="Make Random Move", command=make_random_move)
    button_reset_board = tk.Button(button_frame, text="Reset Board", command=reset_board)
    button_isready = tk.Button(button_frame, text="isready", command=lambda: is_ready(process))
    button_go_test = tk.Button(button_frame, text="go", command=lambda: go_command(process))
    
    for button in [button_make_random_move, button_reset_board, button_isready, button_go_test]:
        button.pack(side=tk.LEFT)

def main():
    global board_label, turn_label, engine_label
    
    window = tk.Tk()
    window.title("Chess")
    window.resizable(False, False)
    window.geometry("+2000+100")

    
    engine_process = subprocess.Popen(["python3", "engine.py"], 
                            stdin=subprocess.PIPE, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            text=True, 
                            bufsize=1,
                            universal_newlines=True)

    setup_buttons(window, engine_process)
    setup_board(window)
    setup_labels(window)
    
    output_queue = queue.Queue()
    engine_thread = threading.Thread(target=engine_output_processor, args=(output_queue, engine_process), daemon=True)
    engine_thread.start()
    
    def on_closing():
        send_command(engine_process, 'quit')
        window.after(200, lambda: (engine_process.terminate() if engine_process.poll() is None else None, window.destroy()))
        engine_thread.join()
        
    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.after(200, process_output_queue, output_queue, window)
    
    
    draw_board()
    choose_player_color()
    window.mainloop()



if __name__ == "__main__":
    main()
