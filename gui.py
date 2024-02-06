import tkinter as tk
from tkinter import ttk
import threading
import subprocess
import queue
import chess, chess.svg
import random
from PIL import Image, ImageTk


from utils import color_text, debug_text, svg_to_photo_image
from chess_logic import *
from engine_comm import engine_output_processor, send_command


        
   
def process_output_queue(output_queue, window):
    while not output_queue.empty():
        output = output_queue.get()
        parse_engine_output(output)
        print(color_text('Received  ', '34') + output)
    window.after(200, process_output_queue, output_queue, window)

def parse_engine_output(output):
    global engine_label
    
    if output.startswith('readyok'):
        engine_label.config(text="readyok")
    elif output.startswith("info string"):
        engine_label.config(text=output[11:])
    elif output.startswith("bestmove"):
        bestmove = output[9:]
        engine_label.config(text="Best move: " + bestmove)
        draw_board(best_move=bestmove)        



def make_random_move():
    legal_moves = list(board.legal_moves)
    move = random.choice(legal_moves)
    board.push(move)
    draw_board()

def reset_board():
    board.reset()
    draw_board()



def draw_board(selected_square=None, best_move=None):
    global board_label, turn_label
    
    check_game_status()
    available_moves = []
    best_move_arrow = []
    
    if best_move:
        best_move_arrow = [chess.svg.Arrow(chess.parse_square(best_move[:2]), chess.parse_square(best_move[2:]))]

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
    window.mainloop()



if __name__ == "__main__":
    main()
