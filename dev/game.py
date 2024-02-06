import chess, chess.svg
import tkinter as tk
import cairosvg
import random
import tkinter.messagebox as messagebox
from tkinter import simpledialog, Toplevel

import subprocess
import os

from PIL import Image, ImageTk
from io import BytesIO

dev_mode = False  # Set to False to enable color selection pop-up
player_color = 'White'










def make_random_move():
    return

def make_engine_move():
    return

def reset_board():
    return

def start_game(player_choice):
    global player_color
    player_color = player_choice
    color_selection_window.destroy()
    print(player_color)



# Tkinter window
window = tk.Tk()
window.title("Chess")
window.withdraw()

if not dev_mode:
    # Pop-up window for color selection
    color_selection_window = Toplevel(window)
    color_selection_window.title("Choose Color")

    # Variable to store the player's color choice
    player_color_var = tk.StringVar(value="White")  # default value

    # Radio buttons for color selection
    tk.Radiobutton(color_selection_window, text="White", variable=player_color_var, value="White").pack(anchor=tk.W)
    tk.Radiobutton(color_selection_window, text="Black", variable=player_color_var, value="Black").pack(anchor=tk.W)

    # Button to confirm color selection
    confirm_button = tk.Button(color_selection_window, text="Confirm", command=lambda: start_game(player_color_var.get()))
    confirm_button.pack()

    # Wait for the player to make a selection
    color_selection_window.wait_window()
else:
    # Set player color to White for development purposes
    player_color = "White"

window.deiconify()

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

window.mainloop()