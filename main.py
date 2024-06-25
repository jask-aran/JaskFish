import sys
# import subprocess
# import threading
# import queue
import chess
from PySide2.QtWidgets import QApplication
from gui import ChessGUI
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fen', help='Set the initial board state to the given FEN string')
    return parser.parse_args()
    # k7/2Q4P/8/8/8/8/8/K2R4

def main():
    args = parse_args()

    app = QApplication(sys.argv)
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    gui = ChessGUI(board, dev=True)
    
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
