import sys
# import subprocess
# import threading
# import queue
import chess
from PySide2.QtWidgets import QApplication
from gui import ChessGUI

def main():
    app = QApplication(sys.argv)
    board = chess.Board()
    gui = ChessGUI(board, dev=True)
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
