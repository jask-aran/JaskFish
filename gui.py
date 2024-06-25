import sys
import chess
import argparse
from PySide2.QtCore import Qt, QSize
from PySide2.QtGui import QFont
from PySide2.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QPushButton, QLabel, QVBoxLayout, QMessageBox, QHBoxLayout, QDialog, QComboBox
import chess_logic


class PromotionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pawn Promotion")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.combo = QComboBox()
        self.combo.addItems(["Queen", "Rook", "Bishop", "Knight"])
        layout.addWidget(self.combo)

        button = QPushButton("OK")
        button.clicked.connect(self.accept)
        layout.addWidget(button)

    def get_promotion_piece(self):
        return self.combo.currentText().lower()

class ChessGUI(QMainWindow):
    def __init__(self, board, dev=False):
        super().__init__()
        self.board = board
        self.selected_square = None
        if dev:
            self.player_is_white = True
        else:
            self.player_is_white = self.get_player_color()  # This will prompt the user to choose their color
            print(self.player_is_white)
            if not self.player_is_white:
                self.board.turn = chess.BLACK  # Set the board turn to Black if the player chooses Black
        self.init_ui()

    def get_player_color(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Choose Your Color")
        msg_box.setText("Do you want to play as White or Black?")
        white_button = msg_box.addButton("White", QMessageBox.YesRole)
        black_button = msg_box.addButton("Black", QMessageBox.NoRole)
        msg_box.exec_()
        
        return msg_box.clickedButton() == white_button

    
    def init_ui(self):
        self.setWindowTitle('JaskFish')
        self.setGeometry(100, 100, 500, 550)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.turn_indicator = QLabel("White's turn" if self.board.turn == chess.WHITE else "Black's turn")
        self.turn_indicator.setAlignment(Qt.AlignCenter)
        self.turn_indicator.setFont(QFont('Arial', 16))
        main_layout.addWidget(self.turn_indicator)

        board_widget = QWidget()
        grid_layout = QGridLayout(board_widget)
        main_layout.addWidget(board_widget)

        label_font = QFont('Arial', 14)
        files = 'abcdefgh'

        # Add file and rank labels
        for i in range(8):
            file_label = QLabel(files[i], alignment=Qt.AlignCenter, font=label_font)
            rank_label = QLabel(str(8 - i), alignment=Qt.AlignCenter, font=label_font)
            grid_layout.addWidget(file_label, 8, i + 1)
            grid_layout.addWidget(rank_label, i, 0)

        # Create squares
        self.squares = {}
        for row in range(8):
            for col in range(8):
                button = QPushButton(fixedSize=QSize(50, 50), font=QFont('Arial', 20))
                button.clicked.connect(self.on_square_clicked)
                grid_layout.addWidget(button, row, col + 1)
                self.squares[chess.square(col, 7 - row)] = button

        restart_button = QPushButton("Restart")
        restart_button.clicked.connect(self.restart_game)
        main_layout.addWidget(restart_button)
        
        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_move)
        main_layout.addWidget(undo_button)

        self.update_board()
        self.center_on_screen()


        
    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        window_size = self.size()
        x = (screen_geometry.width() - window_size.width()) / 2 + screen_geometry.left()
        y = (screen_geometry.height() - window_size.height()) / 2 + screen_geometry.top()
        self.move(x, y)
        
    def get_piece_unicode(self, piece):
        piece_unicode = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        return piece_unicode[piece.symbol()]

    def update_board(self):
        if self.board.move_stack:
            last_move = self.board.move_stack[-1]
            last_move = [chess.square_name(last_move.from_square), chess.square_name(last_move.to_square)]
        else:
            last_move = None
        
        for square, button in self.squares.items():
            piece = self.board.piece_at(square)
            if piece:
                button.setText(self.get_piece_unicode(piece))
            else:
                button.setText('')
            button.setStyleSheet(self.get_square_style(square, last_moved=last_move))
            

        self.turn_indicator.setText("White's turn" if self.board.turn == chess.WHITE else "Black's turn")

        if chess_logic.is_game_over(self.board):
            QMessageBox.information(self, "Game Over", chess_logic.get_game_result(self.board))

    def get_square_style(self, square, last_moved=None):
        light_square = "#F0D9B5"
        dark_square = "#B58863"
        selected_color = "#646F40"
        prev_moved_color = "#ddf0a1"
        attacked_color = "#fca5a5"

        is_light = (chess.square_rank(square) + chess.square_file(square)) % 2 == 1
        base_color = light_square if is_light else dark_square
        
        piece = self.board.piece_at(square)

        if square == self.selected_square:
            return f"background-color: {selected_color};"
        elif piece and piece.piece_type == chess.KING and self.board.is_attacked_by(not piece.color, square):
            return f"background-color: {attacked_color};"
        elif last_moved and square in [chess.parse_square(sq) for sq in last_moved]:
            return f"background-color: {prev_moved_color};"
        else:
            return f"background-color: {base_color};"
        
    def on_square_clicked(self):
        clicked_button = self.sender()
        clicked_square = next(square for square, button in self.squares.items() if button == clicked_button)

        def is_own_color(square):
            piece = self.board.piece_at(square)
            return piece and piece.color == self.board.turn

        if self.selected_square == clicked_square: # Handle unselecting
            self.selected_square = None
        elif is_own_color(clicked_square): # Handle selecting/ reselecting own piece
            self.selected_square = clicked_square
        elif self.selected_square is not None: # Handle move attempt
            move = chess.Move(self.selected_square, clicked_square)
            self.attempt_move(move)
            self.selected_square = None
        else:
            pass
            # print("No Piece on Square/ Wrong color")

        self.update_board()
        
    
    def attempt_move(self, move):
        print(f"Move attempted: {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}")

        if chess_logic.is_pawn_promotion_attempt(self.board, move):
            promotion_choice = self.get_promotion_choice()
            if not promotion_choice:
                print("Promotion required but not selected")
                return
            move.promotion = promotion_choice

        if chess_logic.is_valid_move(self.board, move):
            print(f"Valid Move: {str(move)}")
            chess_logic.make_move(self.board, move)
            self.selected_square = None
            # self.update_board() # Not needed because called in on_square_clicked
        else:
            print("Invalid Move")
        
        
    def get_promotion_choice(self):
        dialog = PromotionDialog(self)
        if dialog.exec_():
            piece = dialog.get_promotion_piece()
            return {'queen': chess.QUEEN, 'rook': chess.ROOK, 'bishop': chess.BISHOP, 'knight': chess.KNIGHT}[piece]
        return None  # Return None if the dialog is cancelled

    def restart_game(self):
        print("Restarting game...")
        self.board.reset()
        self.selected_square = None
        self.update_board()
        
    def undo_move(self):
        chess_logic.undo_move(self.board)
        self.selected_square = None
        self.update_board()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fen', help='Set the initial board state to the given FEN string')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    app = QApplication(sys.argv)
    board = chess.Board("k7/2Q4P/8/8/8/8/8/K2R4")
    # board = chess.Board("k7/8/8/8/p7/8/Q7/K7") # Testing multiple pieces possible for target square
    if args.fen:
        board.set_fen(args.fen)
    chess_gui = ChessGUI(board, dev=True)

    chess_gui.show()
    sys.exit(app.exec_())
