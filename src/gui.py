# GUI
import sys
import chess
import json
import time
import os
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QGridLayout,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QMessageBox,
    QHBoxLayout,
    QDialog,
    QComboBox,
)

import chess_logic
import utils


class PromotionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pawn Promotion")
        self.setMinimumWidth(220)
        self.setStyleSheet(
            """
            QDialog { background-color: #1f232a; color: #f6f7fb; }
            QComboBox {
                background-color: #2d333d;
                color: #f6f7fb;
                border-radius: 6px;
                padding: 6px 8px;
                border: 1px solid #3a414d;
            }
            QComboBox QListView {
                background-color: #2d333d;
                color: #f6f7fb;
                selection-background-color: #4752c4;
                selection-color: #ffffff;
            }
            QPushButton {
                background-color: #5865f2;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #4752c4; }
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self.combo = QComboBox()
        self.combo.addItems(["Queen", "Rook", "Bishop", "Knight"])
        layout.addWidget(self.combo)

        button = QPushButton("OK")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(self.accept)
        layout.addWidget(button)

    def get_promotion_piece(self):
        return self.combo.currentText().lower()


class ChessGUI(QMainWindow):
    def __init__(
        self,
        board,
        dev=False,
        go_callback=None,
        ready_callback=None,
        restart_engine_callback=None,
    ):
        super().__init__()
        self.board = board
        self.initial_fen = self.board.fen()
        self.selected_square = None
        self.dev = dev
        self.square_font = QFont("Segoe UI Symbol", 28)
        self.control_button_font = QFont("Segoe UI", 11)
        self.apply_theme()
        self.player_is_white = self.get_player_color()
        self.board.turn = chess.WHITE if self.player_is_white else chess.BLACK
        self.go_callback = go_callback
        self.ready_callback = ready_callback
        self.restart_engine_callback = restart_engine_callback
        print(utils.info_text("Starting Game..."))

        # Reporting Options
        self.full_reporting = True
        if self.dev:
            print(utils.debug_text("Debug Mode ENABLED"))
            print(
                utils.debug_text(
                    f"Full Reporting {'ENABLED' if self.full_reporting else 'DISABLED'}"
                )
            )
            print(
                utils.debug_text(
                    f"User Color {'WHITE' if self.player_is_white else 'BLACK'}"
                )
            )
        self.init_ui()

    def apply_theme(self):
        app = QApplication.instance()
        if app and app.style().objectName().lower() != "fusion":
            QApplication.setStyle("Fusion")

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1c1f24"))
        palette.setColor(QPalette.WindowText, QColor("#f5f7fb"))
        palette.setColor(QPalette.Base, QColor("#1c1f24"))
        palette.setColor(QPalette.AlternateBase, QColor("#242830"))
        palette.setColor(QPalette.Text, QColor("#f5f7fb"))
        palette.setColor(QPalette.Button, QColor("#2b3038"))
        palette.setColor(QPalette.ButtonText, QColor("#f5f7fb"))
        palette.setColor(QPalette.Highlight, QColor("#5865f2"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

        if app:
            app.setPalette(palette)

        self.setStyleSheet(
            """
            QMainWindow { background-color: #1c1f24; }
            QLabel#turnIndicator { font-size: 18px; font-weight: 600; }
            QLabel#infoIndicator { color: #b0b7c3; font-size: 12px; }
            QWidget#boardContainer {
                background-color: #171a1f;
                border-radius: 12px;
                padding: 6px;                                 /* was 18px */
            }
            QPushButton[panel="control"] {
                background-color: #2d333c;
                color: #f5f7fb;
                border: 1px solid #3a414d;
                border-radius: 8px;
                padding: 6px 10px;                            /* was 8px 16px */
            }
            """
        )

    def style_control_button(self, button):
        button.setProperty("panel", "control")
        button.setFont(self.control_button_font)
        button.setCursor(Qt.PointingHandCursor)
        button.setFocusPolicy(Qt.NoFocus)
        button.setMinimumWidth(72)
        button.style().unpolish(button)
        button.style().polish(button)
        button.update()

    def get_player_color(self):
        if self.dev:
            return True  # Always play as white in dev mode

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Choose Your Color")
        msg_box.setStyleSheet(
            """
            QMessageBox { background-color: #1f232a; color: #f6f7fb; }
            QPushButton {
                background-color: #2d333c;
                color: #f6f7fb;
                border-radius: 6px;
                padding: 6px 14px;
                border: 1px solid #3a414d;
            }
            QPushButton:hover { background-color: #363d48; }
            QPushButton:pressed { background-color: #2b313a; }
            """
        )
        white_button = msg_box.addButton("White", QMessageBox.YesRole)
        black_button = msg_box.addButton("Black", QMessageBox.NoRole)
        utils.center_on_screen(msg_box)
        msg_box.exec()

        return msg_box.clickedButton() == white_button

    def init_ui(self):
        self.setWindowTitle("JaskFish")
        self.resize(560, 600)
        self.setMinimumSize(560, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        self.turn_indicator = QLabel(
            "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        )
        self.turn_indicator.setObjectName("turnIndicator")
        self.turn_indicator.setAlignment(Qt.AlignCenter)
        self.turn_indicator.setFont(QFont("Segoe UI Semibold", 20))
        main_layout.addWidget(self.turn_indicator)

        self.info_indicator = QLabel("Game Started")
        self.info_indicator.setObjectName("infoIndicator")
        self.info_indicator.setAlignment(Qt.AlignCenter)
        self.info_indicator.setFont(QFont("Segoe UI", 11))
        main_layout.addWidget(self.info_indicator)

        board_widget = QWidget()
        board_widget.setObjectName("boardContainer")
        grid_layout = QGridLayout(board_widget)
        grid_layout.setSpacing(1)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.addWidget(board_widget)

        # Tighten rank/file label columns
        grid_layout.setColumnStretch(0, 0)
        grid_layout.setColumnMinimumWidth(0, 18)
        for col in range(1, 9):
            grid_layout.setColumnStretch(col, 1)

        label_font = QFont("Segoe UI", 11)
        label_font.setBold(True)
        label_style = "color: #d5d9e3;"
        files = "abcdefgh"

        for i in range(8):
            file_label = QLabel(files[i])
            file_label.setAlignment(Qt.AlignCenter)
            file_label.setFont(label_font)
            file_label.setStyleSheet(label_style)
            grid_layout.addWidget(file_label, 8, i + 1)

            rank_label = QLabel(str(8 - i))
            rank_label.setAlignment(Qt.AlignCenter)
            rank_label.setFont(label_font)
            rank_label.setStyleSheet(label_style)
            grid_layout.addWidget(rank_label, i, 0)

        self.squares = {}
        for row in range(8):
            for col in range(8):
                button = QPushButton("")
                button.setFixedSize(QSize(48, 48))
                button.setFont(self.square_font)
                button.setCursor(Qt.PointingHandCursor)
                button.setFocusPolicy(Qt.NoFocus)
                button.clicked.connect(self.on_square_clicked)
                grid_layout.addWidget(button, row, col + 1)
                self.squares[chess.square(col, 7 - row)] = button

        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        main_layout.addLayout(button_layout)

        undo_button = QPushButton("Undo")
        undo_button.clicked.connect(self.undo_move)
        self.style_control_button(undo_button)
        button_layout.addWidget(undo_button)

        export_button = QPushButton("Export")
        export_button.clicked.connect(self.export_game)
        self.style_control_button(export_button)
        button_layout.addWidget(export_button)

        reset_button = QPushButton("Reset board")
        reset_button.clicked.connect(self.reset_game)
        self.style_control_button(reset_button)
        button_layout.addWidget(reset_button)

        button_layout2 = QHBoxLayout()
        button_layout2.setSpacing(8)
        main_layout.addLayout(button_layout2)

        ready_button = QPushButton("Ready")
        ready_button.clicked.connect(self.ready_command)
        self.style_control_button(ready_button)
        button_layout2.addWidget(ready_button)

        go_button = QPushButton("Go")
        go_button.clicked.connect(self.go_command)
        self.style_control_button(go_button)
        button_layout2.addWidget(go_button)

        restart_engine_button = QPushButton("Restart Engine")
        restart_engine_button.clicked.connect(self.restart_engine)
        self.style_control_button(restart_engine_button)
        button_layout2.addWidget(restart_engine_button)

        self.update_board()
        utils.center_on_screen(self)

    def _build_game_over_message(self, outcome):
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            return f"Checkmate! {winner} wins."

        draw_messages = {
            "Stalemate": "Draw by Stalemate (No Checkmate).",
            "Insufficient Material": "Draw by Insufficient Material (No Checkmate).",
            "75-move rule": "Draw by 75-move rule (No Checkmate).",
            "Fivefold Repetition": "Draw by Fivefold Repetition (No Checkmate).",
            "Variant-specific Draw": "Draw by Variant-specific Draw (No Checkmate).",
        }
        return draw_messages.get(outcome, outcome)

    # def resizeEvent(self, event):
    #     super().resizeEvent(event)
    #     # available width for 8 files + grid gaps (~9 px total) + labels col
    #     board_area = self.centralWidget().width() - 32
    #     sq = max(36, min((board_area // 10), (self.height() - 240) // 10))
    #     for b in self.squares.values():
    #         b.setFixedSize(sq, sq)
    #     # scale piece font to square size
    #     self.square_font.setPointSizeF(max(18, sq * 0.55))
    #     for b in self.squares.values():
    #         b.setFont(self.square_font)

    def update_board(self, info_text=None):
        last_move = None
        if self.board.move_stack:
            last_move_move = self.board.move_stack[-1]
            last_move = [
                chess.square_name(last_move_move.from_square),
                chess.square_name(last_move_move.to_square),
            ]

        for square, button in self.squares.items():
            piece = self.board.piece_at(square)
            button.setText(utils.get_piece_unicode(piece) if piece else "")
            button.setStyleSheet(self.get_square_style(square, last_moved=last_move))

        self.turn_indicator.setText(
            "White's turn" if self.board.turn == chess.WHITE else "Black's turn"
        )
        self.info_indicator.setText(info_text) if info_text else None

        if chess_logic.is_game_over(self.board):
            outcome = chess_logic.get_game_result(self.board)
            message = self._build_game_over_message(outcome)
            print(utils.info_text(f"Game Over: {message}"))
            QMessageBox.information(self, "Game Over", message)

    def get_square_style(self, square, last_moved=None):
        square_style = {
            "light_square": "#d2b48c",
            "dark_square": "#8e6336",
            "selected_color": "#4f6f52",
            "prev_moved_color": "#6b8f71",
            "attacked_color": "#d75d5d",
        }

        is_light = (chess.square_rank(square) + chess.square_file(square)) % 2 == 1
        square_color = (
            square_style["light_square"] if is_light else square_style["dark_square"]
        )
        piece = self.board.piece_at(square)
        text_color = "#2b2626"
        if piece:
            text_color = "#f9f6f2" if piece.color == chess.WHITE else "#2b2626"

        if square == self.selected_square:
            square_color = square_style["selected_color"]
        elif (
            piece
            and piece.piece_type == chess.KING
            and self.board.is_attacked_by(not piece.color, square)
        ):
            square_color = square_style["attacked_color"]
            print(
                utils.info_text(
                    f"{chess.square_name(square)} In Check ({'White' if piece.color else 'Black'})"
                )
            )
        elif last_moved and square in [chess.parse_square(sq) for sq in last_moved]:
            square_color = square_style["prev_moved_color"]

        return (
            f"background-color: {square_color}; color: {text_color}; "
            f"border-radius: 10px; border: 1px solid rgba(0, 0, 0, 0.2);"
        )

    def on_square_clicked(self):
        clicked_button = self.sender()
        clicked_square = next(
            square
            for square, button in self.squares.items()
            if button == clicked_button
        )

        def is_own_color(square):
            piece = self.board.piece_at(square)
            return piece and piece.color == self.board.turn

        if self.selected_square == clicked_square:  # Handle unselecting
            self.selected_square = None
            (
                print(
                    utils.debug_text(f"{chess.square_name(clicked_square)} Unselected")
                )
                if self.dev and self.full_reporting
                else None
            )
        elif is_own_color(clicked_square):  # Handle selecting/ reselecting own piece
            self.selected_square = clicked_square
            (
                print(utils.debug_text(f"{chess.square_name(clicked_square)} Selected"))
                if self.dev and self.full_reporting
                else None
            )
        elif self.selected_square is not None:  # Handle move attempt
            move = chess.Move(self.selected_square, clicked_square)
            self.attempt_move(move)
            self.selected_square = None
        else:
            (
                print(utils.debug_text("No Piece on Square/ Wrong color"))
                if self.dev and self.full_reporting
                else None
            )

        self.update_board()

    def attempt_move(self, move):
        # Move in form e1e2
        (
            print(
                utils.debug_text(
                    f"Move attempted: {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}"
                )
            )
            if self.dev and self.full_reporting
            else None
        )

        if chess_logic.is_pawn_promotion_attempt(self.board, move):
            promotion_choice = self.get_promotion_choice()
            if not promotion_choice:
                return
            move.promotion = promotion_choice

        if chess_logic.is_valid_move(self.board, move):
            print(
                utils.info_text(
                    f"{str(move)} {utils.color_text('Valid Move', '32')} by {'White' if self.board.turn else 'Black'}"
                )
            )
            chess_logic.make_move(self.board, move)
            self.selected_square = None
        else:
            print(
                utils.info_text(
                    f"{str(move)} {utils.color_text('Invalid Move', '31')} attempted by {'White' if self.board.turn else 'Black'}"
                )
            )

    def attempt_engine_move(self, move_uci):
        move = chess.Move.from_uci(move_uci)
        print(
            utils.info_text(
                f"Engine Move Attempted: {chess.square_name(move.from_square)} -> {chess.square_name(move.to_square)}"
            )
        )
        self.attempt_move(move)
        self.update_board()

    def get_promotion_choice(self):
        dialog = PromotionDialog(self)
        if dialog.exec():
            piece = dialog.get_promotion_piece()
            return {
                "queen": chess.QUEEN,
                "rook": chess.ROOK,
                "bishop": chess.BISHOP,
                "knight": chess.KNIGHT,
            }[piece]

        print(utils.debug_text("Promotion Dialog Cancelled"))
        return None  # Return None if the dialog is cancelled

    def reset_game(self):
        print(utils.info_text("Resetting game..."))
        self.board.reset()
        self.initial_fen = self.board.fen()
        self.selected_square = None
        self.update_board(info_text="Game Reset")

    def undo_move(self):
        last_move = self.board.peek() if self.board.move_stack else None
        if last_move:
            chess_logic.undo_move(self.board)

        if last_move and self.dev:
            print(utils.debug_text(f"{last_move} Undone"))
            last_move = None
        elif self.dev:
            print(utils.debug_text("Move Stack Empty"))

        self.selected_square = None
        self.update_board()

    def game_state(self):
        return self.board.fen()

    def export_game(self):
        print(utils.info_text("---EXPORTING GAME---"))

        game_state = {
            "export-time": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
            ),
            "fen-init": self.initial_fen,
            "fen-final": self.board.fen(),
            "san": chess_logic.export_move_history_san(self.board),
            "uci": chess_logic.export_move_history_uci(self.board),
        }

        gamestates_folder = "gamestates"
        if not os.path.exists(gamestates_folder):
            os.makedirs(gamestates_folder)

        filename = f'{gamestates_folder}/chess_game_{game_state["export-time"]}.json'
        with open(filename, "w") as outfile:
            outfile.write(json.dumps(game_state))

        print(utils.info_text(f"FEN: {game_state['fen-final']}"))
        print(utils.info_text(f"SAN: {game_state['san']}"))
        print(utils.info_text(f"UCI: {game_state['uci']}"))

    def go_command(self):
        if self.go_callback:
            self.go_callback(fen_string=self.board.fen())
        else:
            print(utils.debug_text("Go callback not set"))

    def ready_command(self):
        if self.ready_callback:
            self.ready_callback()
        else:
            print(utils.debug_text("Ready callback not set"))

    def restart_engine(self):
        if self.restart_engine_callback:
            self.restart_engine_callback()
        else:
            print(utils.debug_text("Restart engine callback not set"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # board = chess.Board()
    board = chess.Board("k7/2Q4P/8/8/8/8/8/K2R4")

    chess_gui = ChessGUI(board, dev=True)

    chess_gui.show()
    sys.exit(app.exec())
