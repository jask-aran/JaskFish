from PySide2.QtWidgets import QApplication
def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def debug_text(text):
    return f"{color_text('DEBUG', '31')} {text}"

def info_text(text):
    return f"{color_text('INFO', '32')}  {text}"



def get_piece_unicode(piece):
    piece_unicode = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    }
    return piece_unicode[piece.symbol()]

def center_on_screen(window):
    screen = QApplication.primaryScreen()
    screen_geometry = screen.geometry()
    window_size = window.size()
    x = (screen_geometry.width() - window_size.width()) / 2 + screen_geometry.left()
    y = (screen_geometry.height() - window_size.height()) / 2 + screen_geometry.top()
    window.move(x, y)