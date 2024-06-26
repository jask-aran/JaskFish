import chess

def is_valid_move(board: chess.Board, move: chess.Move) -> bool:
    return move in board.legal_moves

def is_game_over(board: chess.Board) -> bool:
    return board.is_game_over()

def get_game_result(board: chess.Board) -> str:
    if board.is_checkmate():
        return "Checkmate"
    elif board.is_stalemate():
        return "Stalemate"
    elif board.is_insufficient_material():
        return "Insufficient Material"
    elif board.is_seventyfive_moves():
        return "75-move rule"
    elif board.is_fivefold_repetition():
        return "Fivefold Repetition"
    elif board.is_variant_draw():
        return "Variant-specific Draw"
    else:
        return "Game in progress"

def is_in_check(board: chess.Board) -> bool:
    return board.is_check()

def get_possible_moves(board: chess.Board, square: chess.Square) -> list:
    return [move for move in board.legal_moves if move.from_square == square]

def make_move(board: chess.Board, move: chess.Move) -> None:
    board.push(move)

def undo_move(board: chess.Board) -> None:
    if board.move_stack:
        board.pop()
        return True
    return False

    
def is_pawn_promotion_attempt(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    
    move.promotion = chess.QUEEN
    if not is_valid_move(board, move):
        move.promotion = None
        return False
    
    rank = chess.square_rank(move.to_square)
    return (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0)

def export_board_fen(board: chess.Board) -> str:
    return board.fen()


def export_move_history_uci(board: chess.Board) -> str:
    """Exports the move history of a chess game in Universal Chess Interface (UCI) format."""
    moves_uci = [move.uci() for move in board.move_stack]
    return ' '.join(moves_uci)

import chess

def export_move_history_san(board: chess.Board) -> str:
    moves_san = []
    temp_board = board.copy()  # Create a copy of the board
    temp_board.reset()  # Reset to starting position
    
    for move in board.move_stack:
        
        moves_san.append(temp_board.san(move))  # Convert to SAN before applying
        temp_board.push(move)  # Apply the move
    
    return ' '.join(moves_san)

