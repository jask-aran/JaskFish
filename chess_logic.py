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
    else:
        print("No moves to undo")
    
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