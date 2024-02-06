import chess
import random
import sys

def get_random_legal_move(fen):
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if len(legal_moves) == 0:
        return None
    random_move = random.choice(legal_moves)
    return random_move.uci()


fen = sys.stdin.readline().strip()
move = get_random_legal_move(fen)
print(move)
