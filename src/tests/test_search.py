import chess

from search import AlphaBetaSearcher


def test_search_finds_mate_in_one():
    board = chess.Board("rnb1k1nr/pppp1Qpp/5p2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 5")
    searcher = AlphaBetaSearcher(board, max_depth=3, time_limit=None)
    info = searcher.search()

    assert info.move is not None
    assert info.move.uci() == "h5f7"


def test_search_prefers_material_gain():
    board = chess.Board()
    board.clear_board()
    board.set_piece_at(chess.E4, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.D1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.A4, chess.Piece(chess.ROOK, chess.BLACK))
    board.turn = chess.WHITE

    searcher = AlphaBetaSearcher(board, max_depth=2, time_limit=None)
    info = searcher.search()

    assert info.move is not None
    assert info.move.uci() == "d1a4"
