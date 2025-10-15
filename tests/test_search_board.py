import chess
from chess import polyglot

from engines.pvsengine import _SearchBoard


def assert_matches(board: chess.Board, search_board: _SearchBoard) -> None:
    assert search_board.turn == board.turn
    assert search_board.occupied == int(board.occupied)
    assert search_board.occupied_white == int(board.occupied_co[chess.WHITE])
    assert search_board.occupied_black == int(board.occupied_co[chess.BLACK])
    for piece_type in range(1, 7):
        expected_white = int(board.pieces(piece_type, chess.WHITE).mask)
        expected_black = int(board.pieces(piece_type, chess.BLACK).mask)
        assert search_board.piece_bitboards[0][piece_type] == expected_white
        assert search_board.piece_bitboards[1][piece_type] == expected_black
    assert search_board.castling_rights == board.castling_rights
    assert search_board.ep_square == board.ep_square
    expected_hash = getattr(board, "_transposition_key", None)
    if callable(expected_hash):
        expected_hash = expected_hash()
    else:
        expected_hash = polyglot.zobrist_hash(board)
    assert search_board.hash_key == expected_hash


def test_search_board_tracks_basic_sequence() -> None:
    board = chess.Board()
    search_board = _SearchBoard(board)
    assert_matches(board, search_board)

    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("c7c5"),
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("d7d5"),
        chess.Move.from_uci("e4d5"),
        chess.Move.from_uci("g8f6"),
        chess.Move.from_uci("f1b5"),
    ]

    for move in moves:
        search_board.push_move(board, move)
        board.push(move)
        search_board.finalize_move(board, verify=True)
        assert_matches(board, search_board)

    for _ in moves:
        board.pop()
        search_board.pop()
        assert_matches(board, search_board)


def test_search_board_handles_castling_and_ep_flags() -> None:
    board = chess.Board()
    search_board = _SearchBoard(board)

    sequence = [
        chess.Move.from_uci("e2e4"),  # enables EP square e3
        chess.Move.from_uci("e7e5"),  # enables EP square e6
        chess.Move.from_uci("g1f3"),
        chess.Move.from_uci("b8c6"),
        chess.Move.from_uci("d2d4"),
        chess.Move.from_uci("g8f6"),
        chess.Move.from_uci("h1g1"),  # removes white king-side castling right
        chess.Move.from_uci("h8g8"),  # removes black king-side castling right
    ]

    for move in sequence:
        search_board.push_move(board, move)
        board.push(move)
        search_board.finalize_move(board, verify=True)
        assert_matches(board, search_board)

    for _ in sequence:
        board.pop()
        search_board.pop()
        assert_matches(board, search_board)


def test_search_board_handles_null_moves() -> None:
    board = chess.Board()
    search_board = _SearchBoard(board)

    search_board.push_null()
    board.push(chess.Move.null())
    search_board.finalize_move(board, verify=True)
    assert_matches(board, search_board)

    board.pop()
    search_board.pop()
    assert_matches(board, search_board)
