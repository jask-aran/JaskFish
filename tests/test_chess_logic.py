import chess

import main as chess_logic


def test_is_valid_move_and_make_move_updates_board() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    assert chess_logic.is_valid_move(board, move) is True
    chess_logic.make_move(board, move)
    assert board.piece_at(chess.E4).piece_type == chess.PAWN
    assert board.fullmove_number == 1


def test_is_valid_move_rejects_illegal_move() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e5")
    assert chess_logic.is_valid_move(board, move) is False


def test_get_possible_moves_filters_by_origin_square() -> None:
    board = chess.Board()
    moves = chess_logic.get_possible_moves(board, chess.G1)
    uci_moves = sorted(move.uci() for move in moves)
    assert uci_moves == ["g1f3", "g1h3"]


def test_is_game_over_and_result_messages() -> None:
    checkmate = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        checkmate.push_san(san)
    assert chess_logic.is_game_over(checkmate) is True
    assert chess_logic.get_game_result(checkmate) == "Checkmate"

    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert chess_logic.get_game_result(stalemate) == "Stalemate"

    in_progress = chess.Board()
    assert chess_logic.get_game_result(in_progress) == "Game in progress"


def test_is_in_check_identifies_check_state() -> None:
    board = chess.Board("4k3/8/4R3/8/8/8/8/4K3 b - - 0 1")
    assert chess_logic.is_in_check(board) is True


def test_undo_move_returns_boolean_state() -> None:
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    chess_logic.make_move(board, move)
    assert chess_logic.undo_move(board) is True
    assert chess_logic.undo_move(board) is False


def test_pawn_promotion_detection() -> None:
    board = chess.Board("8/5P2/8/8/8/8/8/6k1 w - - 0 1")
    move = chess.Move.from_uci("f7f8q")
    assert chess_logic.is_pawn_promotion_attempt(board, move) is True

    not_promotion = chess.Move.from_uci("e2e4")
    assert chess_logic.is_pawn_promotion_attempt(board, not_promotion) is False


def test_export_board_state_and_history() -> None:
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3"]
    for uci in moves:
        chess_logic.make_move(board, chess.Move.from_uci(uci))

    assert chess_logic.export_board_fen(board) == board.fen()
    assert chess_logic.export_move_history_uci(board) == " ".join(moves)
    assert chess_logic.export_move_history_san(board) == "e4 e5 Nf3"
