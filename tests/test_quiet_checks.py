import types

import chess

from pvsengine import PVSearchBackend, _SearchBoard


def collect_quiet_checks(backend: PVSearchBackend, board: chess.Board, seen: set[chess.Move]) -> list[str]:
    state = types.SimpleNamespace(search_board=_SearchBoard(board))
    moves = backend._generate_quiet_checks(state, board, seen)
    return sorted(move.uci() for move in moves)


def expected_quiet_checks(board: chess.Board, seen: set[chess.Move]) -> list[str]:
    expected = []
    for move in board.generate_legal_moves():
        if move in seen:
            continue
        if board.is_capture(move) or move.promotion:
            continue
        if board.gives_check(move):
            expected.append(move.uci())
    expected.sort()
    return expected


def test_quiet_checks_match_baseline_across_positions() -> None:
    backend = PVSearchBackend()

    # Direct bishop check (Bb5+)
    board_bishop = chess.Board()
    board_bishop.clear()
    board_bishop.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board_bishop.set_piece_at(chess.C1, chess.Piece(chess.KING, chess.WHITE))
    board_bishop.set_piece_at(chess.C4, chess.Piece(chess.BISHOP, chess.WHITE))
    board_bishop.turn = chess.WHITE

    # Discovered rook checks via e2 moves
    board_discovered = chess.Board()
    board_discovered.clear()
    board_discovered.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board_discovered.set_piece_at(chess.C1, chess.Piece(chess.KING, chess.WHITE))
    board_discovered.set_piece_at(chess.E1, chess.Piece(chess.ROOK, chess.WHITE))
    board_discovered.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
    board_discovered.turn = chess.WHITE

    # Knight jump to deliver check
    board_knight = chess.Board()
    board_knight.clear()
    board_knight.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board_knight.set_piece_at(chess.C1, chess.Piece(chess.KING, chess.WHITE))
    board_knight.set_piece_at(chess.D4, chess.Piece(chess.KNIGHT, chess.WHITE))
    board_knight.turn = chess.WHITE

    test_cases = [
        board_bishop,
        board_discovered,
        board_knight,
    ]

    for board in test_cases:
        seen: set[chess.Move] = set()
        assert collect_quiet_checks(backend, board, seen) == expected_quiet_checks(board, seen)

        # Sanity check that seen moves are filtered out
        baseline = expected_quiet_checks(board, seen)
        if baseline:
            move_obj = chess.Move.from_uci(baseline[0])
            seen_with_move = {move_obj}
            result = collect_quiet_checks(backend, board, seen_with_move)
            assert move_obj.uci() not in result
            baseline_without = expected_quiet_checks(board, seen_with_move)
            assert result == baseline_without
