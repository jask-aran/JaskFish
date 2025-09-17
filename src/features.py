"""Feature extraction helpers shared across evaluation backends.

The functions in this module convert :class:`chess.Board` positions into
lightweight numerical representations (bitboards, planes and scalar
summaries) that can be reused by both the classical heuristic evaluator
and potential machine-learning powered evaluators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Tuple

import chess


ColorPiece = Tuple[chess.Color, chess.PieceType]
Plane = List[int]


# Default material values expressed in centipawns.  The constants are defined in
# this module so that they can be shared between classical heuristics and future
# ML feature pipelines without having to duplicate the tables.
DEFAULT_PIECE_VALUES: Mapping[chess.PieceType, float] = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
    chess.KING: 20000.0,
}


@dataclass(frozen=True)
class FeatureBundle:
    """Container aggregating the different feature views of a board position."""

    bitboards: Mapping[ColorPiece, int]
    planes: Mapping[str, Plane]
    scalars: Mapping[str, float]


def _empty_plane() -> Plane:
    return [0] * 64


def board_to_bitboards(board: chess.Board) -> Dict[ColorPiece, int]:
    """Return raw bitboards for every colour/piece combination.

    The returned dictionary maps ``(color, piece_type)`` tuples to integers that
    represent the occupancy bitboard for the respective piece type.  ``int`` is a
    convenient representation because Python exposes ``int.bit_count`` which can
    be used to efficiently compute population counts.
    """

    bitboards: Dict[ColorPiece, int] = {}
    for color in chess.COLORS:
        for piece_type in chess.PIECE_TYPES:
            squares = board.pieces(piece_type, color)
            bitboards[(color, piece_type)] = squares.bb
    return bitboards


def board_to_planes(board: chess.Board) -> Dict[str, Plane]:
    """Convert the current board to a dictionary of 0/1 occupancy planes."""

    planes: Dict[str, Plane] = {}
    for color in chess.COLORS:
        colour_prefix = "white" if color == chess.WHITE else "black"
        for piece_type in chess.PIECE_TYPES:
            name = f"{colour_prefix}_{chess.piece_name(piece_type)}"
            plane = _empty_plane()
            for square in board.pieces(piece_type, color):
                plane[square] = 1
            planes[name] = plane

    occupied_plane = _empty_plane()
    for square in board.occupied:
        occupied_plane[square] = 1
    planes["occupied"] = occupied_plane

    return planes


def board_to_scalar_features(
    board: chess.Board,
    *,
    bitboards: Mapping[ColorPiece, int] | None = None,
    piece_values: Mapping[chess.PieceType, float] | None = None,
) -> Dict[str, float]:
    """Compute scalar summary statistics of a position.

    Parameters
    ----------
    board:
        The board that should be summarised.
    bitboards:
        Optional pre-computed bitboards, typically originating from
        :func:`board_to_bitboards`.  Passing the value avoids recomputing the
        population counts.
    piece_values:
        Material weights, defaulting to :data:`DEFAULT_PIECE_VALUES`.
    """

    if bitboards is None:
        bitboards = board_to_bitboards(board)
    if piece_values is None:
        piece_values = DEFAULT_PIECE_VALUES

    scalars: Dict[str, float] = {
        "turn": 1.0 if board.turn == chess.WHITE else -1.0,
        "halfmove_clock": float(board.halfmove_clock),
        "fullmove_number": float(board.fullmove_number),
        "white_castling_rights": float(board.has_kingside_castling_rights(chess.WHITE))
        + float(board.has_queenside_castling_rights(chess.WHITE)),
        "black_castling_rights": float(board.has_kingside_castling_rights(chess.BLACK))
        + float(board.has_queenside_castling_rights(chess.BLACK)),
    }

    white_material = 0.0
    black_material = 0.0
    for piece_type, value in piece_values.items():
        white_material += bitboards[(chess.WHITE, piece_type)].bit_count() * value
        black_material += bitboards[(chess.BLACK, piece_type)].bit_count() * value

    scalars["material_white"] = white_material
    scalars["material_black"] = black_material
    scalars["material_balance"] = white_material - black_material

    return scalars


def extract_features(board: chess.Board) -> FeatureBundle:
    """Compute all available feature views for a board position."""

    bitboards = board_to_bitboards(board)
    planes = board_to_planes(board)
    scalars = board_to_scalar_features(board, bitboards=bitboards)
    return FeatureBundle(bitboards=bitboards, planes=planes, scalars=scalars)


__all__ = [
    "DEFAULT_PIECE_VALUES",
    "FeatureBundle",
    "board_to_bitboards",
    "board_to_planes",
    "board_to_scalar_features",
    "extract_features",
]

