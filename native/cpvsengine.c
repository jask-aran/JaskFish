#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <ctype.h>
#include <pthread.h>

#define MAX_MOVES 256
#define MAX_PLY 64
#define TT_SIZE (1u << 20)
#define INF 30000

typedef enum {
    COLOR_WHITE = 0,
    COLOR_BLACK = 1
} Color;

typedef struct {
    int squares[128];
    Color side_to_move;
    int castling; /* bit0: WK, bit1: WQ, bit2: BK, bit3: BQ */
    int ep_square;
    int halfmove_clock;
    int fullmove_number;
    uint64_t zobrist_key;
} Position;

typedef struct {
    int move;
    int captured;
    int castling;
    int ep_square;
    int halfmove_clock;
    uint64_t zobrist_key;
} Undo;

typedef struct {
    uint64_t key;
    int value;
    int depth;
    int flag;
    int move;
} TTEntry;

typedef struct {
    int moves[MAX_MOVES];
    int scores[MAX_MOVES];
    int count;
} MoveList;

static Position current_position;
static pthread_t search_thread;
static bool search_running = false;
static volatile bool stop_search = false;
static uint64_t nodes;
static int root_depth;
static int best_move_global = 0;
static int best_score_global = -INF;
static long search_time_limit_ms = 0;
static struct timespec search_start_time;

static TTEntry transposition_table[TT_SIZE];

static int pv_table[MAX_PLY][MAX_PLY];
static int pv_length[MAX_PLY];

static uint64_t zobrist_pieces[12][64];
static uint64_t zobrist_castling[16];
static uint64_t zobrist_ep[8];
static uint64_t zobrist_side;

static const int piece_values[6] = {100, 320, 330, 500, 900, 20000};
static int pst_white[6][64];
static int pst_black[6][64];

static const int pst_reference[6][64] = {
    /* Pawn */
    {
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    },
    /* Knight */
    {
       -50,-40,-30,-30,-30,-30,-40,-50,
       -40,-20,  0,  0,  0,  0,-20,-40,
       -30,  0, 10, 15, 15, 10,  0,-30,
       -30,  5, 15, 20, 20, 15,  5,-30,
       -30,  0, 15, 20, 20, 15,  0,-30,
       -30,  5, 10, 15, 15, 10,  5,-30,
       -40,-20,  0,  5,  5,  0,-20,-40,
       -50,-40,-30,-30,-30,-30,-40,-50
    },
    /* Bishop */
    {
       -20,-10,-10,-10,-10,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5, 10, 10,  5,  0,-10,
       -10,  5,  5, 10, 10,  5,  5,-10,
       -10,  0, 10, 10, 10, 10,  0,-10,
       -10, 10, 10, 10, 10, 10, 10,-10,
       -10,  5,  0,  0,  0,  0,  5,-10,
       -20,-10,-10,-10,-10,-10,-10,-20
    },
    /* Rook */
    {
         0,  0,  0,  0,  0,  0,  0,  0,
         5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         0,  0,  0,  5,  5,  0,  0,  0
    },
    /* Queen */
    {
       -20,-10,-10, -5, -5,-10,-10,-20,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
         0,  0,  5,  5,  5,  5,  0, -5,
       -10,  0,  5,  5,  5,  5,  0,-10,
       -10,  0,  0,  0,  0,  0,  0,-10,
       -20,-10,-10, -5, -5,-10,-10,-20
    },
    /* King */
    {
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-40,-40,-50,-50,-40,-40,-30,
       -30,-30,-30,-40,-40,-30,-30,-30,
       -20,-20,-20,-20,-20,-20,-20,-20,
       -10,-10,-10,-10,-10,-10,-10,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    }
};

static const int knight_offsets[8] = {31, 33, 14, 18, -31, -33, -14, -18};
static const int bishop_offsets[4] = {15, 17, -15, -17};
static const int rook_offsets[4] = {16, -16, 1, -1};
static const int king_offsets[8] = {16, -16, 1, -1, 15, 17, -15, -17};

static inline int square_to_64(int sq) {
    return ((sq >> 4) * 8) + (sq & 7);
}

static inline bool square_on_board(int sq) {
    return !(sq & 0x88);
}

static inline int piece_index(int piece) {
    if (piece > 0) {
        return piece - 1;
    }
    return (-piece - 1) + 6;
}

static inline Color piece_color(int piece) {
    return piece > 0 ? COLOR_WHITE : COLOR_BLACK;
}

static inline int piece_type(int piece) {
    return piece > 0 ? piece : -piece;
}

static uint64_t random_u64(void) {
    uint64_t a = (uint64_t)rand();
    uint64_t b = (uint64_t)rand();
    uint64_t c = (uint64_t)rand();
    uint64_t d = (uint64_t)rand();
    return (a << 48) ^ (b << 32) ^ (c << 16) ^ d;
}

static void init_zobrist(void) {
    srand(20231123);
    for (int i = 0; i < 12; ++i) {
        for (int sq = 0; sq < 64; ++sq) {
            zobrist_pieces[i][sq] = random_u64();
        }
    }
    for (int i = 0; i < 16; ++i) {
        zobrist_castling[i] = random_u64();
    }
    for (int i = 0; i < 8; ++i) {
        zobrist_ep[i] = random_u64();
    }
    zobrist_side = random_u64();
}

static void init_pst(void) {
    for (int p = 0; p < 6; ++p) {
        for (int sq = 0; sq < 64; ++sq) {
            pst_white[p][sq] = pst_reference[p][sq];
            int rank = sq / 8;
            int file = sq % 8;
            int mirrored = (7 - rank) * 8 + file;
            pst_black[p][sq] = pst_reference[p][mirrored];
        }
    }
}

static void reset_transposition(void) {
    memset(transposition_table, 0, sizeof(transposition_table));
}

static uint64_t compute_zobrist(const Position *pos) {
    uint64_t key = 0;
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0) {
            continue;
        }
        int idx = piece_index(piece);
        key ^= zobrist_pieces[idx][square_to_64(sq)];
    }
    key ^= zobrist_castling[pos->castling & 15];
    if (pos->ep_square != -1) {
        key ^= zobrist_ep[(pos->ep_square & 7)];
    }
    if (pos->side_to_move == COLOR_BLACK) {
        key ^= zobrist_side;
    }
    return key;
}

static void clear_position(Position *pos) {
    memset(pos->squares, 0, sizeof(pos->squares));
    pos->side_to_move = COLOR_WHITE;
    pos->castling = 0;
    pos->ep_square = -1;
    pos->halfmove_clock = 0;
    pos->fullmove_number = 1;
    pos->zobrist_key = compute_zobrist(pos);
}

static bool parse_fen(Position *pos, const char *fen) {
    clear_position(pos);
    int rank = 7;
    int file = 0;
    const char *ptr = fen;
    while (*ptr && rank >= 0) {
        char c = *ptr++;
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        if (c == ' ') {
            break;
        }
        if (isdigit((unsigned char)c)) {
            file += c - '0';
            continue;
        }
        if (file >= 8) {
            return false;
        }
        int sq = (rank << 4) | file;
        int piece = 0;
        switch (c) {
            case 'P': piece = 1; break;
            case 'N': piece = 2; break;
            case 'B': piece = 3; break;
            case 'R': piece = 4; break;
            case 'Q': piece = 5; break;
            case 'K': piece = 6; break;
            case 'p': piece = -1; break;
            case 'n': piece = -2; break;
            case 'b': piece = -3; break;
            case 'r': piece = -4; break;
            case 'q': piece = -5; break;
            case 'k': piece = -6; break;
            default: return false;
        }
        pos->squares[sq] = piece;
        file++;
    }
    if (*ptr != ' ') {
        return false;
    }
    ptr++;
    if (*ptr == 'w') {
        pos->side_to_move = COLOR_WHITE;
    } else if (*ptr == 'b') {
        pos->side_to_move = COLOR_BLACK;
    } else {
        return false;
    }
    ptr++;
    if (*ptr != ' ') {
        return false;
    }
    ptr++;
    pos->castling = 0;
    if (*ptr == '-') {
        ptr++;
    } else {
        while (*ptr && *ptr != ' ') {
            switch (*ptr) {
                case 'K': pos->castling |= 1; break;
                case 'Q': pos->castling |= 2; break;
                case 'k': pos->castling |= 4; break;
                case 'q': pos->castling |= 8; break;
                default: return false;
            }
            ptr++;
        }
    }
    if (*ptr != ' ') {
        return false;
    }
    ptr++;
    pos->ep_square = -1;
    if (*ptr == '-') {
        ptr++;
    } else {
        if (!isalpha((unsigned char)ptr[0]) || !isdigit((unsigned char)ptr[1])) {
            return false;
        }
        int file_char = ptr[0] - 'a';
        int rank_char = ptr[1] - '1';
        if (file_char < 0 || file_char > 7 || rank_char < 0 || rank_char > 7) {
            return false;
        }
        pos->ep_square = (rank_char << 4) | file_char;
        ptr += 2;
    }
    if (*ptr != ' ') {
        return false;
    }
    ptr++;
    pos->halfmove_clock = atoi(ptr);
    while (*ptr && *ptr != ' ') {
        ptr++;
    }
    if (*ptr == ' ') {
        ptr++;
        pos->fullmove_number = atoi(ptr);
    } else {
        pos->fullmove_number = 1;
    }
    pos->zobrist_key = compute_zobrist(pos);
    return true;
}

static bool parse_fen_wrapper(Position *pos, const char *fen) {
    if (strcmp(fen, "startpos") == 0) {
        return parse_fen(pos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    return parse_fen(pos, fen);
}

static bool square_attacked(const Position *pos, int square, Color by_color);
static bool make_move(Position *pos, int move, Undo *undo);
static void undo_move(Position *pos, int move, const Undo *undo);
static void generate_moves(const Position *pos, MoveList *list, bool captures_only);
static int search(Position *pos, int depth, int alpha, int beta, int ply);
static int quiescence(Position *pos, int alpha, int beta, int ply);
static void *search_position(void *arg);
static void print_bestmove(int move);
static void send_info(int depth, int score, int time_ms, uint64_t nodes_searched);
static void format_move_uci(int move, char buffer[8]);
static int parse_move(const Position *pos, const char *uci);

static void update_zobrist_piece(Position *pos, int piece, int square) {
    int idx = piece_index(piece);
    pos->zobrist_key ^= zobrist_pieces[idx][square_to_64(square)];
}

static void update_zobrist_castling(Position *pos, int castling) {
    pos->zobrist_key ^= zobrist_castling[castling & 15];
}

static void update_zobrist_ep(Position *pos, int ep_square) {
    if (ep_square != -1) {
        pos->zobrist_key ^= zobrist_ep[ep_square & 7];
    }
}

static void toggle_side(Position *pos) {
    pos->side_to_move = (pos->side_to_move == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
    pos->zobrist_key ^= zobrist_side;
}

static int evaluate(const Position *pos) {
    int score = 0;
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0) {
            continue;
        }
        int type = piece_type(piece) - 1;
        int sq64 = square_to_64(sq);
        if (piece > 0) {
            score += piece_values[type] + pst_white[type][sq64];
        } else {
            score -= piece_values[type] + pst_black[type][sq64];
        }
    }
    return (pos->side_to_move == COLOR_WHITE) ? score : -score;
}

static bool square_attacked(const Position *pos, int square, Color by_color) {
    int direction = (by_color == COLOR_WHITE) ? -16 : 16;
    int left = direction - 1;
    int right = direction + 1;
    int target = square + left;
    if (square_on_board(target)) {
        int piece = pos->squares[target];
        if (piece == (by_color == COLOR_WHITE ? 1 : -1)) {
            return true;
        }
    }
    target = square + right;
    if (square_on_board(target)) {
        int piece = pos->squares[target];
        if (piece == (by_color == COLOR_WHITE ? 1 : -1)) {
            return true;
        }
    }
    for (int i = 0; i < 8; ++i) {
        int sq = square + knight_offsets[i];
        if (!square_on_board(sq)) {
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0) {
            continue;
        }
        if (piece == (by_color == COLOR_WHITE ? 2 : -2)) {
            return true;
        }
    }
    for (int i = 0; i < 4; ++i) {
        int offset = bishop_offsets[i];
        int sq = square + offset;
        while (square_on_board(sq)) {
            int piece = pos->squares[sq];
            if (piece != 0) {
                if (piece_color(piece) == by_color && (piece_type(piece) == 3 || piece_type(piece) == 5)) {
                    return true;
                }
                break;
            }
            sq += offset;
        }
    }
    for (int i = 0; i < 4; ++i) {
        int offset = rook_offsets[i];
        int sq = square + offset;
        while (square_on_board(sq)) {
            int piece = pos->squares[sq];
            if (piece != 0) {
                if (piece_color(piece) == by_color && (piece_type(piece) == 4 || piece_type(piece) == 5)) {
                    return true;
                }
                break;
            }
            sq += offset;
        }
    }
    for (int i = 0; i < 8; ++i) {
        int sq = square + king_offsets[i];
        if (!square_on_board(sq)) {
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == (by_color == COLOR_WHITE ? 6 : -6)) {
            return true;
        }
    }
    return false;
}

static int encode_move(int from, int to, int promotion, int flags) {
    return (from & 0x7F) | ((to & 0x7F) << 7) | ((promotion & 7) << 14) | ((flags & 0xF) << 17);
}

static int move_from(int move) { return move & 0x7F; }
static int move_to(int move) { return (move >> 7) & 0x7F; }
static int move_promotion(int move) { return (move >> 14) & 7; }
static int move_flags(int move) { return (move >> 17) & 0xF; }

enum MoveFlags {
    FLAG_NONE = 0,
    FLAG_CAPTURE = 1,
    FLAG_DOUBLE_PAWN = 2,
    FLAG_EN_PASSANT = 4,
    FLAG_CASTLING = 8
};

static void add_move(const Position *pos, MoveList *list, int from, int to, int promotion, int flags) {
    if (list->count >= MAX_MOVES) {
        return;
    }
    int move = encode_move(from, to, promotion, flags);
    list->moves[list->count] = move;
    int captured = pos->squares[to];
    int score = 0;
    if (captured != 0) {
        int victim = piece_type(captured) - 1;
        int attacker = piece_type(pos->squares[from]) - 1;
        score = 10 * piece_values[victim] - piece_values[attacker];
    }
    if (promotion) {
        score += piece_values[promotion - 1];
    }
    list->scores[list->count] = score;
    list->count++;
}

static void generate_moves(const Position *pos, MoveList *list, bool captures_only) {
    list->count = 0;
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0 || piece_color(piece) != pos->side_to_move) {
            continue;
        }
        int type = piece_type(piece);
        if (type == 1) {
            int forward = (pos->side_to_move == COLOR_WHITE) ? 16 : -16;
            int start_rank = (pos->side_to_move == COLOR_WHITE) ? 1 : 6;
            int promotion_rank = (pos->side_to_move == COLOR_WHITE) ? 6 : 1;
            int to = sq + forward;
            if (!captures_only && square_on_board(to) && pos->squares[to] == 0) {
                if ((sq >> 4) == promotion_rank) {
                    add_move(pos, list, sq, to, 5, FLAG_NONE);
                    add_move(pos, list, sq, to, 4, FLAG_NONE);
                    add_move(pos, list, sq, to, 3, FLAG_NONE);
                    add_move(pos, list, sq, to, 2, FLAG_NONE);
                } else {
                    add_move(pos, list, sq, to, 0, FLAG_NONE);
                    if ((sq >> 4) == start_rank) {
                        int two = to + forward;
                        if (square_on_board(two) && pos->squares[two] == 0) {
                            add_move(pos, list, sq, two, 0, FLAG_DOUBLE_PAWN);
                        }
                    }
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                int capture_sq = sq + forward + df;
                if (!square_on_board(capture_sq)) {
                    continue;
                }
                if (capture_sq == pos->ep_square) {
                    add_move(pos, list, sq, capture_sq, 0, FLAG_EN_PASSANT | FLAG_CAPTURE);
                } else {
                    int target_piece = pos->squares[capture_sq];
                    if (target_piece != 0 && piece_color(target_piece) != pos->side_to_move) {
                        int flags = FLAG_CAPTURE;
                        if ((sq >> 4) == promotion_rank) {
                            add_move(pos, list, sq, capture_sq, 5, flags);
                            add_move(pos, list, sq, capture_sq, 4, flags);
                            add_move(pos, list, sq, capture_sq, 3, flags);
                            add_move(pos, list, sq, capture_sq, 2, flags);
                        } else {
                            add_move(pos, list, sq, capture_sq, 0, flags);
                        }
                    }
                }
            }
        } else if (type == 2) {
            for (int i = 0; i < 8; ++i) {
                int to = sq + knight_offsets[i];
                if (!square_on_board(to)) {
                    continue;
                }
                int target_piece = pos->squares[to];
                if (target_piece == 0 && !captures_only) {
                    add_move(pos, list, sq, to, 0, FLAG_NONE);
                } else if (target_piece != 0 && piece_color(target_piece) != pos->side_to_move) {
                    add_move(pos, list, sq, to, 0, FLAG_CAPTURE);
                }
            }
        } else if (type == 3 || type == 5) {
            for (int i = 0; i < 4; ++i) {
                int offset = bishop_offsets[i];
                int to = sq + offset;
                while (square_on_board(to)) {
                    int target_piece = pos->squares[to];
                    if (target_piece == 0) {
                        if (!captures_only) {
                            add_move(pos, list, sq, to, 0, FLAG_NONE);
                        }
                    } else {
                        if (piece_color(target_piece) != pos->side_to_move) {
                            add_move(pos, list, sq, to, 0, FLAG_CAPTURE);
                        }
                        break;
                    }
                    to += offset;
                }
            }
            if (type == 3) {
                continue;
            }
            // queen continues with rook offsets below
        }
        if (type == 4 || type == 5) {
            for (int i = 0; i < 4; ++i) {
                int offset = rook_offsets[i];
                int to = sq + offset;
                while (square_on_board(to)) {
                    int target_piece = pos->squares[to];
                    if (target_piece == 0) {
                        if (!captures_only) {
                            add_move(pos, list, sq, to, 0, FLAG_NONE);
                        }
                    } else {
                        if (piece_color(target_piece) != pos->side_to_move) {
                            add_move(pos, list, sq, to, 0, FLAG_CAPTURE);
                        }
                        break;
                    }
                    to += offset;
                }
            }
        }
        if (type == 6) {
            for (int i = 0; i < 8; ++i) {
                int to = sq + king_offsets[i];
                if (!square_on_board(to)) {
                    continue;
                }
                int target_piece = pos->squares[to];
                if (target_piece == 0 && !captures_only) {
                    add_move(pos, list, sq, to, 0, FLAG_NONE);
                } else if (target_piece != 0 && piece_color(target_piece) != pos->side_to_move) {
                    add_move(pos, list, sq, to, 0, FLAG_CAPTURE);
                }
            }
            if (!captures_only) {
                if (pos->side_to_move == COLOR_WHITE) {
                    if ((pos->castling & 1) && pos->squares[5] == 0 && pos->squares[6] == 0 &&
                        !square_attacked(pos, 4, COLOR_BLACK) && !square_attacked(pos, 5, COLOR_BLACK) &&
                        !square_attacked(pos, 6, COLOR_BLACK)) {
                        add_move(pos, list, sq, sq + 2, 0, FLAG_CASTLING);
                    }
                    if ((pos->castling & 2) && pos->squares[3] == 0 && pos->squares[2] == 0 && pos->squares[1] == 0 &&
                        !square_attacked(pos, 4, COLOR_BLACK) && !square_attacked(pos, 3, COLOR_BLACK) &&
                        !square_attacked(pos, 2, COLOR_BLACK)) {
                        add_move(pos, list, sq, sq - 2, 0, FLAG_CASTLING);
                    }
                } else {
                    if ((pos->castling & 4) && pos->squares[117] == 0 && pos->squares[118] == 0 &&
                        !square_attacked(pos, 116, COLOR_WHITE) && !square_attacked(pos, 117, COLOR_WHITE) &&
                        !square_attacked(pos, 118, COLOR_WHITE)) {
                        add_move(pos, list, sq, sq + 2, 0, FLAG_CASTLING);
                    }
                    if ((pos->castling & 8) && pos->squares[115] == 0 && pos->squares[114] == 0 && pos->squares[113] == 0 &&
                        !square_attacked(pos, 116, COLOR_WHITE) && !square_attacked(pos, 115, COLOR_WHITE) &&
                        !square_attacked(pos, 114, COLOR_WHITE)) {
                        add_move(pos, list, sq, sq - 2, 0, FLAG_CASTLING);
                    }
                }
            }
        }
    }
}

static int find_king(const Position *pos, Color color) {
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0) {
            continue;
        }
        if (piece_color(piece) == color && piece_type(piece) == 6) {
            return sq;
        }
    }
    return -1;
}

static void update_castling_rights(Position *pos, int square) {
    switch (square) {
        case 4: pos->castling &= ~3; break;
        case 0: pos->castling &= ~2; break;
        case 7: pos->castling &= ~1; break;
        case 116: pos->castling &= ~12; break;
        case 112: pos->castling &= ~8; break;
        case 119: pos->castling &= ~4; break;
        default: break;
    }
}

static bool make_move(Position *pos, int move, Undo *undo) {
    int from = move_from(move);
    int to = move_to(move);
    int promotion = move_promotion(move);
    int flags = move_flags(move);
    int piece = pos->squares[from];
    int captured = pos->squares[to];

    undo->move = move;
    undo->captured = captured;
    undo->castling = pos->castling;
    undo->ep_square = pos->ep_square;
    undo->halfmove_clock = pos->halfmove_clock;
    undo->zobrist_key = pos->zobrist_key;

    pos->halfmove_clock++;

    if (piece_type(piece) == 1 || captured != 0) {
        pos->halfmove_clock = 0;
    }

    update_zobrist_piece(pos, piece, from);
    if (captured != 0) {
        update_zobrist_piece(pos, captured, to);
    }

    pos->ep_square = -1;
    pos->squares[to] = piece;
    pos->squares[from] = 0;
    update_zobrist_piece(pos, piece, to);

    update_castling_rights(pos, from);
    update_castling_rights(pos, to);

    if (flags & FLAG_DOUBLE_PAWN) {
        pos->ep_square = (pos->side_to_move == COLOR_WHITE) ? to - 16 : to + 16;
    }

    if (flags & FLAG_EN_PASSANT) {
        int capture_sq = (pos->side_to_move == COLOR_WHITE) ? to - 16 : to + 16;
        int captured_piece = pos->squares[capture_sq];
        update_zobrist_piece(pos, captured_piece, capture_sq);
        pos->squares[capture_sq] = 0;
        captured = captured_piece;
    }

    if (promotion) {
        update_zobrist_piece(pos, pos->squares[to], to);
        int promoted_piece = (pos->side_to_move == COLOR_WHITE) ? promotion : -promotion;
        pos->squares[to] = promoted_piece;
        update_zobrist_piece(pos, pos->squares[to], to);
    }

    if (flags & FLAG_CASTLING) {
        if (to == from + 2) {
            int rook_from = to + 1;
            int rook_to = to - 1;
            int rook = pos->squares[rook_from];
            update_zobrist_piece(pos, rook, rook_from);
            pos->squares[rook_to] = rook;
            pos->squares[rook_from] = 0;
            update_zobrist_piece(pos, rook, rook_to);
        } else if (to == from - 2) {
            int rook_from = to - 2;
            int rook_to = to + 1;
            int rook = pos->squares[rook_from];
            update_zobrist_piece(pos, rook, rook_from);
            pos->squares[rook_to] = rook;
            pos->squares[rook_from] = 0;
            update_zobrist_piece(pos, rook, rook_to);
        }
    }

    update_zobrist_castling(pos, undo->castling);
    update_zobrist_castling(pos, pos->castling);
    if (undo->ep_square != -1) {
        update_zobrist_ep(pos, undo->ep_square);
    }
    if (pos->ep_square != -1) {
        update_zobrist_ep(pos, pos->ep_square);
    }

    toggle_side(pos);

    int king_sq = find_king(pos, (pos->side_to_move == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE);
    if (king_sq == -1 || square_attacked(pos, king_sq, pos->side_to_move)) {
        undo_move(pos, move, undo);
        return false;
    }

    if (pos->side_to_move == COLOR_WHITE) {
        pos->fullmove_number++;
    }

    return true;
}

static void undo_move(Position *pos, int move, const Undo *undo) {
    int from = move_from(move);
    int to = move_to(move);
    int promotion = move_promotion(move);
    int flags = move_flags(move);
    toggle_side(pos);
    pos->zobrist_key = undo->zobrist_key;
    pos->castling = undo->castling;
    pos->ep_square = undo->ep_square;
    pos->halfmove_clock = undo->halfmove_clock;
    if (pos->side_to_move == COLOR_WHITE) {
        pos->fullmove_number--;
    }
    int piece = pos->squares[to];
    if (promotion) {
        piece = (pos->side_to_move == COLOR_WHITE) ? 1 : -1;
    }
    pos->squares[from] = piece;
    pos->squares[to] = undo->captured;
    if (flags & FLAG_EN_PASSANT) {
        int capture_sq = (pos->side_to_move == COLOR_WHITE) ? to - 16 : to + 16;
        pos->squares[to] = 0;
        pos->squares[capture_sq] = (pos->side_to_move == COLOR_WHITE) ? -1 : 1;
    }
    if (flags & FLAG_CASTLING) {
        if (to == from + 2) {
            int rook_from = to - 1;
            int rook_to = to + 1;
            pos->squares[rook_to] = pos->squares[rook_from];
            pos->squares[rook_from] = 0;
        } else if (to == from - 2) {
            int rook_from = to + 1;
            int rook_to = to - 2;
            pos->squares[rook_to] = pos->squares[rook_from];
            pos->squares[rook_from] = 0;
        }
    }
}

static void sort_moves(MoveList *list) {
    for (int i = 1; i < list->count; ++i) {
        int move = list->moves[i];
        int score = list->scores[i];
        int j = i - 1;
        while (j >= 0 && list->scores[j] < score) {
            list->moves[j + 1] = list->moves[j];
            list->scores[j + 1] = list->scores[j];
            j--;
        }
        list->moves[j + 1] = move;
        list->scores[j + 1] = score;
    }
}

static inline long elapsed_ms(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    long seconds = now.tv_sec - search_start_time.tv_sec;
    long nanos = now.tv_nsec - search_start_time.tv_nsec;
    return seconds * 1000 + nanos / 1000000;
}

static bool time_exceeded(void) {
    if (search_time_limit_ms <= 0) {
        return stop_search;
    }
    if (stop_search) {
        return true;
    }
    return elapsed_ms() >= search_time_limit_ms;
}

static int quiescence(Position *pos, int alpha, int beta, int ply) {
    int stand_pat = evaluate(pos);
    if (stand_pat >= beta) {
        return beta;
    }
    if (alpha < stand_pat) {
        alpha = stand_pat;
    }
    MoveList list;
    generate_moves(pos, &list, true);
    sort_moves(&list);
    for (int i = 0; i < list.count; ++i) {
        int move = list.moves[i];
        if (!(move_flags(move) & FLAG_CAPTURE) && !(move_flags(move) & FLAG_EN_PASSANT)) {
            continue;
        }
        Undo undo;
        if (!make_move(pos, move, &undo)) {
            continue;
        }
        nodes++;
        int score = -quiescence(pos, -beta, -alpha, ply + 1);
        undo_move(pos, move, &undo);
        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    return alpha;
}

static int probe_tt(uint64_t key, int depth, int alpha, int beta, int *tt_move) {
    TTEntry *entry = &transposition_table[key & (TT_SIZE - 1)];
    if (entry->key == key && entry->depth >= depth) {
        if (tt_move) {
            *tt_move = entry->move;
        }
        if (entry->flag == 0) {
            return entry->value;
        }
        if (entry->flag == -1 && entry->value <= alpha) {
            return alpha;
        }
        if (entry->flag == 1 && entry->value >= beta) {
            return beta;
        }
    }
    if (tt_move) {
        *tt_move = (entry->key == key) ? entry->move : 0;
    }
    return INF + 1;
}

static void store_tt(uint64_t key, int depth, int value, int flag, int move) {
    TTEntry *entry = &transposition_table[key & (TT_SIZE - 1)];
    entry->key = key;
    entry->value = value;
    entry->depth = depth;
    entry->flag = flag;
    entry->move = move;
}

static int search(Position *pos, int depth, int alpha, int beta, int ply) {
    int original_alpha = alpha;
    if (time_exceeded()) {
        return alpha;
    }
    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply);
    }

    int tt_move = 0;
    int tt_value = probe_tt(pos->zobrist_key, depth, alpha, beta, &tt_move);
    if (tt_value != INF + 1) {
        return tt_value;
    }

    pv_length[ply] = 0;

    MoveList list;
    generate_moves(pos, &list, false);
    if (list.count == 0) {
        int king_sq = find_king(pos, pos->side_to_move);
        if (king_sq != -1 && square_attacked(pos, king_sq, pos->side_to_move == COLOR_WHITE ? COLOR_BLACK : COLOR_WHITE)) {
            return -INF + ply;
        }
        return 0;
    }

    if (tt_move) {
        for (int i = 0; i < list.count; ++i) {
            if (list.moves[i] == tt_move) {
                list.scores[i] = 1000000;
                break;
            }
        }
    }

    sort_moves(&list);

    int best_value = -INF;
    int best_move = 0;
    bool first_move = true;

    for (int i = 0; i < list.count; ++i) {
        int move = list.moves[i];
        Undo undo;
        if (!make_move(pos, move, &undo)) {
            continue;
        }
        nodes++;
        int score;
        if (first_move) {
            score = -search(pos, depth - 1, -beta, -alpha, ply + 1);
            first_move = false;
        } else {
            score = -search(pos, depth - 1, -alpha - 1, -alpha, ply + 1);
            if (score > alpha && score < beta) {
                score = -search(pos, depth - 1, -beta, -alpha, ply + 1);
            }
        }
        undo_move(pos, move, &undo);

        if (score > best_value) {
            best_value = score;
            best_move = move;
        }
        if (score > alpha) {
            alpha = score;
            pv_table[ply][0] = move;
            pv_length[ply] = pv_length[ply + 1] + 1;
            for (int j = 0; j < pv_length[ply + 1]; ++j) {
                pv_table[ply][j + 1] = pv_table[ply + 1][j];
            }
        }
        if (alpha >= beta) {
            break;
        }
    }

    if (best_move == 0) {
        return best_value;
    }

    int flag = 0;
    if (best_value <= original_alpha) {
        flag = -1;
    } else if (best_value >= beta) {
        flag = 1;
    }
    store_tt(pos->zobrist_key, depth, best_value, flag, best_move);

    return best_value;
}

static void send_info(int depth, int score, int time_ms, uint64_t nodes_searched) {
    printf("info depth %d score cp %d time %d nodes %llu nps %llu", depth, score, time_ms,
           (unsigned long long)nodes_searched,
           time_ms > 0 ? (unsigned long long)(nodes_searched * 1000 / time_ms) : 0ULL);
    printf(" pv");
    for (int i = 0; i < pv_length[0]; ++i) {
        char buffer[8];
        format_move_uci(pv_table[0][i], buffer);
        printf(" %s", buffer);
    }
    printf("\n");
    fflush(stdout);
}

static void format_move_uci(int move, char buffer[8]) {
    int from = move_from(move);
    int to = move_to(move);
    int promotion = move_promotion(move);
    buffer[0] = 'a' + (from & 7);
    buffer[1] = '1' + (from >> 4);
    buffer[2] = 'a' + (to & 7);
    buffer[3] = '1' + (to >> 4);
    int idx = 4;
    if (promotion) {
        char promo_char = 'q';
        switch (promotion) {
            case 2: promo_char = 'n'; break;
            case 3: promo_char = 'b'; break;
            case 4: promo_char = 'r'; break;
            case 5: promo_char = 'q'; break;
        }
        buffer[idx++] = promo_char;
    }
    buffer[idx] = '\0';
}

static int parse_move(const Position *pos, const char *uci) {
    if (strlen(uci) < 4) {
        return 0;
    }
    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file = uci[2] - 'a';
    int to_rank = uci[3] - '1';
    if (from_file < 0 || from_file > 7 || to_file < 0 || to_file > 7 ||
        from_rank < 0 || from_rank > 7 || to_rank < 0 || to_rank > 7) {
        return 0;
    }
    int from = (from_rank << 4) | from_file;
    int to = (to_rank << 4) | to_file;
    int promotion = 0;
    if (strlen(uci) >= 5) {
        switch (uci[4]) {
            case 'q': promotion = 5; break;
            case 'r': promotion = 4; break;
            case 'b': promotion = 3; break;
            case 'n': promotion = 2; break;
            default: promotion = 0; break;
        }
    }
    MoveList list;
    generate_moves(pos, &list, false);
    for (int i = 0; i < list.count; ++i) {
        int move = list.moves[i];
        if (move_from(move) == from && move_to(move) == to) {
            if (!promotion || move_promotion(move) == promotion) {
                Position tmp = *pos;
                Undo undo;
                if (make_move(&tmp, move, &undo)) {
                    return move;
                }
            }
        }
    }
    return 0;
}

static void *search_position(void *arg) {
    (void)arg;
    Position pos = current_position;
    nodes = 0;
    best_move_global = 0;
    best_score_global = -INF;
    clock_gettime(CLOCK_MONOTONIC, &search_start_time);
    pv_length[0] = 0;
    for (int depth = 1; depth <= root_depth; ++depth) {
        pv_length[depth] = 0;
        int score = search(&pos, depth, -INF, INF, 0);
        if (time_exceeded()) {
            break;
        }
        best_move_global = pv_table[0][0];
        best_score_global = score;
        int time_ms = (int)elapsed_ms();
        send_info(depth, score, time_ms, nodes);
        if (search_time_limit_ms > 0 && elapsed_ms() > search_time_limit_ms) {
            break;
        }
    }
    if (best_move_global == 0) {
        MoveList list;
        generate_moves(&pos, &list, false);
        if (list.count > 0) {
            best_move_global = list.moves[0];
        }
    }
    print_bestmove(best_move_global);
    search_running = false;
    return NULL;
}

static void print_bestmove(int move) {
    char buffer[8] = {0};
    if (move != 0) {
        format_move_uci(move, buffer);
    } else {
        strcpy(buffer, "0000");
    }
    printf("bestmove %s\n", buffer);
    fflush(stdout);
}

static void ensure_search_finished(void) {
    if (search_running) {
        stop_search = true;
        pthread_join(search_thread, NULL);
        search_running = false;
    }
    stop_search = false;
}

static void handle_position_command(const char *args) {
    ensure_search_finished();
    Position pos;
    clear_position(&pos);
    const char *ptr = args;
    if (strncmp(ptr, "startpos", 8) == 0) {
        parse_fen_wrapper(&pos, "startpos");
        ptr += 8;
    } else if (strncmp(ptr, "fen", 3) == 0) {
        ptr += 3;
        while (*ptr == ' ') ptr++;
        char fen[256];
        int idx = 0;
        int spaces = 0;
        while (*ptr && !(spaces >= 5 && *ptr == ' ')) {
            if (*ptr == ' ') {
                spaces++;
            }
            if (idx < 255) {
                fen[idx++] = *ptr;
            }
            ptr++;
        }
        fen[idx] = '\0';
        parse_fen_wrapper(&pos, fen);
    } else {
        return;
    }
    while (*ptr == ' ') ptr++;
    if (strncmp(ptr, "moves", 5) == 0) {
        ptr += 5;
        while (*ptr == ' ') ptr++;
        char move_buf[16];
        while (*ptr) {
            int len = 0;
            while (*ptr && *ptr != ' ') {
                if (len < 15) move_buf[len++] = *ptr;
                ptr++;
            }
            move_buf[len] = '\0';
            if (len == 0) {
                break;
            }
            int move = parse_move(&pos, move_buf);
            if (move) {
                Undo undo;
                make_move(&pos, move, &undo);
            }
            while (*ptr == ' ') ptr++;
        }
    }
    current_position = pos;
}

static void handle_go_command(const char *args) {
    ensure_search_finished();
    search_time_limit_ms = 0;
    root_depth = 64;
    const char *ptr = args;
    while (*ptr) {
        while (*ptr == ' ') ptr++;
        if (strncmp(ptr, "depth", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            root_depth = atoi(ptr);
        } else if (strncmp(ptr, "movetime", 8) == 0) {
            ptr += 8;
            while (*ptr == ' ') ptr++;
            search_time_limit_ms = atol(ptr);
        } else if (strncmp(ptr, "wtime", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            long wtime = atol(ptr);
            if (current_position.side_to_move == COLOR_WHITE) {
                search_time_limit_ms = wtime / 30;
            }
        } else if (strncmp(ptr, "btime", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            long btime = atol(ptr);
            if (current_position.side_to_move == COLOR_BLACK) {
                search_time_limit_ms = btime / 30;
            }
        }
        while (*ptr && *ptr != ' ') ptr++;
    }
    if (root_depth <= 0) {
        root_depth = 1;
    }
    stop_search = false;
    search_running = true;
    pthread_create(&search_thread, NULL, (void *(*)(void *))search_position, NULL);
}

static void handle_stop_command(void) {
    if (search_running) {
        stop_search = true;
        pthread_join(search_thread, NULL);
        search_running = false;
    }
}

static void handle_uci(void) {
    printf("id name CPVSEngine\n");
    printf("id author JaskFish Native Port\n");
    printf("uciok\n");
    fflush(stdout);
}

static void handle_isready(void) {
    printf("readyok\n");
    fflush(stdout);
}

static void handle_ucinewgame(void) {
    ensure_search_finished();
    parse_fen_wrapper(&current_position, "startpos");
    reset_transposition();
}

static void handle_quit(void) {
    handle_stop_command();
    exit(0);
}

int main(void) {
    init_zobrist();
    init_pst();
    parse_fen_wrapper(&current_position, "startpos");
    reset_transposition();
    char line[512];
    while (fgets(line, sizeof(line), stdin)) {
        char *newline = strchr(line, '\n');
        if (newline) {
            *newline = '\0';
        }
        if (strncmp(line, "uci", 3) == 0) {
            handle_uci();
        } else if (strncmp(line, "isready", 7) == 0) {
            handle_isready();
        } else if (strncmp(line, "ucinewgame", 10) == 0) {
            handle_ucinewgame();
        } else if (strncmp(line, "position", 8) == 0) {
            handle_position_command(line + 8);
        } else if (strncmp(line, "go", 2) == 0) {
            handle_go_command(line + 2);
        } else if (strncmp(line, "stop", 4) == 0) {
            handle_stop_command();
        } else if (strncmp(line, "quit", 4) == 0) {
            handle_quit();
            break;
        }
    }
    handle_stop_command();
    return 0;
}
