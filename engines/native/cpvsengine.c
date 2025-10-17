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
#define MATE_THRESHOLD (INF - MAX_PLY)

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
    int phase; /* Cached game phase: 256=opening, 0=endgame */
    int king_square[2];
    int material_score[2];
    int pst_score[2];
    int piece_count[2][6];
    int piece_list[2][6][16];
    int piece_list_size[2][6];
    uint64_t occupancy[2];
    uint64_t occupancy_both;
} Position;

typedef struct {
    int move;
    int captured;
    int castling;
    int ep_square;
    int halfmove_clock;
    uint64_t zobrist_key;
    int phase;
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
static uint64_t nodes_total;
static uint64_t nodes_search;
static uint64_t nodes_qsearch;
static uint64_t tt_probes;
static uint64_t tt_hits;
static uint64_t tt_cutoffs;
static uint64_t beta_cutoffs;
static uint64_t first_move_cutoffs;
static uint64_t pv_change_count;
static uint64_t q_delta_prunes;
static uint64_t q_see_prunes;
static uint64_t q_check_expansions;
static uint64_t null_move_tried;
static uint64_t null_move_pruned;
static uint64_t lmr_applied;
static uint64_t lmr_researched;
static uint64_t aspiration_fail_low_count;
static uint64_t aspiration_fail_high_count;
static uint64_t aspiration_research_count;
static uint64_t move_generation_calls;
static uint64_t see_evaluations;
static int max_sel_depth;
static int root_depth;
static int best_move_global = 0;
static int best_score_global = -INF;
static long search_time_limit_ms = 0;
static struct timespec search_start_time;

static TTEntry transposition_table[TT_SIZE];

static int pv_table[MAX_PLY][MAX_PLY];
static int pv_length[MAX_PLY];
static int history_table[2][128][128];
static int killer_moves[MAX_PLY][2];

static uint64_t zobrist_pieces[12][64];
static uint64_t zobrist_castling[16];
static uint64_t zobrist_ep[8];
static uint64_t zobrist_side;

static const int piece_values[6] = {100, 320, 330, 500, 900, 20000};
static int pst_white[6][64];
static int pst_black[6][64];

/* Phase values for game phase computation */
static const int phase_values[6] = {0, 1, 1, 2, 4, 0}; /* P, N, B, R, Q, K */
static const int initial_piece_counts[6] = {16, 4, 4, 4, 2, 2};
static int max_phase_value = 0;

/* Evaluation weights - Midgame */
static const int PASSED_PAWN_BONUS_MG[8] = {0, 10, 20, 40, 70, 120, 200, 0};
static const int DOUBLED_PAWN_PENALTY_MG = 15;
static const int ISOLATED_PAWN_PENALTY_MG = 20;
static const int BISHOP_PAIR_BONUS_MG = 50;
static const int ROOK_OPEN_FILE_BONUS_MG = 30;
static const int ROOK_SEMI_OPEN_FILE_BONUS_MG = 18;
static const int KNIGHT_OUTPOST_BONUS_MG = 35;
static const int MOBILITY_BONUS_MG = 6;
static const int KING_PAWN_SHIELD_BONUS_MG = 12;
static const int KING_ATTACK_WEIGHT_MG = 25;

/* Evaluation weights - Endgame */
static const int PASSED_PAWN_BONUS_EG[8] = {0, 20, 40, 80, 140, 240, 400, 0};
static const int DOUBLED_PAWN_PENALTY_EG = 20;
static const int ISOLATED_PAWN_PENALTY_EG = 25;
static const int BISHOP_PAIR_BONUS_EG = 60;
static const int ROOK_OPEN_FILE_BONUS_EG = 20;
static const int ROOK_SEMI_OPEN_FILE_BONUS_EG = 12;
static const int KNIGHT_OUTPOST_BONUS_EG = 15;
static const int MOBILITY_BONUS_EG = 4;
static const int KING_PAWN_SHIELD_BONUS_EG = 5;
static const int KING_ATTACK_WEIGHT_EG = 10;
static const int KING_CENTRALIZATION_BONUS_EG = 8;

static void format_move_uci(int move, char buffer[8]);
static void send_perf_summary(int depth, int sel_depth, int score_cp, int score_delta_cp, double elapsed_seconds, double budget_seconds);

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

static inline uint64_t bit_for_square(int sq) {
    return 1ULL << square_to_64(sq);
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

static int get_piece_value(int piece) {
    if (piece == 0) {
        return 0;
    }
    int type = piece_type(piece);
    if (type < 1 || type > 6) {
        return 0;
    }
    return piece_values[type - 1];
}

static bool attacks_square_from_board(const int board[128], int from, int to, int piece) {
    int type = piece_type(piece);
    if (type == 0) {
        return false;
    }
    int from_file = from & 7;
    int from_rank = from >> 4;
    int to_file = to & 7;
    int to_rank = to >> 4;
    if (type == 1) {
        int forward = piece > 0 ? 16 : -16;
        if (to == from + forward - 1 || to == from + forward + 1) {
            return true;
        }
        return false;
    }
    if (type == 2) {
        for (int i = 0; i < 8; ++i) {
            int target = from + knight_offsets[i];
            if (target == to) {
                return square_on_board(target);
            }
        }
        return false;
    }
    bool diag_aligned = abs(from_file - to_file) == abs(from_rank - to_rank);
    bool straight_aligned = (from_file == to_file) || (from_rank == to_rank);
    if (type == 3 || type == 5) {
        if (!diag_aligned) {
            if (type == 3) {
                return false;
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                int offset = bishop_offsets[i];
                int sq = from + offset;
                while (square_on_board(sq)) {
                    if (sq == to) {
                        return true;
                    }
                    if (board[sq] != 0) {
                        break;
                    }
                    sq += offset;
                }
            }
        }
        if (type == 3) {
            return false;
        }
    }
    if (type == 4 || type == 5) {
        if (!straight_aligned) {
            return (type == 5) && diag_aligned;
        }
        for (int i = 0; i < 4; ++i) {
            int offset = rook_offsets[i];
            int sq = from + offset;
            while (square_on_board(sq)) {
                if (sq == to) {
                    return true;
                }
                if (board[sq] != 0) {
                    break;
                }
                sq += offset;
            }
        }
        if (type == 4) {
            return false;
        }
    }
    if (type == 6) {
        for (int i = 0; i < 8; ++i) {
            int target = from + king_offsets[i];
            if (target == to) {
                return square_on_board(target);
            }
        }
        return false;
    }
    return false;
}

static int find_least_valuable_attacker(const int board[128], int square, Color color, int *out_value) {
    int best_square = -1;
    int best_value = 1 << 30;
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = board[sq];
        if (piece == 0 || piece_color(piece) != color) {
            continue;
        }
        if (!attacks_square_from_board(board, sq, square, piece)) {
            continue;
        }
        int value = get_piece_value(piece);
        if (value < best_value || (value == best_value && best_square != -1 && piece_type(piece) < piece_type(board[best_square]))) {
            best_value = value;
            best_square = sq;
        }
    }
    if (best_square != -1 && out_value) {
        *out_value = best_value;
    }
    return best_square;
}

static int value_to_tt(int value, int ply) {
    if (value > MATE_THRESHOLD) {
        return value + ply;
    }
    if (value < -MATE_THRESHOLD) {
        return value - ply;
    }
    return value;
}

static int value_from_tt(int value, int ply) {
    if (value > MATE_THRESHOLD) {
        return value - ply;
    }
    if (value < -MATE_THRESHOLD) {
        return value + ply;
    }
    return value;
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
    
    /* Compute max phase value */
    max_phase_value = 0;
    for (int p = 0; p < 6; ++p) {
        max_phase_value += phase_values[p] * initial_piece_counts[p];
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
    pos->zobrist_key = 0;
    pos->phase = 0;
    pos->king_square[0] = -1;
    pos->king_square[1] = -1;
}

static void add_piece(Position *pos, int piece, int square) {
    (void)pos;
    (void)piece;
    (void)square;
}

static void remove_piece(Position *pos, int piece, int square) {
    (void)pos;
    (void)piece;
    (void)square;
}

static void move_piece(Position *pos, int piece, int from, int to) {
    (void)pos;
    (void)piece;
    (void)from;
    (void)to;
}

static bool parse_fen(Position *pos, const char *fen) {
    clear_position(pos);
    int rank = 7;
    int file = 0;
    const char *ptr = fen;
    int phase_sum = 0;
    while (*ptr && rank >= 0) {
        char c = *ptr++;
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        if (c == ' ') {
            ptr--;
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
        int type = piece_type(piece) - 1;
        if (type >= 0 && type < 6) {
            phase_sum += phase_values[type];
            add_piece(pos, piece, sq);
        }
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

    /* Compute initial phase */
    if (max_phase_value > 0) {
        pos->phase = (phase_sum * 256) / max_phase_value;
    } else {
        pos->phase = 128;
    }

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
static int move_from(int move);
static int move_to(int move);
static int move_promotion(int move);
static int move_flags(int move);
static int get_piece_value(int piece);
static int find_least_valuable_attacker(const int board[128], int square, Color color, int *out_value);
static int see(const Position *pos, int move);
static int value_to_tt(int value, int ply);
static int value_from_tt(int value, int ply);
static void make_null_move(Position *pos, Undo *undo);
static void undo_null_move(Position *pos, const Undo *undo);
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

static int compute_phase(const Position *pos) {
    /* Returns phase value: 256 = opening, 0 = endgame */
    int current_phase = 0;
    
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0) continue;
        
        int type = piece_type(piece) - 1;
        if (type >= 0 && type < 6) {
            current_phase += phase_values[type];
        }
    }
    
    if (max_phase_value == 0) return 128;
    
    /* Scale to 0-256 range */
    return (current_phase * 256) / max_phase_value;
}

static int taper_score(int mg_score, int eg_score, int phase) {
    /* Interpolate between midgame and endgame scores based on phase */
    /* phase: 256 = pure midgame, 0 = pure endgame */
    return ((mg_score * phase) + (eg_score * (256 - phase))) / 256;
}

static int eval_pawn_structure(const Position *pos, Color color, int phase) {
    int score = 0;
    int c = (color == COLOR_WHITE) ? 0 : 1;
    int pawn_piece = (color == COLOR_WHITE) ? 1 : -1;
    int file_counts[8] = {0};
    int file_pawns[8][8];
    int file_sizes[8] = {0};
    int count = pos->piece_list_size[c][0];
    for (int i = 0; i < count; ++i) {
        int sq = pos->piece_list[c][0][i];
        int file = sq & 7;
        file_pawns[file][file_sizes[file]++] = sq;
        file_counts[file]++;
    }
    for (int file = 0; file < 8; ++file) {
        int pawns_on_file = file_counts[file];
        if (pawns_on_file == 0) {
            continue;
        }
        if (pawns_on_file > 1) {
            score -= taper_score(DOUBLED_PAWN_PENALTY_MG, DOUBLED_PAWN_PENALTY_EG, phase) * (pawns_on_file - 1);
        }
        bool has_neighbor = (file > 0 && file_counts[file - 1] > 0) || (file < 7 && file_counts[file + 1] > 0);
        if (!has_neighbor) {
            score -= taper_score(ISOLATED_PAWN_PENALTY_MG, ISOLATED_PAWN_PENALTY_EG, phase) * pawns_on_file;
        }
        for (int i = 0; i < file_sizes[file]; ++i) {
            int sq = file_pawns[file][i];
            int rank = sq >> 4;
            bool passed = true;
            int forward = (color == COLOR_WHITE) ? 16 : -16;
            int max_ranks = (color == COLOR_WHITE) ? (7 - rank) : rank;
            for (int r = 1; r <= max_ranks && passed; ++r) {
                int check_rank = rank + r * (forward >> 4);
                for (int df = -1; df <= 1; ++df) {
                    int check_file = file + df;
                    if (check_file < 0 || check_file > 7) {
                        continue;
                    }
                    int check_sq = (check_rank << 4) | check_file;
                    if (square_on_board(check_sq) && pos->squares[check_sq] == -pawn_piece) {
                        passed = false;
                        break;
                    }
                }
            }
            if (passed) {
                score += taper_score(PASSED_PAWN_BONUS_MG[rank], PASSED_PAWN_BONUS_EG[rank], phase);
            }
        }
    }
    return score;
}

static int eval_mobility(const Position *pos, Color color, int phase) {
    int mobility = 0;
    
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0 || piece_color(piece) != color) continue;
        
        int type = piece_type(piece);
        
        /* Skip pawns and king for mobility */
        if (type == 1 || type == 6) continue;
        
        if (type == 2) { /* Knight */
            for (int i = 0; i < 8; ++i) {
                int to = sq + knight_offsets[i];
                if (square_on_board(to)) {
                    int target = pos->squares[to];
                    if (target == 0 || piece_color(target) != color) {
                        mobility++;
                    }
                }
            }
        } else if (type == 3) { /* Bishop */
            for (int i = 0; i < 4; ++i) {
                int offset = bishop_offsets[i];
                int to = sq + offset;
                while (square_on_board(to)) {
                    int target = pos->squares[to];
                    if (target == 0) {
                        mobility++;
                        to += offset;
                    } else {
                        if (piece_color(target) != color) mobility++;
                        break;
                    }
                }
            }
        } else if (type == 4) { /* Rook */
            for (int i = 0; i < 4; ++i) {
                int offset = rook_offsets[i];
                int to = sq + offset;
                while (square_on_board(to)) {
                    int target = pos->squares[to];
                    if (target == 0) {
                        mobility++;
                        to += offset;
                    } else {
                        if (piece_color(target) != color) mobility++;
                        break;
                    }
                }
            }
        } else if (type == 5) { /* Queen */
            /* Simplified queen mobility - just count immediate squares */
            for (int i = 0; i < 8; ++i) {
                int to = sq + king_offsets[i];
                if (square_on_board(to)) {
                    int target = pos->squares[to];
                    if (target == 0 || piece_color(target) != color) {
                        mobility += 2; /* Weight queen mobility higher */
                    }
                }
            }
        }
    }
    
    return mobility * taper_score(MOBILITY_BONUS_MG, MOBILITY_BONUS_EG, phase) / 10;
}

static int eval_king_safety(const Position *pos, Color color, int phase) {
    int score = 0;
    int king_piece = (color == COLOR_WHITE) ? 6 : -6;
    int king_sq = -1;
    
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        if (pos->squares[sq] == king_piece) {
            king_sq = sq;
            break;
        }
    }
    
    if (king_sq == -1) return 0;
    
    int pawn_piece = (color == COLOR_WHITE) ? 1 : -1;
    int forward = (color == COLOR_WHITE) ? 16 : -16;
    
    /* Pawn shield */
    int shield_count = 0;
    for (int df = -1; df <= 1; ++df) {
        int file = (king_sq & 7) + df;
        if (file < 0 || file > 7) continue;
        
        int shield_sq = king_sq + forward + df;
        if (square_on_board(shield_sq) && pos->squares[shield_sq] == pawn_piece) {
            shield_count++;
        }
        
        int shield_sq2 = king_sq + 2 * forward + df;
        if (square_on_board(shield_sq2) && pos->squares[shield_sq2] == pawn_piece && pos->squares[shield_sq] != pawn_piece) {
            shield_count++;
        }
    }
    score += shield_count * taper_score(KING_PAWN_SHIELD_BONUS_MG, KING_PAWN_SHIELD_BONUS_EG, phase) / 10;
    
    /* King attackers penalty */
    Color opponent = (color == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
    int attackers = 0;
    for (int i = 0; i < 8; ++i) {
        int to = king_sq + king_offsets[i];
        if (square_on_board(to) && square_attacked(pos, to, opponent)) {
            attackers++;
        }
    }
    score -= attackers * taper_score(KING_ATTACK_WEIGHT_MG, KING_ATTACK_WEIGHT_EG, phase) / 10;
    
    /* King centralization in endgame */
    if (phase < 128) {
        int file = king_sq & 7;
        int rank = king_sq >> 4;
        int central_distance = (abs(file - 3) + abs(file - 4)) / 2 + (abs(rank - 3) + abs(rank - 4)) / 2;
        score += (256 - phase) * (7 - central_distance) * KING_CENTRALIZATION_BONUS_EG / 256 / 10;
    }
    
    return score;
}

static int eval_piece_bonuses(const Position *pos, Color color, int phase) {
    int score = 0;
    int bishop_count = 0;
    
    for (int sq = 0; sq < 128; ++sq) {
        if (!square_on_board(sq)) {
            sq += 7;
            continue;
        }
        int piece = pos->squares[sq];
        if (piece == 0 || piece_color(piece) != color) continue;
        
        int type = piece_type(piece);
        
        /* Bishop pair */
        if (type == 3) {
            bishop_count++;
        }
        
        /* Rook on open/semi-open file */
        if (type == 4) {
            int file = sq & 7;
            bool has_our_pawn = false;
            bool has_their_pawn = false;
            int our_pawn = (color == COLOR_WHITE) ? 1 : -1;
            
            for (int r = 0; r < 8; ++r) {
                int check_sq = (r << 4) | file;
                if (square_on_board(check_sq)) {
                    int p = pos->squares[check_sq];
                    if (piece_type(p) == 1) {
                        if (p == our_pawn) has_our_pawn = true;
                        if (p == -our_pawn) has_their_pawn = true;
                    }
                }
            }
            
            if (!has_our_pawn && !has_their_pawn) {
                score += taper_score(ROOK_OPEN_FILE_BONUS_MG, ROOK_OPEN_FILE_BONUS_EG, phase);
            } else if (!has_our_pawn) {
                score += taper_score(ROOK_SEMI_OPEN_FILE_BONUS_MG, ROOK_SEMI_OPEN_FILE_BONUS_EG, phase);
            }
        }
        
        /* Knight outpost */
        if (type == 2) {
            int rank = sq >> 4;
            int file = sq & 7;
            bool is_outpost = false;
            
            if ((color == COLOR_WHITE && rank >= 4 && rank <= 6) || 
                (color == COLOR_BLACK && rank >= 1 && rank <= 3)) {
                int pawn_piece = (color == COLOR_WHITE) ? 1 : -1;
                int support_rank = (color == COLOR_WHITE) ? rank - 1 : rank + 1;
                
                for (int df = -1; df <= 1; df += 2) {
                    if (file + df >= 0 && file + df < 8) {
                        int support_sq = (support_rank << 4) | (file + df);
                        if (square_on_board(support_sq) && pos->squares[support_sq] == pawn_piece) {
                            is_outpost = true;
                            break;
                        }
                    }
                }
            }
            
            if (is_outpost) {
                score += taper_score(KNIGHT_OUTPOST_BONUS_MG, KNIGHT_OUTPOST_BONUS_EG, phase);
            }
        }
    }
    
    if (bishop_count >= 2) {
        score += taper_score(BISHOP_PAIR_BONUS_MG, BISHOP_PAIR_BONUS_EG, phase);
    }
    
    return score;
}

static int evaluate_simple(const Position *pos) {
    int score = pos->material_score[0] - pos->material_score[1];
    score += pos->pst_score[0] - pos->pst_score[1];
    return (pos->side_to_move == COLOR_WHITE) ? score : -score;
}

static int evaluate(const Position *pos) {
    int phase = pos->phase;
    int score = pos->material_score[0] - pos->material_score[1];
    score += pos->pst_score[0] - pos->pst_score[1];
    score += eval_pawn_structure(pos, COLOR_WHITE, phase);
    score -= eval_pawn_structure(pos, COLOR_BLACK, phase);
    if (phase > 32) {
        score += eval_mobility(pos, COLOR_WHITE, phase);
        score -= eval_mobility(pos, COLOR_BLACK, phase);
    }
    if (phase > 64) {
        score += eval_king_safety(pos, COLOR_WHITE, phase);
        score -= eval_king_safety(pos, COLOR_BLACK, phase);
    }
    score += eval_piece_bonuses(pos, COLOR_WHITE, phase);
    score -= eval_piece_bonuses(pos, COLOR_BLACK, phase);
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

static int see(const Position *pos, int move) {
    see_evaluations++;
    int from = move_from(move);
    int to = move_to(move);
    int promotion = move_promotion(move);
    int flags = move_flags(move);
    int board[128];
    memcpy(board, pos->squares, sizeof(board));
    int mover = board[from];
    if (mover == 0) {
        return 0;
    }
    int captured_value = 0;
    if (flags & FLAG_EN_PASSANT) {
        int ep_sq = (pos->side_to_move == COLOR_WHITE) ? (to - 16) : (to + 16);
        captured_value = get_piece_value(board[ep_sq]);
        board[ep_sq] = 0;
    } else {
        captured_value = get_piece_value(board[to]);
    }
    if (!(flags & FLAG_CAPTURE) && !(flags & FLAG_EN_PASSANT) && !promotion) {
        return 0;
    }
    board[from] = 0;
    if (promotion) {
        captured_value += piece_values[promotion - 1] - piece_values[0];
        mover = (pos->side_to_move == COLOR_WHITE) ? promotion : -promotion;
    }
    board[to] = mover;
    int gain[MAX_PLY];
    int depth = 0;
    gain[0] = captured_value;
    Color stm = (pos->side_to_move == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
    while (true) {
        int attacker_value = 0;
        int attacker_sq = find_least_valuable_attacker(board, to, stm, &attacker_value);
        if (attacker_sq == -1) {
            break;
        }
        int attacker_piece = board[attacker_sq];
        board[attacker_sq] = 0;
        depth++;
        if (depth >= MAX_PLY) {
            depth--;
            board[attacker_sq] = attacker_piece;
            break;
        }
        gain[depth] = attacker_value - gain[depth - 1];
        board[to] = attacker_piece;
        stm = (stm == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
        if (piece_type(attacker_piece) == 6) {
            break;
        }
    }
    while (depth > 0) {
        int alt = gain[depth];
        int prev = gain[depth - 1];
        if (-prev > alt) {
            alt = -prev;
        }
        gain[depth - 1] = -alt;
        depth--;
    }
    return gain[0];
}

static void add_move(const Position *pos, MoveList *list, int from, int to, int promotion, int flags) {
    if (list->count >= MAX_MOVES) {
        return;
    }
    int move = encode_move(from, to, promotion, flags);
    list->moves[list->count] = move;
    int captured = pos->squares[to];
    int score = 0;
    if (flags & FLAG_EN_PASSANT) {
        captured = (pos->side_to_move == COLOR_WHITE) ? -1 : 1;
    }
    if ((flags & FLAG_CAPTURE) || (flags & FLAG_EN_PASSANT)) {
        int victim = piece_type(captured) - 1;
        int attacker = piece_type(pos->squares[from]) - 1;
        int mvv_lva = 0;
        if (victim >= 0 && attacker >= 0) {
            mvv_lva = 10 * piece_values[victim] - piece_values[attacker];
        }
        int see_score = see(pos, move);
        score = 200000 + mvv_lva + see_score;
    } else if (promotion) {
        score = 150000 + piece_values[promotion - 1];
    }
    list->scores[list->count] = score;
    list->count++;
}

static void generate_moves(const Position *pos, MoveList *list, bool captures_only) {
    move_generation_calls++;
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
    undo->phase = pos->phase;

    pos->halfmove_clock++;

    if (piece_type(piece) == 1 || captured != 0) {
        pos->halfmove_clock = 0;
    }

    update_zobrist_piece(pos, piece, from);
    if (captured != 0) {
        update_zobrist_piece(pos, captured, to);
        /* Update phase for captured piece */
        int cap_type = piece_type(captured) - 1;
        if (cap_type >= 0 && cap_type < 6 && max_phase_value > 0) {
            int phase_delta = (phase_values[cap_type] * 256) / max_phase_value;
            pos->phase -= phase_delta;
            if (pos->phase < 0) pos->phase = 0;
        }
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
        /* Update phase for en passant capture (pawn) */
        if (max_phase_value > 0) {
            int pawn_phase = (phase_values[0] * 256) / max_phase_value;
            pos->phase -= pawn_phase;
            if (pos->phase < 0) pos->phase = 0;
        }
    }

    if (promotion) {
        update_zobrist_piece(pos, pos->squares[to], to);
        int promoted_piece = (pos->side_to_move == COLOR_WHITE) ? promotion : -promotion;
        pos->squares[to] = promoted_piece;
        update_zobrist_piece(pos, pos->squares[to], to);
        /* Update phase for promotion (remove pawn, add promoted piece) */
        if (max_phase_value > 0) {
            int pawn_phase = (phase_values[0] * 256) / max_phase_value;
            int promo_phase = (phase_values[promotion - 1] * 256) / max_phase_value;
            pos->phase = pos->phase - pawn_phase + promo_phase;
            if (pos->phase > 256) pos->phase = 256;
        }
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
    pos->phase = undo->phase;
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

static void make_null_move(Position *pos, Undo *undo) {
    undo->move = 0;
    undo->captured = 0;
    undo->castling = pos->castling;
    undo->ep_square = pos->ep_square;
    undo->halfmove_clock = pos->halfmove_clock;
    undo->zobrist_key = pos->zobrist_key;
    if (pos->ep_square != -1) {
        update_zobrist_ep(pos, pos->ep_square);
    }
    pos->ep_square = -1;
    pos->halfmove_clock++;
    toggle_side(pos);
    if (pos->side_to_move == COLOR_WHITE) {
        pos->fullmove_number++;
    }
}

static void undo_null_move(Position *pos, const Undo *undo) {
    if (pos->side_to_move == COLOR_WHITE) {
        pos->fullmove_number--;
    }
    toggle_side(pos);
    pos->castling = undo->castling;
    pos->ep_square = undo->ep_square;
    pos->halfmove_clock = undo->halfmove_clock;
    pos->zobrist_key = undo->zobrist_key;
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
    if (time_exceeded()) {
        return alpha;
    }
    if (ply > max_sel_depth) {
        max_sel_depth = ply;
    }
    /* Use fast evaluation in quiescence (material + PST only) */
    int stand_pat = evaluate_simple(pos);
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
        int flags = move_flags(move);
        if (!(flags & FLAG_CAPTURE) && !(flags & FLAG_EN_PASSANT)) {
            continue;
        }
        int target = move_to(move);
        int capture_value = 0;
        if (flags & FLAG_EN_PASSANT) {
            capture_value = piece_values[0];
        } else {
            capture_value = get_piece_value(pos->squares[target]);
        }
        int promotion = move_promotion(move);
        if (promotion) {
            capture_value += piece_values[promotion - 1] - piece_values[0];
        }
        const int delta_margin = 80;
        if (stand_pat + capture_value + delta_margin < alpha) {
            q_delta_prunes++;
            continue;
        }
        if (see(pos, move) < 0) {
            q_see_prunes++;
            continue;
        }
        Undo undo;
        if (!make_move(pos, move, &undo)) {
            continue;
        }
        nodes_total++;
        nodes_qsearch++;
        int score = -quiescence(pos, -beta, -alpha, ply + 1);
        undo_move(pos, move, &undo);
        if (score >= beta) {
            return beta;
        }
        if (score > alpha) {
            alpha = score;
        }
    }
    MoveList quiet_list;
    generate_moves(pos, &quiet_list, false);
    for (int i = 0; i < quiet_list.count; ++i) {
        int move = quiet_list.moves[i];
        int flags = move_flags(move);
        if ((flags & FLAG_CAPTURE) || (flags & FLAG_EN_PASSANT) || (flags & FLAG_CASTLING)) {
            continue;
        }
        Undo undo;
        if (!make_move(pos, move, &undo)) {
            continue;
        }
        int king_sq = find_king(pos, pos->side_to_move);
        Color attacker = (pos->side_to_move == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
        bool gives_check = (king_sq != -1) && square_attacked(pos, king_sq, attacker);
        if (!gives_check) {
            undo_move(pos, move, &undo);
            continue;
        }
        q_check_expansions++;
        nodes_total++;
        nodes_qsearch++;
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

static int probe_tt(uint64_t key, int depth, int alpha, int beta, int ply, int *tt_move) {
    TTEntry *entry = &transposition_table[key & (TT_SIZE - 1)];
    tt_probes++;
    bool key_match = (entry->key == key);
    if (tt_move) {
        *tt_move = key_match ? entry->move : 0;
    }
    if (key_match && entry->depth >= depth) {
        tt_hits++;
        int stored = value_from_tt(entry->value, ply);
        if (entry->flag == 0) {
            tt_cutoffs++;
            return stored;
        }
        if (entry->flag == -1 && stored <= alpha) {
            tt_cutoffs++;
            return stored;
        }
        if (entry->flag == 1 && stored >= beta) {
            tt_cutoffs++;
            return stored;
        }
    }
    return INF + 1;
}

static void store_tt(uint64_t key, int depth, int ply, int value, int flag, int move) {
    TTEntry *entry = &transposition_table[key & (TT_SIZE - 1)];
    entry->key = key;
    entry->value = value_to_tt(value, ply);
    entry->depth = depth;
    entry->flag = flag;
    entry->move = move;
}

static int search(Position *pos, int depth, int alpha, int beta, int ply) {
    int original_alpha = alpha;
    if (time_exceeded()) {
        return alpha;
    }
    if (ply > max_sel_depth) {
        max_sel_depth = ply;
    }

    Color us = pos->side_to_move;
    Color them = (us == COLOR_WHITE) ? COLOR_BLACK : COLOR_WHITE;
    int king_sq = find_king(pos, us);
    bool in_check = (king_sq != -1) && square_attacked(pos, king_sq, them);

    if (depth == 0) {
        return quiescence(pos, alpha, beta, ply);
    }

    bool pv_node = (beta - alpha) > 1;

    int tt_move = 0;
    int tt_value = probe_tt(pos->zobrist_key, depth, alpha, beta, ply, &tt_move);
    if (tt_value != INF + 1) {
        return tt_value;
    }

    pv_length[ply] = 0;

    if (!pv_node && !in_check && depth >= 3 && pos->halfmove_clock > 0) {
        Undo undo_null;
        make_null_move(pos, &undo_null);
        null_move_tried++;
        int reduction = 2 + depth / 4;
        if (reduction > depth - 1) {
            reduction = depth - 1;
        }
        int null_score = -search(pos, depth - 1 - reduction, -beta, -beta + 1, ply + 1);
        undo_null_move(pos, &undo_null);
        if (null_score >= beta) {
            null_move_pruned++;
            return beta;
        }
    }

    MoveList list;
    generate_moves(pos, &list, false);
    if (list.count == 0) {
        if (in_check) {
            return -INF + ply;
        }
        return 0;
    }

    for (int i = 0; i < list.count; ++i) {
        int move = list.moves[i];
        if (move == tt_move) {
            list.scores[i] = 2000000000;
            continue;
        }
        int flags = move_flags(move);
        if (flags & (FLAG_CAPTURE | FLAG_EN_PASSANT)) {
            list.scores[i] += 1000000;
            continue;
        }
        if (killer_moves[ply][0] == move) {
            list.scores[i] = 900000;
            continue;
        }
        if (killer_moves[ply][1] == move) {
            list.scores[i] = 800000;
            continue;
        }
        list.scores[i] = history_table[us][move_from(move)][move_to(move)];
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
        nodes_total++;
        nodes_search++;

        int flags = move_flags(move);
        bool is_capture = (flags & (FLAG_CAPTURE | FLAG_EN_PASSANT)) != 0;
        bool is_promotion = move_promotion(move) != 0;

        int search_depth = depth - 1;
        int score;
        if (first_move) {
            score = -search(pos, search_depth, -beta, -alpha, ply + 1);
            first_move = false;
        } else {
            int reduction = 0;
            if (!pv_node && !in_check && search_depth > 0 && !is_capture && !is_promotion) {
                reduction = 1 + (depth >= 5 && i >= 5);
                if (reduction > search_depth) {
                    reduction = search_depth;
                }
            }
            if (reduction > 0) {
                lmr_applied++;
                score = -search(pos, search_depth - reduction, -alpha - 1, -alpha, ply + 1);
                if (score > alpha) {
                    lmr_researched++;
                    score = -search(pos, search_depth, -alpha - 1, -alpha, ply + 1);
                }
            } else {
                score = -search(pos, search_depth, -alpha - 1, -alpha, ply + 1);
            }
            if (score > alpha && score < beta) {
                score = -search(pos, search_depth, -beta, -alpha, ply + 1);
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
            beta_cutoffs++;
            if (i == 0) {
                first_move_cutoffs++;
            }
            if (!is_capture && !is_promotion) {
                int from_sq = move_from(move);
                int to_sq = move_to(move);
                int delta = depth * depth;
                history_table[us][from_sq][to_sq] += delta;
                if (history_table[us][from_sq][to_sq] > 1000000) {
                    history_table[us][from_sq][to_sq] = 1000000;
                }
                if (killer_moves[ply][0] != move) {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = move;
                }
            }
            break;
        }
    }

    if (best_move == 0) {
        return in_check ? (-INF + ply) : 0;
    }

    int flag = 0;
    if (best_value <= original_alpha) {
        flag = -1;
    } else if (best_value >= beta) {
        flag = 1;
    }
    store_tt(pos->zobrist_key, depth, ply, best_value, flag, best_move);

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

static void send_perf_summary(
    int depth,
    int sel_depth,
    int score_cp,
    int score_delta_cp,
    double elapsed_seconds,
    double budget_seconds
) {
    uint64_t total_nodes = nodes_total;
    uint64_t regular_nodes = nodes_search;
    uint64_t quiescence_nodes = nodes_qsearch;
    double denom = (elapsed_seconds > 1e-6) ? elapsed_seconds : 1e-6;
    unsigned long long nps = (unsigned long long)(total_nodes / denom);
    double q_ratio = (total_nodes > 0) ? (double)quiescence_nodes / (double)total_nodes : 0.0;
    double tt_hit_rate = (tt_probes > 0) ? (double)tt_hits / (double)tt_probes : 0.0;
    uint64_t total_cuts = beta_cutoffs + tt_cutoffs;
    double tt_cut_share = (total_cuts > 0) ? (double)tt_cutoffs / (double)total_cuts : 0.0;
    double first_move_rate = (total_cuts > 0) ? (double)first_move_cutoffs / (double)total_cuts : 0.0;
    uint64_t q_prunes = q_delta_prunes + q_see_prunes;
    double q_prune_ratio = (quiescence_nodes > 0) ? (double)q_prunes / (double)quiescence_nodes : 0.0;
    double check_ratio = (quiescence_nodes > 0) ? (double)q_check_expansions / (double)quiescence_nodes : 0.0;

    char pv_json[2048];
    int pv_offset = snprintf(pv_json, sizeof(pv_json), "[");
    for (int i = 0; i < pv_length[0] && pv_offset >= 0 && pv_offset < (int)sizeof(pv_json); ++i) {
        char move_buf[8];
        format_move_uci(pv_table[0][i], move_buf);
        pv_offset += snprintf(
            pv_json + pv_offset,
            sizeof(pv_json) - (size_t)pv_offset,
            "\"%s\"%s",
            move_buf,
            (i + 1 < pv_length[0]) ? "," : ""
        );
    }
    if (pv_offset < 0 || pv_offset >= (int)sizeof(pv_json)) {
        pv_json[sizeof(pv_json) - 2] = ']';
        pv_json[sizeof(pv_json) - 1] = '\0';
    } else {
        snprintf(pv_json + pv_offset, sizeof(pv_json) - (size_t)pv_offset, "]");
    }

    char pv_display[256];
    pv_display[0] = '\0';
    for (int i = 0; i < pv_length[0] && i < 5; ++i) {
        char move_buf[8];
        format_move_uci(pv_table[0][i], move_buf);
        size_t used = strlen(pv_display);
        size_t remaining = sizeof(pv_display) - used;
        if (remaining <= 1) {
            break;
        }
        if (used > 0) {
            strncat(pv_display, " ", remaining - 1);
            used++;
            remaining--;
        }
        strncat(pv_display, move_buf, remaining - 1);
    }
    if (pv_length[0] > 5) {
        strncat(pv_display, "...", sizeof(pv_display) - strlen(pv_display) - 1);
    }
    if (pv_display[0] == '\0') {
        strncpy(pv_display, "(empty)", sizeof(pv_display) - 1);
        pv_display[sizeof(pv_display) - 1] = '\0';
    }

    double score_value = (double)score_cp;
    double score_delta = (double)score_delta_cp;
    int q_pct = (int)(q_ratio * 100.0);
    int tt_hit_pct = (int)(tt_hit_rate * 100.0);
    int tt_cut_pct = (int)(tt_cut_share * 100.0);
    int first_cut_pct = (int)(first_move_rate * 100.0);

    char nodes_str[64];
    snprintf(
        nodes_str,
        sizeof(nodes_str),
        "%llu(r:%llu,q:%llu)",
        (unsigned long long)total_nodes,
        (unsigned long long)regular_nodes,
        (unsigned long long)quiescence_nodes
    );

    char time_summary[64];
    if (budget_seconds >= 0.0) {
        snprintf(time_summary, sizeof(time_summary), "%.2f/%.2fs", elapsed_seconds, budget_seconds);
    } else {
        snprintf(time_summary, sizeof(time_summary), "%.2fs", elapsed_seconds);
    }

    char budget_json[32];
    if (budget_seconds >= 0.0) {
        snprintf(budget_json, sizeof(budget_json), "%.3f", budget_seconds);
    } else {
        strncpy(budget_json, "null", sizeof(budget_json) - 1);
        budget_json[sizeof(budget_json) - 1] = '\0';
    }

    char payload[4096];
    snprintf(
        payload,
        sizeof(payload),
        "{\"strategy\":\"cpvs\","
        "\"depth\":%d,"
        "\"seldepth\":%d,"
        "\"nodes\":{\"total\":%llu,\"regular\":%llu,\"quiescence\":%llu},"
        "\"nps\":%llu,"
        "\"time\":{\"elapsed\":%.3f,\"budget\":%s},"
        "\"score\":{\"value\":%.3f,\"delta\":%.3f},"
        "\"pv\":%s,"
        "\"pv_changes\":%llu,"
        "\"tt\":{\"probes\":%llu,\"hits\":%llu,\"hit_rate\":%.3f,\"cuts\":%llu,\"cut_share\":%.3f},"
        "\"aspiration\":{\"fail_low\":0,\"fail_high\":0,\"researches\":0},"
        "\"cuts\":{\"tt\":%llu,\"killer\":0,\"history\":0,\"capture\":0,\"null\":0,\"futility\":0,\"other\":0,\"total\":%llu,\"first_move_rate\":%.3f},"
        "\"quiescence\":{\"ratio\":%.3f,\"cutoffs\":0,\"stand_pat_cuts\":0,\"delta_prunes\":0,\"see_prunes\":0,\"prune_ratio\":0.0},"
        "\"reductions\":{\"lmr\":{\"applied\":0,\"researched\":0,\"success_rate\":0.0},\"null\":{\"tried\":0,\"success\":0,\"success_rate\":0.0}}"
        "}",
        depth,
        sel_depth,
        (unsigned long long)total_nodes,
        (unsigned long long)regular_nodes,
        (unsigned long long)quiescence_nodes,
        nps,
        elapsed_seconds,
        budget_json,
        score_value,
        score_delta,
        pv_json,
        (unsigned long long)pv_change_count,
        (unsigned long long)tt_probes,
        (unsigned long long)tt_hits,
        tt_hit_rate,
        (unsigned long long)tt_cutoffs,
        tt_cut_share,
        (unsigned long long)tt_cutoffs,
        (unsigned long long)total_cuts,
        first_move_rate,
        q_ratio
    );

    printf("info string perf payload=%s\n", payload);
    printf(
        "info string perf summary core depth=%d sel=%d nodes=%s nps=%llu time=%s\n",
        depth,
        sel_depth,
        nodes_str,
        nps,
        time_summary
    );
    printf(
        "info string perf summary eval score=%.1f(Δ%+.1f) pv=%s swaps=%llu\n",
        score_value,
        score_delta,
        pv_display,
        (unsigned long long)pv_change_count
    );
    printf(
        "info string perf summary pruning tt probes=%llu hits=%llu hit_rate=%d%% cut_share=%d%% cuts=%llu\n",
        (unsigned long long)tt_probes,
        (unsigned long long)tt_hits,
        tt_hit_pct,
        tt_cut_pct,
        (unsigned long long)tt_cutoffs
    );
    printf(
        "info string perf summary pruning aspiration fail_low=0 fail_high=0 researches=0\n"
    );
    printf(
        "info string perf summary pruning cuts total=%llu first_move_rate=%d%% breakdown=tt:%llu(%d%%) k:0 h:0 c:0 n:0\n",
        (unsigned long long)total_cuts,
        first_cut_pct,
        (unsigned long long)tt_cutoffs,
        tt_cut_pct
    );
    printf("info string perf summary heuristics q=%d%%\n", q_pct);
    printf(
        "info string perf summary qsearch prunes delta=%llu see=%llu pruned=%.1f%% checks=%llu (%.1f%%)\n",
        (unsigned long long)q_delta_prunes,
        (unsigned long long)q_see_prunes,
        q_prune_ratio * 100.0,
        (unsigned long long)q_check_expansions,
        check_ratio * 100.0
    );
    printf(
        "info string perf summary reductions lmr=%llu/%llu null=%llu/%llu\n",
        (unsigned long long)lmr_applied,
        (unsigned long long)lmr_researched,
        (unsigned long long)null_move_tried,
        (unsigned long long)null_move_pruned
    );
    printf(
        "info string perf summary aspiration fail_low=%llu fail_high=%llu researches=%llu\n",
        (unsigned long long)aspiration_fail_low_count,
        (unsigned long long)aspiration_fail_high_count,
        (unsigned long long)aspiration_research_count
    );
    printf(
        "info string perf summary instrumentation moves=%llu see=%llu\n",
        (unsigned long long)move_generation_calls,
        (unsigned long long)see_evaluations
    );
    fflush(stdout);
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
    nodes_total = 0;
    nodes_search = 0;
    nodes_qsearch = 0;
    tt_probes = 0;
    tt_hits = 0;
    tt_cutoffs = 0;
    beta_cutoffs = 0;
    first_move_cutoffs = 0;
    pv_change_count = 0;
    q_delta_prunes = 0;
    q_see_prunes = 0;
    q_check_expansions = 0;
    null_move_tried = 0;
    null_move_pruned = 0;
    lmr_applied = 0;
    lmr_researched = 0;
    aspiration_fail_low_count = 0;
    aspiration_fail_high_count = 0;
    aspiration_research_count = 0;
    move_generation_calls = 0;
    see_evaluations = 0;
    max_sel_depth = 0;
    best_move_global = 0;
    best_score_global = -INF;
    clock_gettime(CLOCK_MONOTONIC, &search_start_time);
    memset(history_table, 0, sizeof(history_table));
    memset(killer_moves, 0, sizeof(killer_moves));
    pv_length[0] = 0;
    int completed_depth = 0;
    int score_history[64];
    int score_history_count = 0;
    int previous_root_move = 0;
    for (int depth = 1; depth <= root_depth; ++depth) {
        pv_length[depth] = 0;
        int alpha_window = -INF;
        int beta_window = INF;
        if (depth > 1 && score_history_count > 0) {
            int prev_score = score_history[score_history_count - 1];
            int window = 50 + depth * 5;
            if (window > INF) {
                window = INF;
            }
            alpha_window = prev_score - window;
            beta_window = prev_score + window;
            if (alpha_window < -INF) {
                alpha_window = -INF;
            }
            if (beta_window > INF) {
                beta_window = INF;
            }
        }
        int score = search(&pos, depth, alpha_window, beta_window, 0);
        if (!time_exceeded() && ((alpha_window != -INF && score <= alpha_window) || (beta_window != INF && score >= beta_window))) {
            if (alpha_window != -INF && score <= alpha_window) {
                aspiration_fail_low_count++;
            }
            if (beta_window != INF && score >= beta_window) {
                aspiration_fail_high_count++;
            }
            aspiration_research_count++;
            score = search(&pos, depth, -INF, INF, 0);
        }
        if (time_exceeded()) {
            break;
        }
        best_move_global = pv_table[0][0];
        best_score_global = score;
        int time_ms = (int)elapsed_ms();
        send_info(depth, score, time_ms, nodes_total);
        if (search_time_limit_ms > 0 && elapsed_ms() > search_time_limit_ms) {
            break;
        }
        if (previous_root_move != 0 && best_move_global != previous_root_move) {
            pv_change_count++;
        }
        previous_root_move = best_move_global;
        if (score_history_count < 64) {
            score_history[score_history_count++] = score;
        }
        completed_depth = depth;
    }
    if (best_move_global == 0) {
        MoveList list;
        generate_moves(&pos, &list, false);
        if (list.count > 0) {
            best_move_global = list.moves[0];
            best_score_global = 0;
        }
    }
    if (best_score_global <= -INF) {
        best_score_global = 0;
    }
    int final_time_ms = (int)elapsed_ms();
    double elapsed_seconds = final_time_ms / 1000.0;
    int score_delta = 0;
    if (score_history_count >= 2) {
        score_delta = score_history[score_history_count - 1] - score_history[score_history_count - 2];
    }
    double budget_seconds = (search_time_limit_ms > 0) ? (search_time_limit_ms / 1000.0) : -1.0;
    send_perf_summary(
        completed_depth,
        max_sel_depth,
        best_score_global,
        score_delta,
        elapsed_seconds,
        budget_seconds
    );
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
    while (*ptr == ' ') {
        ptr++;
    }
    if (strncmp(ptr, "startpos", 8) == 0) {
        if (!parse_fen_wrapper(&pos, "startpos")) {
            printf("info string Failed to set position to startpos\n");
            fflush(stdout);
            return;
        }
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
        if (!parse_fen_wrapper(&pos, fen)) {
            printf("info string Invalid FEN provided to position command: %s\n", fen);
            fflush(stdout);
            return;
        }
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
    long movetime = 0;
    long wtime = 0;
    long btime = 0;
    long winc = 0;
    long binc = 0;
    int movestogo = 0;
    bool infinite = false;
    while (*ptr) {
        while (*ptr == ' ') ptr++;
        if (strncmp(ptr, "depth", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            root_depth = atoi(ptr);
        } else if (strncmp(ptr, "movetime", 8) == 0) {
            ptr += 8;
            while (*ptr == ' ') ptr++;
            movetime = atol(ptr);
        } else if (strncmp(ptr, "wtime", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            wtime = atol(ptr);
        } else if (strncmp(ptr, "btime", 5) == 0) {
            ptr += 5;
            while (*ptr == ' ') ptr++;
            btime = atol(ptr);
        } else if (strncmp(ptr, "winc", 4) == 0) {
            ptr += 4;
            while (*ptr == ' ') ptr++;
            winc = atol(ptr);
        } else if (strncmp(ptr, "binc", 4) == 0) {
            ptr += 4;
            while (*ptr == ' ') ptr++;
            binc = atol(ptr);
        } else if (strncmp(ptr, "movestogo", 9) == 0) {
            ptr += 9;
            while (*ptr == ' ') ptr++;
            movestogo = atoi(ptr);
        } else if (strncmp(ptr, "infinite", 8) == 0) {
            ptr += 8;
            infinite = true;
        }
        while (*ptr && *ptr != ' ') ptr++;
    }
    if (root_depth <= 0) {
        root_depth = 1;
    }
    if (!infinite) {
        if (movetime > 0) {
            search_time_limit_ms = movetime;
        } else {
            Color side = current_position.side_to_move;
            long remaining = (side == COLOR_WHITE) ? wtime : btime;
            long increment = (side == COLOR_WHITE) ? winc : binc;
            if (remaining > 0) {
                int moves = movestogo > 0 ? movestogo : 30;
                long budget = remaining / (moves + 1);
                if (budget < 10) {
                    budget = remaining / 20;
                }
                if (budget < 10) {
                    budget = 10;
                }
                long increment_bonus = increment > 0 ? increment / 2 : 0;
                budget += increment_bonus;
                long max_budget = remaining - remaining / 10;
                if (budget > max_budget) {
                    budget = max_budget;
                }
                if (budget < 10) {
                    budget = 10;
                }
                search_time_limit_ms = budget;
            }
        }
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
