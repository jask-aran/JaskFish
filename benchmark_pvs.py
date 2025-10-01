#!/usr/bin/env python3
"""Benchmark script for PVS engine performance testing."""

import time
import chess
from typing import List, Dict, Any
from pvsengine import (
    PVSearchBackend,
    SearchTuning,
    MetaParams,
    SearchLimits,
    SearchReporter,
    StrategyContext,
    build_search_tuning,
)


# Test positions covering different game phases
BENCHMARK_POSITIONS = [
    ("startpos", chess.STARTING_FEN, "Opening - Start position"),
    ("midgame1", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10", "Middlegame - Complex"),
    ("midgame2", "r1b2rk1/ppq1bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1QBPPP/R1B2RK1 b - - 0 10", "Middlegame - Tactical"),
    ("endgame1", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "Endgame - K+P vs K"),
    ("endgame2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Endgame - Complex pawn"),
]


def create_pvs_engine(depth: int = 7) -> tuple[PVSearchBackend, SearchTuning, MetaParams]:
    """Create and configure a PVS engine for benchmarking."""
    from dataclasses import replace
    
    meta = MetaParams()  # balanced preset
    tuning, limits = build_search_tuning(meta)
    
    # Override the search depth for benchmarking
    tuning = replace(tuning, search_depth=depth)
    
    engine = PVSearchBackend(max_threads=1)
    engine.configure(tuning, meta)
    
    return engine, tuning, meta


def benchmark_position(engine: PVSearchBackend, tuning: SearchTuning, meta: MetaParams, fen: str) -> Dict[str, Any]:
    """Benchmark a single position at given depth."""
    board = chess.Board(fen)
    
    context = StrategyContext(
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=fen,
        legal_moves_count=board.legal_moves.count(),
        repetition_info={},
        time_controls=None,
        in_check=board.is_check(),
        opponent_mate_in_one_threat=False,
    )
    
    limits = SearchLimits(
        min_time=tuning.min_time_limit,
        max_time=tuning.max_time_limit,
        base_time=tuning.base_time_limit,
        time_factor=tuning.time_allocation_factor,
    )
    reporter = SearchReporter(logger=lambda msg: None)  # Silent reporter for benchmarking
    
    start = time.perf_counter()
    result = engine.search(board, context, limits, reporter, budget_seconds=None)
    elapsed = time.perf_counter() - start
    
    nps = result.nodes / elapsed if elapsed > 0 else 0
    
    return {
        'fen': fen,
        'depth': tuning.search_depth,
        'nodes': result.nodes,
        'time': elapsed,
        'nps': int(nps),
        'score': result.score,
        'pv': [m.uci() for m in result.principal_variation[:5]],
        'completed_depth': result.completed_depth,
    }


def run_benchmark_suite(depths: List[int] = [4, 5]) -> None:
    """Run full benchmark suite."""
    print("=" * 80)
    print("PVS Engine Performance Benchmark")
    print("=" * 80)
    print()
    
    all_results = []
    
    for pos_id, fen, description in BENCHMARK_POSITIONS:
        print(f"\nPosition: {description} ({pos_id})")
        print(f"FEN: {fen}")
        print("-" * 80)
        
        for depth in depths:
            engine, tuning, meta = create_pvs_engine(depth)
            result = benchmark_position(engine, tuning, meta, fen)
            all_results.append({
                'pos_id': pos_id,
                'description': description,
                **result
            })
            
            pv_str = " ".join(result['pv'][:3])
            print(f"  Depth {depth:2d}: {result['nodes']:7d} nodes  "
                  f"{result['time']:5.2f}s  {result['nps']:7d} NPS  "
                  f"score={result['score']:6.1f}  pv: {pv_str}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    total_nodes = sum(r['nodes'] for r in all_results)
    total_time = sum(r['time'] for r in all_results)
    avg_nps = total_nodes / total_time if total_time > 0 else 0
    
    print(f"\nTotal nodes:   {total_nodes:,}")
    print(f"Total time:    {total_time:.2f}s")
    print(f"Average NPS:   {int(avg_nps):,}")
    
    # Per-depth averages
    print("\nAverage NPS by depth:")
    for depth in depths:
        depth_results = [r for r in all_results if r['depth'] == depth]
        if depth_results:
            avg = sum(r['nps'] for r in depth_results) / len(depth_results)
            print(f"  Depth {depth}: {int(avg):,} NPS")
    
    # Per-phase averages
    print("\nAverage NPS by game phase:")
    for phase in ["Opening", "Middlegame", "Endgame"]:
        phase_results = [r for r in all_results if phase in r['description']]
        if phase_results:
            avg = sum(r['nps'] for r in phase_results) / len(phase_results)
            print(f"  {phase:12s}: {int(avg):,} NPS")
    
    print("\n" + "=" * 80)


def profile_position(pos_id: str = "midgame1", depth: int = 5) -> str:
    """Profile a specific position and save results."""
    # Find the position
    fen = None
    for pid, pfen, desc in BENCHMARK_POSITIONS:
        if pid == pos_id:
            fen = pfen
            break
    
    if fen is None:
        print(f"Position '{pos_id}' not found")
        return ""
    
    engine, tuning, meta = create_pvs_engine(depth)
    board = chess.Board(fen)
    output_file = f"pvs_profile_{pos_id}_d{depth}.txt"
    
    print(f"Profiling position '{pos_id}' at depth {depth}...")
    print(f"FEN: {fen}")
    
    # Note: profile_search is not implemented in PVSearchBackend
    # This would need to be added to the engine
    print("WARNING: profile_search method not implemented in PVSearchBackend")
    print("Consider using the benchmark mode instead")
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        # Profile mode
        pos_id = sys.argv[2] if len(sys.argv) > 2 else "midgame1"
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        profile_position(pos_id, depth)
    else:
        # Benchmark mode
        depths = [4, 5] if len(sys.argv) == 1 else [int(d) for d in sys.argv[1:]]
        run_benchmark_suite(depths)
