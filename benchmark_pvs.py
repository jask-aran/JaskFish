#!/usr/bin/env python3
"""Utilities for benchmarking and profiling the PV search backend."""

from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Sequence

import chess

from pvsengine import (
    MetaParams,
    PVSearchBackend,
    SearchLimits,
    SearchReporter,
    SearchTuning,
    StrategyContext,
    build_search_tuning,
    run_profile as engine_run_profile,
)


@dataclass(frozen=True)
class Position:
    slug: str
    fen: str
    label: str
    phase: str


POSITIONS: Sequence[Position] = (
    Position("start", chess.STARTING_FEN, "Opening - Start position", "Opening"),
    Position(
        "mid-complex",
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10",
        "Middlegame - Complex center",
        "Middlegame",
    ),
    Position(
        "mid-tactical",
        "r1b2rk1/ppq1bppp/2n1pn2/3p4/2PP4/2N1PN2/PP1QBPPP/R1B2RK1 b - - 0 10",
        "Middlegame - Tactical imbalance",
        "Middlegame",
    ),
    Position(
        "mid-pressure",
        "2r2rk1/1b2bppp/p2p1n2/1p1Pp3/1P2P3/P1N1BN2/1B3PPP/2RR2K1 w - - 0 21",
        "Middlegame - Pressure on king",
        "Middlegame",
    ),
    Position("end-kp", "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "Endgame - K+P vs K", "Endgame"),
    Position(
        "end-pawns",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "Endgame - Complex pawn race",
        "Endgame",
    ),
)

POSITION_INDEX = {pos.slug: pos for pos in POSITIONS}

BENCHMARK_DEFAULTS = ["start", "mid-complex", "mid-tactical", "end-kp", "end-pawns"]
PROFILE_DEFAULTS = ["start", "mid-complex", "mid-pressure"]


def create_engine(depth: int, threads: int) -> tuple[PVSearchBackend, SearchTuning, SearchLimits]:
    meta = MetaParams()
    tuning, limits = build_search_tuning(meta)
    tuning = replace(tuning, search_depth=depth)
    engine = PVSearchBackend(max_threads=max(1, threads))
    engine.configure(tuning, meta)
    return engine, tuning, limits


def benchmark_position(
    engine: PVSearchBackend,
    tuning: SearchTuning,
    limits: SearchLimits,
    position: Position,
) -> dict[str, object]:
    board = chess.Board(position.fen)
    context = StrategyContext(
        fullmove_number=board.fullmove_number,
        halfmove_clock=board.halfmove_clock,
        piece_count=len(board.piece_map()),
        material_imbalance=0,
        turn=board.turn,
        fen=position.fen,
        legal_moves_count=board.legal_moves.count(),
        repetition_info={},
        time_controls=None,
        in_check=board.is_check(),
        opponent_mate_in_one_threat=False,
    )
    reporter = SearchReporter(logger=lambda *_: None)
    budget = limits.resolve_budget(context, reporter)

    start = time.perf_counter()
    result = engine.search(board, context, limits, reporter, budget)
    elapsed = getattr(result, "time_spent", time.perf_counter() - start)

    nps = result.nodes / elapsed if elapsed > 0 else 0

    return {
        "position": position,
        "depth": tuning.search_depth,
        "nodes": result.nodes,
        "time": elapsed,
        "nps": int(nps),
        "score": result.score,
        "pv": [move.uci() for move in result.principal_variation[:5]],
        "completed_depth": result.completed_depth,
    }


def run_benchmarks(depths: Sequence[int], positions: Sequence[Position], threads: int) -> None:
    print("=" * 80)
    print("PVS Engine Benchmark")
    print("=" * 80)

    all_results: List[dict[str, object]] = []

    for depth in depths:
        engine, tuning, limits = create_engine(depth, threads)
        for position in positions:
            result = benchmark_position(engine, tuning, limits, position)
            all_results.append(result)
            pv_preview = " ".join(result["pv"][:3])
            print(
                f"{position.label:<35} depth {depth:2d}  nodes={result['nodes']:9d}  "
                f"time={result['time']:.2f}s  nps={result['nps']:8d}  "
                f"score={result['score']:6.1f}  pv: {pv_preview}"
            )

    if not all_results:
        print("No benchmark results.")
        return

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_nodes = sum(r["nodes"] for r in all_results)
    total_time = sum(r["time"] for r in all_results)
    avg_nps = int(total_nodes / total_time) if total_time else 0

    print(f"Total nodes: {total_nodes:,}")
    print(f"Total time:  {total_time:.2f}s")
    print(f"Average NPS: {avg_nps:,}")

    print("\nAverage NPS by depth:")
    for depth in depths:
        depth_results = [r for r in all_results if r["depth"] == depth]
        if not depth_results:
            continue
        avg = sum(r["nps"] for r in depth_results) / len(depth_results)
        print(f"  depth {depth:2d}: {int(avg):,}")

    print("\nAverage NPS by phase:")
    for phase in sorted({pos.phase for pos in positions}):
        phase_results = [r for r in all_results if r["position"].phase == phase]
        if not phase_results:
            continue
        avg = sum(r["nps"] for r in phase_results) / len(phase_results)
        print(f"  {phase:12s}: {int(avg):,}")


def capture_profile_output(fen: str, threads: int) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        engine_run_profile(fen, threads)
    return buffer.getvalue().rstrip()


def run_profiles(positions: Sequence[Position], threads: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sections: List[str] = []

    for position in positions:
        print(f"Profiling {position.slug} ({position.label})...")
        output = capture_profile_output(position.fen, threads)
        sections.append(
            "\n".join(
                [
                    "=" * 80,
                    f"Scenario: {position.label}",
                    f"FEN: {position.fen}",
                    "=" * 80,
                    output,
                    "",
                ]
            )
        )

    combined = "\n".join(sections).rstrip() + "\n"
    output_path.write_text(combined)
    print(f"Profile output written to {output_path}")


def resolve_positions(slugs: Iterable[str]) -> List[Position]:
    resolved: List[Position] = []
    for slug in slugs:
        try:
            resolved.append(POSITION_INDEX[slug])
        except KeyError as exc:
            raise SystemExit(f"Unknown position slug '{slug}'. Use 'list' to view options.") from exc
    return resolved


def list_positions() -> None:
    print("Available positions:")
    for pos in POSITIONS:
        print(f"  {pos.slug:<13} {pos.phase:<11} {pos.label}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark and profile PV search utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench.add_argument("--depth", action="append", type=int, dest="depths", help="Search depth to benchmark")
    bench.add_argument(
        "--positions",
        nargs="*",
        default=None,
        help="Position slugs to include (default: curated benchmark set)",
    )
    bench.add_argument("--threads", type=int, default=1, help="Number of threads for the backend")

    profile = subparsers.add_parser("profile", help="Run cProfile across scenarios")
    profile.add_argument(
        "--positions",
        nargs="*",
        default=None,
        help="Position slugs to profile (default: curated profile set)",
    )
    profile.add_argument(
        "--output",
        type=Path,
        default=Path("profiles/pvs_profile.txt"),
        help="File to write combined profile output",
    )
    profile.add_argument("--threads", type=int, default=1, help="Number of threads for the backend")

    subparsers.add_parser("list", help="List available positions")

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "list":
        list_positions()
        return

    if args.command == "benchmark":
        depths = args.depths or [4, 5]
        positions = resolve_positions(args.positions or BENCHMARK_DEFAULTS)
        run_benchmarks(depths, positions, args.threads)
        return

    if args.command == "profile":
        positions = resolve_positions(args.positions or PROFILE_DEFAULTS)
        run_profiles(positions, args.threads, args.output)
        return

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    main()
