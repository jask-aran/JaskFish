#!/usr/bin/env python3
"""Unified benchmarking and profiling harness for the PVS engine via UCI."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

DEFAULT_ENGINE_CMD: Sequence[str] = (sys.executable, "pvsengine.py")


@dataclass(frozen=True)
class FenScenario:
    slug: str
    fen: str
    label: str
    phase: str


CURATED_SCENARIOS: Sequence[FenScenario] = (
    FenScenario("start", "rn1qkbnr/pppb1ppp/4p3/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R w KQkq - 0 5", "Opening – Semi-Slav structure", "Opening"),
    FenScenario(
        "kingside_pressure",
        "r2q1rk1/pp1b1ppp/2n2n2/2bp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10",
        "Middlegame – Kingside pressure",
        "Middlegame",
    ),
    FenScenario(
        "tactical_fireworks",
        "r1b2rk1/1pp1qppp/p1n2n2/3p4/2PP4/2N1PN2/PPQ2PPP/R1B2RK1 b - - 0 11",
        "Middlegame – Tactical imbalance",
        "Middlegame",
    ),
    FenScenario(
        "blocked_center",
        "r1bq1rk1/3n1pbp/pppp1np1/4p3/PP1PP3/2N1BN2/2P1BPPP/R2Q1RK1 w - - 0 12",
        "Middlegame – Blocked center",
        "Middlegame",
    ),
    FenScenario(
        "opposite_wings",
        "2r2rk1/1b2bppp/p2p1n2/1p1Pp3/1P2P3/P1NB1N2/1B3PPP/2RR2K1 w - - 0 21",
        "Middlegame – Opposite-wing plans",
        "Middlegame",
    ),
    FenScenario(
        "rook_endgame",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "Endgame – Complex rook ending",
        "Endgame",
    ),
    FenScenario(
        "minor_piece_endgame",
        "8/5k2/1p4p1/1P2p3/4P3/2K3P1/5N2/8 w - - 0 50",
        "Endgame – Knight vs pawns",
        "Endgame",
    ),
    FenScenario(
        "fortress_test",
        "8/8/5k2/4p3/3pP3/3K4/6R1/8 w - - 0 58",
        "Endgame – Fortress probe",
        "Endgame",
    ),
)

CURATED_DEFAULT_SLUGS: Sequence[str] = (
    "start",
    "kingside_pressure",
    "tactical_fireworks",
    "opposite_wings",
    "rook_endgame",
    "minor_piece_endgame",
)

SCENARIO_INDEX = {scenario.slug: scenario for scenario in CURATED_SCENARIOS}


class EngineSessionError(RuntimeError):
    """Error raised when engine interaction fails."""


@dataclass
class GoRequest:
    fen: str
    go_tokens: List[str]
    infinite_duration: Optional[float]
    max_wait: float
    echo_info: bool
    label: str


@dataclass
class GoResult:
    request: GoRequest
    lines: List[str]
    payloads: List[dict]
    bestmove: Optional[str]
    ponder: Optional[str]
    elapsed: float
    finished: bool

    @property
    def final_payload(self) -> Optional[dict]:
        return self.payloads[-1] if self.payloads else None


class EngineSession:
    """Maintain a UCI session with the engine."""

    def __init__(
        self,
        command: Sequence[str] = DEFAULT_ENGINE_CMD,
        ready_timeout: float = 10.0,
        include_stderr: bool = True,
    ) -> None:
        self.command = list(command)
        self.ready_timeout = ready_timeout
        self.include_stderr = include_stderr
        self._proc: Optional[subprocess.Popen[str]] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        stderr = subprocess.STDOUT if self.include_stderr else None
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise EngineSessionError("Failed to open pipes to engine process")
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._send("uci")
        self._await_token("uciok", self.ready_timeout)
        self.await_ready()
        self._send("ucinewgame")

    def await_ready(self) -> None:
        self._send("isready")
        self._await_token("readyok", self.ready_timeout)

    def set_position(self, fen: str) -> None:
        self._send(f"position fen {fen}")

    def go(self, request: GoRequest) -> GoResult:
        go_command = " ".join(["go", *request.go_tokens]).strip()
        if not request.go_tokens:
            go_command = "go"

        self.await_ready()
        self.set_position(request.fen)
        self._send(go_command)

        payloads: List[dict] = []
        lines: List[str] = []
        bestmove: Optional[str] = None
        ponder: Optional[str] = None
        start = time.perf_counter()
        stop_sent = False
        finished = False

        while True:
            remaining = request.max_wait - (time.perf_counter() - start)
            timeout = max(0.05, min(1.0, remaining)) if request.max_wait > 0 else 1.0
            line = self._read_line(timeout=timeout)
            if line is None:
                # If engine exited unexpectedly, break.
                if self._proc and self._proc.poll() is not None:
                    break
                if not stop_sent and request.max_wait > 0 and (time.perf_counter() - start) >= request.max_wait:
                    self._send("stop")
                    stop_sent = True
                    continue
                continue

            stripped = line.strip()
            if request.echo_info:
                print(stripped)
            lines.append(stripped)

            if stripped.startswith("info string perf payload="):
                payload_text = stripped.split("info string perf payload=", 1)[1].strip()
                try:
                    payloads.append(json.loads(payload_text))
                except json.JSONDecodeError:
                    payloads.append({"raw": payload_text, "error": "json decode failure"})

            if stripped.startswith("bestmove"):
                tokens = stripped.split()
                if len(tokens) >= 2:
                    bestmove = tokens[1]
                if len(tokens) >= 4 and tokens[2] == "ponder":
                    ponder = tokens[3]
                finished = True
                break

            if request.infinite_duration and not stop_sent:
                if (time.perf_counter() - start) >= request.infinite_duration:
                    self._send("stop")
                    stop_sent = True

        elapsed = time.perf_counter() - start
        return GoResult(
            request=request,
            lines=lines,
            payloads=payloads,
            bestmove=bestmove,
            ponder=ponder,
            elapsed=elapsed,
            finished=finished,
        )

    def shutdown(self) -> None:
        if not self._running:
            return
        try:
            self._send("quit")
        except EngineSessionError:
            pass
        if self._proc:
            try:
                self._proc.stdin and self._proc.stdin.flush()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._running = False

    def _reader_loop(self) -> None:
        assert self._proc and self._proc.stdout
        for raw in self._proc.stdout:
            self._queue.put(raw.rstrip("\r\n"))
        self._queue.put(None)

    def _send(self, command: str) -> None:
        if not self._running or not self._proc or not self._proc.stdin:
            raise EngineSessionError("Engine session is not active")
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def _await_token(self, token: str, timeout: float) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            remaining = max(0.05, deadline - time.time())
            line = self._read_line(timeout=remaining)
            if line is None:
                continue
            if line.strip() == token:
                return
        raise EngineSessionError(f"Timed out waiting for '{token}' from engine")

    def _read_line(self, timeout: float = 1.0) -> Optional[str]:
        try:
            line = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        return line


def build_go_tokens(args: argparse.Namespace) -> List[str]:
    tokens: List[str] = []
    for key in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth"):
        value = getattr(args, key, None)
        if value is not None:
            tokens.extend([key, str(value)])
    if getattr(args, "infinite", False):
        tokens.append("infinite")
    if getattr(args, "mate", None) is not None:
        tokens.extend(["mate", str(args.mate)])
    if getattr(args, "ponder", False):
        tokens.append("ponder")
    return tokens


def iter_user_fens(args: argparse.Namespace) -> Iterator[FenScenario]:
    counter = 1
    for fen in args.fen or []:
        slug = f"custom_{counter}"
        yield FenScenario(slug=slug, fen=fen, label=f"Custom #{counter}", phase="Custom")
        counter += 1

    if args.fen_file:
        path = Path(args.fen_file)
        if not path.exists():
            raise SystemExit(f"Provided FEN file '{path}' does not exist")
        for line in path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            slug = f"file_{counter}"
            yield FenScenario(slug=slug, fen=stripped, label=f"File #{counter}", phase="Custom")
            counter += 1


def resolve_scenarios(selected: Optional[Sequence[str]], include_defaults: bool) -> List[FenScenario]:
    scenarios: List[FenScenario] = []
    if include_defaults or not selected:
        scenarios.extend(SCENARIO_INDEX[slug] for slug in CURATED_DEFAULT_SLUGS)
    if selected:
        for slug in selected:
            if slug not in SCENARIO_INDEX:
                raise SystemExit(f"Unknown scenario slug '{slug}'. Use 'list' to inspect catalogue.")
            scenarios.append(SCENARIO_INDEX[slug])
    # Deduplicate while preserving order.
    seen = set()
    unique: List[FenScenario] = []
    for scenario in scenarios:
        if scenario.slug in seen:
            continue
        seen.add(scenario.slug)
        unique.append(scenario)
    return unique


def run_benchmark(args: argparse.Namespace) -> List[GoResult]:
    scenarios = resolve_scenarios(args.positions, include_defaults=not args.skip_defaults)
    scenarios.extend(iter_user_fens(args))
    if not scenarios:
        raise SystemExit("No scenarios provided. Use --positions or --fen/--fen-file.")

    go_tokens = build_go_tokens(args)
    infinite_duration = args.infinite_duration
    if args.infinite and infinite_duration is None:
        infinite_duration = 5.0

    request_template = {
        "go_tokens": go_tokens,
        "infinite_duration": infinite_duration,
        "max_wait": args.max_wait,
        "echo_info": args.echo_info,
    }

    session = EngineSession(command=args.engine_cmd or DEFAULT_ENGINE_CMD)
    session.start()

    results: List[GoResult] = []
    try:
        for scenario in scenarios:
            request = GoRequest(
                fen=scenario.fen,
                label=scenario.label,
                **request_template,
            )
            print(f"\n=== {scenario.label} ({scenario.phase}) ===")
            print(f"FEN: {scenario.fen}")
            result = session.go(request)
            results.append(result)
            summarise_result(result)
    finally:
        session.shutdown()
    return results


def summarise_result(result: GoResult) -> None:
    payload = result.final_payload or {}
    nodes_block = payload.get("nodes", {})
    time_block = payload.get("time", {})
    score_block = payload.get("score", {})
    depth = payload.get("depth")
    seldepth = payload.get("seldepth")
    nodes = nodes_block.get("total")
    nps = payload.get("nps")
    elapsed = time_block.get("elapsed", result.elapsed)
    budget = time_block.get("budget")
    score = score_block.get("value")
    pv = payload.get("pv") or []
    pv_preview = " ".join(pv[:6])
    bestmove = result.bestmove or "(none)"

    print(f"Best move: {bestmove}" + (f" | ponder: {result.ponder}" if result.ponder else ""))
    print(
        "Depth: "
        + ", ".join(
            filter(
                None,
                [
                    f"search={depth}" if depth is not None else None,
                    f"sel={seldepth}" if seldepth is not None else None,
                ],
            )
        )
    )
    print(
        "Nodes: "
        + ", ".join(
            filter(
                None,
                [
                    f"total={nodes:,}" if nodes is not None else None,
                    f"regular={nodes_block.get('regular', '–')}",
                    f"q={nodes_block.get('quiescence', '–')}",
                ],
            )
        )
    )
    print(
        "Time: "
        + ", ".join(
            filter(
                None,
                [
                    f"elapsed={elapsed:.3f}s" if isinstance(elapsed, (int, float)) else None,
                    f"budget={budget:.3f}s" if isinstance(budget, (int, float)) else None,
                    f"NPS={int(nps):,}" if isinstance(nps, (int, float)) else None,
                ],
            )
        )
    )
    print(
        "Score: "
        + ", ".join(
            filter(
                None,
                [
                    f"value={score:.1f}" if isinstance(score, (int, float)) else None,
                    f"delta={score_block.get('delta', '–')}",
                ],
            )
        )
    )
    print(f"PV: {pv_preview or '(empty)'}")


def run_profile(args: argparse.Namespace) -> None:
    from pvsengine import run_profile as engine_run_profile

    scenarios = resolve_scenarios(args.positions, include_defaults=not args.skip_defaults)
    scenarios.extend(iter_user_fens(args))
    if not scenarios:
        raise SystemExit("No scenarios provided for profiling.")

    for scenario in scenarios:
        print(f"\n=== Profiling {scenario.label} ===")
        engine_run_profile(scenario.fen, args.threads)


def list_scenarios() -> None:
    print("Available curated scenarios:")
    for scenario in CURATED_SCENARIOS:
        print(f"  {scenario.slug:<20} {scenario.phase:<12} {scenario.label}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark or profile the PVS engine using UCI interactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine-cmd",
        nargs="+",
        default=None,
        help="Override engine command (default: python pvsengine.py)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    common_benchmark = argparse.ArgumentParser(add_help=False)
    common_benchmark.add_argument(
        "--positions",
        nargs="*",
        default=None,
        help="Catalogue slugs to include.",
    )
    common_benchmark.add_argument(
        "--skip-defaults",
        action="store_true",
        help="Do not preload the curated default scenario set.",
    )
    common_benchmark.add_argument(
        "--fen",
        action="append",
        default=None,
        help="Additional FEN strings to evaluate (repeatable).",
    )
    common_benchmark.add_argument(
        "--fen-file",
        default=None,
        help="Path to a text file with one FEN per line (comments starting with '#').",
    )
    common_benchmark.add_argument(
        "--echo-info",
        action="store_true",
        help="Echo every engine info line during searches.",
    )

    time_group = common_benchmark.add_argument_group("Budget")
    time_group.add_argument("--wtime", type=int, help="White time remaining in ms.")
    time_group.add_argument("--btime", type=int, help="Black time remaining in ms.")
    time_group.add_argument("--winc", type=int, help="White increment in ms.")
    time_group.add_argument("--binc", type=int, help="Black increment in ms.")
    time_group.add_argument("--movestogo", type=int, help="Moves to the next time control.")
    time_group.add_argument("--movetime", type=int, help="Fixed time per move in ms.")
    time_group.add_argument("--nodes", type=int, help="Limit by number of searched nodes.")
    time_group.add_argument("--depth", type=int, help="Limit by search depth.")
    time_group.add_argument("--mate", type=int, help="Search for mate in given moves.")
    time_group.add_argument("--infinite", action="store_true", help="Run go infinite.")
    time_group.add_argument(
        "--infinite-duration",
        type=float,
        default=None,
        help="Seconds to wait before sending stop when using --infinite (default 5s).",
    )
    time_group.add_argument(
        "--ponder",
        action="store_true",
        help="Set ponder flag on go command.",
    )
    time_group.add_argument(
        "--max-wait",
        type=float,
        default=90.0,
        help="Maximum seconds to wait for bestmove before issuing stop.",
    )

    benchmark = subparsers.add_parser("benchmark", parents=[common_benchmark], help="Run benchmark scenarios via UCI.")
    benchmark.set_defaults(handler=run_benchmark)

    profile = subparsers.add_parser("profile", parents=[common_benchmark], help="Collect cProfile traces using engine internals.")
    profile.add_argument("--threads", type=int, default=1, help="Number of worker threads for profiling backend.")
    profile.set_defaults(handler=run_profile)

    list_parser = subparsers.add_parser("list", help="List curated scenario slugs.")
    list_parser.set_defaults(handler=lambda *_: list_scenarios())

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = getattr(args, "handler", None)
    if not handler:
        parser.print_help()
        return
    handler(args)


if __name__ == "__main__":
    main()
