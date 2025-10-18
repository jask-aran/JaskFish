#!/usr/bin/env python3
"""Unified benchmarking, profiling, and comparison harness for JaskFish engines."""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import textwrap
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import chess

from main import compute_adaptive_movetime


class SmartFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Formatter that shows defaults while preserving custom epilog layout."""


@dataclass(frozen=True)
class FenScenario:
    slug: str
    fen: str
    label: str
    phase: str


CURATED_SCENARIOS: Sequence[FenScenario] = (
    FenScenario(
        "start",
        "rn1qkbnr/pppb1ppp/4p3/3p4/3P4/2P2N2/PP2PPPP/RNBQKB1R w KQkq - 0 5",
        "Opening – Semi-Slav structure",
        "Opening",
    ),
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


ProfileHandler = Callable[["EngineSpec", str, int], None]


@dataclass(frozen=True)
class EngineSpec:
    key: str
    display_name: str
    command: Sequence[str]
    aliases: Tuple[str, ...]
    profile_handler: ProfileHandler
    supports_profile: bool
    profile_note: str

    @property
    def all_aliases(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys((self.key, *self.aliases)))

    @property
    def canonical_alias(self) -> str:
        return self.key


def profile_python_pvs_engine(spec: EngineSpec, fen: str, threads: int) -> None:
    from engines.pvsengine import run_profile as pvs_run_profile

    pvs_run_profile(fen, threads)


def profile_not_supported(spec: EngineSpec, fen: str, threads: int) -> None:
    raise SystemExit(
        f"Profiling for '{spec.display_name}' is not available yet. {spec.profile_note}"
    )


ENGINE_SPECS: Tuple[EngineSpec, ...] = (
    EngineSpec(
        key="pvs",
        display_name="Python PVS engine",
        command=(sys.executable, "engines/pvsengine.py"),
        aliases=("python", "python-pvs"),
        profile_handler=profile_python_pvs_engine,
        supports_profile=True,
        profile_note="Runs engines.pvsengine.run_profile with cProfile statistics.",
    ),
    EngineSpec(
        key="simple",
        display_name="Simple Python reference engine",
        command=(sys.executable, "engines/simple_engine.py"),
        aliases=("simple-python",),
        profile_handler=profile_not_supported,
        supports_profile=False,
        profile_note="Lightweight engine; profiling hook not implemented.",
    ),
    EngineSpec(
        key="cpvs",
        display_name="Native CPVS engine",
        command=("engines/native/cpvsengine",),
        aliases=("native", "c"),
        profile_handler=profile_not_supported,
        supports_profile=False,
        profile_note="Native profiler plumbing will land in a future update.",
    ),
)

ENGINE_ALIAS_MAP: Dict[str, EngineSpec] = {}
for spec in ENGINE_SPECS:
    for alias in spec.all_aliases:
        lowered = alias.lower()
        if lowered in ENGINE_ALIAS_MAP:
            raise RuntimeError(
                f"Duplicate engine alias '{alias}' for {ENGINE_ALIAS_MAP[lowered].display_name}"
            )
        ENGINE_ALIAS_MAP[lowered] = spec


def format_engine_overview() -> str:
    lines: List[str] = ["Available engine aliases:"]
    for spec in ENGINE_SPECS:
        aliases = ", ".join(spec.all_aliases)
        command = " ".join(str(token) for token in spec.command)
        profile_status = (
            "profiling supported" if spec.supports_profile else "profiling unavailable"
        )
        lines.append(f"  {aliases}")
        lines.append(f"    name: {spec.display_name}")
        lines.append(f"    command: {command}")
        lines.append(f"    {profile_status}: {spec.profile_note}")
    return "\n".join(lines)


def validate_engine_alias(value: str) -> str:
    alias = value.strip().lower()
    if alias not in ENGINE_ALIAS_MAP:
        available = ", ".join(sorted(ENGINE_ALIAS_MAP))
        raise argparse.ArgumentTypeError(
            f"Unknown engine alias '{value}'. Choose from: {available}"
        )
    return alias


def resolve_engine_alias(alias: str) -> EngineSpec:
    try:
        return ENGINE_ALIAS_MAP[alias]
    except KeyError as exc:  # pragma: no cover - safeguarded by validation
        raise SystemExit(f"Unknown engine alias '{alias}'.") from exc


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


@dataclass
class EngineMetrics:
    depth: Optional[int]
    seldepth: Optional[int]
    time_ms: Optional[int]
    nodes: Optional[int]
    nps: Optional[int]
    score_cp: Optional[float]
    pv: List[str]
    bestmove: Optional[str]


class EngineSession:
    """Maintain a UCI session with the engine."""

    def __init__(
        self,
        command: Sequence[str],
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
                if self._proc and self._proc.poll() is not None:
                    break
                if (
                    not stop_sent
                    and request.max_wait > 0
                    and (time.perf_counter() - start) >= request.max_wait
                ):
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
                    payloads.append(
                        {"raw": payload_text, "error": "json decode failure"}
                    )

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
    for key in (
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
        "movetime",
        "nodes",
        "depth",
    ):
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
        yield FenScenario(
            slug=slug, fen=fen, label=f"Custom #{counter}", phase="Custom"
        )
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
            yield FenScenario(
                slug=slug, fen=stripped, label=f"File #{counter}", phase="Custom"
            )
            counter += 1


def resolve_scenarios(
    selected: Optional[Sequence[str]], include_defaults: bool
) -> List[FenScenario]:
    scenarios: List[FenScenario] = []
    if include_defaults or not selected:
        scenarios.extend(SCENARIO_INDEX[slug] for slug in CURATED_DEFAULT_SLUGS)
    if selected:
        for slug in selected:
            if slug not in SCENARIO_INDEX:
                raise SystemExit(
                    f"Unknown scenario slug '{slug}'. Use 'list' to inspect the catalogue."
                )
            scenarios.append(SCENARIO_INDEX[slug])
    seen = set()
    unique: List[FenScenario] = []
    for scenario in scenarios:
        if scenario.slug in seen:
            continue
        seen.add(scenario.slug)
        unique.append(scenario)
    return unique


def run_benchmark(args: argparse.Namespace) -> List[GoResult]:
    spec = resolve_engine_alias(args.engine)
    scenarios = resolve_scenarios(
        args.positions, include_defaults=not args.skip_defaults
    )
    scenarios.extend(iter_user_fens(args))
    if not scenarios:
        raise SystemExit("No scenarios provided. Use --positions or --fen/--fen-file.")

    base_go_tokens = build_go_tokens(args)
    infinite_duration = args.infinite_duration
    if args.infinite and infinite_duration is None:
        infinite_duration = 5.0

    request_template = {
        "infinite_duration": infinite_duration,
        "max_wait": args.max_wait,
        "echo_info": args.echo_info,
    }

    session = EngineSession(command=spec.command)
    session.start()

    print(
        f"Selected engine: {spec.display_name} [{spec.canonical_alias}] (command: {' '.join(map(str, spec.command))})"
    )

    results: List[GoResult] = []
    try:
        for scenario in scenarios:
            if base_go_tokens:
                go_tokens = list(base_go_tokens)
            else:
                board = chess.Board(scenario.fen)
                movetime_ms = compute_adaptive_movetime(board)
                go_tokens = ["movetime", str(movetime_ms)]
            request = GoRequest(
                fen=scenario.fen,
                label=scenario.label,
                go_tokens=go_tokens,
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

    print(
        f"Best move: {bestmove}"
        + (f" | ponder: {result.ponder}" if result.ponder else "")
    )
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
                    f"total={nodes:,}" if isinstance(nodes, int) else None,
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
                    f"elapsed={elapsed:.3f}s"
                    if isinstance(elapsed, (int, float))
                    else None,
                    f"budget={budget:.3f}s"
                    if isinstance(budget, (int, float))
                    else None,
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


def parse_info_line(lines: Sequence[str]) -> Dict[str, Optional[object]]:
    data: Dict[str, Optional[object]] = {
        "depth": None,
        "seldepth": None,
        "time_ms": None,
        "nodes": None,
        "nps": None,
        "score_cp": None,
        "pv": [],
    }
    for raw in reversed(lines):
        stripped = raw.strip()
        if not stripped.startswith("info depth"):
            continue
        tokens = stripped.split()
        pv_index: Optional[int] = None
        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token == "depth" and i + 1 < len(tokens):
                try:
                    data["depth"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if token == "seldepth" and i + 1 < len(tokens):
                try:
                    data["seldepth"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if token == "score" and i + 2 < len(tokens):
                mode = tokens[i + 1]
                value_token = tokens[i + 2]
                if mode == "cp":
                    try:
                        data["score_cp"] = float(value_token)
                    except ValueError:
                        data["score_cp"] = None
                elif mode == "mate":
                    try:
                        mate_in = int(value_token)
                        data["score_cp"] = 32000.0 if mate_in > 0 else -32000.0
                    except ValueError:
                        data["score_cp"] = None
                i += 3
                continue
            if token == "time" and i + 1 < len(tokens):
                try:
                    data["time_ms"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if token == "nodes" and i + 1 < len(tokens):
                try:
                    data["nodes"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if token == "nps" and i + 1 < len(tokens):
                try:
                    data["nps"] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
                continue
            if token == "pv":
                pv_index = i + 1
                break
            i += 1
        if pv_index is not None:
            data["pv"] = tokens[pv_index:]
        else:
            data["pv"] = []
        return data
    return data


def extract_metrics(result: GoResult) -> EngineMetrics:
    payload = result.final_payload
    if payload:
        time_block_obj = payload.get("time")
        time_block = time_block_obj if isinstance(time_block_obj, dict) else {}
        elapsed = time_block.get("elapsed")
        time_ms = (
            int(round(elapsed * 1000)) if isinstance(elapsed, (int, float)) else None
        )

        nodes_block_obj = payload.get("nodes")
        nodes_block = nodes_block_obj if isinstance(nodes_block_obj, dict) else {}
        nodes_value = nodes_block.get("total")
        nodes = nodes_value if isinstance(nodes_value, int) else None

        score_block_obj = payload.get("score")
        score_block = score_block_obj if isinstance(score_block_obj, dict) else {}
        score_val = score_block.get("value")
        score_cp = (
            float(score_val) * 100.0 if isinstance(score_val, (int, float)) else None
        )

        pv_raw = payload.get("pv")
        pv = [str(move) for move in pv_raw] if isinstance(pv_raw, list) else []

        depth_val = payload.get("depth")
        depth = depth_val if isinstance(depth_val, int) else None

        seldepth_val = payload.get("seldepth")
        seldepth = seldepth_val if isinstance(seldepth_val, int) else None

        nps_val = payload.get("nps")
        nps = nps_val if isinstance(nps_val, int) else None

        return EngineMetrics(
            depth=depth,
            seldepth=seldepth,
            time_ms=time_ms,
            nodes=nodes,
            nps=nps,
            score_cp=score_cp,
            pv=pv,
            bestmove=result.bestmove,
        )

    info_data = parse_info_line(result.lines)

    depth_raw = info_data.get("depth")
    depth = depth_raw if isinstance(depth_raw, int) else None

    seldepth_raw = info_data.get("seldepth")
    seldepth = seldepth_raw if isinstance(seldepth_raw, int) else None

    time_raw = info_data.get("time_ms")
    time_ms = time_raw if isinstance(time_raw, int) else None

    nodes_raw = info_data.get("nodes")
    nodes = nodes_raw if isinstance(nodes_raw, int) else None

    nps_raw = info_data.get("nps")
    nps = nps_raw if isinstance(nps_raw, int) else None

    score_raw = info_data.get("score_cp")
    score_cp = score_raw if isinstance(score_raw, (int, float)) else None

    pv_raw = info_data.get("pv")
    pv = [str(move) for move in pv_raw] if isinstance(pv_raw, list) else []

    return EngineMetrics(
        depth=depth,
        seldepth=seldepth,
        time_ms=time_ms,
        nodes=nodes,
        nps=nps,
        score_cp=score_cp,
        pv=pv,
        bestmove=result.bestmove,
    )


def describe_metrics(label: str, metrics: EngineMetrics) -> str:
    pieces = [f"{label:<20}"]
    pieces.append(f"depth={metrics.depth}" if metrics.depth is not None else "depth=?")
    if metrics.seldepth is not None:
        pieces.append(f"sel={metrics.seldepth}")
    pieces.append(
        f"time={metrics.time_ms / 1000:.3f}s"
        if metrics.time_ms is not None
        else "time=?"
    )
    if metrics.nodes is not None:
        pieces.append(f"nodes={metrics.nodes:,}")
    if metrics.nps is not None:
        pieces.append(f"nps={metrics.nps:,}")
    if metrics.score_cp is not None:
        pieces.append(f"score={metrics.score_cp:.1f}cp")
    best = metrics.bestmove or "(none)"
    pieces.append(f"best={best}")
    pv_preview = " ".join(metrics.pv[:6])
    if pv_preview:
        pieces.append(f"pv={pv_preview}")
    return "  " + " | ".join(pieces)


def summarise_deltas(
    primary: EngineMetrics,
    secondary: EngineMetrics,
    primary_label: str,
    secondary_label: str,
) -> str:
    details: List[str] = []
    if primary.depth is not None and secondary.depth is not None:
        diff = primary.depth - secondary.depth
        details.append(f"depthΔ={diff:+d}")
    if (
        primary.time_ms is not None
        and secondary.time_ms is not None
        and secondary.time_ms > 0
    ):
        ratio = primary.time_ms / secondary.time_ms
        details.append(f"time {primary_label}/{secondary_label}={ratio:.2f}x")
    if (
        primary.nodes is not None
        and secondary.nodes is not None
        and secondary.nodes > 0
    ):
        ratio = primary.nodes / secondary.nodes
        details.append(f"nodes {primary_label}/{secondary_label}={ratio:.2f}x")
    return "    " + " | ".join(details) if details else ""


def run_profile(args: argparse.Namespace) -> None:
    spec = resolve_engine_alias(args.engine)
    scenarios = resolve_scenarios(
        args.positions, include_defaults=not args.skip_defaults
    )
    scenarios.extend(iter_user_fens(args))
    if not scenarios:
        raise SystemExit("No scenarios provided for profiling.")

    print(f"Profiling engine: {spec.display_name} [{spec.canonical_alias}]")
    print(f"Profile note: {spec.profile_note}")

    if not spec.supports_profile:
        spec.profile_handler(spec, scenarios[0].fen, args.threads)
        return

    for scenario in scenarios:
        print(f"\n=== Profiling {scenario.label} ({scenario.phase}) ===")
        spec.profile_handler(spec, scenario.fen, args.threads)


def list_scenarios() -> None:
    print("Available curated scenarios:")
    for scenario in CURATED_SCENARIOS:
        print(f"  {scenario.slug:<20} {scenario.phase:<12} {scenario.label}")


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    return float(mean(numeric))


def run_compare(args: argparse.Namespace) -> None:
    alias_a = args.engine_a
    alias_b = args.engine_b
    if alias_a == alias_b:
        raise SystemExit("Compare mode requires two distinct engine aliases.")

    spec_a = resolve_engine_alias(alias_a)
    spec_b = resolve_engine_alias(alias_b)

    scenarios = resolve_scenarios(
        args.positions, include_defaults=not args.skip_defaults
    )
    scenarios.extend(iter_user_fens(args))
    if not scenarios:
        raise SystemExit("No scenarios provided. Use --positions or --fen/--fen-file.")

    base_go_tokens = build_go_tokens(args)
    request_common = {
        "infinite_duration": args.infinite_duration,
        "max_wait": args.max_wait,
        "echo_info": args.echo_info,
    }

    session_a = EngineSession(command=spec_a.command)
    session_b = EngineSession(command=spec_b.command)

    session_a.start()
    session_b.start()

    label_a = f"{spec_a.display_name} ({alias_a})"
    label_b = f"{spec_b.display_name} ({alias_b})"

    metrics_a: List[EngineMetrics] = []
    metrics_b: List[EngineMetrics] = []
    time_ratios: List[float] = []
    node_ratios: List[float] = []
    depth_deltas: List[int] = []

    try:
        for scenario in scenarios:
            print(f"\n=== {scenario.label} ({scenario.phase}) ===")
            print(f"FEN: {scenario.fen}")

            if base_go_tokens:
                go_tokens = list(base_go_tokens)
            else:
                board = chess.Board(scenario.fen)
                movetime_ms = compute_adaptive_movetime(board)
                go_tokens = ["movetime", str(movetime_ms)]

            request = GoRequest(
                fen=scenario.fen,
                label=scenario.label,
                go_tokens=go_tokens,
                **request_common,
            )

            result_a = session_a.go(request)
            metrics_first = extract_metrics(result_a)
            metrics_a.append(metrics_first)
            print(describe_metrics(label_a, metrics_first))

            result_b = session_b.go(request)
            metrics_second = extract_metrics(result_b)
            metrics_b.append(metrics_second)
            print(describe_metrics(label_b, metrics_second))

            delta_line = summarise_deltas(
                metrics_first, metrics_second, alias_a, alias_b
            )
            if delta_line:
                print(delta_line)

            if (
                metrics_first.time_ms is not None
                and metrics_second.time_ms is not None
                and metrics_second.time_ms > 0
            ):
                time_ratios.append(metrics_first.time_ms / metrics_second.time_ms)
            if (
                metrics_first.nodes is not None
                and metrics_second.nodes is not None
                and metrics_second.nodes > 0
            ):
                node_ratios.append(metrics_first.nodes / metrics_second.nodes)
            if metrics_first.depth is not None and metrics_second.depth is not None:
                depth_deltas.append(metrics_first.depth - metrics_second.depth)
    finally:
        session_a.shutdown()
        session_b.shutdown()

    print("\n=== Aggregate ===")
    for alias, label, data in (
        (alias_a, label_a, metrics_a),
        (alias_b, label_b, metrics_b),
    ):
        depth_avg = _safe_mean([m.depth for m in data])
        time_avg = _safe_mean([m.time_ms for m in data])
        nodes_avg = _safe_mean([m.nodes for m in data])
        nps_avg = _safe_mean([m.nps for m in data])
        score_avg = _safe_mean([m.score_cp for m in data])
        parts = [f"{label:<30}"]
        if depth_avg is not None:
            parts.append(f"avg depth={depth_avg:.2f}")
        if time_avg is not None:
            parts.append(f"avg time={time_avg / 1000:.3f}s")
        if nodes_avg is not None:
            parts.append(f"avg nodes={int(nodes_avg):,}")
        if nps_avg is not None:
            parts.append(f"avg nps={int(nps_avg):,}")
        if score_avg is not None:
            parts.append(f"avg score={score_avg:.1f}cp")
        print("  " + " | ".join(parts))

    if time_ratios:
        print(f"  Average time ratio ({alias_a}/{alias_b}): {mean(time_ratios):.2f}x")
    if node_ratios:
        print(f"  Average node ratio ({alias_a}/{alias_b}): {mean(node_ratios):.2f}x")
    if depth_deltas:
        print(f"  Mean depth delta ({alias_a}-{alias_b}): {mean(depth_deltas):+.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark, profile, or compare registered JaskFish engines via UCI.",
        formatter_class=SmartFormatter,
        epilog=textwrap.dedent(format_engine_overview()),
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
    time_group.add_argument(
        "--movestogo", type=int, help="Moves to the next time control."
    )
    time_group.add_argument("--movetime", type=int, help="Fixed time per move in ms.")
    time_group.add_argument(
        "--nodes", type=int, help="Limit by number of searched nodes."
    )
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

    benchmark = subparsers.add_parser(
        "benchmark",
        parents=[common_benchmark],
        help="Run benchmark scenarios against a single engine.",
    )
    benchmark.add_argument(
        "--engine",
        type=validate_engine_alias,
        required=True,
        help="Engine alias to benchmark.",
    )
    benchmark.set_defaults(handler=run_benchmark)

    profile = subparsers.add_parser(
        "profile",
        parents=[common_benchmark],
        help="Collect profiling output using the engine's configured hook.",
    )
    profile.add_argument(
        "--engine",
        type=validate_engine_alias,
        required=True,
        help="Engine alias to profile.",
    )
    profile.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of worker threads for profiling backend (engine-specific).",
    )
    profile.set_defaults(handler=run_profile)

    compare = subparsers.add_parser(
        "compare",
        parents=[common_benchmark],
        help="Compare two engines on identical scenarios.",
    )
    compare.add_argument(
        "--engine-a",
        type=validate_engine_alias,
        required=True,
        help="Primary engine alias.",
    )
    compare.add_argument(
        "--engine-b",
        type=validate_engine_alias,
        required=True,
        help="Secondary engine alias.",
    )
    compare.set_defaults(handler=run_compare)

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
