# MAIN
import argparse
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
import chess
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

try:  # pragma: no cover - optional dependency for GUI mode
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover
    QProcess = None  # type: ignore[assignment]
    QApplication = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from PySide6.QtCore import QProcess as QProcessType
    from gui import ChessGUI  # pragma: no cover
else:  # pragma: no cover
    QProcessType = Any

import chess_logic
from utils import ReportingLevel
from utils import cleanup, debug_text, info_text, recieved_text, sending_text
from pvsengine import MetaRegistry, build_search_tuning

ENGINE_SPECS = {
    "engine1": {
        "number": 1,
        "default_script": "pvsengine.py",
        "default_name": "PyPVS",
        "preferred_color": chess.WHITE,
        "time_preset": "balanced",
    },
    "engine2": {
        "number": 2,
        "default_script": "native/cpvsengine_shim.py",
        "default_name": "cPVS",
        "preferred_color": chess.BLACK,
        "time_preset": "balanced",
    },
}
ENGINE_ID_ORDER = tuple(ENGINE_SPECS.keys())
COLOR_NAME = {chess.WHITE: "White", chess.BLACK: "Black"}


EngineProcess = Any
SendCommand = Callable[[EngineProcess, str], None]

DEFAULT_TIME_PRESET = "balanced"


def _mate_in_one_threat(board: chess.Board) -> bool:
    probe = board.copy(stack=False)
    try:
        probe.push(chess.Move.null())
    except ValueError:
        return False
    for move in probe.legal_moves:
        probe.push(move)
        try:
            if probe.is_checkmate():
                return True
        finally:
            probe.pop()
    return False


def compute_adaptive_movetime(
    board: chess.Board,
    *,
    meta_preset: str = DEFAULT_TIME_PRESET,
    time_controls: Optional[Dict[str, int]] = None,
) -> int:
    meta = MetaRegistry.resolve(meta_preset)
    _, limits = build_search_tuning(meta)
    min_time = float(limits.min_time)
    max_time = float(limits.max_time)
    base_time = float(limits.base_time)
    time_factor = float(limits.time_factor)

    tc = dict(time_controls) if time_controls else {}
    if tc.get("infinite"):
        return 0
    if "movetime" in tc:
        return max(1, int(tc["movetime"]))

    colour = board.turn
    time_key = "wtime" if colour == chess.WHITE else "btime"
    inc_key = "winc" if colour == chess.WHITE else "binc"
    if time_key in tc:
        time_left = max(0, int(tc.get(time_key, 0)))
        increment = max(0, int(tc.get(inc_key, 0)))
        moves_to_go = int(tc.get("movestogo", 0))
        if moves_to_go > 0:
            budget_ms = time_left // max(1, moves_to_go)
        else:
            budget_ms = int(time_left * time_factor)
        budget_ms += increment
        seconds = budget_ms / 1000.0
        clamped = max(min_time, min(max_time, seconds))
        return max(1, int(clamped * 1000))

    legal_moves = board.legal_moves.count()
    piece_count = len(board.piece_map())
    in_check = board.is_check()
    mate_threat = _mate_in_one_threat(board)

    phase_factor = 0.3 + min(1.0, max(0.0, (piece_count - 2) / 30.0)) * 0.7
    if legal_moves <= 10:
        complexity_factor = 0.25
    elif legal_moves <= 20:
        complexity_factor = 0.55
    elif legal_moves <= 35:
        complexity_factor = 1.10
    elif legal_moves <= 60:
        complexity_factor = 1.45
    else:
        complexity_factor = 1.9 + min(0.6, (legal_moves - 60) * 0.01)

    tension_factor = 1.35 if (in_check or mate_threat) else 1.0
    budget_seconds = base_time * complexity_factor * phase_factor * tension_factor
    budget_seconds = max(min_time, min(max_time, budget_seconds))

    return max(1, int(budget_seconds * 1000))


def build_go_command(
    board: chess.Board,
    *,
    meta_preset: str = DEFAULT_TIME_PRESET,
    time_controls: Optional[Dict[str, int]] = None,
) -> str:
    """Construct a UCI `go` command with an adaptive movetime budget."""
    movetime_ms = compute_adaptive_movetime(
        board,
        meta_preset=meta_preset,
        time_controls=time_controls,
    )
    if movetime_ms <= 0:
        return "go infinite"
    return f"go movetime {movetime_ms}"


class _SelfPlayUI(Protocol):
    """Protocol describing the UI hooks required by :class:`SelfPlayManager`."""

    board: chess.Board

    def set_self_play_active(self, active: bool) -> None:
        ...

    def set_board_interaction_enabled(self, enabled: bool) -> None:
        ...

    def set_manual_controls_enabled(self, enabled: bool) -> None:
        ...

    def set_info_message(self, message: str) -> None:
        ...

    def indicate_engine_activity(self, engine_label: str, context: str) -> None:
        ...

    def clear_engine_activity(self, message: Optional[str] = None) -> None:
        ...

    def self_play_evaluation_complete(self, engine_label: str) -> None:
        ...


class SelfPlayManager:
    """Coordinates automated play between engine processes."""

    def __init__(
        self,
        gui: _SelfPlayUI,
        engines: Dict[bool, EngineProcess],
        send_command: SendCommand,
        engine_names: Dict[bool, str],
        *,
        engine_time_presets: Optional[Dict[bool, str]] = None,
        capture_payload: bool = False,
        trace_directory: Optional[Union[Path, str]] = None,
    ) -> None:
        if chess.WHITE not in engines or chess.BLACK not in engines:
            raise ValueError("SelfPlayManager requires engines for both colors")

        self._gui = gui
        self._engines = engines
        self._send_command = send_command
        self._engine_names = engine_names
        default_presets = {
            chess.WHITE: DEFAULT_TIME_PRESET,
            chess.BLACK: DEFAULT_TIME_PRESET,
        }
        if engine_time_presets:
            for color, preset in engine_time_presets.items():
                if color in default_presets and isinstance(preset, str) and preset:
                    default_presets[color] = preset
        self._engine_time_presets = default_presets

        self._active = False
        self._current_engine_color: Optional[bool] = None
        self._pending_ignore_color: Optional[bool] = None
        self._waiting_for_move = False
        self._trace_directory = Path(trace_directory) if trace_directory else Path.cwd() / "self_play_traces"
        self._capture_payload = capture_payload
        self._session_traces: Dict[bool, List[str]] = {chess.WHITE: [], chess.BLACK: []}
        self._session_start_fen: Optional[str] = None
        self._session_started_at: Optional[datetime] = None
        self._last_trace_path: Optional[Path] = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def last_trace_path(self) -> Optional[Path]:
        return self._last_trace_path

    def current_expected_color(self) -> Optional[bool]:
        if not self._waiting_for_move:
            return None
        return self._current_engine_color

    def update_engines(
        self,
        engines: Dict[bool, EngineProcess],
        engine_names: Dict[bool, str],
        engine_time_presets: Optional[Dict[bool, str]] = None,
    ) -> None:
        if chess.WHITE not in engines or chess.BLACK not in engines:
            raise ValueError("SelfPlayManager requires engines for both colors")

        if self._active:
            self.stop("Self-play stopped (engine assignments updated)")

        self._engines = engines
        self._engine_names = engine_names
        if engine_time_presets:
            for color, preset in engine_time_presets.items():
                if color in self._engine_time_presets and isinstance(preset, str) and preset:
                    self._engine_time_presets[color] = preset
        self._current_engine_color = None
        self._pending_ignore_color = None
        self._waiting_for_move = False

    def start(self) -> bool:
        if self._active:
            return False

        self._active = True
        self._pending_ignore_color = None
        self._current_engine_color = None
        self._waiting_for_move = False
        self._session_traces = {chess.WHITE: [], chess.BLACK: []}
        self._session_start_fen = self._gui.board.fen()
        self._session_started_at = datetime.now(timezone.utc)
        self._last_trace_path = None

        self._gui.set_self_play_active(True)
        self._gui.set_board_interaction_enabled(False)
        self._gui.set_manual_controls_enabled(False)
        self._gui.set_info_message("Self-play running")

        unique_engines = []
        seen_ids = set()
        for engine in self._engines.values():
            engine_id = id(engine)
            if engine_id in seen_ids:
                continue
            seen_ids.add(engine_id)
            unique_engines.append(engine)

        for engine in unique_engines:
            self._dispatch(engine, "ucinewgame")

        self._request_move(self._gui.board.turn)
        return True

    def stop(self, message: Optional[str] = None) -> bool:
        if not self._active and not self._waiting_for_move:
            return False

        if self._waiting_for_move and self._current_engine_color is not None:
            engine = self._engines[self._current_engine_color]
            self._pending_ignore_color = self._current_engine_color
            self._dispatch(engine, "stop")

        self._active = False
        self._waiting_for_move = False
        self._current_engine_color = None

        self._gui.set_self_play_active(False)
        self._gui.set_board_interaction_enabled(True)
        self._gui.set_manual_controls_enabled(True)
        self._gui.clear_engine_activity(message or "Self-play stopped")
        self._export_traces(message)
        return True

    def should_apply_move(self, color: bool) -> bool:
        if self._pending_ignore_color is not None and color == self._pending_ignore_color:
            self._pending_ignore_color = None
            return False

        if not self._active:
            return True

        return color == self._current_engine_color

    def on_engine_move(self, color: bool, move_uci: str) -> None:
        if not self._active or color != self._current_engine_color:
            return

        self._waiting_for_move = False
        engine_label = self._engine_names.get(color, "Engine")
        self._gui.self_play_evaluation_complete(engine_label)

        if chess_logic.is_game_over(self._gui.board):
            outcome = chess_logic.get_game_result(self._gui.board)
            self.stop(message=f"Self-play finished: {outcome}")
            return

        next_color = not color
        self._request_move(next_color)

    def on_engine_output(self, color: bool, line: str) -> None:
        if not (self._active or self._waiting_for_move):
            return
        if not self._capture_payload and line.startswith("info string perf payload="):
            return
        if color not in self._session_traces:
            self._session_traces[color] = []
        self._session_traces[color].append(line)

    def _request_move(self, color: bool) -> None:
        if not self._active:
            return

        engine = self._engines[color]
        fen = self._gui.board.fen()
        self._dispatch(engine, f"position fen {fen}")
        preset = self._engine_time_presets.get(color, DEFAULT_TIME_PRESET)
        go_command = build_go_command(self._gui.board, meta_preset=preset)
        self._dispatch(engine, go_command)

        self._current_engine_color = color
        self._waiting_for_move = True
        engine_label = self._engine_names.get(color, "Engine")
        self._gui.indicate_engine_activity(engine_label, "Self-play")

    def _dispatch(self, engine: EngineProcess, command: str) -> None:
        self._send_command(engine, command)

    def _export_traces(self, stop_message: Optional[str]) -> None:
        has_trace = any(self._session_traces[color] for color in self._session_traces)
        if not has_trace:
            return
        try:
            self._trace_directory.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        timestamp = self._session_started_at or datetime.now(timezone.utc)
        iso_stamp = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        pattern = re.compile(r"^(\d+)_selfplay(?:\.[^.]+)?$")
        next_index = 1
        try:
            existing_indices = []
            for entry in self._trace_directory.iterdir():
                if not entry.is_file():
                    continue
                match = pattern.match(entry.name)
                if not match:
                    continue
                try:
                    existing_indices.append(int(match.group(1)))
                except ValueError:
                    continue
            if existing_indices:
                next_index = max(existing_indices) + 1
        except Exception:
            next_index = 1

        filename = f"{next_index}_selfplay.txt"
        path = self._trace_directory / filename
        header_lines = [
            f"Self-play trace recorded at {iso_stamp}",
            f"Initial FEN: {self._session_start_fen or 'unknown'}",
            f"Final FEN: {self._gui.board.fen()}",
        ]
        if stop_message:
            header_lines.append(f"Stop reason: {stop_message}")

        try:
            with path.open("w", encoding="utf-8") as trace_file:
                trace_file.write("\n".join(header_lines))
                trace_file.write("\n\n")
                for color in (chess.WHITE, chess.BLACK):
                    label = self._engine_names.get(color, "Engine")
                    trace_file.write(f"[{label}]\n")
                    for line in self._session_traces.get(color, []):
                        trace_file.write(f"  {line}\n")
                    trace_file.write("\n")
        except Exception:
            return

        self._last_trace_path = path
        self._session_traces = {chess.WHITE: [], chess.BLACK: []}
        self._session_started_at = None
        self._session_start_fen = None


class HeadlessEngineProcess:
    """Minimal subprocess wrapper for headless self-play."""

    def __init__(self, path: str, *, workdir: str) -> None:
        self.path = path
        self._proc = subprocess.Popen(
            [sys.executable, path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=workdir,
        )
        self._write_lock = threading.Lock()

    def send(self, command: str) -> None:
        with self._write_lock:
            if not self._proc.stdin:
                return
            try:
                self._proc.stdin.write(command + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                pass

    def readline(self) -> str:
        if not self._proc.stdout:
            return ""
        try:
            return self._proc.stdout.readline()
        except Exception:
            return ""

    def poll(self) -> Optional[int]:
        return self._proc.poll()

    def stop(self, timeout: float = 2.0) -> None:
        self.send("quit")
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                try:
                    self._proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass


class HeadlessSelfPlayUI:
    """Lightweight UI adapter that satisfies SelfPlayManager's expectations."""

    def __init__(self, board: chess.Board, *, quiet: bool = False) -> None:
        self.board = board
        self.quiet = quiet
        self.info_message: str = ""

    def _log(self, message: str) -> None:
        if not self.quiet and message:
            print(info_text(message))

    def set_self_play_active(self, active: bool) -> None:
        state = "Self-play started" if active else "Self-play inactive"
        self._log(state)

    def set_board_interaction_enabled(self, enabled: bool) -> None:
        # No-op for headless mode.
        pass

    def set_manual_controls_enabled(self, enabled: bool) -> None:
        # No-op for headless mode.
        pass

    def set_info_message(self, message: str) -> None:
        self.info_message = message
        self._log(message)

    def indicate_engine_activity(self, engine_label: str, context: str) -> None:
        self._log(f"{context}: {engine_label} evaluating")

    def clear_engine_activity(self, message: Optional[str] = None) -> None:
        if message:
            self.set_info_message(message)
        else:
            self._log("Engines idle")

    def self_play_evaluation_complete(self, engine_label: str) -> None:
        self._log(f"Self-play: {engine_label} move received")


def run_headless_self_play(args, script_dir: str) -> None:
    board = chess.Board(args.fen) if args.fen else chess.Board()
    quiet = bool(args.self_play_quiet)
    include_payload = bool(getattr(args, "include_perf_payload", False))

    ui = HeadlessSelfPlayUI(board, quiet=quiet)
    stop_event = threading.Event()
    board_lock = threading.Lock()

    def resolve_engine_path(default_script: str) -> str:
        path = os.path.join(script_dir, default_script)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Engine script not found: {path}")
        return path

    white_spec = ENGINE_SPECS["engine1"]
    black_spec = ENGINE_SPECS["engine2"]
    white_path = resolve_engine_path(white_spec["default_script"])
    black_path = resolve_engine_path(black_spec["default_script"])

    white_engine = HeadlessEngineProcess(white_path, workdir=script_dir)
    black_engine = HeadlessEngineProcess(black_path, workdir=script_dir)

    engines: Dict[bool, HeadlessEngineProcess] = {
        chess.WHITE: white_engine,
        chess.BLACK: black_engine,
    }

    engine_labels = {
        chess.WHITE: f"{COLOR_NAME[chess.WHITE]} - {white_spec['default_name']} [{white_spec['number']}]",
        chess.BLACK: f"{COLOR_NAME[chess.BLACK]} - {black_spec['default_name']} [{black_spec['number']}]",
    }
    monitor_labels = {
        white_engine: f"{white_spec['default_name']}[{white_spec['number']}][W]",
        black_engine: f"{black_spec['default_name']}[{black_spec['number']}][B]",
    }
    engine_time_presets = {
        chess.WHITE: white_spec.get("time_preset", DEFAULT_TIME_PRESET),
        chess.BLACK: black_spec.get("time_preset", DEFAULT_TIME_PRESET),
    }

    def send_command(engine: HeadlessEngineProcess, command: str) -> None:
        if not quiet:
            label = monitor_labels.get(engine, "Engine")
            print(sending_text(f"{label} {command}"))
        engine.send(command)

    manager = SelfPlayManager(
        ui,
        engines,
        send_command,
        engine_labels,
        engine_time_presets=engine_time_presets,
        capture_payload=include_payload,
    )

    def handle_bestmove(color: bool, move_line: str) -> Optional[str]:
        parts = move_line.strip().split()
        if len(parts) < 2:
            return None
        move_uci = parts[1]
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            if not quiet:
                label = engine_labels.get(color, "Engine")
                print(info_text(f"{label} produced invalid move: {move_uci}"))
            return None
        with board_lock:
            if move not in ui.board.legal_moves:
                if not quiet:
                    label = engine_labels.get(color, "Engine")
                    print(info_text(f"{label} move illegal in current position: {move_uci}"))
                return None
            ui.board.push(move)
        return move_uci

    def reader(color: bool, engine: HeadlessEngineProcess) -> None:
        label = monitor_labels.get(engine, "Engine")
        def emit(line: str) -> None:
            if quiet:
                return
            print(recieved_text(f"{label} {line}"))

        def bestmove_handler(line: str) -> Optional[str]:
            return handle_bestmove(color, line)

        while not stop_event.is_set():
            line = engine.readline()
            if line == "":
                if engine.poll() is not None:
                    if manager.active:
                        manager.stop(f"{label} terminated unexpectedly")
                    break
                time.sleep(0.01)
                continue
            line = line.strip()
            if not line:
                continue
            process_engine_output_line(
                line,
                expected_color=color,
                manager=manager,
                include_payload=include_payload,
                emit=emit,
                handle_bestmove=bestmove_handler,
            )

    threads = [
        threading.Thread(target=reader, args=(chess.WHITE, white_engine), daemon=True),
        threading.Thread(target=reader, args=(chess.BLACK, black_engine), daemon=True),
    ]
    for thread in threads:
        thread.start()

    try:
        started = manager.start()
        if not started and not quiet:
            print(info_text("Self-play already active; nothing to do"))
        while manager.active or manager.current_expected_color() is not None:
            time.sleep(0.05)
    except KeyboardInterrupt:
        manager.stop("Self-play interrupted by user")
        if not quiet:
            print(info_text("Self-play interrupted by user"))
    finally:
        stop_event.set()
        for engine in engines.values():
            engine.stop()
        for thread in threads:
            thread.join(timeout=1.0)

        trace_path = manager.last_trace_path
        if not quiet and trace_path:
            print(info_text(f"Self-play trace written -> {trace_path}"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fen", help="Set the initial board state to the given FEN string"
    )
    parser.add_argument("-dev", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--self-play",
        action="store_true",
        help="Run headless self-play and exit instead of launching the GUI",
    )
    parser.add_argument(
        "--self-play-quiet",
        action="store_true",
        help="Reduce console logging while running headless self-play",
    )
    parser.add_argument(
        "--include-perf-payload",
        action="store_true",
        help="Include perf payload JSON lines in terminal output and self-play traces",
    )
    return parser.parse_args()

def handle_bestmove_line(bestmove_line: str, gui: "ChessGUI", engine_label: str) -> Optional[str]:
    parts = bestmove_line.strip().split()
    if len(parts) >= 2:
        move_uci = parts[1]
        gui.attempt_engine_move(move_uci)
        return move_uci
    print(f"info string [{engine_label}] No best move found.")
    return None


def process_engine_output_line(
    line: str,
    *,
    expected_color: Optional[bool],
    manager: Optional[SelfPlayManager],
    include_payload: bool,
    emit: Callable[[str], None],
    handle_bestmove: Callable[[str], Optional[str]],
    manual_complete: Optional[Callable[[], None]] = None,
) -> None:
    if manager and expected_color is not None:
        manager.on_engine_output(expected_color, line)

    if line.startswith("bestmove"):
        if manager and expected_color is not None and not manager.should_apply_move(expected_color):
            return
        emit(line)
        move_uci = handle_bestmove(line)
        if move_uci and manager and expected_color is not None:
            manager.on_engine_move(expected_color, move_uci)
        if manual_complete:
            manual_complete()
        return

    if line.startswith("info string perf payload=") and not include_payload:
        return

    emit(line)


def engine_output_processor(
    proc: QProcessType,
    gui: "ChessGUI",
    *,
    engine_id: Optional[str],
    resolve_log_label,
    resolve_expected_color,
    resolve_color_label,
    resolve_monitor_label,
    self_play_manager: Optional[SelfPlayManager] = None,
    manual_pending: Optional[Dict[str, bool]] = None,
    manual_pending_color: Optional[Dict[str, Optional[bool]]] = None,
    include_payload: bool = False,
) -> None:
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if not output:
            continue

        expected_color = None
        if engine_id and resolve_expected_color is not None:
            expected_color = resolve_expected_color(engine_id)
        if (
            expected_color is None
            and manual_pending_color is not None
            and engine_id is not None
        ):
            expected_color = manual_pending_color.get(engine_id)

        engine_label = "Engine"
        monitor_label = "Engine"
        if engine_id:
            if expected_color is not None and resolve_color_label is not None:
                engine_label = resolve_color_label(engine_id, expected_color)
            elif resolve_log_label is not None:
                engine_label = resolve_log_label(engine_id)
            if resolve_monitor_label is not None:
                monitor_label = resolve_monitor_label(engine_id, expected_color)
            else:
                monitor_label = engine_label
        else:
            if resolve_monitor_label is not None:
                monitor_label = resolve_monitor_label(engine_id, expected_color)

        def emit(line: str) -> None:
            print(recieved_text(f"{monitor_label} {line}"))

        def bestmove_handler(line: str) -> Optional[str]:
            return handle_bestmove_line(line, gui, engine_label)

        manual_complete: Optional[Callable[[], None]] = None
        if manual_pending is not None and engine_id is not None:
            def _manual_complete() -> None:
                if manual_pending.get(engine_id):
                    manual_pending[engine_id] = False
                    if manual_pending_color is not None:
                        manual_pending_color[engine_id] = None
                    gui.manual_evaluation_complete(engine_label)

            manual_complete = _manual_complete

        process_engine_output_line(
            output,
            expected_color=expected_color,
            manager=self_play_manager,
            include_payload=include_payload,
            emit=emit,
            handle_bestmove=bestmove_handler,
            manual_complete=manual_complete,
        )


def start_engine_process(path: str) -> QProcessType:
    if QProcess is None:  # pragma: no cover - defensive
        raise ImportError("PySide6 is required to start GUI engine processes")
    proc = QProcess()
    proc.setProcessChannelMode(QProcess.MergedChannels)
    proc.start(sys.executable, [path])
    if not proc.waitForStarted(5000):
        print(f"info string Engine failed to start within timeout: {path}")
    return proc


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.self_play:
        run_headless_self_play(args, script_dir)
        return

    if QApplication is None or QProcess is None:
        raise ImportError("PySide6 is required for GUI mode; install PySide6 or use --self-play.")

    from gui import ChessGUI  # Local import to avoid PySide requirement for headless use

    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev
    reporting_level = ReportingLevel.BASIC if dev else ReportingLevel.QUIET
    include_perf_payload = bool(args.include_perf_payload)

    app = QApplication(sys.argv)

    color_assignments: Dict[bool, str] = {}

    engine_slots: Dict[str, Dict[str, object]] = {}
    for engine_id in ENGINE_ID_ORDER:
        spec = ENGINE_SPECS[engine_id]
        engine_path = os.path.join(script_dir, spec["default_script"])
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine script not found: {engine_path}")
        engine_name = spec.get("default_name") or f"Engine {spec['number']}"
        preferred_color = spec["preferred_color"]
        color_assignments.setdefault(preferred_color, engine_id)
        number = int(spec["number"])
        caption = f"{engine_name} [{number}]" if number else engine_name
        monitor_token = f"{engine_name}[{number}]" if number else engine_name
        engine_slots[engine_id] = {
            "id": engine_id,
            "name": engine_name,
            "number": number,
            "preferred_color": preferred_color,
            "time_preset": spec.get("time_preset", DEFAULT_TIME_PRESET),
            "default_script": spec["default_script"],
            "path": engine_path,
            "process": None,
            "active": True,
            "caption": caption,
            "monitor": monitor_token,
        }

    primary_engine = ENGINE_ID_ORDER[0]
    for color in (chess.WHITE, chess.BLACK):
        color_assignments.setdefault(color, primary_engine)

    def engine_caption(engine_id: str) -> str:
        slot = engine_slots.get(engine_id, {})
        caption = slot.get("caption")
        if isinstance(caption, str) and caption:
            return caption
        name = slot.get("name")
        number = slot.get("number")
        base_name = str(name) if name else engine_id
        if isinstance(number, int) and number:
            return f"{base_name} [{number}]"
        return base_name

    def engine_monitor_token(engine_id: str) -> str:
        slot = engine_slots.get(engine_id, {})
        token = slot.get("monitor")
        if isinstance(token, str) and token:
            return token
        name = slot.get("name")
        number = slot.get("number")
        base_name = str(name) if name else engine_id
        if isinstance(number, int) and number:
            return f"{base_name}[{number}]"
        return base_name

    def engine_monitor_label(engine_id: Optional[str], color: Optional[bool]) -> str:
        if not engine_id or engine_id not in engine_slots:
            return "Engine"
        token = engine_monitor_token(engine_id)
        if color is not None:
            suffix = "[W]" if color == chess.WHITE else "[B]"
            return f"{token}{suffix}"
        assigned = colors_for_engine(engine_id)
        if len(assigned) == 1:
            suffix = "[W]" if assigned[0] == chess.WHITE else "[B]"
            return f"{token}{suffix}"
        if len(assigned) > 1:
            return f"{token}[WB]"
        return token

    manual_pending: Dict[str, bool] = {engine_id: False for engine_id in ENGINE_ID_ORDER}
    manual_pending_color: Dict[str, Optional[bool]] = {
        engine_id: None for engine_id in ENGINE_ID_ORDER
    }
    process_slot_lookup: Dict[int, str] = {}

    gui: Optional["ChessGUI"] = None
    self_play_manager: Optional[SelfPlayManager] = None

    def colors_for_engine(engine_id: str) -> Tuple[bool, ...]:
        return tuple(
            color for color, assigned in color_assignments.items() if assigned == engine_id
        )

    def engine_log_label(engine_id: Optional[str]) -> str:
        if not engine_id or engine_id not in engine_slots:
            return "Engine"
        colors = colors_for_engine(engine_id)
        if not colors:
            return f"Inactive - {engine_caption(engine_id)}"
        color_text = "/".join(COLOR_NAME[color] for color in colors)
        return f"{color_text} - {engine_caption(engine_id)}"

    def color_specific_label(engine_id: str, color: bool) -> str:
        return f"{COLOR_NAME[color]} - {engine_caption(engine_id)}"

    def color_assignment_label(color: bool) -> str:
        engine_id = color_assignments[color]
        return engine_caption(engine_id)

    def manual_engine_provider() -> Tuple[str, str]:
        if gui is None:
            return ENGINE_ID_ORDER[0], engine_caption(ENGINE_ID_ORDER[0])
        turn_color = gui.board.turn
        engine_id = color_assignments.get(turn_color, ENGINE_ID_ORDER[0])
        slot = engine_slots.get(engine_id)
        if not slot or not slot.get("active"):
            fallback = next((eid for eid in ENGINE_ID_ORDER if engine_slots[eid].get("active")), ENGINE_ID_ORDER[0])
            engine_id = fallback
        assigned_colors = colors_for_engine(engine_id)
        if turn_color in assigned_colors:
            label_color = COLOR_NAME[turn_color]
        elif assigned_colors:
            label_color = "/".join(COLOR_NAME[color] for color in assigned_colors)
        else:
            label_color = "Engine"
        return engine_id, f"{label_color} - {engine_caption(engine_id)}"

    def build_self_play_engine_map() -> Dict[bool, QProcessType]:
        engines: Dict[bool, QProcessType] = {}
        for color in (chess.WHITE, chess.BLACK):
            engine_id = color_assignments[color]
            slot = engine_slots[engine_id]
            proc = slot.get("process")
            if proc is None:
                raise RuntimeError(f"No engine process assigned for {COLOR_NAME[color]}")
            engines[color] = proc  # type: ignore[assignment]
        return engines

    def build_self_play_time_presets() -> Dict[bool, str]:
        presets: Dict[bool, str] = {}
        for color in (chess.WHITE, chess.BLACK):
            engine_id = color_assignments[color]
            slot = engine_slots.get(engine_id, {})
            preset = slot.get("time_preset", DEFAULT_TIME_PRESET)
            presets[color] = preset if isinstance(preset, str) and preset else DEFAULT_TIME_PRESET
        return presets

    def rebuild_engine_labels_by_color() -> Dict[bool, str]:
        return {
            chess.WHITE: f"{COLOR_NAME[chess.WHITE]} - {color_assignment_label(chess.WHITE)}",
            chess.BLACK: f"{COLOR_NAME[chess.BLACK]} - {color_assignment_label(chess.BLACK)}",
        }

    def active_engine_ids() -> Tuple[str, ...]:
        return tuple(
            engine_id for engine_id, slot in engine_slots.items() if slot.get("active")
        )

    def can_swap_colors() -> bool:
        return color_assignments[chess.WHITE] != color_assignments[chess.BLACK]

    def send_command_with_lookup(proc: QProcessType, command: str) -> None:
        engine_id = process_slot_lookup.get(id(proc))
        monitor_label = engine_monitor_label(engine_id, None)
        print(sending_text(f"{monitor_label} {command}"))
        proc.write((command + "\n").encode())
        proc.waitForBytesWritten()

    def resolve_expected_color(engine_id: str) -> Optional[bool]:
        if not self_play_manager:
            return None
        expected = self_play_manager.current_expected_color()
        if expected is None:
            return None
        if color_assignments.get(expected) == engine_id:
            return expected
        return None

    def attach_engine_output(engine_id: str, proc: QProcessType) -> None:
        if gui is None:
            return
        proc.readyReadStandardOutput.connect(
            lambda eid=engine_id, process=proc: engine_output_processor(
                process,
                gui,
                engine_id=eid,
                resolve_log_label=engine_log_label,
                resolve_expected_color=resolve_expected_color,
                resolve_color_label=color_specific_label,
                resolve_monitor_label=engine_monitor_label,
                self_play_manager=self_play_manager,
                manual_pending=manual_pending,
                manual_pending_color=manual_pending_color,
                include_payload=include_perf_payload,
            )
        )

    def start_engine_instance(engine_id: str) -> QProcessType:
        slot = engine_slots[engine_id]
        process = start_engine_process(slot["path"])
        slot["process"] = process
        slot["active"] = process.state() == QProcess.Running
        process_slot_lookup[id(process)] = engine_id
        attach_engine_output(engine_id, process)
        print(info_text(f"{engine_log_label(engine_id)} -> {slot['path']}"))
        return process

    def stop_process(proc: QProcessType) -> None:
        try:
            if proc.state() != QProcess.NotRunning:
                send_command_with_lookup(proc, "quit")
                proc.closeWriteChannel()
                if not proc.waitForFinished(3000):
                    proc.terminate()
                    if not proc.waitForFinished(2000):
                        proc.kill()
                        proc.waitForFinished(1000)
        except Exception:
            pass
        try:
            proc.readyReadStandardOutput.disconnect()
        except Exception:
            pass
        process_slot_lookup.pop(id(proc), None)

    def manual_go_callback(engine_id: str, engine_label: str, fen_string: str) -> None:
        if gui is None:
            return
        slot = engine_slots[engine_id]
        proc = slot.get("process")
        if not slot.get("active") or proc is None or proc.state() != QProcess.Running:
            manual_pending[engine_id] = False
            manual_pending_color[engine_id] = None
            gui.manual_engine_busy = False
            gui.set_manual_controls_enabled(True)
            gui.clear_engine_activity(f"{engine_label} is not running")
            return

        manual_pending_color[engine_id] = gui.board.turn
        manual_pending[engine_id] = True
        print(info_text(f"Manual evaluation started using {engine_label}"))
        send_command_with_lookup(proc, f"position fen {fen_string}")
        preset = slot.get("time_preset", DEFAULT_TIME_PRESET)
        if not isinstance(preset, str) or not preset:
            preset = DEFAULT_TIME_PRESET
        go_command = build_go_command(gui.board, meta_preset=preset)
        send_command_with_lookup(proc, go_command)

    def manual_ready_callback(engine_id: str, engine_label: str) -> None:
        if gui is None:
            return
        slot = engine_slots[engine_id]
        proc = slot.get("process")
        if not slot.get("active") or proc is None or proc.state() != QProcess.Running:
            gui.set_info_message(f"{engine_label} is not running")
            return
        print(info_text(f"Readiness check using {engine_label}"))
        send_command_with_lookup(proc, "isready")

    gui = ChessGUI(
        board,
        dev=dev,
        reporting_level=reporting_level,
        go_callback=manual_go_callback,
        ready_callback=manual_ready_callback,
        restart_engine_callback=None,
        swap_colors_callback=None,
        toggle_engine_callback=None,
    )

    for engine_id in ENGINE_ID_ORDER:
        start_engine_instance(engine_id)

    active_defaults = [
        engine_id for engine_id in ENGINE_ID_ORDER if engine_slots[engine_id].get("active")
    ]
    if not active_defaults:
        raise RuntimeError("No engines are available to run")
    for color, engine_id in list(color_assignments.items()):
        if not engine_slots[engine_id].get("active"):
            color_assignments[color] = active_defaults[0]

    engine_labels_by_color = rebuild_engine_labels_by_color()
    white_label = engine_labels_by_color[chess.WHITE]
    black_label = engine_labels_by_color[chess.BLACK]
    print(info_text(f"Engine assignments -> {white_label}; {black_label}"))

    self_play_manager = SelfPlayManager(
        gui,
        build_self_play_engine_map(),
        send_command_with_lookup,
        engine_labels_by_color.copy(),
        engine_time_presets=build_self_play_time_presets(),
        capture_payload=include_perf_payload,
    )

    def refresh_engine_configuration(info_message: Optional[str] = None) -> None:
        nonlocal engine_labels_by_color
        engine_labels_by_color = rebuild_engine_labels_by_color()
        if self_play_manager:
            self_play_manager.update_engines(
                build_self_play_engine_map(),
                engine_labels_by_color.copy(),
                build_self_play_time_presets(),
            )
        if gui:
            gui.set_engine_labels(
                {engine_id: engine_caption(engine_id) for engine_id in ENGINE_ID_ORDER}
            )
            gui.set_engine_activation_states(
                {engine_id: engine_slots[engine_id].get("active", False) for engine_id in ENGINE_ID_ORDER}
            )
            gui.set_swap_button_enabled(can_swap_colors())
            gui.set_engine_assignments(
                engine_caption(color_assignments[chess.WHITE]),
                engine_caption(color_assignments[chess.BLACK]),
            )
            gui.set_manual_engine_provider(manual_engine_provider)
            if info_message:
                gui.set_info_message(info_message)
        white = engine_labels_by_color[chess.WHITE]
        black = engine_labels_by_color[chess.BLACK]
        print(info_text(f"Engine assignments -> {white}; {black}"))

    def start_self_play() -> bool:
        if not self_play_manager:
            return False
        for color in (chess.WHITE, chess.BLACK):
            engine_id = color_assignments[color]
            slot = engine_slots[engine_id]
            proc = slot.get("process")
            if not slot.get("active") or proc is None or proc.state() != QProcess.Running:
                message = f"{engine_log_label(engine_id)} is not running"
                gui.set_info_message(message)
                print(info_text(message))
                return False
        started = self_play_manager.start()
        if started:
            print(
                info_text(
                    f"Self-play started: {engine_labels_by_color[chess.WHITE]} vs {engine_labels_by_color[chess.BLACK]}"
                )
            )
        return started

    def stop_self_play(message: str = "Self-play stopped by user") -> bool:
        if not self_play_manager:
            return False
        stopped = self_play_manager.stop(message)
        if stopped:
            print(info_text("Self-play stopped"))
        return stopped

    def swap_engine_colors() -> None:
        if not can_swap_colors():
            gui.set_info_message("Activate both engines to swap colors")
            return
        stop_self_play("Self-play stopped (engine colors swapped)")
        color_assignments[chess.WHITE], color_assignments[chess.BLACK] = (
            color_assignments[chess.BLACK],
            color_assignments[chess.WHITE],
        )
        refresh_engine_configuration("Engine colors swapped")

    def toggle_engine_activation(engine_id: str, activate: bool) -> None:
        slot = engine_slots[engine_id]
        if activate:
            if slot.get("active") and slot.get("process") and slot["process"].state() == QProcess.Running:
                gui.set_info_message(f"{engine_log_label(engine_id)} is already active")
                return
            process = start_engine_instance(engine_id)
            if process.state() != QProcess.Running:
                slot["active"] = False
                gui.set_info_message(f"{engine_log_label(engine_id)} failed to start")
                return
            slot["active"] = True
            preferred = slot["preferred_color"]
            color_assignments[preferred] = engine_id
            refresh_engine_configuration(f"{engine_caption(engine_id)} activated")
        else:
            remaining = [eid for eid in active_engine_ids() if eid != engine_id]
            if not remaining:
                gui.set_info_message("Cannot deactivate the last active engine")
                return
            stop_self_play("Self-play stopped (engine deactivated)")
            if manual_pending.get(engine_id):
                manual_pending[engine_id] = False
                manual_pending_color[engine_id] = None
                gui.manual_engine_busy = False
                gui.set_manual_controls_enabled(True)
                gui.clear_engine_activity(f"Manual evaluation canceled: {engine_log_label(engine_id)}")
            proc = slot.get("process")
            if proc is not None:
                stop_process(proc)
                proc.deleteLater()
            slot["process"] = None
            slot["active"] = False
            replacement = remaining[0]
            for color, assigned in list(color_assignments.items()):
                if assigned == engine_id:
                    color_assignments[color] = replacement
            refresh_engine_configuration(f"{engine_caption(engine_id)} deactivated")

    def restart_engines() -> None:
        print(info_text("Restarting engines..."))
        stop_self_play("Self-play stopped (engines restarting)")
        if gui:
            gui.manual_engine_busy = False
            gui.clear_engine_activity("Restarting engines")

        for engine_id, slot in engine_slots.items():
            if not slot.get("active"):
                continue
            manual_pending[engine_id] = False
            manual_pending_color[engine_id] = None
            proc = slot.get("process")
            if proc is not None:
                stop_process(proc)
                proc.deleteLater()
            new_proc = start_engine_instance(engine_id)
            if new_proc.state() != QProcess.Running:
                gui.set_info_message(f"{engine_log_label(engine_id)} failed to restart")
        refresh_engine_configuration("Engines restarted")

    gui.set_self_play_callbacks(start_self_play, stop_self_play)
    gui.restart_engine_callback = restart_engines
    gui.set_engine_management_callbacks(swap_engine_colors, toggle_engine_activation)
    refresh_engine_configuration()

    def shutdown():
        if self_play_manager:
            self_play_manager.stop()
        for engine_id, slot in engine_slots.items():
            proc = slot.get("process")
            if proc is None:
                continue
            label = engine_log_label(engine_id)
            if proc.state() != QProcess.NotRunning:
                try:
                    send_command_with_lookup(proc, "quit")
                    proc.closeWriteChannel()
                except Exception as exc:  # pragma: no cover - defensive logging
                    if dev:
                        print(debug_text(f"Failed to send quit to {label}: {exc}"))
            cleanup(proc, None, app, dev=dev, quit_app=False)
            process_slot_lookup.pop(id(proc), None)
            proc.deleteLater()
            slot["process"] = None

    app.aboutToQuit.connect(shutdown)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
