# MAIN
import argparse
import os
import sys
from typing import Dict, Optional, Tuple

import chess
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from gui import ChessGUI
from self_play import SelfPlayManager
from utils import cleanup, debug_text, info_text, recieved_text, sending_text

ENGINE_ID_ORDER = ("engine1", "engine2")
ENGINE_ID_TO_NUMBER = {"engine1": 1, "engine2": 2}
ENGINE_PREFERRED_COLORS = {"engine1": chess.WHITE, "engine2": chess.BLACK}
COLOR_NAME = {chess.WHITE: "White", chess.BLACK: "Black"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fen", help="Set the initial board state to the given FEN string"
    )
    parser.add_argument("-dev", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--engine1",
        dest="engine1",
        help="Path to the engine executable/script used for Engine 1",
    )
    parser.add_argument(
        "--engine2",
        dest="engine2",
        help="Path to the engine executable/script used for Engine 2",
    )
    parser.add_argument(
        "--engine1-name",
        dest="engine1_name",
        help="Display label for Engine 1",
    )
    parser.add_argument(
        "--engine2-name",
        dest="engine2_name",
        help="Display label for Engine 2",
    )
    return parser.parse_args()


def resolve_engine_path(default_path: str, override: Optional[str]) -> str:
    if not override:
        return default_path
    candidate = os.path.abspath(override)
    if not os.path.exists(candidate):
        print(info_text(f"Engine path not found: {candidate}. Falling back to {default_path}"))
        return default_path
    return candidate


def build_engine_base_label(engine_id: str, friendly_name: Optional[str]) -> str:
    number = ENGINE_ID_TO_NUMBER[engine_id]
    base = f"Engine {number}"
    if friendly_name:
        base = f"{base} – {friendly_name}"
    return base


def handle_bestmove_line(bestmove_line: str, gui: ChessGUI, engine_label: str) -> Optional[str]:
    parts = bestmove_line.strip().split()
    if len(parts) >= 2:
        move_uci = parts[1]
        gui.attempt_engine_move(move_uci)
        return move_uci
    print(f"info string [{engine_label}] No best move found.")
    return None


def engine_output_processor(
    proc: QProcess,
    gui: ChessGUI,
    *,
    engine_id: Optional[str],
    resolve_log_label,
    resolve_expected_color,
    resolve_color_label,
    self_play_manager: Optional[SelfPlayManager] = None,
    manual_pending: Optional[Dict[str, bool]] = None,
    manual_pending_color: Optional[Dict[str, Optional[bool]]] = None,
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
        if engine_id:
            if expected_color is not None and resolve_color_label is not None:
                engine_label = resolve_color_label(engine_id, expected_color)
            elif resolve_log_label is not None:
                engine_label = resolve_log_label(engine_id)

        if output.startswith("bestmove"):
            if self_play_manager and expected_color is not None:
                self_play_manager.on_engine_output(expected_color, output)
            if self_play_manager and expected_color is not None:
                if not self_play_manager.should_apply_move(expected_color):
                    continue

            print(recieved_text(f"[{engine_label}] {output}"))
            move_uci = handle_bestmove_line(output, gui, engine_label)
            if move_uci and self_play_manager and expected_color is not None:
                self_play_manager.on_engine_move(expected_color, move_uci)

            if manual_pending is not None and engine_id is not None:
                if manual_pending.get(engine_id):
                    manual_pending[engine_id] = False
                    if manual_pending_color is not None:
                        manual_pending_color[engine_id] = None
                    gui.manual_evaluation_complete(engine_label)
        else:
            if self_play_manager and expected_color is not None:
                self_play_manager.on_engine_output(expected_color, output)
            print(recieved_text(f"[{engine_label}] {output}"))


def start_engine_process(path: str) -> QProcess:
    proc = QProcess()
    proc.setProcessChannelMode(QProcess.MergedChannels)
    proc.start(sys.executable, [path])
    if not proc.waitForStarted(5000):
        print(f"info string Engine failed to start within timeout: {path}")
    return proc


def main():
    args = parse_args()
    board = chess.Board() if not args.fen else chess.Board(args.fen)
    dev = not args.dev

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_engine_path = os.path.join(script_dir, "engine.py")

    app = QApplication(sys.argv)

    color_assignments: Dict[bool, str] = {
        chess.WHITE: "engine1",
        chess.BLACK: "engine2",
    }

    engine_slots: Dict[str, Dict[str, object]] = {}
    for engine_id in ENGINE_ID_ORDER:
        override_path = getattr(args, engine_id)
        friendly_name = getattr(args, f"{engine_id}_name")
        engine_path = resolve_engine_path(default_engine_path, override_path)
        engine_slots[engine_id] = {
            "id": engine_id,
            "path": engine_path,
            "process": None,
            "active": True,
            "base_label": build_engine_base_label(engine_id, friendly_name),
            "preferred_color": ENGINE_PREFERRED_COLORS.get(engine_id, chess.WHITE),
        }

    manual_pending: Dict[str, bool] = {engine_id: False for engine_id in ENGINE_ID_ORDER}
    manual_pending_color: Dict[str, Optional[bool]] = {
        engine_id: None for engine_id in ENGINE_ID_ORDER
    }
    process_slot_lookup: Dict[int, str] = {}

    gui: Optional[ChessGUI] = None
    self_play_manager: Optional[SelfPlayManager] = None

    def colors_for_engine(engine_id: str) -> Tuple[bool, ...]:
        return tuple(
            color for color, assigned in color_assignments.items() if assigned == engine_id
        )

    def engine_number_label(engine_id: str) -> str:
        number = ENGINE_ID_TO_NUMBER.get(engine_id, 0)
        return f"Engine {number}" if number else "Engine"

    def engine_log_label(engine_id: Optional[str]) -> str:
        if not engine_id or engine_id not in engine_slots:
            return "Engine"
        colors = colors_for_engine(engine_id)
        if not colors:
            return f"Inactive - {engine_number_label(engine_id)}"
        color_text = "/".join(COLOR_NAME[color] for color in colors)
        return f"{color_text} - {engine_number_label(engine_id)}"

    def color_specific_label(engine_id: str, color: bool) -> str:
        return f"{COLOR_NAME[color]} - {engine_number_label(engine_id)}"

    def color_assignment_label(color: bool) -> str:
        engine_id = color_assignments[color]
        return engine_number_label(engine_id)

    def manual_engine_provider() -> Tuple[str, str]:
        if gui is None:
            return ENGINE_ID_ORDER[0], engine_number_label(ENGINE_ID_ORDER[0])
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
        return engine_id, f"{label_color} - {engine_number_label(engine_id)}"

    def build_self_play_engine_map() -> Dict[bool, QProcess]:
        engines: Dict[bool, QProcess] = {}
        for color in (chess.WHITE, chess.BLACK):
            engine_id = color_assignments[color]
            slot = engine_slots[engine_id]
            proc = slot.get("process")
            if proc is None:
                raise RuntimeError(f"No engine process assigned for {COLOR_NAME[color]}")
            engines[color] = proc  # type: ignore[assignment]
        return engines

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

    def send_command_with_lookup(proc: QProcess, command: str) -> None:
        engine_id = process_slot_lookup.get(id(proc))
        label = engine_log_label(engine_id)
        print(sending_text(f"[{label}] {command}"))
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

    def attach_engine_output(engine_id: str, proc: QProcess) -> None:
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
                self_play_manager=self_play_manager,
                manual_pending=manual_pending,
                manual_pending_color=manual_pending_color,
            )
        )

    def start_engine_instance(engine_id: str) -> QProcess:
        slot = engine_slots[engine_id]
        process = start_engine_process(slot["path"])
        slot["process"] = process
        slot["active"] = process.state() == QProcess.Running
        process_slot_lookup[id(process)] = engine_id
        attach_engine_output(engine_id, process)
        if process.state() == QProcess.Running and dev:
            send_command_with_lookup(process, "debug on")
        print(info_text(f"{engine_log_label(engine_id)} -> {slot['path']}"))
        return process

    def stop_process(proc: QProcess) -> None:
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
        send_command_with_lookup(proc, "go")

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
    )

    def refresh_engine_configuration(info_message: Optional[str] = None) -> None:
        nonlocal engine_labels_by_color
        engine_labels_by_color = rebuild_engine_labels_by_color()
        if self_play_manager:
            self_play_manager.update_engines(
                build_self_play_engine_map(),
                engine_labels_by_color.copy(),
            )
        if gui:
            gui.set_engine_activation_states(
                {engine_id: engine_slots[engine_id].get("active", False) for engine_id in ENGINE_ID_ORDER}
            )
            gui.set_swap_button_enabled(can_swap_colors())
            gui.set_engine_assignments(
                engine_number_label(color_assignments[chess.WHITE]),
                engine_number_label(color_assignments[chess.BLACK]),
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
            refresh_engine_configuration(f"{slot['base_label']} activated")
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
            refresh_engine_configuration(f"{slot['base_label']} deactivated")

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
