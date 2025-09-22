# MAIN
import argparse
import os
import sys
from typing import Dict, Optional

import chess
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication

from gui import ChessGUI
from self_play import SelfPlayManager
from utils import cleanup, debug_text, info_text, recieved_text, sending_text

ENGINE_ID_ORDER = ("engine1", "engine2")
ENGINE_ID_TO_NUMBER = {"engine1": 1, "engine2": 2}
ENGINE_COLOR_ASSIGNMENT = {
    "engine1": chess.WHITE,
    "engine2": chess.BLACK,
}
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


def build_engine_labels(engine_id: str, friendly_name: Optional[str], color: bool) -> Dict[str, str]:
    number = ENGINE_ID_TO_NUMBER[engine_id]
    base = f"Engine {number}"
    if friendly_name:
        base = f"{base} – {friendly_name}"
    color_text = COLOR_NAME[color]
    display_label = f"{base} – {color_text}"
    log_label = f"{base} [{color_text}]"
    return {"display": display_label, "log": log_label, "base": base}


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
    engine_label: str,
    engine_color: Optional[bool],
    self_play_manager: Optional[SelfPlayManager] = None,
    manual_pending: Optional[Dict[str, bool]] = None,
) -> None:
    while proc.canReadLine():
        output = bytes(proc.readLine()).decode().strip()
        if not output:
            continue

        if output.startswith("bestmove"):
            if self_play_manager and engine_color is not None:
                if not self_play_manager.should_apply_move(engine_color):
                    continue

            print(recieved_text(f"[{engine_label}] {output}"))
            move_uci = handle_bestmove_line(output, gui, engine_label)
            if move_uci and self_play_manager and engine_color is not None:
                self_play_manager.on_engine_move(engine_color, move_uci)

            if manual_pending is not None and engine_id is not None:
                if manual_pending.get(engine_id):
                    manual_pending[engine_id] = False
                    gui.manual_evaluation_complete(engine_label)
        else:
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

    engine_slots: Dict[str, Dict[str, object]] = {}
    process_slot_lookup: Dict[int, Dict[str, object]] = {}

    def send_command_with_lookup(proc: QProcess, command: str) -> None:
        slot = process_slot_lookup.get(id(proc))
        label = slot.get("log_label", "Engine") if slot else "Engine"
        print(sending_text(f"[{label}] {command}"))
        proc.write((command + "\n").encode())
        proc.waitForBytesWritten()

    manual_pending: Dict[str, bool] = {engine_id: False for engine_id in ENGINE_ID_ORDER}

    for engine_id in ENGINE_ID_ORDER:
        override_path = getattr(args, engine_id)
        friendly_name = getattr(args, f"{engine_id}_name")
        color = ENGINE_COLOR_ASSIGNMENT[engine_id]

        engine_path = resolve_engine_path(default_engine_path, override_path)
        labels = build_engine_labels(engine_id, friendly_name, color)
        process = start_engine_process(engine_path)

        slot = {
            "id": engine_id,
            "path": engine_path,
            "process": process,
            "color": color,
            "display_label": labels["display"],
            "log_label": labels["log"],
        }
        engine_slots[engine_id] = slot
        process_slot_lookup[id(process)] = slot

        if dev and process.state() == QProcess.Running:
            send_command_with_lookup(process, "debug on")

    engine_labels_by_color = {
        slot["color"]: slot["log_label"] for slot in engine_slots.values()
    }

    white_label = engine_labels_by_color[chess.WHITE]
    black_label = engine_labels_by_color[chess.BLACK]
    print(info_text(f"Engine assignments -> White: {white_label}; Black: {black_label}"))

    def manual_go_callback(engine_id: str, engine_label: str, fen_string: str) -> None:
        slot = engine_slots[engine_id]
        proc: QProcess = slot["process"]  # type: ignore[index]
        if proc.state() != QProcess.Running:
            manual_pending[engine_id] = False
            gui.manual_engine_busy = False
            gui.set_manual_controls_enabled(True)
            gui.clear_engine_activity(f"{slot['log_label']} is not running")
            return

        manual_pending[engine_id] = True
        print(info_text(f"Manual evaluation started using {slot['log_label']}"))
        send_command_with_lookup(proc, f"position fen {fen_string}")
        send_command_with_lookup(proc, "go")

    def manual_ready_callback(engine_id: str, engine_label: str) -> None:
        slot = engine_slots[engine_id]
        proc: QProcess = slot["process"]  # type: ignore[index]
        if proc.state() != QProcess.Running:
            gui.set_info_message(f"{slot['log_label']} is not running")
            return
        print(info_text(f"Readiness check using {slot['log_label']}"))
        send_command_with_lookup(proc, "isready")

    gui = ChessGUI(
        board,
        dev=dev,
        go_callback=manual_go_callback,
        ready_callback=manual_ready_callback,
        restart_engine_callback=None,
    )

    gui.set_manual_engine_options(
        [
            (engine_id, engine_slots[engine_id]["display_label"])
            for engine_id in ENGINE_ID_ORDER
            if engine_id in engine_slots
        ]
    )

    self_play_manager = SelfPlayManager(
        gui,
        {
            slot["color"]: slot["process"]  # type: ignore[index]
            for slot in engine_slots.values()
        },
        send_command_with_lookup,
        engine_labels_by_color,
    )

    def start_self_play() -> bool:
        started = self_play_manager.start()
        if started:
            print(
                info_text(
                    f"Self-play started: {engine_labels_by_color[chess.WHITE]} vs {engine_labels_by_color[chess.BLACK]}"
                )
            )
        return started

    def stop_self_play() -> bool:
        stopped = self_play_manager.stop("Self-play stopped by user")
        if stopped:
            print(info_text("Self-play stopped"))
        return stopped

    gui.set_self_play_callbacks(start_self_play, stop_self_play)

    def restart_engines() -> None:
        print(info_text("Restarting engines..."))
        self_play_manager.stop(message="Self-play stopped (engines restarting)")
        gui.manual_engine_busy = False
        gui.clear_engine_activity("Restarting engines")

        for engine_id, slot in engine_slots.items():
            manual_pending[engine_id] = False
            proc: QProcess = slot["process"]  # type: ignore[index]
            if proc.state() == QProcess.Running:
                proc.kill()
                proc.waitForFinished()

            proc.start(sys.executable, [slot["path"]])
            if not proc.waitForStarted(5000):
                message = f"{slot['log_label']} failed to restart"
                print(f"info string {message}")
                gui.set_info_message(message)
            elif dev:
                send_command_with_lookup(proc, "debug on")

        gui.set_manual_controls_enabled(True)
        gui.set_info_message("Engines restarted")
        print(info_text("Engines restarted"))

    gui.restart_engine_callback = restart_engines

    for engine_id in ENGINE_ID_ORDER:
        slot = engine_slots[engine_id]
        proc: QProcess = slot["process"]  # type: ignore[index]
        engine_color = slot["color"]
        engine_label = slot["log_label"]

        proc.readyReadStandardOutput.connect(
            lambda eid=engine_id, ecol=engine_color, elabel=engine_label: engine_output_processor(
                engine_slots[eid]["process"],  # type: ignore[index]
                gui,
                engine_id=eid,
                engine_label=elabel,
                engine_color=ecol,
                self_play_manager=self_play_manager,
                manual_pending=manual_pending,
            )
        )

    def shutdown():
        self_play_manager.stop()
        for slot in engine_slots.values():
            proc: QProcess = slot["process"]  # type: ignore[index]
            label = slot["log_label"]
            if proc.state() != QProcess.NotRunning:
                try:
                    send_command_with_lookup(proc, "quit")
                    proc.closeWriteChannel()
                except Exception as exc:  # pragma: no cover - defensive logging
                    if dev:
                        print(debug_text(f"Failed to send quit to {label}: {exc}"))
            cleanup(proc, None, app, dev=dev, quit_app=False)

    app.aboutToQuit.connect(shutdown)

    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
