"""Self-play orchestration for engine-versus-engine games."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Protocol

import chess

try:
    from src import chess_logic
except ImportError:  # pragma: no cover - fallback when executed as a script
    import chess_logic  # type: ignore


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


EngineProcess = Any
SendCommand = Callable[[EngineProcess, str], None]


class SelfPlayManager:
    """Coordinates automated play between engine processes."""

    def __init__(
        self,
        gui: _SelfPlayUI,
        engines: Dict[bool, EngineProcess],
        send_command: SendCommand,
        engine_names: Dict[bool, str],
    ) -> None:
        if chess.WHITE not in engines or chess.BLACK not in engines:
            raise ValueError("SelfPlayManager requires engines for both colors")

        self._gui = gui
        self._engines = engines
        self._send_command = send_command
        self._engine_names = engine_names

        self._active = False
        self._current_engine_color: Optional[bool] = None
        self._pending_ignore_color: Optional[bool] = None
        self._waiting_for_move = False

    @property
    def active(self) -> bool:
        return self._active

    def current_expected_color(self) -> Optional[bool]:
        if not self._waiting_for_move:
            return None
        return self._current_engine_color

    def update_engines(
        self, engines: Dict[bool, EngineProcess], engine_names: Dict[bool, str]
    ) -> None:
        if chess.WHITE not in engines or chess.BLACK not in engines:
            raise ValueError("SelfPlayManager requires engines for both colors")

        if self._active:
            self.stop("Self-play stopped (engine assignments updated)")

        self._engines = engines
        self._engine_names = engine_names
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

    def _request_move(self, color: bool) -> None:
        if not self._active:
            return

        engine = self._engines[color]
        fen = self._gui.board.fen()
        self._dispatch(engine, f"position fen {fen}")
        self._dispatch(engine, "go")

        self._current_engine_color = color
        self._waiting_for_move = True
        engine_label = self._engine_names.get(color, "Engine")
        self._gui.indicate_engine_activity(engine_label, "Self-play")

    def _dispatch(self, engine: EngineProcess, command: str) -> None:
        self._send_command(engine, command)
