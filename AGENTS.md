# JaskFish Agent Handbook

This document captures how the engine, GUI, and tooling in JaskFish fit together so an agent can reason about the codebase without spelunking through every module. It emphasises runtime interactions, default behaviours, and the test harnesses that are safe to trigger headlessly.

## 1. Architectural Overview
- **Engine core (`src/engine.py`)** – single-process UCI engine with a strategy stack. It owns the `chess.Board`, exposes handlers for the usual commands (`uci`, `isready`, `position`, `go`, `ucinewgame`, `quit`), and performs move search on a background thread so stdin stays responsive.
- **GUI shell (`src/gui.py`)** – PySide6 application responsible for board rendering, user controls, and delegating engine requests. It never evaluates positions directly; instead, it relays commands to engine processes and reacts to their output.
- **Launcher (`src/main.py`)** – boots two `QProcess` instances running `src/engine.py`, funnels console IO, and glues the GUI, self-play controls, and debugging options into a single executable.
- **Self-play coordination (`src/self_play.py`)** – `SelfPlayManager` abstracts the “issue `ucinewgame` → `position` → `go`” protocol for any UI that can expose a minimal callback interface. Both the GUI and the dev-only headless harness use the same manager, so behaviour matches across environments.
- **Shared helpers**
  - `src/chess_logic.py` supplies board utilities (move validation, repetition/fivefold detection, exporting PGN/SAN/uci histories).
  - `src/utils.py` contains ANSI logging helpers, Qt process cleanup, Unicode glyph mapping, and window placement helpers.
  - Opening books live under `src/books/` (default `books/opening/gm2001.bin`), while reproducible positions sit in `gamestates/`. The `legacy/` tree is reference-only.

## 2. Engine Internals
- **Strategy orchestration** – `MoveStrategy` defines the interface; `StrategySelector` maintains a priority-ordered list and chooses the first strategy that returns a `StrategyResult`. If all strategies decline, the engine falls back to `random_move` to keep the protocol legal.
- **Default toggle flags (`STRATEGY_ENABLE_FLAGS`)**
  - `mate_in_one` → `True`
  - `opening_book` → `False` (polyglot reader is available but disabled by default)
  - `repetition_avoidance` → `False` (stub slot for future work)
  - `heuristic` → `True`
  - `fallback_random` → `False`
- **Built-in strategies**
  1. `MateInOneStrategy` (priority 100, short-circuits) – brute-forces legal replies from the opponent’s perspective to find immediate mates.
  2. `OpeningBookStrategy` (priority 90) – queries the on-disk polyglot book if the flag is enabled, otherwise falls back to a curated FEN dictionary.
  3. `HeuristicSearchStrategy` (priority 70) – iterative deepening alpha-beta with quiescence, killer/history heuristics, null-move pruning, LMR, aspiration windows, and a transposition table capped by `transposition_table_size`.
  4. `FallbackRandomStrategy` (priority 0) – injected only when the `fallback_random` flag is enabled; ensures a legal move is always produced.
- **Search budgeting** – `HeuristicSearchStrategy` resolves a time budget each `go` based on explicit time controls or a dynamic heuristic that considers the move count, material, and branching factor. Budgets are clamped between `min_time_limit` (0.25 s) and `max_time_limit` (12 s) with a default `base_time_limit` of 4 s. When `debug on` is active the engine prints per-depth summaries (`depth=`, `nodes`, `time`, principal variation) and reports when a depth is cut short due to time.
- **Strategy context** – `StrategyContext` snapshots everything a strategy might need (`fen`, repetition claims, legal move count, material imbalance, check status, opponent mate threat, time controls). The same context instance is passed to all strategies to keep evaluation deterministic.
- **Threading model** – `handle_go` spawns a worker thread that calls `process_go_command`. The thread selects a move, prints `bestmove <uci>`, then prints `readyok`. `state_lock` protects access to the shared board and bookkeeping (`move_calculating`, debug mode, etc.).

## 3. GUI & Self-Play Flow
- When the GUI starts, `main.py` launches two engine subprocesses and assigns them to White/Black slots. Commands are relayed via `send_command_with_lookup`, and stdout is parsed to update the UI.
- Activating self-play simply calls `SelfPlayManager.start()`. The manager issues `ucinewgame` to each engine once, then loops:
  1. Send `position fen <current>` to the side to move.
  2. Send `go`.
  3. Wait for `bestmove`; once received, push it on the shared board, update UI indicators, and repeat for the opposing colour.
- `SelfPlayManager.stop()` dispatches `stop` to the in-flight engine, cancels pending moves from that colour, and restores GUI controls.
- Manual analysis buttons in the GUI piggyback on the same machinery: they send `position …` and `go`, then disable UI controls until the engine replies.

## 4. Testing Matrix
| Area | File | Markers | Notes |
| --- | --- | --- | --- |
| Engine protocol & strategies | `src/tests/test_engine_core.py` | default + `dev` | Covers UCI handlers, strategy registration, time-control parsing, context snapshots, and heuristic instrumentation. Dev-only tests drive full self-play cycles.
| Self-play coordinator | `src/tests/test_self_play_manager.py` | default | Uses stub engines to confirm alternating requests, stop semantics, and engine reassignment.
| GUI integration | `src/tests/test_gui.py` | `gui` (implicit via `pytest.importorskip`) | Requires PySide6 and a display (or a virtual framebuffer). Exercises layout, move interaction, promotion dialog, and export actions.

### Headless Heuristic Trace Tests (Dev Marker)
These tests use the same `SelfPlayManager` loop that the GUI relies on, but run entirely in-process with stubbed UI surfaces:
1. `python3 -m pytest -m dev src/tests/test_engine_core.py::test_headless_self_play_debug_trace` – plays 12 opening plies from the initial position, asserting that the heuristic search completes depth 3/5 with adaptive budgets and logs the expected trace fragments.
2. `python3 -m pytest -m dev src/tests/test_engine_core.py::test_headless_self_play_midgame_trace` – starts from `r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10`, pushes eight plies, and asserts that timeouts and aspiration-window adjustments appear under heavy branching.

Both tests rely on the helper classes defined in `test_engine_core.py`:
- `HeadlessSelfPlayUI` implements the minimal callbacks `SelfPlayManager` expects (toggle controls, report activity, apply moves to a shared `chess.Board`).
- `EngineHarness` wraps an in-process `ChessEngine`, forces `debug on`, and routes commands directly to handler methods. `process_go_command` is called synchronously so tests can capture the full trace.
- `SelfPlayTestHarness` wires the pieces together, exposing `start()`, `play_plies(n)`, and `stop()` helpers so tests can sequence move generation deterministically.

Because these are marked `dev`, they only run when explicitly requested (`-m dev`). Expect ~26 s for the opening scenario and ~23 s for the midgame stress test on a typical workstation.

## 5. Default Behaviours & Operational Tips
- **Debug logging** – `debug on` is the main switch. In GUI launches it’s controlled by the “Debug Mode” toggle; in tests it is enabled within `EngineHarness` for deterministic traces.
- **Opening book** – disabled by default via `STRATEGY_ENABLE_FLAGS`. To enable globally, flip the flag before instantiating the engine or pass `ChessEngine(opening_book_path="books/opening/gm2001.bin")`.
- **Random fallback** – also disabled by default. To guarantee a move even when every other strategy fails, enable `fallback_random` or register `FallbackRandomStrategy` manually.
- **Time controls** – supplying explicit `go` arguments (`wtime`, `btime`, `movetime`, etc.) overrides the adaptive heuristics. The GUI’s self-play loop sends bare `go`, so the adaptive budget dominates.
- **Thread safety** – if you script direct access to `ChessEngine`, use the provided handlers (`handle_position`, `handle_go`, etc.) and respect `readyok` responses. Modifying `.board` directly without the lock can desynchronise strategies.
- **PySide6 tests** – `test_gui.py` creates a real `QApplication`. Run it only in environments with GUI support (or a headless display configured via Xvfb). Agents without a GUI can skip it.

## 6. Quick Command Reference
- Launch GUI: `python src/main.py [-fen <FEN>] [-dev]`
- Run engine standalone: `python src/engine.py`
- Full test suite (excluding GUI/DEV markers): `python3 -m pytest src/tests`
- Focused engine tests: `python3 -m pytest src/tests/test_engine_core.py -q`
- Headless heuristic traces: 
  1. `python3 -m pytest -m dev src/tests/test_engine_core.py::test_headless_self_play_debug_trace`
  2. `python3 -m pytest -m dev src/tests/test_engine_core.py::test_headless_self_play_midgame_trace`

Armed with this cheat-sheet, an agent can reason about how changes ripple through the engine, GUI, and self-play systems, and can select the right test harness for validating heuristic behaviour without spinning up the full PySide6 interface.
