# JaskFish Codebase Snapshot

## High-Level Layout
- Runtime source lives in `src/`; tests mirror that layout under `src/tests/`.
- `engine.py` implements the UCI speaking engine and the strategy stack; `gui.py` renders the PySide6 board and player controls; `main.py` wires two engine processes to the GUI.
- Shared helpers: `chess_logic.py` wraps common `python-chess` calls, `utils.py` contains CLI formatting and process cleanup helpers, and `self_play.py` coordinates engine-versus-engine matches.
- Assets ship with the repo: opening books under `src/books/` (default `books/opening/gm2001.bin`) and sample positions in `gamestates/`. The `legacy/` tree contains reference implementations and stays read-only.
- `pytest.ini` registers the custom markers (`logic`, `gui`, `edge_case`, `performance`, `dev`). `src/conftest.py` amends `sys.path` so tests can import modules as `from src import engine`.

## Engine Architecture (`src/engine.py`)
- `ChessEngine` exposes a UCI-like protocol (`uci`, `isready`, `position`, `go`, `ucinewgame`, etc.) and runs `process_go_command` on a worker thread to keep stdin responsive.
- Strategy execution is pluggable. `MoveStrategy` defines the interface, `StrategySelector` prioritises registered strategies, and selection falls back to `random_move` if nothing returns a move.
- Default strategies (registered according to `STRATEGY_ENABLE_FLAGS`) are:
  - `MateInOneStrategy` (priority 100) that scans legal replies for immediate mates.
  - `OpeningBookStrategy` (priority 90) that consults an optional polyglot reader or the built-in FEN dictionary. Note: the global flag currently disables this strategy by default even though `gm2001.bin` is available.
  - `HeuristicSearchStrategy` (priority 70) that runs iterative deepening alpha-beta with quiescence search, killer/history heuristics, a transposition table, and simple mobility/king safety/pawn structure terms. It adapts depth and time budget from parsed `go` arguments.
  - `FallbackRandomStrategy`, appended last to guarantee a legal move when everything else fails.
- `StrategyContext` snapshots board state (piece counts, repetition claims, opponent mate-in-one threat detection, time controls, etc.) before strategies run, making them deterministic when handed the same FEN.
- Debug tracing can be toggled with `debug on/off`; when enabled the engine prints `info string` entries for context prep, strategy choices, and errors.

## GUI & Process Orchestration (`src/gui.py`, `src/main.py`)
- `ChessGUI` builds a themed 8x8 board with move highlighting, undo/reset/export actions, manual engine controls (`Ready`, `Go`), engine restarts, and a self-play toggle. Piece glyphs come from Unicode, and pawn promotion is resolved through a custom dialog.
- Manual analysis is mediated through callbacks. The GUI disables inputs while an engine search is pending and re-enables them once `manual_evaluation_complete` fires.
- Self-play controls surface status messaging (`Self-play: Engine … evaluating…`) and prevent board interaction while games are automated.
- `main.py` parses CLI flags (`-fen`, `-dev`, `--engine1/2`, name overrides), starts two `QProcess` instances that both run `src/engine.py`, and routes I/O via `engine_output_processor`. Console logging is colorised through helpers in `utils`.
- The launcher also builds human-readable labels (e.g., `Engine 1 – White`), mirrors UCI commands to the child processes, and coordinates self-play via `SelfPlayManager`.

## Self-Play Coordination (`src/self_play.py`)
- `SelfPlayManager` controls engine-versus-engine loops for a GUI implementation that exposes basic hooks (enable/disable controls, show activity, acknowledge completion).
- Starting self-play issues `ucinewgame`, sends `position fen …` followed by `go`, then alternates between white/black engines until a result is detected via `chess_logic.is_game_over`.
- `stop()` cancels outstanding searches by dispatching `stop` and suppresses the first move that arrives after cancellation to keep the board consistent.

## Game Logic & Utilities
- `chess_logic.py` wraps common board queries (`is_valid_move`, `get_game_result`, repetition/fivefold/75 move checks) and exports move history in UCI or SAN form. Pawn promotion helpers assume queen promotion unless overridden at the GUI layer.
- `utils.py` centralises ANSI-colour log helpers, Qt process cleanup (with a dev-mode verbose toggle), piece-to-Unicode conversion, and window centering.

## Testing (`src/tests`)
- `test_engine_core.py` focuses on the UCI handlers, strategy registration, opening book metadata, time-control parsing, and `StrategyContext` construction. It stubs strategies to keep behaviour deterministic.
- `test_self_play_manager.py` verifies the GUI contract, alternating engine requests, stop semantics, and command dispatch when cycling between players.
- No GUI rendering tests currently exist; the suite stays headless and fast (`pytest src/tests`).

## Operational Notes
- Recommended workflow still matches the README: create a virtualenv, install `python-chess Pillow cairosvg PySide6 pytest`, run the GUI via `python src/main.py [-fen <FEN>] [-dev]`, or interact with the engine directly using `python src/engine.py`.
- Opening book support requires either re-enabling the `opening_book` flag or passing a custom `ChessEngine(opening_book=...)`/`opening_book_path`; otherwise the heuristic search drives move selection.
- When extending the strategy stack, follow the `handle_*` naming for new UCI commands and prefer registering strategies through `_register_default_strategies` to keep prioritisation explicit.
