# Repository Guidelines

## Environment Setup
- Always work inside the repository-managed virtual environment `.venv` so the engine runs against the pinned dependency set. Activate it with `source .venv/bin/activate` on Unix-like shells or `.venv\Scripts\activate` on Windows PowerShell before running any commands.
- Install or refresh dependencies with `pip install -r requirements.txt` while the virtual environment is active.

## Project Structure & Module Organization
- Core engine logic lives in `pvsengine.py`; simplified or legacy variants sit in `simple_engine.py`, `hsengine.py`, and `legacy/`.
- GUI entry point is `main.py` with supporting widgets in `gui.py`; reusable utilities live in `utils.py`.
- Gameplay traces, saved games, and profiling artefacts are stored in `self_play_traces/`, `gamestates/`, and `profiles/` respectively.
- Tests are in `tests/` with shared fixtures in `conftest.py`; benchmarks and profiling helpers are in `benchmark_pvs.py` and `pvs_profile_output.txt`.

## Test Execution & Tooling
- `pytest -m "not gui"` — default regression sweep; covers search heuristics, UCI handling, and self-play orchestration without the PySide6 dependency.
- `pytest tests/test_engine_core.py` — validates the full UCI command surface (e.g. `uci`, `isready`, `position`, `go` parsing) with deterministic fixtures.
- `pytest tests/test_search_backend.py -k "principal_variation"` — exercises the iterative-deepening and PVS pipeline; use `-k` filters to focus tight subtests.
- `pytest tests/test_simple_engine.py` — regression tests for the lightweight search engine variant.
- `pytest tests/test_self_play_manager.py` — headless verification of the self-play coordinator used by both GUI and tooling.
- `pytest -m gui` — opt-in UI suite that boots PySide6; run inside a display-capable environment only.

## Headless Engine Interaction
- `python pvsengine.py` — spins up the primary engine in a blocking stdin/stdout loop. Send commands such as:
  - `uci` ➝ expect `id`/`uciok`.
  - `isready` ➝ wait for `readyok` (engine reports busy if a search is in flight).
  - `position startpos` or `position fen <FEN>` to seed the board.
  - `go depth 4`, `go movetime 2000`, or `stop` to control analysis; monitor best-move lines via the streamed UCI output.
- `python simple_engine.py` / `python hsengine.py` — launch alternative engines with the same UCI contract for differential testing.
- Pipe scripted command sequences from fixtures (see `tests/test_engine_core.py`) to reproduce UCI edge cases without the GUI.
- Terminal output from the GUI and helpers is color coded via `utils.sending_text`/`utils.recieved_text`; expect mirrored `SENDING`/`RECIEVED` lines for every UCI command and engine response.

## GUI Command Pipeline & Time Budgets
- Manual analysis uses `ChessGUI.go_command`, which issues `position fen <current>` followed by bare `go`. No explicit `depth`/`movetime` is supplied; `pvsengine.SearchLimits.resolve_budget` infers budgets from the board state. Early moves commonly receive ~2.5 s with a nominal depth limit of 5, while late middlegame searches drop toward ~1.3 s as seen in `self_play_traces/1_selfplay.txt`.
- The GUI mirrors this through terminal logs: for every action you will see `SENDING` lines (e.g. `SENDING   White[W] position fen …`) followed by `RECIEVED` lines echoing `info string …`, `bestmove …`, and `readyok`. Expect patterns such as `info string HS: start search depth_limit=5 time_budget=2.51s legal_moves=22` and subsequent iterative deepening summaries down to aspiration-window adjustments or `search timeout` when the allocated window is exhausted.
- Self-play reuses the same pipeline via `SelfPlayManager._request_move`, so the most recent `<X>_selfplay.txt` file reflects exactly what a GUI-initiated evaluation prints in the terminal, including `info string perf … time=1.66/2.51s` summaries and post-search `bestmove` announcements.
- When comparing against benchmarks or scripted runs, match this GUI behaviour by letting the engine manage time autonomously (omit `go depth <n>`), then validate that the streamed `time_budget=` and `time=…/…` figures align with expectations.

## Testing & Validation Methods

### Pytest Suites
- `pytest -m "not gui"` — primary regression sweep; covers strategy selection, UCI handlers, and self-play orchestration without PySide6.
- `pytest tests/test_engine_core.py` — exercises the full UCI surface (`uci`, `isready`, `position`, `go`) and validates time-budget parsing edge cases.
- `pytest tests/test_search_backend.py -k "principal_variation"` — focuses on iterative deepening, aspiration windows, and PV stability.
- `pytest tests/test_simple_engine.py` / `pytest tests/test_self_play_manager.py` — verify the lightweight engine and self-play controller independently.
- `pytest -m gui` — optional UI harness; run only in environments with a display server.

### Headless Self-Play Harness
- `python self_play.py` — launches the same coordination logic used by the GUI, logging traces to `<X>_selfplay.txt` under `self_play_traces/`. Inspect the latest (highest `X`) file to review raw `info`/`bestmove` transcripts, inferred budgets, and stop reasons.
- Use these traces to cross-check GUI expectations (time budgets, depth ceilings, timeout behaviour) or to diff engine output across revisions.

### Benchmarks & Profiling CLI
- `python benchmark_pvs.py benchmark --threads 1 --depth 5` — measures nodes-per-second over curated FENs; compare reported PVs and scores to GUI/self-play transcripts.
- `python benchmark_pvs.py profile --threads 1 --output profiles/pvs_profile.txt` — writes combined `cProfile` output for hotspot analysis.
- Prefer the same depth/budget settings as GUI runs when validating behavioural regressions; combine with self-play logs to ensure consistent PV ordering.

### Direct UCI Sessions
- `python pvsengine.py` (or `simple_engine.py` / `hsengine.py`) — interact via stdin/stdout using UCI commands. Send `uci`, `isready`, `position startpos`, and `go` to simulate the GUI pipeline; capture `info string` and `bestmove` responses directly.
- Reuse scripted command batches from `tests/test_engine_core.py` to recreate corner cases, or issue commands manually to probe new heuristics while observing the live budget calculations and `info string perf …` summaries.

## Build & Development Shortcuts
- `python main.py` — launch the PySide6 GUI with the default engine pairing.
- Use the `SearchReporter` hooks and `utils.debug_text` when instrumenting new heuristics so test and benchmark output stays consistent.

## Artifact Locations
- `profiles/` — contains profiler captures; `benchmark_pvs.py profile` defaults to `profiles/pvs_profile.txt`, summarising each scenario with 80-character separators around `cProfile` output.
- `self_play_traces/` — chronological logs for automated matches stored as `<X>_selfplay.txt`. Sort numerically on `X` to find the latest run, then parse the `[White - …]`/`[Black - …]` sections to reconstruct engine dialogue.
- `gamestates/` — GUI exports produce `chess_game_<timestamp>.json` with `fen-init`, `fen-final`, and SAN/UCI move lists; leverage these snapshots to rerun reproductions via `position` commands.
- `pvs_profile_output.txt` — historical benchmark report for quick diffing; regenerate via `python benchmark_pvs.py profile --output pvs_profile_output.txt` when comparing branches.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation; type hints and dataclasses are the norm for new Python code.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and keep modules cohesive around a single concern.
- Logging should use the existing helper hooks (`utils.debug_text`, `SearchReporter.trace`) rather than ad-hoc `print` calls.

## Testing Guidelines
- Tests use `pytest`; store new cases under `tests/` mirroring the module under test (e.g., `tests/test_pvsengine.py`).
- Mark slow or GUI-dependent checks with the configured markers from `pytest.ini` (`@pytest.mark.gui`, `@pytest.mark.performance`).
- Provide deterministic FEN fixtures and assert principal variation or score deltas rather than raw stdout.

## Commit & Pull Request Guidelines
- Craft short, imperative commit titles (e.g., `Improve aspiration window logging`), mirroring the existing history.
- Reference related issues in the body and note user-facing changes, benchmarks, or new CLI options.
- Pull requests should summarise the change, list test evidence (`pytest`, benchmarks), and include screenshots when UI behaviour shifts.

## Performance & Profiling Tips
- Use `python benchmark_pvs.py profile --threads 1` to generate aggregated profiler output in `profiles/`.
- Within the engine, prefer `SearchReporter` timing hooks to instrument new strategies so benchmark output stays coherent.
