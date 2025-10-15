# Agent Guide

We track work in Beads instead of Markdown. Run `bd quickstart` to see how.

## Overview
- **Purpose**: JaskFish is a UCI-compliant chess engine with a PySide6 GUI. The
  engine stack prioritises composable search strategies, deterministic testing,
  and traceable time budgeting.
- **Entrypoints**: `pvsengine.py` for the primary PVS engine, `simple_engine.py`
  for the lightweight heuristic engine, `hsengine.py` for the hybrid stack, and
  `main.py` for the GUI/self-play harness.
- **Docs**: `docs/testing_architecture.md` (test layout) and
  `docs/sunfish_engine_analysis.md` (legacy sunfish notes) are the authoritative
  references beyond this file.

## Environment Setup
- Always activate the repository-managed virtual environment: `source .venv/bin/activate`
  (Unix) or `.venv\Scripts\activate` (Windows PowerShell).
- Refresh dependencies with `pip install -r requirements.txt` while the virtualenv
  is active. The test suite and GUI expect the pinned versions.
- Use `python -m pip` for per-module installs so the interpreter inside the
  virtualenv is used explicitly.

## Quick Command Reference

### Application & Self-Play
- `python main.py` — launch the GUI with default engine pairings.
- `python main.py -fen "<fen>"` — start the GUI at a custom position.
- `python main.py --self-play [--include-perf-payload]` — run headless self-play;
  logs land in `self_play_traces/<n>_selfplay.txt`.

### Engines (UCI loops)
- `python pvsengine.py` — primary engine; responds to `uci`, `position`, `go`.
- `python simple_engine.py` — deterministic heuristic engine for fast testing.
- `python hsengine.py` — hybrid strategy stack useful for regression comparisons.

### Testing (Pytest markers are short options)
- `pytest` — fast regression sweep (GUI and search smoke tests are skipped).
- `pytest -S` or `pytest --search` — include PVS search smoke tests that drive
  real engine searches.
- `QT_QPA_PLATFORM=offscreen pytest -G` (or `--gui`) — run the PySide6 GUI tests
  with an offscreen backend; omit the env var if a display server is available.
- See `docs/testing_architecture.md` for suite-by-suite detail and extension tips.

### Benchmarking & Profiling
- `python benchmark_pvs.py benchmark --threads 1 --depth 5` — throughput check on
  curated FENs; compares nodes/sec and PVs.
- `python benchmark_pvs.py profile --threads 1 --output profiles/pvs_profile.txt`
  — emit a combined cProfile trace suitable for hotspot analysis.

## Architecture

### Engine Stack (`pvsengine.py`)
- **StrategySelector** orchestrates ordered strategies. Each `MoveStrategy`
  returns a `StrategyResult` and may short-circuit by setting `definitive=True`.
- **Built-in strategies**:
  - `MateInOneStrategy` (highest priority) brute-forces mate threats and returns
    immediately when a forced mate is found.
  - `HeuristicSearchStrategy` performs iterative deepening PV search with
    aspiration windows, quiescence, killer/history tables, null-move pruning,
    late-move reductions, and repetition penalties.
- **StrategyContext** snapshots board state (FEN, legal move count, repetition
  info, check status, material data, resolved time controls) so strategies stay
  side-effect free.
- **Meta configuration**: `MetaParams` presets (`balanced`, `fastblitz`,
  `tournament`) feed into `build_search_tuning`, which produces both `SearchTuning`
  heuristics and `SearchLimits` (budget clamps, aspiration margins, depth goals).
  Apply alternate configs with `HeuristicSearchStrategy.apply_config`.
- **Budgeting & threading**: `SearchLimits.resolve_budget` clamps search windows
  between safe minimum/maximum bounds. `ChessEngine.handle_go` runs the search in a
  worker thread (via `ThreadPoolExecutor` for multi-root evaluation) to keep the
  UCI loop responsive.
- **Reporting**: `SearchReporter` standardises trace output: iterative deepening
  summaries, aspiration adjustments, timeout reasons, and per-depth NPS metrics.

### GUI & Self-Play (`main.py`, `gui.py`)
- **Process model**: The GUI launches engine subprocesses defined in `ENGINE_SPECS`.
  All analysis commands (`uci`, `position`, `go`, `stop`) flow through
  `send_command_with_lookup`, and stdout is parsed by `engine_output_processor`
  to update boards, logs, and clocks.
- **SelfPlayManager** (headless and GUI) issues `ucinewgame`, manages alternating
  `position`/`go` loops, and records transcripts to `self_play_traces/`. Both GUI
  and headless runs therefore share identical command sequencing.
- **Logging**: Use `utils.sending_text`, `utils.recieved_text`, and
  `SearchReporter.trace` rather than ad-hoc prints so terminal colour coding and
  tests remain consistent.

### Key Modules & Data
- `pvsengine.py` — primary engine and strategy stack described above.
- `hsengine.py` — historical hybrid engine kept for comparison regressions.
- `simple_engine.py` — minimal heuristic-only engine (no deep search).
- `gui.py` — PySide6 widget layer for boards, clocks, and control panels.
- `utils.py` — logging helpers, ANSI colouring, Qt cleanup utilities.
- `chess_logic.py` — board helpers, repetition detection, PGN/SAN utilities.
- `tests/` — full regression suite (see Testing Strategy below).
- `docs/` — long-form documentation (`testing_architecture.md`,
  `sunfish_engine_analysis.md`).
- `self_play_traces/`, `gamestates/`, `profiles/` — generated artefacts for
  self-play logs, exported game states, and profiler outputs respectively.

## Testing Strategy
- Tests default to **instant feedback**; long-running GUI and search suites are
  opt-in (`-G`, `-S`). This keeps `pytest` suitable for quick pre-commit runs.
- Suite organisation mirrors engine components:
  - `tests/test_pvsengine_engine.py`, `test_pvsengine_strategy_selector.py`,
    `test_pvsengine_search_limits.py`, and `test_pvsengine_heuristic_strategy.py`
    cover the UCI façade, strategy ordering, budget resolution, and search config.
  - `tests/test_pvsengine_pvsearch.py` is marked `search` and gated behind `-S`.
  - `tests/test_self_play_manager.py`, `tests/test_simple_engine.py`,
    `tests/test_chess_logic.py`, and `tests/test_utils.py` exercise supporting
    modules and coordination helpers.
  - `tests/test_gui.py` is marked `gui` and depends on PySide6 plus a display or
    offscreen backend.
- Markers live in `pytest.ini`; `conftest.py` exposes fixtures for engine harnesses,
  FEN repositories, and GUI bootstrapping. Extend these instead of duplicating
  harness code.
- For a deep dive into intended coverage, fixture relationships, and extension
  recipes, read `docs/testing_architecture.md`.

## Development Practices
- Follow PEP 8 with 4-space indentation; prefer type hints and dataclasses for
  new constructs.
- Stick with `snake_case` for functions/variables, `PascalCase` for classes.
- Avoid stray prints—route logging through `utils.debug_text`, `SearchReporter`,
  or existing structured channels.
- When changing search behaviour, capture evidence via `pytest -S` and, if
  relevant, `benchmark_pvs.py benchmark`. For GUI-affecting work, add or update
  cases in `tests/test_gui.py`.
- Commit style: short imperative titles (`Improve aspiration window logging`),
  reference related issues in bodies, and note any benchmarks or GUI screenshots.

## Additional Resources
- `README.md` — quickstart flow for developers outside the agent workflow.
- `docs/testing_architecture.md` — canonical reference for the test matrix.
- `docs/sunfish_engine_analysis.md` — preserved analysis of the Sunfish adapter.
- `agent_plans/` — coordination notes for ongoing investigative efforts.
- Reach out in this file when new workflows or conventions need to be surfaced;
  keeping AGENTS.md current avoids divergence between contributors and automation.
