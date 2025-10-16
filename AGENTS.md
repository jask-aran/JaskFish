## Project Overview

JaskFish is a UCI-compliant chess engine orchestrator with a PySide6 GUI and a bundle of chess engines that supports engine-versus-engine self-play and human-versus-engine play. The purpose of the project is to build performant chess engines and interact with/ benchmark/ improve them. The architecture emphasizes composable strategy selection, UCI protocol compliance, and headless testing capabilities.


## Agent Guidelines

### UV Workflow (Mandatory)
- Use `uv` for all environment and command execution; avoid `pip`, `venv`, or direct `python` invocations.
- Synchronize dependencies with `uv sync`; commit resulting `uv.lock` updates as needed.
- Run tools and scripts through `uv run <command>` (e.g., `uv run python gui.py`).
- Add or update dependencies via `uv add <package>` or `uv pip install` equivalents so the lockfile stays authoritative.
- Build the the native engine with `make -C engines/native` (use `make clean` before rebuilding if binaries look stale).

### Linting & Formatting
- Ruff is installed; lint with `uv run ruff check .` and address findings before submission.
- Follow PEP 8 formatting with 4-space indentation and ~100 character lines; mirror existing spacing patterns.
- Order imports stdlib → third-party → local modules; remove unused or wildcard imports.

### Beads



#### Commands
```bash
# Create new issue
bd create "Issue title" -t bug|feature|task|epic|chore -p 0-4 -d "Description" --json

# Create with labels
bd create "Issue title" -t bug  -l label1,label2,label3,... --json

# Create and link in one command (new way)
bd create "Issue title" -t bug -p 1 --deps discovered-from:<parent-id> --json

# Create with explicit ID, overrides default ID schema
bd create "Issue title" --id worker1-100 -p 1 --json

# Update issue status
bd update <id> --status in_progress --json

# Show dependency tree from an ID
bd dep tree <id>

# Use JQ to query the returned json from any comand
bd ready --json | jq '.[0]'
```

#### Guidelines
- **Issue Types:** `bug`, `feature`, `task`, `epic`, `chore`.
- **Priority Levels:** `0` critical, `1` high, `2` medium, `3` low, `4` backlog.
- **Dependency Types:** `blocks`, `related`, `parent-child`, `discovered-from`; only `blocks` suppresses items from `bd ready`. 
- **Statuses** (`open`, `in_progress`, `blocked`, `closed`)
- **Queues & Details:** `bd list` surfaces backlog slices, `bd show <id>` reveals metadata, and `bd dep tree <id>` maps deliverables and blockers.
- **Epics & Hierarchies:** `bd list --type epic` highlights umbrellas; refine scope via `bd show <epic>` or `bd dep tree <epic>`; connect child work with `bd dep add <epic> <child> --type parent-child` and keep every child updated through `bd update`.
- **Capture New Work:** Record every bug/TODO found—even outside the current task—via `bd create`; attach it to the correct parent or seek approval to introduce a new epic before proceeding.



- **Updates & Notes:** `bd update <id>` keeps status (`open`, `in_progress`, `blocked`, `closed`), assignee, description, and acceptance criteria in sync with reality.
- **Reference:** `bd quickstart` and `bd -h` provide command-level detail when needed.

- **Daily Loop:** `bd ready` → `bd update <id> --status in_progress --assignee <name>` → build/test/document → log discoveries with `bd create` → `bd close <id> --reason "Done: <summary>"` → rerun `bd ready`.


## Important Implementation Details

### Testing
- Testing is about ensuring validity of outputs to quickly detect feature regressions, rather than to validate performance.
- Tests default to **instant feedback** only; long-running GUI and search suites are opt-in (`-G`, `-S`). This keeps `pytest` suitable for quick pre-commit runs.
  - Execute the default test suite with `uv run pytest` from the repo root.
- run `pytest --collect-only` to see all available tests with information, or simply run the default suite in verbose mode `pytest -v`
- For a deep dive into intended coverage, fixture relationships, and extension
  recipes, read `docs/testing_architecture.md`.

### Benchmarking & Profiling
- **Entrypoint:** Run scenarios with `uv run python benchmark.py <subcommand>`; override the engine via `--engine-cmd <cmd ...>` when testing alternates.
- **Scenario Catalogue:** Inspect curated slugs and phases with `uv run python benchmark.py list`. Use `--positions <slug ...>` to focus, `--skip-defaults` to start empty, or feed extra FENs via `--fen` / `--fen-file`.
- **Benchmark Runs:** `uv run python benchmark.py benchmark` executes UCI searches, prints per-scenario summaries (bestmove, depth, nodes, NPS, score), and honours budget flags like `--movetime`, `--depth`, `--nodes`, and `--infinite-duration`. Add `--echo-info` to stream raw `info` lines.
- **Profiling:** `uv run python benchmark.py profile [--threads N]` calls the engine’s internal profiler on each scenario, emitting cProfile tables (top 25 cumulative functions) plus quick best-move stats.
- **Engine Comparison:** `uv run python benchmark.py compare --movetime <ms>|--depth <n>` runs matched searches against the native CPvsEngine (override with `--native-cmd`). Output includes per-scenario metric diffs and aggregate depth/time/node ratios.
- **Workflow Tips:** Combine curated and custom FENs to mirror regressions, keep `--max-wait` conservative for long searches, and redirect output if you need archival logs. Use `--ponder` or time-control flags to mimic tournament constraints.


### Self-Play and Headless Self-Play
- **GUI Mode:** `uv run python main.py` launches the PySide6 board with start/stop self-play controls. Activating self play hands control to a self-play manager class that calls each engine in turn. At the end of a self play session, traces are saved to `self_play_traces/`.
- **Headless Mode:** `uv run main.py --self-play` runs the same manager without the GUI, desirable for agent led testing.
- **Configuration:** Pass `-fen <FEN>` for bespoke starts, toggle perf payloads via `--include-perf-payload`, and adjust engine pairings/time presets in `ENGINE_SPECS` within `main.py`, just like normal interactions with `main.py`
- **Workflow:** Headless runs auto-rotate engines, log UCI transcripts, and respect `SelfPlayManager` heuristics; GUI mode shares code paths so parity bugs should be filed once.
- **Tracing Tips:** Inspect trace files for move-by-move logs, diff runs to catch regressions, and keep quiet mode off when diagnosing engine chatter.


### UCI Protocol Compliance

All engines must respond to:
- `uci`: Print `id name`, `id author`, `uciok`
- `isready`: Print `readyok` (or `info string Engine is busy processing a move` if calculating)
- `ucinewgame`: Reset board state
- `position startpos moves ...`: Set board to starting position and apply moves
- `position fen <fen> moves ...`: Set board to FEN and apply moves
- `go [wtime <ms>] [btime <ms>] [winc <ms>] [binc <ms>] [movestogo <n>] [movetime <ms>] [infinite]`: Start calculation
- `stop`: Halt current calculation (graceful)
- `quit`: Shutdown engine
- `debug on/off`: Enable/disable debug logging

### Changing Engine Matchups

Edit `ENGINE_SPECS` in `main.py`:
```python
ENGINE_SPECS = {
    "engine1": {
        "default_script": "engines/pvsengine.py",  # or "cpvsengine_shim.py", "simple_engine.py"
        "default_name": "JaskFish", # displayed name in terminal outputs and GUI
        "preferred_color": chess.WHITE,
    },
    # ...
}
```
Restart GUI to load new engine assignments.