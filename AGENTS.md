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
`bd` powers work tracking. Start with `bd quickstart --json` for the canonical overview, and lean on `bd -h` plus `bd <command> -h` whenever you need specifics. 
- Log every discovered bug, follow-up, or idea with `bd create ... --json`, even when uncovered during unrelated tasks; choose the correct issue type, priority, and assignee immediately. Link it to an appropraite epic/ related item, ask if one does not exist.

- **Capture clearly**: always assign ownership, supply rich descriptions, flesh out acceptance criteria, and add labels/priority when creating or updating issues.
- **Stay linked**: wire dependencies (`blocks`, `parent-child`, `discovered-from`) so related efforts remain traceable.
- **Lifecycle hygiene**: update statuses (`open`, `in_progress`, `blocked`, `closed`) as work moves; use `bd list`, `bd ready`, and `bd blocked` to slice queues quickly.
- **Signal with filters**: combine status, priority, assignee, and label filters to spotlight the next best task.
- **Keep it tidy**: prune stale notes, close finished work with meaningful summaries, and run periodic cleanups (compaction, renumbering, sync) to ensure the tracker stays trustworthy.

#### `bd create`
```bash
# Capture a high-priority regression with rich metadata
bd create "Fix PV search timeout" -t bug -p 0 -a jask --labels engine,urgent -d "Timeout when depth>25" --json

# Attach dependencies, acceptance criteria, and design context in one go
bd create "Wire GUI toggle" --deps discovered-from:bd-18,blocks:bd-12 --acceptance "Toggle lives under Settings"  --json

```
- **Metadata**: combine `-t`, `-p`, `-a`, and `-l` to capture type, priority, owner, and labels at creation time.
- **Rich fields**: use `-d`, `--acceptance`, `--design`, and `--external-ref` to keep context with the issue.

#### `bd list`
```bash
# Filter urgent open bugs
bd list --status open --type bug -p 0 --json

# Review your top assignments with label filters
bd list -a jask --label engine,search --limit 5 --json

# Render a custom report with templates
bd list --format '{{range .}}{{.ID}}\t{{.Title}}\t{{.Priority}}\n{{end}}'
```
- **Stack filters**: status, type, priority, assignee, labels, and title substring can all be combined.
- **Right-sized views**: `--limit` trims long backlogs; use multiple labels to require all tags.
- **Reporting**: switch to Go templates via `--format` or graph visualizations with `--format dot|digraph`.

#### `bd update`
```bash
# Move work in progress and reassign
bd update bd-42 --status in_progress -a jask --notes "Investigating move ordering" --json

# Adjust priority and add external cross-reference
bd update bd-51 -p 1 --external-ref gh-123 --json

# Rename and tighten acceptance criteria before handoff
bd update bd-37 --title "Finalize PV heuristic" --acceptance-criteria "Benchmark regression <= 2%" --json
```
- **Status hygiene**: update status, assignee, and notes together whenever you pick up or drop work.
- **Signal priority**: `-p` keeps triage honest; combine with `--external-ref` for cross-system tracking.
- **Scope clarity**: rewrite titles and acceptance criteria as understanding evolves to avoid churn later.

#### Workflow Views
```bash
bd ready --json -n 5
bd blocked --json --assignee jask
bd show bd-18 --json
bd stats --json
```
- **Pull queue**: `bd ready` surfaces unblocked candidates; tighten with `-p` or `-a`.
- **Surface blockers**: `bd blocked` highlights work that needs intervention.
- **Spot check**: `bd show` gives full issue context inline; `bd stats` summarizes workload distribution.

#### Dependency Management

```bash
bd dep add bd-45 bd-12 --type parent-child --json
bd dep tree bd-45 --json
bd dep cycles --json
```
- **Explicit relationships**: default `blocks` hides items from `bd ready`; use `parent-child` for epic trees.
- **Visualize scope**: `dep tree` maps hierarchy from a given issue.
- **Guard health**: run `dep cycles` before major planning sessions to prevent loops.

#### Identifier Maintenance
```bash
bd rename-prefix wk- --dry-run
bd renumber --dry-run
bd renumber --force
```
- **Prefix shifts**: `rename-prefix` rebrands IDs across database text; always preview first.
- **Sequential cleanup**: `renumber` closes gaps—only proceed with `--force` once backups are confirmed.

#### Data Stewardship
```bash
bd compact --dry-run --stats
bd compact --id bd-90 --tier 2 --force
bd delete bd-113 --json
```
- **Audit first**: `--dry-run` plus `--stats` reveals compaction impact before touching history.
- **Targeted decay**: choose tiers per issue based on age and archival needs; `--force` bypasses safety rails.
- **Last resort**: `bd delete` is permanent—only use when data must disappear entirely.

#### Sync & Automation
```bash
bd sync --dry-run --no-push
bd sync --message "Sync beads database"
bd daemon --status
bd daemon --auto-commit --auto-push --interval 10m
bd export --output .beads/snapshot.jsonl
bd import --input .beads/snapshot.jsonl --resolve-collisions
```
- **One-shot sync**: `bd sync` wraps export, commit, pull, import, and push; opt out of steps with `--no-pull|--no-push`.
- **Background harmony**: `bd daemon` keeps mirrors fresh; stop or inspect via `--status` and `--stop`.
- **Portability**: `export`/`import` shuttle data between machines; `--resolve-collisions` tames ID clashes.



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