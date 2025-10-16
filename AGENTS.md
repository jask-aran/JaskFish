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
- We track work in Beads instead of Markdown.
- Use `bd init` to set up the tracker (add `--prefix` for custom IDs).
- Create issues with `bd create "<title>" [-p <priority>] [-t <type>]` and optional details.
- Inspect queues via `bd list`, `bd show <id>`, and `bd ready` to claim unblocked work.
- Manage dependencies with `bd dep add <issue> <blocker>` and visualize using `bd dep tree <id>`.
- Follow the loop: `bd ready` → `bd update <id> --status in_progress` → implement/test/document → record new work with `bd create` → finish with `bd close <id> --reason "Done: <summary>"` → rerun `bd ready`.
- Issue types: `bug`, `feature`, `task`, `epic`, `chore`.
- Priority levels: `0` critical, `1` high, `2` medium, `3` low, `4` backlog.
- Dependency types: `blocks` (gates `bd ready`), `related`, `parent-child`, `discovered-from`.
- Run `bd quickstart` or `bd -h` anytime for the complete command reference.


## Important Implementation Details

### Testing
- Testing is about validity of outputs to quickly detect feature regressions, rather than validate performance.
- Tests default to **instant feedback** only; long-running GUI and search suites are opt-in (`-G`, `-S`). This keeps `pytest` suitable for quick pre-commit runs.
  - Execute the default test suite with `uv run pytest` from the repo root.
- run `pytest --collect-only` to see all available tests with information, or simply run the default suite in verbose mode `pytest -v`
- For a deep dive into intended coverage, fixture relationships, and extension
  recipes, read `docs/testing_architecture.md`.

### Benchmarking & Profiling
- `benchmark.py` contains tools for benchmarking engine performance, profiling and recording.
<!-- Impliment concise explanation of benchmarking and profiling apparatus, specifically what commands are available and their expected outputs/ behaviours..-->


### Self-Play and Headless Self Play
<!-- Impliment concise explanation of self play with reference file for understanding implementation -->
Both GUI and headless (`python main.py --self-play`) harnesses use the same manager, ensuring behavioral consistency. Use `--include-perf-payload` when launching to preserve the raw JSON perf payload lines in logs and traces (they are filtered by default).


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