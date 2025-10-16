## Project Overview

JaskFish is a UCI-compliant chess engine orchestrator with a PySide6 GUI and a bundle of chess engines that supports engine-versus-engine self-play and human-versus-engine play. The purpose of the project is to build performant chess engines and interact with/ benchmark/ improve them. The architecture emphasizes composable strategy selection, UCI protocol compliance, and headless testing capabilities.


## Agent Guidelines

### UV Workflow (Mandatory)
- Use `uv` for all environment and command execution; avoid `pip`, `venv`, or direct `python` invocations.
- Synchronize dependencies with `uv sync`; commit resulting `uv.lock` updates as needed.
- Run tools and scripts through `uv run <command>` (e.g., `uv run python gui.py`).
- Add or update dependencies via `uv add <package>` or `uv pip install` equivalents so the lockfile stays authoritative.

### Linting & Formatting
- Ruff is installed; lint with `uv run ruff check .` and address findings before submission.
- Follow PEP 8 formatting with 4-space indentation and ~100 character lines; mirror existing spacing patterns.
- Order imports stdlib → third-party → local modules; remove unused or wildcard imports.

### Beads
- We track work in Beads instead of Markdown. 
<!-- Proivde an extremely concise explanation for use of the beads tool through command bd for issue tracking and linking.-->
- Run `bd quickstart` or `bd -h` to see how.


## Important Implementation Details

### Build & Test
- Target Python 3.10+; ensure virtual envs are managed by `uv`.
- Build the the native engine with `make -C engines/native` (use `make clean` 
  before rebuilding if binaries look stale).
- Execute the default test suite with `uv run pytest` from the repo root.
- Tests default to **instant feedback**; long-running GUI and search suites are
  opt-in (`-G`, `-S`). This keeps `pytest` suitable for quick pre-commit runs.
- Markers live in `pytest.ini`; `conftest.py` exposes fixtures for engine harnesses,
  FEN repositories, and GUI bootstrapping. Extend these instead of duplicating
  harness code.
- Run a focused test via `uv run pytest tests/test_simple_engine.py::test_random_move`, 
  swapping paths/test IDs as needed.
- For a deep dive into intended coverage, fixture relationships, and extension
  recipes, read `docs/testing_architecture.md`.

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


## Common Development Scenarios

### Running Headless Self-Play

<!-- Impliment concise explanation of self play with reference file for understanding implementation -->
Both GUI and headless (`python main.py --self-play`) harnesses use the same manager, ensuring behavioral consistency. Use `--include-perf-payload` when launching to preserve the raw JSON perf payload lines in logs and traces (they are filtered by default).


### Changing Engine Matchups

Edit `ENGINE_SPECS` in `main.py`:
```python
ENGINE_SPECS = {
    "engine1": {
        "default_script": "engine.py",  # or "pvsengine.py", "simple_engine.py"
        "default_name": "JaskFish",
        "preferred_color": chess.WHITE,
    },
    # ...
}
```

Restart GUI to load new engine assignments.