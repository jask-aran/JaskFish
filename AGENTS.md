# Repository Guidelines

## Project Structure & Module Organization
Source lives in `src/`. `engine.py` exposes the UCI-facing chess engine and strategy registry; `gui.py` handles the PySide6 board and process wiring; `main.py` glues the GUI and engine processes. Utility helpers sit in `utils.py`. Test modules live under `src/tests/` and mirror the engine/GUI split. Opening books and sample states are stored in `src/books/` and `gamestates/`. The `legacy/` folder keeps reference implementations—do not modify without coordination. Run tools from the repository root so relative imports resolve correctly.

## Build, Test, and Development Commands
Create an isolated environment first: `python -m venv .venv && source .venv/bin/activate`. Install runtime deps with `pip install -U python-chess Pillow cairosvg PySide6 pytest`. Launch the GUI-driven app via `python src/main.py [-fen <FEN>] [-dev]`. Headless engine checks can be run with `python src/engine.py` to interact over UCI. Execute the test suite using `pytest src/tests`; add `-m <marker>` to focus on `logic`, `gui`, or other marked scenarios. Use `pytest --maxfail=1` for fast feedback when iterating.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation. Prefer explicit imports (`from src import engine`) and type hints when signatures touch engine state. Keep module-level constants upper snake case, classes in PascalCase, and functions/variables in snake_case. When adding CLI flags or UCI commands, mirror the existing `handle_*` naming in `engine.py`. Run `python -m compileall src` if you need a quick syntax check before pushing.

## Testing Guidelines
All new features require pytest coverage with descriptive `test_` function names. Co-locate fixtures in `src/tests/conftest.py` to keep reuse centralized. Use the provided markers (`edge_case`, `performance`, `dev`, `logic`, `gui`) to scope slow or GUI-bound tests; default CI paths should pass without selecting optional markers. For deterministic assertions, stub strategies as shown in `test_engine_core.py`.

## Commit & Pull Request Guidelines
Write short, imperative commit titles (e.g., “add move pruning check”), matching the existing history. Group related changes into a single commit; avoid unrelated formatting noise. Pull requests should describe the change, mention impacted modules, and link any tracked issues. Include before/after screenshots when GUI behavior changes and list any manual test commands you ran. Request a review whenever engine evaluation logic or GUI interactions shift.
