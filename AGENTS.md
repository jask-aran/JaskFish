# Repository Guidelines

## Project Structure & Module Organization
- Core engine logic lives in `pvsengine.py`; simplified or legacy variants sit in `simple_engine.py`, `hsengine.py`, and `legacy/`.
- GUI entry point is `main.py` with supporting widgets in `gui.py`; reusable utilities live in `utils.py`.
- Gameplay traces, saved games, and profiling artefacts are stored in `self_play_traces/`, `gamestates/`, and `profiles/` respectively.
- Tests are in `tests/` with shared fixtures in `conftest.py`; benchmarks and profiling helpers are in `benchmark_pvs.py` and `pvs_profile_output.txt`.

## Build, Test, and Development Commands
- `python main.py` — launch the PySide6 GUI with the default engine pairing.
- `python pvsengine.py` — run the UCI-compatible engine directly on stdin/stdout.
- `python benchmark_pvs.py benchmark --threads 1 --depth 5` — measure PV-search NPS across curated FENs.
- `pytest -m "not gui"` — execute automated tests while skipping the GUI suite.

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
