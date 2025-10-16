# Testing Architecture Overview

This document explains how the test suite is organised, which markers control optional coverage, and the commands you can use to target each layer of functionality.

## 1. Philosophy

The default `pytest` run is deliberately fast. It covers pure unit and lightweight integration behaviour (protocol handling, logic helpers, self-play coordination) without launching a GUI or performing deep searches. Longer-running checks are still available, but you opt into them explicitly so day-to-day iterations stay quick.

## 2. Markers & Opt-in Flags

| Marker | Description | Enable with |
| --- | --- | --- |
| `gui` | PySide6 GUI interaction tests that need a display or virtual framebuffer. | `pytest -G` or `pytest --gui` |
| `search_slow` | PV search smoke tests that run actual engine searches across curated FENs. | `pytest -S` or `pytest --search` |
| `performance`, `dev`, `edge_case`, `logic` | Additional domain markers reused in legacy plans. These remain opt-in via `-m`. | `pytest -m performance` (etc.) |

Both `-G`/`--gui` and `-S`/`--search` accept optional file or node filters, e.g. `pytest -G tests/test_gui.py::test_export_game`.

## 3. Suite Breakdown

### Core & Helpers (default `pytest`)
- `tests/test_pvsengine_engine.py` — UCI façade behaviour and deterministic mate strategy smoke test.
- `tests/test_pvsengine_strategy_selector.py` — ordering, tie-breaking, and exception handling.
- `tests/test_pvsengine_search_limits.py` — time-budget resolution and meta clamping.
- `tests/test_pvsengine_heuristic_strategy.py` — backend contract + metadata normalisation with a stub outcome.
- `tests/test_self_play_manager.py` — engine orchestration, trace export, and state toggles.
- `tests/test_simple_engine.py` — lightweight engine command handling and move legality.
- `tests/test_chess_logic.py`, `tests/test_utils.py` — board utilities, export helpers, and ANSI/logging helpers exposed via `main.py`.

### PV Search Smoke (add `-S`/`--search`)
- `tests/test_pvsengine_pvsearch.py` — iterative deepening backend sanity checks across multiple edge-case positions, including repeated runs to ensure the board state stays untouched.

### GUI Suite (add `-G`/`--gui`)
- `tests/test_gui.py` — widget layout, click interactions, promotion dialog, manual engine callbacks, self-play toggles, export logic. Relies on PySide6 and uses `pytest.importorskip` to fail gracefully when the dependency is missing.

## 4. Common Command Recipes

```bash
# Fast default (skips GUI and slow search)
pytest

# Run everything including smoke and GUI (use a display or virtual framebuffer)
QT_QPA_PLATFORM=offscreen pytest -S -G

# Only the PV search smoke checks
pytest -S tests/test_pvsengine_pvsearch.py

# GUI suite in isolation (headless mode)
QT_QPA_PLATFORM=offscreen pytest -G

# Drill into a single test node
pytest -S tests/test_pvsengine_pvsearch.py::test_pvsearch_backend_edge_positions
```

## 5. Adding New Tests

1. Decide whether the test should run in the default fast path. If it invokes the GUI or requires the engine to search for measurable time, mark it with `@pytest.mark.gui` or `@pytest.mark.search_slow`.
2. Update `pytest.ini` if a new marker is introduced.
3. Document the addition here (and in `AGENTS.md`) so contributors know how to run the new coverage.
4. If the test needs PySide6, wrap the import in `pytest.importorskip("PySide6")` to keep headless environments friendly.

## 6. CI / Automation Notes

- Continuous integration can run `pytest` for the quick signal and schedule periodic or nightly jobs that append `-S`/`--search` and `-G`/`--gui`.
- Because flag parsing happens in `conftest.py`, you can combine them with other pytest selections (`-k`, `-m`, etc.) without extra plugin configuration.

For historical context on how this structure evolved, see `agent_plans/testing_rebuild_plan_20251104.md`. For contributor-facing quick references, see `AGENTS.md` and `CLAUDE.md`.
