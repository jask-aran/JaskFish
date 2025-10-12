# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JaskFish is a UCI-compliant chess engine with a PySide6 GUI that supports engine-versus-engine self-play. The architecture emphasizes composable strategy selection, UCI protocol compliance, and headless testing capabilities.

## Key Commands

### Running the Application
```bash
# Launch GUI with default starting position
python main.py

# Launch GUI with custom FEN position
python main.py -fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Launch in development mode (enables debug logging)
python main.py -dev
```

### Running Engines Standalone
```bash
# Run main engine in UCI mode (accepts uci, position, go, etc.)
python engine.py

# Run simple heuristic engine
python simple_engine.py

# Run PVS engine
python pvsengine.py
```

### Testing
```bash
# Run fast unit suite (GUI and slow search smoke tests are skipped by default)
pytest

# Run specific test file
pytest tests/test_pvsengine_engine.py

# Run only engine core tests (quiet)
pytest tests/test_pvsengine_engine.py -q

# Include PV search smoke tests (longer-running engine searches)
pytest -S tests/test_pvsengine_pvsearch.py

# Run GUI tests (requires display or Xvfb)
QT_QPA_PLATFORM=offscreen pytest -G tests/test_gui.py

# Run with verbose output
pytest -v
```

### Profiling
```bash
# Profile the PV search backend on starting position
python engine.py --profile

# Profile with custom FEN and threads
python engine.py --profile --fen "r2q1rk1/pp2bppp/2n1pn2/2pp4/3P4/2P1PN2/PP1NBPPP/R1BQ1RK1 w - - 0 10" --threads 4
```

## Architecture

### Engine Core (`engine.py` and `pvsengine.py`)

The engines use a **strategy stack** architecture:
- `StrategySelector` evaluates registered strategies by priority
- Strategies return `StrategyResult` with move, score, confidence, and metadata
- Strategies marked `definitive=True` short-circuit further evaluation
- If all strategies decline, the engine returns no move (protocol-legal fallback)

**Built-in Strategies** (in priority order):
1. **MateInOneStrategy** (priority 100): Brute-forces opponent responses to find immediate checkmates, returns `definitive=True`
2. **HeuristicSearchStrategy** (priority 70): Iterative deepening alpha-beta with aspiration windows, PV search, quiescence, transposition table, killer/history heuristics, null-move pruning, LMR, and repetition penalties

**Strategy Context**: Each strategy receives a `StrategyContext` snapshot containing FEN, time controls, repetition claims, legal move count, material imbalance, check status, and opponent mate threat.

**Search Budgeting**: The heuristic strategy resolves time budgets from explicit `go` time controls or dynamically based on move complexity, material, and position tension. Budgets are clamped between `min_time_limit` (0.25s) and `max_time_limit` (12s).

**Threading Model**: `handle_go` spawns a worker thread that calls `process_go_command`. The worker selects a move via the strategy selector, prints `bestmove <uci>`, then prints `readyok`. Access to the shared `chess.Board` is protected by `state_lock`.

**Multi-threaded Search**: `pvsengine.py` supports parallel root move evaluation via `ThreadPoolExecutor` when `max_threads > 1`. Uses persistent executor to reduce per-iteration overhead.

### GUI Integration (`main.py`, `gui.py`)

- `main.py` launches two engine subprocesses (configurable via `ENGINE_SPECS`)
- Commands are relayed via `send_command_with_lookup`
- Stdout is parsed in `engine_output_processor` to update the UI and relay moves
- The GUI never evaluates positions directly; all analysis is delegated to engine processes

### Self-Play Coordination (`main.py`)

`SelfPlayManager` (now defined in `main.py`) abstracts the "issue `ucinewgame` → `position fen <...>` → `go`" protocol:
1. Sends `ucinewgame` to each engine once at session start
2. Loops: send `position fen <current>` to side-to-move, send `go`, wait for `bestmove`, apply move, repeat
3. `stop()` dispatches `stop` to the in-flight engine, cancels pending moves, restores UI controls
4. Exports trace logs to `self_play_traces/` directory

Both GUI and headless (`python main.py --self-play`) harnesses use the same manager, ensuring behavioral consistency. Use `--include-perf-payload` when launching to preserve the raw JSON perf payload lines in logs and traces (they are filtered by default).

### Meta Configuration (`MetaParams`, `SearchTuning`)

The engine derives search parameters from high-level meta-parameters:
- **strength** (0-1): Controls search depth and time allocation
- **speed_bias** (0-1): Trades depth for faster responses
- **risk** (0-1): Adjusts pruning margins (futility, razoring)
- **stability** (0-1): Controls aspiration window size and history decay
- **tt_budget_mb**: Transposition table size in MB
- **avoid_repetition**: Penalizes moves that recreate earlier positions

**Presets**: `"balanced"`, `"fastblitz"`, `"tournament"`

To change engine behavior, modify the meta preset in `ChessEngine.__init__()` or pass different `MetaParams` to `HeuristicSearchStrategy.apply_config()`.

### Testing Strategy

**Headless Heuristic Trace Tests** (marked `dev`, ~50s total):
- `test_headless_self_play_debug_trace`: 12 plies from start position
- `test_headless_self_play_midgame_trace`: 8 plies from complex midgame

These tests use:
- `HeadlessSelfPlayUI`: Minimal callback interface for `SelfPlayManager`
- `EngineHarness`: Wraps `ChessEngine`, forces `debug on`, routes commands synchronously
- `SelfPlayTestHarness`: Sequences move generation deterministically

**Test Markers** (defined in `pytest.ini`):
- `dev`: Developer-only tests, expensive, run with `-m dev`
- `gui`: GUI tests requiring display/Xvfb
- `performance`: Performance benchmarks
- `logic`: Game logic tests
- `edge_case`: Edge case tests

### File Structure

**Core Modules:**
- `engine.py` / `pvsengine.py`: UCI engine with strategy stack and PV search backend
- `simple_engine.py`: Lightweight heuristic-only engine for head-to-head testing
- `main.py`: Launcher, boots two engine subprocesses and PySide6 GUI, and hosts the self-play manager/headless loop
- `gui.py`: PySide6 chess board UI and controls
- `chess_logic.py`: Board utilities (move validation, repetition, PGN/SAN export)
- `utils.py`: ANSI logging, Qt cleanup, glyph mapping, window placement

**Configuration:**
- `requirements.txt`: Python dependencies (chess, PySide6)
- `pytest.ini`: Test markers

**Data:**
- `gamestates/`: Reproducible test positions
- `self_play_traces/`: Auto-generated self-play logs
- `legacy/`: Reference-only historical code

## Important Implementation Details

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

### Repetition Avoidance

When `MetaParams.avoid_repetition=True`, the heuristic search applies penalties:
- Fivefold repetition: `MATE_SCORE` penalty (forced draw)
- Threefold repetition or fifty-move rule: `repetition_strong_penalty` (default 90cp)
- Twofold repetition: `repetition_penalty` (default 45cp)

Penalties are subtracted from the score *from the perspective of the side to move*, discouraging loops.

### Time Controls

Explicit time controls from `go` command override adaptive heuristics. If no time controls provided, the engine computes a budget based on:
- Complexity (legal move count)
- Phase (piece count)
- Tension (in check, mate threat)

Budget is clamped: `[min_time_limit, max_time_limit]`.

### Debug Logging

Set `debug on` to enable trace logging:
- Strategy selection rationale
- Per-depth search summaries (depth, nodes, time, NPS, PV)
- Aspiration window adjustments
- Budget consumption warnings
- Transposition table hit rates

In GUI, toggle via "Debug Mode" checkbox. In tests, `EngineHarness` forces debug mode for deterministic traces.

## Common Development Scenarios

### Adding a New Strategy

1. Subclass `MoveStrategy` in `engine.py`
2. Implement `is_applicable(context)` and `generate_move(board, context)`
3. Set `priority` (higher = earlier evaluation)
4. Optionally set `definitive=True` to short-circuit after this strategy
5. Register in `ChessEngine.__init__()` via `self.selector.register(...)`

### Modifying Search Heuristics

Edit `PVSearchBackend._evaluate()` in `pvsengine.py`. Current terms:
- Material + piece-square tables
- Bishop pair bonus
- Passed pawns
- Mobility
- King safety (tapered by game phase)

Add custom heuristics, update `SearchTuning` parameters, or adjust piece-square tables.

### Tuning Time Management

Adjust `SearchLimits.resolve_budget()` in `pvsengine.py`. Factors:
- `complexity_factor`: Legal move count buckets
- `phase_factor`: Piece count scaling
- `tension_factor`: Check/mate threat multiplier

Or modify `MetaParams` to change base allocations.

### Running Headless Self-Play

See `tests/test_engine_core.py` for reference:
```python
ui = HeadlessSelfPlayUI(board)
harness_white = EngineHarness(ChessEngine(), chess.WHITE)
harness_black = EngineHarness(ChessEngine(), chess.BLACK)
manager = SelfPlayTestHarness(ui, harness_white, harness_black)
manager.start()
manager.play_plies(12)
manager.stop()
```

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
