# Testing Improvements Plan — 2025-10-10

## Overview
Recent review of the automated test matrix shows solid coverage of UCI plumbing and GUI interactions but lighter checks on deep search behaviour, heuristic time budgeting, and regression tracking across releases. This plan captures the current state and outlines targeted enhancements so contributors can prioritise testing work.

## Current Coverage Snapshot
- **`tests/test_engine_core.py`** — exercises the full UCI surface (`uci`, `isready`, `position`, `go`, error paths) and validates strategy context construction, go-argument parsing, and self-play trace export. Integration harnesses simulate engines to ensure deterministic command sequencing.
- **`tests/test_self_play_manager.py`** — covers GUI coordination: start/stop flows, engine swapping, readiness checks, and manual-control toggles. Confirms signalling to the UI and ignores late moves post-stop.
- **`tests/test_search_backend.py`** — smoke-tests `PVSearchBackend` for move production, depth ≥1, and basic multi-thread consistency under a 1 s budget.
- **`tests/test_simple_engine.py`** — verifies the lightweight engine handles `position` commands, maintains legal boards, and emits a `bestmove`.
- **`tests/test_gui.py`** (Qt-only) — checks window layout, square interaction, invalid-move messaging, promotion dialog, undo/reset, and export scaffolding.

## Identified Gaps
1. **Budget Verification** — No assertions cover the `time_budget=` or `info string perf` output that the GUI/self-play pipeline emits, leaving heuristic regressions undetected.
2. **Search Behaviour Depth** — PV search tests stop at “returns a move,” without validating score trends, aspiration-window responses, or PV stability across iterations.
3. **Edge-Case Positions** — Limited fixtures for zugzwang, forced mates, repetitions, and high-branching midgames; strategy selection, null-move pruning, and repetition handling remain largely untested.
4. **Benchmark Regression Safety Nets** — Benchmark CLI output is not compared against historical baselines, so throughput or PV regressions rely on manual inspection.
5. **GUI Logging & Busy-State Handling** — Manual `Go` flow is not asserted to check for log formatting or prevention of overlapping searches when the engine reports busy.
6. **Generative/Property Checks** — Tests do not randomise positions to assert invariants such as legal `bestmove`, stable hash state, or absence of illegal board mutations.

## Recommended Enhancements
1. **Budget & Perf Assertions**
   - Extend self-play harness tests to read the latest `<X>_selfplay.txt` and assert expected `time_budget` ranges for opening, middlegame, and late positions.
   - Capture stdout from GUI-triggered `go` commands with mocked engines to verify `info string perf … time=X/Ys` formatting and readiness signalling.
2. **Search Behaviour Regression Tests**
   - Add table-driven tests covering aspiration-window fail-high/low paths, ensuring the engine widens windows and eventually converges.
   - Record PV/score sequences for curated FENs (e.g., tactic, fortress) and assert monotonic depth growth plus bounded score deltas.
3. **Edge-Case Fixtures**
   - Introduce `tests/fixtures/fen/` with deterministic scenarios (zugzwang, claimable repetition, forced mate in N, stalemate traps).
   - Write targeted pytest cases validating move legality, context flags (`in_check`, `opponent_mate_in_one_threat`), and time-budget adjustments for each fixture.
4. **Benchmark Snapshot Comparisons**
   - Create a lightweight harness that runs `benchmark_pvs.py benchmark --depth 5 --threads 1 --positions …` and compares nodes/NPS/PV against JSON “golden” data stored in `tests/data/benchmarks/`.
   - Allow configurable tolerances so small numeric drift is acceptable while detecting large regressions.
5. **GUI Busy-State & Logging Tests**
   - Mock engine processes inside GUI tests to emit staged outputs; assert that `Go` is disabled until `readyok` returns and that terminal logs follow the expected `SENDING/RECIEVED` pattern.
   - Validate that manual stop/start clears pending evaluations and updates the info indicator.
6. **Property-Based Checks**
   - Use `hypothesis` (or a custom generator if dependencies must remain minimal) to randomise legal midgame positions, run a short search, and assert invariants: legal `bestmove`, unchanged board state post-search, and non-negative node counts.

## Next Steps
1. Prioritise enhancements 1–3 to harden the GUI/self-play pipeline and catch budget regressions.
2. Prototype a benchmark snapshot test to assess feasibility and runtime overhead.
3. Revisit dependency policy before introducing property-based testing; if approved, integrate `hypothesis` in a targeted module.
4. Update contributor documentation (`AGENTS.md`) once new suites are in place so the expanded coverage is discoverable.

## Tracking
- Owner: TBD (assign once roadmap finalised)
- Target milestone: Align with the next performance tuning cycle to ensure benchmarks are gated by automated checks.
