# Benchmark Harness Overhaul Plan

## 1. Context
- `benchmark_pvs.py` currently offers limited benchmarking/profiling flows and does not mirror the GUI/self-play orchestration of `go` commands, time budgeting, or the new JSON `info` payload emitted by the engine.
- Recent infra changes enforce an informative `info` string and emphasise deterministic budgeting (`SearchLimits.resolve_budget`); the harness should exercise those exact pathways when analysing runtime behaviour.
- Optimising the PVS engine now relies on rapidly iterating hypotheses across curated FENs, configurable budgets, and profiling hooks without rewriting scripts per experiment.

## 2. Desired Capabilities
- **Unified entrypoint** that can benchmark, profile, or run exploratory search loops using the same orchestration logic as the GUI/self-play manager.
- **Flexible budget control**: default to engine-managed time (matching GUI behaviour) but allow overrides for fixed movetime, depth, nodes, or total time, including “infinite” analysis.
- **Scenario-driven CLI** supporting per-FEN, batch FEN lists, or PGN/puzzles, with consistent logging/reporting and optional result export (JSON/CSV).
- **Engine telemetry capture** including the enforced JSON parameter blob, iterative deepening traces, and search statistics for post-run analysis.
- **Extensibility** so new experiment types (e.g., hyperparameter sweeps, concurrency tests) plug into a shared session/runner API.

## 3. Architectural Approach
- **Module layout**: split `benchmark_pvs.py` into (a) CLI parse layer, (b) session management that speaks UCI (mirroring `SelfPlayManager`), (c) scenario runners (benchmark, profile, free-form), and (d) reporters/exporters.
- **Engine session reuse**: factor a lightweight `EngineSession` class that reuses `engine_output_processor` conventions, issues `uci` / `ucinewgame` / `position` / `go` commands, and captures JSON `info` blobs.
- **Budget strategies**: encapsulate budget handling in pluggable “budget adapters” (e.g., GUI-style adaptive, fixed movetime, depth-limited) that map CLI flags to `go` arguments and optionally tweak `MetaParams`.
- **Extensible runners**: define abstract `Scenario` protocols (e.g., `prepare()`, `execute()`, `collect()`) so new experiments can compose existing building blocks without touching the CLI core.
- **Result schema**: store run metadata (timestamp, CLI args, engine config), per-position telemetry, and optional profiling artefacts; consider JSON Lines for machine ingestion plus human-readable summaries.

## 4. Implementation Roadmap
1. **Audit Current Script**  
   - Document existing CLI flags, UCI sequencing, and profiling steps; note gaps versus GUI/self-play orchestration.
2. **Define CLI & Configuration**  
   - Use `argparse` subcommands (`benchmark`, `profile`, `analyze`) or flag groups; support loading scenario configs from YAML/JSON.
3. **Engine Session Abstraction**  
   - Implement `EngineSession` with context management, logging hooks, JSON info parsing, and reuse of time-budget resolution code.
4. **Budget Adapter Layer**  
   - Map CLI options to UCI `go` payloads; include default adapter that defers to engine-managed budgeting.
5. **Scenario Implementations**  
   - Port existing benchmark/profile logic into new `Scenario` classes; add batch FEN runner and unlimited-analysis mode.
6. **Reporting & Export**  
   - Standardise console output, optional file exports, and profiling artefact paths; include the engine JSON blob in outputs.
7. **Documentation & Examples**  
   - Update `README.md`/`docs/testing_architecture.md` with usage patterns and sample workflows aligning with GUI expectation.

## 5. Instrumentation & Reporting
- Parse and surface the JSON `info` parameter block per depth or final iteration; optionally diff against requested budgets.
- Record per-depth metrics (NPS, nodes, selective depth) and budget utilisation to spot throttling or overrun anomalies.
- Provide toggles for saving raw UCI transcripts, aggregated summaries, and profiling stats for offline analysis.

## 6. Testing Strategy
- Add unit tests for CLI parsing, budget adapter selection, and scenario orchestration using mocked engine processes or fixtures from `conftest.py`.
- Integrate targeted regression tests that spawn the real engine with short FENs under deterministic budgets to ensure parity with GUI/self-play command ordering.
- Validate profiling pipeline via smoke tests that run cProfile capture in a temporary directory and verify artefact creation.

## 7. Open Questions / Risks
- Confirm whether profiling should capture multi-threaded runs and how to aggregate JSON telemetry when multiple engines run concurrently.
- Decide on storage format/location for long-running experiment outputs (potential need for rotation or naming scheme).
- Determine how meta-parameter overrides (`--preset balanced`) interact with GUI defaults and whether to surface per-run tuning overrides.

## 8. Immediate Next Steps
- Review `benchmark_pvs.py`, `main.py`, and `SelfPlayManager` to align command sequencing.
- Draft CLI spec and budget adapter interface for stakeholder feedback before large refactor begins.
- Prototype the `EngineSession` abstraction with basic benchmark mode to validate feasibility before migrating all scenarios.
