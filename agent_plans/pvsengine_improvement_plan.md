# PVS Engine Improvement Plan

## 1. Context
- Recent analysis of Sunfish’s architecture (`docs/sunfish_engine_analysis.md`) highlighted how a compact engine derives strength from incremental evaluation and aggressive pruning rather than elaborate piece-square tables (PSTs).
- Self-play logs (e.g., `self_play_traces/3_selfplay.txt`) show PVS and HS entering nearly identical knight-first openings due to shared heuristics and deterministic search.

## 2. Benchmark & Profiling Findings (2024-05 harness)
- **Depth stalls under default budgeting**: The curated “Kingside pressure” run (`python benchmark_pvs.py benchmark --positions kingside_pressure --echo-info`) repeatedly halts at depth 3 with a depth‑4 timeout after ~2.0 s despite a 3.22 s budget. This is consistent with the dynamic budget resolver (`SearchLimits.resolve_budget`, `pvsengine.py:279-338`) assigning ~3.2 s and the deadline guard firing inside `_iterative_deepening` (`pvsengine.py:1173-1306`) before depth 4 can complete.
- **Stop command absent**: The engine never recognises `stop`; the UCI dispatch table omits it (`pvsengine.py:2103-2112`), so `go infinite` runs until the depth loop exhausts its search_depth. Our head-to-head test on the same FEN needed ~138 s to finish depth 7 even after issuing `stop`, proving we currently rely entirely on deadline expiry.
- **Throughput bottlenecks**: Profiling (`python benchmark_pvs.py profile --positions kingside_pressure`) shows 3.2 M calls in 3.3 s, dominated by `_alpha_beta`/`_quiescence` (`pvsengine.py:1454`, `pvsengine.py:1647`) plus `python-chess` move generation (`Board.generate_legal_moves`/`push`). `_evaluate` (`pvsengine.py:1698`) alone accounts for ~1.45 s over 3 k evaluations, which keeps depth 3 expensive.
- **JSON payload confirms low NPS**: The perf telemetry for the same scenario reports 9 345 total nodes, 2.9 kN/s, and depth delta +62.5 cp; the engine never reaches the configured `search_depth=7` derived from the default meta preset (`build_search_tuning`, `pvsengine.py:405-479`).

## 3. Recent Improvements (2024-05)
- **Budget parity + stop support**: `depth_stop_ratio` now allows 60–85 % of the allocated time before aborting the next iteration (`pvsengine.py:420-454`), and the UCI façade wires `stop`/`quit` into a shared event with worker joins (`pvsengine.py:2081-2310`). Benchmarks and hand-issued UCI sessions now halt immediately when a stop arrives.
- **Evaluation caching**: `_evaluate` switched to bitboard accumulation (no `piece_map` scans) and writes to a 100 K-entry LRU keyed by `board._transposition_key()` (`pvsengine.py:1730-1794`). Mobility and king-safety reuse the same key-based caches to avoid repeated legal-move counting (`pvsengine.py:1830-1892`).
- **SEE / quiescence caching**: SEE values are memoised per hash (`pvsengine.py:1912-1947`), and quiescence move ordering caches the sorted capture/order list (`pvsengine.py:1894-1930`). Captures are now generated via a custom bitboard walker rather than iterating `board.legal_moves` (`pvsengine.py:1932-1994`).
- **Prototype custom movegen**: `HeuristicSearchStrategy._generate_capture_moves` builds legal captures through direct bitboard attacks plus legality checks, dodging the full python-chess generator. Quiet promotions are derived explicitly from promotion ranks (`pvsengine.py:1996-2013`).
- **Benchmark delta**: On the curated set (`benchmark_pvs.py benchmark --max-wait 20`), middlegame nodes rose by 25–40 % with unchanged budgets (e.g., Kingside pressure 11.4 K → 17.5 K nodes, 2.9 kN/s → 4.5 kN/s) and Opening Semi-Slav now reaches depth 4. Profiling still highlights `board.generate_legal_moves/push/gives_check` as the dominant cost, but `_evaluate` shrank from 2.46 s to ~1.92 s per profile run.

## 4. Open Areas & Next Experiments
1. **Python-chess hotspots**
   - `board.generate_legal_moves`, `board.generate_pseudo_legal_moves`, and `board.push/pop` remain the heaviest routines (~2 s combined per 6.7 s profile). Replacing these with a compiled generator or a full Sunfish-style move engine is the top throughput lever.
   - `board.gives_check` and `board.attacks` still fire for almost every move. We should fold check detection into the custom generator (precompute attack masks and reuse SEE deltas) to avoid repeated python-chess calls.
2. **Incremental feature stack**
   - Current caching avoids recomputation but still rebuilds PST totals the first time a node appears. Introduce a feature stack tied to the search push/pop cycle (material, PST, bishop counts, phase counters) so `_evaluate` becomes a pure delta update per move.
   - With a feature stack in place, we can expose phase and material deltas to mobility/king-safety without extra board queries.
3. **Quiescence coverage**
   - Quiet checking moves are currently omitted from the prototype generator. Re-introduce a lightweight check finder (e.g., targeting the opposing king square with ray masks) to keep tactical coverage while avoiding full `board.legal_moves` scans.
4. **Hyperparameter follow-up**
   - Re-run `benchmark_pvs.py` with higher `MetaParams.strength` / lower `speed_bias` to gauge depth scaling now that throughput improved.
   - Compare quiescence SEE thresholds after caching (current -50 cp) and consider tuning to reduce `board.gives_check` invocations further.

## 5. PST Strategy
- Maintain current tables while profiling their contribution; if future tuning is warranted, derive adjustments through data-driven methods (e.g., self-play gradient tuning) rather than importing Sunfish weights.
- Any PST experiments must subtract base material and align indices before comparison; reference section 7 of the Sunfish analysis for transformation requirements.

## 6. Next Actions
- Finish incremental feature stack integration (push/pop deltas + phase counters) and measure impact on `_evaluate` plus `_passed_pawn_bonus`.
- Extend the custom move generator beyond captures/promotions—start with knight/rook checking moves derived from the opponent king ray to cut remaining `board.legal_moves` reliance.
- Prototype a shared-memory or C-extension move generator for bulk profiling; collect before/after stats on `generate_legal_moves` and `push/pop`.
- Capture benchmark + profile snapshots after each major change to maintain a historical performance log (nodes, depth, NPS, high-cost functions).

## 7. Python-chess Escape Blueprint (2025-10-12)

- **Current blockers**: `_generate_quiet_checks` still loops over `board.generate_legal_moves()` and filters with `board.gives_check()`, while the main search path triggers `board.push()`/`pop()` for every node. Profiles identify these three python-chess calls as ~2.75 s of a 6.7 s search sample.
- **SearchBoard scaffold**: Introduce a lightweight bitboard-backed state (`SearchBoard`) that mirrors python-chess fields (piece bitboards, occupancy, side-to-move, castling flags, ep square, hash). Extend `_SearchState.push_eval/pop_eval` to keep this structure in sync so evaluation stays incremental even when python-chess is phased out.
- **Custom legal move generation**:
  - Precompute directional masks for sliders plus knight/king attack tables we already use in evaluation.
  - Emit pseudo-legal moves per piece type directly from bitboards, tagging capture/quiet/promotion subsets as we do today.
  - Enforce legality by tracking pinned pieces and verifying king safety via fast attack maps—no `board.gives_check()` calls.
  - Retain python-chess as an assertion-only fallback until unit tests confirm parity on a FEN suite (normal, in-check, pinned, promotions, en-passant).
- **Push/Pop replacement**: Once `SearchBoard` can make/unmake moves (including castling, EP, promotions) and updates Zobrist keys plus repetition stacks, swap search routines to operate on it. Keep python-chess only for FEN I/O/UCI echo until confidence is high.
- **Sunfish lessons**: Reuse Sunfish’s 0x88 movegen patterns (direction lists, inline pawn logic, rook/king castling hooks) as design guidance, but adapt them to our bitboard structure so evaluation deltas and ordering heuristics remain compatible.
- **Validation plan**: 
  1. Build parity tests that compare move lists and check flags between python-chess and the new generator on curated positions.
  2. Profile after switching captures/quiet checks to the new path, ensuring the python-chess hotspot times collapse.
  3. Record before/after benchmarks (depth, NPS) to verify throughput gains offset the refactor risk.
