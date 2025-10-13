# PVS Engine Improvement Plan — Status 2025-10-13

_Last updated: 2025-10-13 12:57:25 UTC on Linux Desktop (WSL2, 6.6.87.2-microsoft-standard-WSL2) running Python 3.12.3._

## 1. Current Performance Snapshot

### 1.1 Benchmark summary (`python benchmark_pvs.py benchmark --depth 5`, 1 thread)
| Scenario | Phase | Depth (search/sel) | Nodes (total) | Time (s) | NPS | Score Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Opening – Semi-Slav structure | Opening | 5 / 11 | 54,823 | 4.112 | 13,331 | +15.0 cp |
| Middlegame – Kingside pressure | Middlegame | 5 / 11 | 50,495 | 3.929 | 12,850 | +75.6 cp |
| Middlegame – Tactical imbalance | Middlegame | 5 / 10 | 61,628 | 4.799 | 12,842 | 0.0 cp |
| Middlegame – Opposite-wing plans | Middlegame | 5 / 11 | 47,577 | 3.649 | 13,037 | –5.0 cp |
| Endgame – Complex rook ending | Endgame | 5 / 10 | 19,595 | 1.005 | 19,505 | +8.0 cp |
| Endgame – Knight vs pawns | Endgame | 7 / 9 | 10,051 | 0.374 | 26,894 | +14.0 cp |

Observations:
- Compared with the May 2024 baseline (≈10 kN/s middlegame), the engine sustains ~13 kN/s while reliably finishing depth 5 and occasionally depth 7 in low-branching endgames, aligning with the user’s reported ~30 % NPS gain.
- Time budgets are being consumed almost entirely (≤0.02 s slack), validating the recent `depth_stop_ratio` and stop-handling work.

### 1.2 Profiling highlights (`python benchmark_pvs.py profile --positions kingside_pressure --threads 1`)
- Sample outputs:
  - Opening Semi-Slav: depth 4 in 7.06 s, NPS 2.4 k.
  - Kingside Pressure: depth 3 in 6.77 s, NPS 1.9 k.
  - Knight vs pawns: depth 8 in 1.62 s, NPS 6.5 k.
- Dominant cumulative costs (Opening scenario):
  - `_alpha_beta` + `_quiescence`: 6.95 s combined.
  - `python-chess` move APIs: `generate_legal_moves` (2.07 s), `generate_pseudo_legal_moves` (1.32 s), `board.push` (1.29 s), `gives_check` (1.60 s), `attackers_mask` (0.28 s).
  - Engine-side helpers: `_order_moves` (2.19 s), `_mobility_term` (1.74 s), `_passed_pawn_bonus` (0.32 s), `_generate_qmoves` (0.78 s).
- Takeaway: python-chess move generation/push remain the heaviest bottlenecks despite caching and SearchBoard scaffolding.

## 2. Completed Improvements

| Area | Status | Notes |
| --- | --- | --- |
| SearchBoard scaffold | ✅ | `_SearchBoard` mirrors key bitboards, castling, ep, hash; linked to `_SearchState.push_eval/pop_eval` to support incremental stats (`pvsengine.py:1030-1480`). Hash verification still reuses python-chess but groundwork for standalone state exists. |
| Incremental evaluation | ✅ (material/PST/bishops) | Push/pop deltas update material, PST, bishop counters; evaluation caches (OrderedDict) reduce repeat computations (`pvsengine.py:1272-2290`). |
| Quiescence enhancements | ✅ | Custom capture/promotion/check generators with SEE pruning, q-move cache, quiet-check enumeration using SearchBoard bitboards (`pvsengine.py:2421-2525`). |
| Null/LMR heuristics | ✅ | Null-move gating via tuning, late-move reductions with history-assisted re-search, depth-based killers/history heuristics updated (`pvsengine.py:1998-2149`). |
| Time management | ✅ | `SearchLimits` derived from preset tuning; `depth_stop_ratio` ensures margin; `profile_search` now reuses configured limits (fixes recent regression). |
| UCI responsiveness | ✅ | `stop`, `quit`, and shared stop event implemented; worker join ensures immediate halts (`pvsengine.py:2888-2965`). |
| Benchmark harness | ✅ | `benchmark_pvs.py` covers curated scenarios, CLI wrappers for profiling, integrates with presets. |

## 3. Outstanding Workstreams

### 3.1 Python-chess Dependency Reduction
- **Root move ordering & quiescence** still call `board.legal_moves`, `board.gives_check`, `board.is_legal`. Replace with `_SearchBoard`-backed enumerators to cut 2–3 s per search in profiling traces.
- **Push/pop**: every node triggers python-chess `push`/`pop`. Implement make/unmake directly on `_SearchBoard` (including castling, en passant, promotions, EP squares, repetition hashes). Use python-chess for validation only during testing.
- **Check detection**: fold legality and check detection into move generation (pin masks, occupied ray scans) to avoid repeated `board.gives_check` calls.

### 3.2 Full Incremental Feature Stack
- Track passed-pawn lanes, mobility differentials, king-safety terms inside `_SearchState` so `_evaluate` becomes a pure delta function (no fresh board scans). This will also shrink the OrderedDict cache churn.
- With mobility cached per push, remove `board.push(chess.Move.null())` in `_mobility_term` by evaluating opponent mobility via precomputed attack bitboards.

### 3.3 Search Heuristic Refinements
- **Hyperparameter re-tuning**: explore higher `MetaParams.strength` / lower `speed_bias` now that throughput improved; confirm depth scaling via `benchmark_pvs.py benchmark --depth 6` and targeted `--positions`.
- **Quiescence coverage**: reintroduce efficient quiet-check generation for pinned pieces/underpromotions once movegen is custom; ensure SEE thresholds are retuned afterwards.
- **Transposition table**: evaluate raising TT size and storing incremental evaluation to reduce `_evaluate` calls during re-searches.

### 3.4 Validation & Tooling
- Build parity tests comparing `_SearchBoard` move lists and check flags against python-chess across curated FEN suites (normal, in-check, pins, EP, promotions, castling).
- Automate benchmark logging (nodes, NPS, depth per scenario) after each major change to maintain a performance timeline.
- Capture profile snapshots post-refactor to ensure python-chess hotspots collapse as custom generators roll out.

## 4. Near-Term Action Plan
1. **Custom move enumeration (captures + quiets)**  
   - Extend `_generate_capture_moves` to operate purely on `_SearchBoard` bitboards.  
   - Implement `_generate_quiet_moves`/`_order_moves` using SearchBoard data, including check detection via attack masks and pin tracking.  
   - Validate against python-chess via dedicated tests before switching default path.
2. **Make/unmake without python-chess**  
   - Add `_SearchBoard.apply_move` returning metadata for unmake; update `_SearchState.push_eval/pop_eval` to avoid calling python-chess during search except for verification in debug mode.  
   - Update repetition bookkeeping to use SearchBoard hash exclusively.
3. **Incremental feature deltas**  
   - Maintain passed-pawn shields, mobility counts, king-safety metrics on the eval stack.  
   - Refactor `_mobility_term`, `_passed_pawn_bonus`, `_king_safety` to read from the incremental data instead of recomputing from the board.
4. **Re-benchmark & profile**  
   - After each milestone above, run:  
     - `python benchmark_pvs.py benchmark --depth 5` (baseline)  
     - `python benchmark_pvs.py benchmark --depth 6 --positions kingside_pressure` (depth stretch)  
     - `python benchmark_pvs.py profile --positions kingside_pressure` (hotspot verification)  
   - Log timestamped results back into this document.

## 5. Long-Term Vision
- Replace python-chess move generation entirely with SearchBoard-native routines inspired by Sunfish’s 0x88 patterns and bitboard attacks, retaining python-chess only for FEN I/O and fallback verification.
- Explore C-extension or Rust move generator once Python implementation proves stable, targeting ≥25 kN/s in middlegame benchmarks and consistent depth 6+ within current budgets.
- Integrate automated tuning (e.g., self-play gradient descent) once throughput plateaus, ensuring evaluation parameters adapt to the faster search pipeline.
