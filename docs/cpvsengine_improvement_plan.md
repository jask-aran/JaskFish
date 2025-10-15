## cpvsengine Improvement Plan (Remaining Scope)

The July round of work implemented tactical sanity checks (SEE, quiescence pruning),
TT mate normalisation, selective search heuristics (null-move, LMR, history/killer),
aspiration windows, time allocation fixes, and richer instrumentation. The plan
below captures the work that still remains from the later phases of the original
roadmap.

### 1. Evaluation overhaul (bring parity with Python PVS)

1. **Audit & parity baselines**
   - Snapshot current C vs. Python eval scores over the benchmark suite (reuse
     `benchmark_pvs.py compare`) to quantify drift per phase/material class.
   - Catalogue the Python-side features we want (phase scaling, mobility,
     king safety, passed pawns, pawn shield, bishop pair bonuses, etc.) with
     representative FENs to use as regression anchors.

2. **Introduce tapered evaluation scaffolding**
   - Port the `_game_phase` logic and define phase weights in C (0.0–1.0).
   - Extend PST tables to expose separate midgame/endgame values or maintain
     dual tables with interpolated scoring like the Python engine.
   - Thread phase into `evaluate` and make sure tempo adjustments remain
     symmetric after scaling.

3. **Implement incremental state bookkeeping**
   - Mirror the Python `_SearchState` push/pop by storing material, phase, and
     PST deltas in the move stack so `make_move`/`undo_move` can update scores
     without full board scans.
   - Add lightweight caches for king squares, pawn bitboards, and mobility
     counts to avoid recomputation in quiescence.

4. **Bring in feature terms iteratively**
   - **Pawn structure:** passed pawns (directional lane scans), doubled/isolated
     penalties, and pawn shield counts for king files.
   - **Mobility:** reuse legal move counters and SEE data to score piece freedom
     with asymmetrical weights per piece type, tapering by phase > 0.2.
   - **King safety:** attackers-on-king, pawn shield strength, and endgame
     centralisation bonus; gate expensive work behind phase thresholds as in
     Python.
   - **Piece-specific tweaks:** bishop pair bonus, trapped piece detection,
     rook-on-open/semi-open files, knight outpost bonuses.

5. **Tune & validate**
   - Start with Python tuning defaults; expose weights in a config header.
   - Run targeted self-play/Texel loops to re-tune for C search horizons.
   - Extend regression set with positions stressing each term to ensure parity.

### 2. Advanced profiling & micro-optimisation
- Re-profile after each feature drop to locate evaluation hotspots (mobility
  counters, pawn scanning, cache churn) and steer data layout changes.
- Investigate struct-of-arrays layouts for incremental eval caches and measure
  false sharing when king safety runs in parallel.
- Extend instrumentation with opt-in timers around new eval terms so nightly
  comparisons can attribute slowdowns quickly.

### 3. Parallel and search-depth extensions
- Explore Young Brothers Wait Concept (YBWC) or similar threaded search once the
  single-thread baseline stabilises; guard behind a build flag.
- Investigate speculative move prefetching or multi-PV output with minimal
  overhead.

### 4. Validation & release automation
- Assemble a regression harness covering perft, STS tactics, and the curated
  evaluation parity suite used in step 1; wire it into CI for deterministic
  score checks.
- Run extended gauntlets against Python PVS, Sunfish, and other baselines to
  quantify Elo gains after each evaluation milestone, tracking results with the
  benchmark compare notebook.
- Document tuning scripts and publish best-known weight sets alongside release
  notes once the new evaluation stabilises.

### 5. Search throughput uplift (offset evaluation cost)
1. **Benchmark & regression tracking**
   - Add NPS tracking to `benchmark_pvs.py compare` outputs and store baselines
     before each eval feature drops; flag >5% regressions in CI.
   - Expand perf counters in the native engine (nodes, qnodes, move-gen calls,
     eval calls) so hotspots are visible when the new terms land.

2. **Incremental evaluation & caching**
   - Prioritise incremental PST/material/phase updates in `make_move` to avoid
     full-board scans, reducing eval calls from O(64) to O(1).
   - Cache per-piece mobility and king-safety metadata on the stack; invalidate
     with precise deltas to keep quiescence cheap.

3. **Move generation & SEE optimisations**
   - Replace 0x88 move gen with bitboard-based attack tables (precomputed magic
     sliders) to drop branching and improve cache locality.
   - Vectorise SEE capture chain using bit operations; specialise pawn/king
     offsets to avoid repeated function calls.

4. **Transposition & memory layout**
   - Upgrade TT to 8-way buckets with aging and 128-bit entries to cut probe
     collisions while keeping cache lines aligned.
   - Convert history/killer tables to contiguous arrays per ply to improve prefetch.

5. **Parallel and speculative search**
   - Enable parallel aspiration/YBWC at the root with task stealing across
     available cores; ensure locking is minimised via split point queues.
   - Experiment with speculative move prefetch (copy-make) or lazy SMP to
     expose multi-thread NPS wins once single-thread baseline is restored.

6. **Compiler & build tuning**
   - Evaluate `-Ofast`, profile-guided optimisation (PGO), and LTO builds; keep
     a fallback non-fast-math profile for correctness checks.
   - Integrate perf/`perf record` runs and hardware counter analysis (L1 misses,
     branch mispredicts) into the profiling loop to guide future micro-tuning.