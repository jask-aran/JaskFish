## cpvsengine Improvement Plan (Remaining Scope)

The July round of work implemented tactical sanity checks (SEE, quiescence pruning),
TT mate normalisation, selective search heuristics (null-move, LMR, history/killer),
aspiration windows, time allocation fixes, and richer instrumentation. The plan
below captures the work that still remains from the later phases of the original
roadmap.

### 1. Evaluation overhaul
- Introduce tapered midgame/endgame evaluation with phase scaling.
- Add mobility, king-safety, pawn-structure, passed-pawn, and trapped-piece
  heuristics; tune weights via Texel or self-play data.
- Implement incremental evaluation updates to keep new features cheap at node
  boundaries.

### 2. Advanced profiling & micro-optimisation
- Re-profile after the evaluation overhaul to identify hotspots (attack maps,
  TT contention, cache misses) and guide layout changes.
- Optimise board/state structures for cache friendliness; consider SIMD for
  pawn/king evaluation paths if beneficial.
- Expand instrumentation with opt-in lightweight timers suitable for CI runs.

### 3. Parallel and search-depth extensions
- Explore Young Brothers Wait Concept (YBWC) or similar threaded search once the
  single-thread baseline stabilises; guard behind a build flag.
- Investigate speculative move prefetching or multi-PV output with minimal
  overhead.

### 4. Validation & release automation
- Assemble a regression harness covering perft, STS tactics, and long-form
  self-play, integrated with CI.
- Run extended gauntlets against Python PVS, Sunfish, and other baselines to
  quantify Elo gains post-evaluation overhaul.
- Document tuning scripts and publish best-known weight sets alongside release
  notes.