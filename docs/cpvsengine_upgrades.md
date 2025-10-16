# CPvsEngine Evaluation and Throughput Upgrade Roadmap


## Track 1: Profiling
### Establish evaluation parity anchors
- Capture current C-versus-Python evaluation score deltas across the benchmark suite using the compare tooling to quantify drift per phase and material class.
- Catalogue the full Python-side feature set—phase scaling, mobility, king safety, passed pawns, pawn shields, bishop pair bonuses, and related heuristics—paired with representative FEN anchors for regression checks.
- Maintain parity baselines so future evaluation changes have deterministic regression gates and auditable acceptance criteria. 

### Instrument hot paths and automate validation
- Profile after each feature drop to locate hotspots such as mobility counters, pawn scanning, cache churn, and related pressure points, guiding data-layout decisions.
- Experiment with struct-of-arrays layouts for incremental evaluation caches and measure false sharing when king-safety logic runs in parallel scenarios.
- Expand opt-in timers around new evaluation terms so nightly comparisons surface slowdowns with actionable traces.
- Assemble a regression harness spanning perft, Strategic Test Suite tactics, and the curated evaluation parity suite, wiring it into continuous integration for deterministic score checks.
- Schedule extended gauntlets against Python PVS, Sunfish, and other baselines to quantify Elo gains after each evaluation milestone, archiving results alongside the benchmark notebook.
- Document tuning scripts and publish best-known weight sets with release notes once the upgraded evaluation stabilises.


## Track 2: Performance & Evaluation
### Deliver tapered & incremental evaluation framework
- Introduce game-phase weighting by porting phase computation logic and establishing midgame/endgame weights on the C side.
- Extend piece-square tables to surface distinct midgame and endgame values or maintain dual tables with interpolated scoring, mirroring the Python implementation.
- Thread phase into the evaluation path while preserving tempo symmetry after scaling.
- Mirror search-state push/pop behaviour by storing material, phase, and PST deltas on the move stack so make/undo cycles avoid full-board rescans.
- Maintain lightweight caches for king squares, pawn bitboards, and mobility counts to keep quiescence evaluation predictable and ready for throughput mitigations.

### Expand feature coverage and tuning loops
- **Pawn structure:** implement passed-pawn detection with directional lane scans, doubled and isolated pawn penalties, and pawn-shield counts on the king files.
- **Mobility:** reuse legal-move counters and SEE data to score freedom per piece with asymmetric weights, tapering contributions once the phase exceeds 0.2.
- **King safety:** score attackers on the king, evaluate pawn-shield strength, and add endgame centralisation bonuses while gating expensive work behind phase thresholds.
- **Piece-specific tweaks:** add bishop-pair bonuses, trapped-piece detection, rook bonuses on open or semi-open files, and knight outpost incentives.
- Seed evaluation weights with the current Python defaults, expose them via a configurable header, and run targeted self-play plus Texel tuning loops tailored to C-side search horizons.
- Extend regression suites with positions stressing every new term so parity stays locked as features evolve.

### Reclaim nodes through incremental caching and move generation
- Prioritise incremental updates for piece-square tables, material, and phase within make-move routines to avoid O(64) board scans.
- Cache per-piece mobility and king-safety metadata on the stack, invalidating only affected entries so quiescence remains lean as evaluation richness grows.
- Replace 0x88 move generation with bitboard-based attack tables (precomputed magic sliders) to reduce branching and improve cache locality.
- Vectorise the SEE capture chain with bitwise operations and specialise pawn and king offsets to avoid repeated function calls.

### Optimise memory layout and search tables
- Upgrade the transposition table to use wider buckets with aging and 128-bit entries, cutting probe collisions while aligning with cache-line boundaries.
- Convert history and killer tables to contiguous per-ply arrays, improving hardware prefetch behaviour as depth increases.

### Scale parallel and speculative search
- Explore Young Brothers Wait Concept (YBWC) or comparable threaded search models once the single-thread baseline settles, guarding parallel builds behind feature flags.
- Enable parallel aspiration searches at the root with task stealing across cores, using split-point queues to minimise locking overhead.
- Investigate speculative move prefetching (copy-make) and optional multi-PV output while keeping overhead minimal so single-thread performance does not regress.

### Tune compiler and build profiles
- Evaluate `-Ofast`, profile-guided optimisation, and link-time optimisation builds while retaining a conservative profile without fast-math for correctness checks.
- Integrate hardware-counter analysis (L1 cache misses, branch mispredicts, and related metrics) via profiling tools such as `perf record` to steer future micro-optimisation passes.
