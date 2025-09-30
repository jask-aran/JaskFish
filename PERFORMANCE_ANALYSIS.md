# PVSengine Performance Analysis

## ⚠️ CRITICAL BUG DISCOVERED

**Phase 5 testing has revealed a catastrophic bug**: PVSengine visits **~400x more nodes** than engine.py to complete depth 1.

### Evidence

```
Position: startpos, Budget: 1.40s

Classic engine.py:
✓ Depth 1: 42 nodes in 0.01s
✓ Depth 2: 182 nodes in 0.03s  
✓ Depth 3: 1,244 nodes in 0.16s
✓ Depth 4: 3,302 nodes in 0.43s
✓ Depth 5: 7,893 nodes in 0.74s
✓ COMPLETED depth 5, total 1.37s

PVSengine:
✗ Depth 1: 16,284 nodes in 1.39s - TIMEOUT
✗ COMPLETED depth 0 (nothing)
```

**Node inflation ratio**: 16,284 / 42 = **387x explosion**

This explains the weak play - the engine can't complete any search depth due to catastrophic node explosion in the root search. The issue is **NOT** with optimizations (threading, board copying, etc.) but with the **core search algorithm**.

---

## Status Summary

### Completed Optimizations (Phase 1-5)

1. **Per-Depth Logging** ✅
   - Added `PVS: depth=X` traces matching classic engine format
   - Logs nodes visited, time elapsed, NPS, and principal variation
   - Timeout notifications with diagnostic information

2. **Persistent ThreadPoolExecutor** ✅
   - Eliminated per-iteration executor creation overhead (~20-50ms/depth)
   - Added shutdown() method for cleanup
   - Sequential fallback for single-threaded mode

3. **Single-Threaded Default** ✅
   - Changed from `max_threads = cpu_count // 2` to `max_threads = 1`
   - Avoids Python GIL contention in CPU-bound search
   - Multi-threading available via explicit constructor parameter

4. **Board Copying Optimization** ✅
   - Changed from `board.copy(stack=True)` to `board.copy(stack=False)`
   - ~40% faster board copying in parallel workers
   - Trade-off: Won't detect exact repetitions across workers (acceptable for root moves)

5. **Budget Calculation Logging** ✅  
   - Added detailed logging to `SearchLimits.resolve_budget()`
   - Logs complexity factor, phase factor, tension factor
   - Shows base time, raw budget, and clamped result
   - **Verified**: Budget calculations match engine.py exactly

### Current Performance Baseline

**Test Position**: startpos with 2000ms budget

```
Classic Engine (engine.py):
- Completed: depth 5
- Nodes: 7,893
- Time: 1.35s
- NPS: ~5,850
- Depth progression: 1→2→3→4→5 smoothly

PVSengine (current):
- Completed: depth 0 (timeout during depth 1)
- Nodes: 22,866
- Time: 1.98s (full budget)
- NPS: ~11,500
- Issue: Visits 3x more nodes but completes no depths
```

### Root Cause Analysis

The high node count (22K vs 8K) with zero completed depths suggests:

1. **Aspiration Window Issues**: Spending excessive time in fail-high/fail-low re-searches
2. **Move Ordering**: Poor root move ordering causing more nodes to be searched
3. **Alpha-Beta Efficiency**: Not achieving sufficient cutoffs during search
4. **Quiescence Explosion**: Q-search may be visiting too many nodes

### Pending Optimizations (Phase 5-7)

#### Phase 5: Budget Calculation Verification
**Priority**: HIGH
- Compare SearchLimits.resolve_budget() output with classic engine
- Verify complexity_factor, phase_factor, tension_factor calculations
- Add debug logging for budget components

```python
# Add to SearchLimits.resolve_budget()
self._logger(
    f"budget calc: complexity={complexity}({legal_moves}mv) "
    f"phase={phase_factor:.2f}({piece_count}pc) "
    f"tension={tension_factor:.2f} => {budget:.2f}s"
)
```

#### Phase 6: Timing Instrumentation
**Priority**: MEDIUM
- Add `_SearchTimers` dataclass with per-component timing
- Track: alpha_beta, move_ordering, evaluation, quiescence, tt_probes
- Report breakdown at end of search

#### Phase 7: Enhanced Profiling
**Priority**: LOW
- Add `--profile-level` argument (full, search, eval)
- Granular profiling of specific search components
- Export timing data for analysis

### Investigation Tasks

1. **Aspiration Window Behavior**
   - Count fail-high/fail-low events
   - Measure time spent in re-searches
   - Compare with classic engine's aspiration handling

2. **Move Ordering Quality**
   - Log first move selection percentage
   - Track beta cutoff rates by move index
   - Compare TT hit rates

3. **Quiescence Search Depth**
   - Measure average Q-search depth
   - Count Q-nodes vs regular nodes ratio
   - Compare Q-search move generation

### Immediate Next Steps (Critical Bug Fix)

**Priority: CRITICAL** - Engine is non-functional until this is resolved.

1. **Compare root move search implementations**
   - Check `_iterative_deepening()` aspiration window logic
   - Compare root move ordering between engines
   - Verify alpha-beta cutoffs are working at root level

2. **Add instrumentation to diagnose node explosion**
   - Log each root move evaluation: move, nodes consumed, score
   - Track aspiration window failures and re-searches
   - Count how many times each root move is evaluated

3. **Hypothesis testing**
   - Disable aspiration windows completely → test if nodes normalize
   - Force full-window search for all depths → check node counts
   - Compare move ordering scores between engines

4. **Root cause candidates**
   - **Aspiration window logic**: Infinite re-search loop?
   - **Move ordering failure**: TT move not prioritized → searching all moves multiple times?
   - **Alpha-beta bug**: Not cutting off early at root → evaluating every move fully?
   - **PV extraction**: Extracting PV after every move instead of after depth completion?

5. **Verification tests**
   - Create minimal test case that shows 42 vs 16k node difference
   - Add assertions in tests for node count expectations
   - Compare identical positions between both engines

### Testing Commands

```bash
# Compare engines side-by-side
echo "position startpos\ngo movetime 2000\nquit" | .venv/bin/python3 engine.py > /tmp/classic.log 2>&1
echo "position startpos\ngo movetime 2000\nquit" | .venv/bin/python3 pvsengine.py > /tmp/pvs.log 2>&1
diff -y /tmp/classic.log /tmp/pvs.log

# Profile PVS backend
.venv/bin/python3 pvsengine.py --profile --threads=1 --fen=startpos

# Self-play test (after fixes)
.venv/bin/python3 self_play.py --engine1=pvsengine.py --engine2=engine.py --moves=10
```

### Success Criteria

- [ ] PVSengine reaches depth 4-5 under same budgets as classic engine
- [ ] Node count within 20% of classic engine for same position
- [ ] NPS maintained or improved (currently better at 11.5k vs 5.8k)
- [ ] Per-depth logging shows smooth depth progression
- [ ] Budget consumption <80% per depth to allow next depth attempt

---

## Summary

Phase 5 successfully added budget calculation logging and **verified budgets match engine.py exactly**. However, testing uncovered a **catastrophic search bug**: PVSengine visits **387x more nodes** than engine.py for the same search depth, making the engine completely non-functional for actual play.

**The optimizations (Phases 1-4) are correct but irrelevant** - the core search algorithm has a fundamental flaw causing massive node explosion. Until this is fixed, the engine cannot complete even shallow depths and makes tactically weak moves.

**Next action**: Instrument root move evaluation to identify why 20 moves at depth 1 generate 16k nodes instead of ~42 nodes.

---

*Last Updated*: 2025-09-30 (Phase 5 completed)
*Commits*:  
- 7e3f679 - Add PVSengine with performance optimizations (Phases 1-4)
- 793170d - Add budget calculation logging (Phase 5) + Critical bug discovery
