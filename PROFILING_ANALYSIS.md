# PVS Engine Performance Profiling Analysis

## Overview

Profiling was run on the starting position with depth 5, completing in ~2.4 seconds with 5,280 nodes searched.

**Command**: `python pvsengine.py --profile --threads 1`

## Top Performance Bottlenecks

### 1. **chess.gives_check()** - 0.853s (35.9% of total time)
- Called 75,527 times
- Used in quiescence search and move ordering
- **Optimization opportunity**: Cache check status or reduce redundant calls

### 2. **_evaluate()** - 0.817s (34.4% of total time)
- Called 4,455 times
- Already has eval cache (100K entries)
- Includes expensive calls to:
  - `_mobility_term()`: 0.331s (13.9%)
  - `_king_safety()`: Included in evaluate overhead
  - `_passed_pawn_bonus()`: Included in evaluate overhead

### 3. **_generate_qmoves()** - 0.799s (33.6% of total time)
- Called 2,208 times in quiescence search
- Generates and scores all legal captures/checks
- **Optimization opportunity**: Early pruning, better move generation

### 4. **chess.push()** - 0.608s (25.6% of total time)
- Called 84,377 times
- Core board manipulation (unavoidable but expensive)

### 5. **chess.generate_legal_moves()** - 0.606s (25.5% of total time)
- Called 171,264 times
- Used for legal move generation and counting
- **Optimization opportunity**: Cache legal move count

### 6. **_order_moves()** - 0.420s (17.7% of total time)
- Called 1,050 times
- Sorting overhead: 0.356s (15.0%)
- **Optimization opportunity**: Partial sorting or lazy evaluation

### 7. **_mobility_term()** - 0.331s (13.9% of total time)
- Called 1,850 times
- Generates legal moves twice (for us and opponent)
- **Optimization opportunity**: Skip in endgames, cache results

### 8. **zobrist_hash()** - 0.231s (9.7% of total time)
- Called 5,871 times for TT probes
- Relatively efficient but high call count

## Performance Breakdown by Category

### Time Distribution:
1. **Chess library calls** (gives_check, push, legal_moves): ~2.07s (87%)
2. **Evaluation** (_evaluate, mobility, king_safety): ~0.82s (34%)
3. **Move generation/ordering**: ~1.22s (51%)
4. **Search logic** (alpha_beta, quiescence): ~0.03s (1%)

### Node Metrics:
- **Total nodes**: 5,280 regular + ~3,874 quiescence = 9,154 total
- **NPS**: 2,222 nodes/sec (regular nodes only)
- **Effective NPS**: ~3,850 total nodes/sec
- **Time per node**: ~0.45ms (regular), ~0.26ms (total)

## Optimization Recommendations

### High Impact (>10% improvement potential):

1. **Reduce gives_check() calls** (35.9% time)
   - Cache check status in TT entries
   - Skip in deep quiescence (only check captures)
   - Pre-compute check status during move generation

2. **Optimize mobility calculation** (13.9% time)
   - Skip entirely in endgame (phase < 0.3) ✅ Already done
   - Cache legal_moves count in board state
   - Use piece mobility approximation instead of full legal moves

3. **Improve move ordering** (17.7% time)
   - Use partial sorting (only sort top N moves)
   - Implement lazy move generation
   - Better history heuristic initialization

### Medium Impact (5-10% improvement):

4. **Optimize qmoves generation** (33.6% time)
   - More aggressive SEE pruning
   - Skip delta pruning in check
   - Generate captures only (skip quiet checks in deep q-search)

5. **Reduce eval cache misses**
   - Current: 100K entries, OrderedDict LRU
   - Consider increasing cache size or better eviction policy
   - Profile cache hit rate

6. **Optimize king safety** (included in eval)
   - Already skips in endgame ✅
   - Consider simpler attacker counting
   - Cache pawn shield patterns

### Low Impact (<5% improvement):

7. **TT optimization**
   - Current hit rate: Unknown (needs instrumentation)
   - Consider packed entry format
   - Better replacement scheme

8. **Reduce board.copy() overhead**
   - Use stack=False where safe ✅ Already done in parallel search
   - Minimize copies in tight loops

## Current Optimizations Already in Place ✅

1. **Eval cache**: 100K entries with OrderedDict LRU eviction
2. **King safety skipped in endgame**: phase < 0.3 threshold
3. **Stack-free board copies**: Used in parallel root search
4. **Mobility skipped in endgame**: phase > 0.2 threshold
5. **Persistent thread pool**: Avoids executor creation overhead

## Profiling Commands

### Run profiling on starting position:
```bash
python pvsengine.py --profile
```

### Profile with custom position:
```bash
python pvsengine.py --profile --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
```

### Profile with multiple threads:
```bash
python pvsengine.py --profile --threads 4
```

### Using benchmark script:
```bash
python benchmark_pvs.py 5
```

## Next Steps

1. **Add instrumentation** to track:
   - TT hit rate per depth
   - Eval cache hit rate
   - Move ordering effectiveness (first-move cutoff rate) ✅ Already tracked
   - gives_check() call frequency

2. **Experiment with optimizations**:
   - Implement check caching in TT
   - Try partial move sorting
   - Test mobility approximation methods

3. **Compare performance across positions**:
   - Opening (current benchmark)
   - Middlegame tactical
   - Endgame

4. **Profile parallel search** (threads > 1):
   - Measure overhead and speedup
   - Identify lock contention
   - Test scalability

## Conclusion

The engine spends most time in chess library calls (gives_check, legal_moves generation) and evaluation. The search logic itself is efficient. Main optimization targets:

1. **Reduce gives_check() overhead** - biggest single bottleneck
2. **Cache mobility calculations** - expensive legal move generation
3. **Improve move ordering** - reduce sorting overhead

The current implementation already has good optimizations (eval cache, endgame pruning, LRU eviction). Focus on reducing redundant chess library calls for maximum impact.
