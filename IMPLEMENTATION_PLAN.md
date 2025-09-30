# PVSengine Optimization Plan

## Phase 6: Comprehensive Instrumentation & Node Inflation Fix

### Current Status
- ✅ Engine functional (depth 5 consistently)
- ⚠️ 2x node inflation vs classic engine (16k vs 8k nodes)
- ⚠️ Missing detailed performance metrics

### Goals
1. **Add comprehensive stats tracking**
2. **Diagnose 2x node inflation**
3. **Increase search budgets** (engine has room to go deeper)
4. **Real-time performance monitoring**

---

## Metrics to Track

### Per-Move Summary Line
```
perf depth=5 sel=7 nodes=16160 nps=13477 time=1.20/1.40s score=2.5(+0.0) 
pv=g1f3(5) swaps=0 fail[L/H]=0/0 tt=41% cuts(tt:79,k:389,h:11,c:58)=537 
order[50/90]=0/0 q=49% eff=0.15
```

### Depth/Throughput
- `depth`: completed depth
- `sel_depth`: max quiescence depth reached
- `nodes`: regular nodes
- `nps`: nodes per second
- `time_used/budget`: e.g., "1.20/1.40s"

### Score Stability
- `score`: centipawns or mate score
- `delta`: change from previous depth
- `pv_length`: principal variation moves
- `pv_swaps`: how often best move changed

### Aspiration Window Health
- `fail_low/fail_high`: counts
- `window`: current aspiration size
- `research_overhead%`: time spent in re-searches

### Move Ordering Quality
- `root_move_rank`: where best move was in initial ordering
- `first_move_cut%`: beta cutoffs on first move
- `order_p50/p90`: median/90th percentile cutoff index

### Transposition Table
- `tt_hit%`: probe hits
- `tt_cut%`: cutoffs from TT
- `occupancy%`: table fullness
- `collisions`: replacement conflicts

### Cut Breakdown
- `tt`: TT move caused cutoff
- `killer`: Killer move caused cutoff  
- `history`: History heuristic move
- `capture`: Good capture (SEE)
- `null`: Null move pruning
- `futility`: Futility pruning
- `lmr`: LMR reductions
- `other`: Everything else

### LMR/Null Effectiveness
- `lmr_applied/researched`: reduction attempts vs full re-searches
- `null_tried/verified`: null move attempts vs successful prunes

### Quiescence Load
- `qnodes`: quiescence nodes
- `q%`: qnodes / (nodes + qnodes)
- `q_cutoffs`: stand-pat + move cutoffs

### Budget Profile (Time Breakdown)
- `order_ms`: move ordering time
- `eval_ms`: evaluation time
- `qsearch_ms`: quiescence time
- `alpha_ms`: alpha-beta time
- `overhead_ms`: other

### Composite Metrics
- `efficiency`: cp_gain per 1k nodes
- `tt_util`: tt_cut% + 0.5 * tt_hit%
- `pruning%`: pruned_moves / considered_moves
- `tactical%`: q% (quiescence ratio)
- `stability`: 1 / (1 + score_jitter)
- `order_idx`: first_move_cut%

---

## Implementation Steps

### Step 1: Create SearchStats Class
```python
@dataclass
class SearchStats:
    # Counters
    nodes: int = 0
    qnodes: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_cuts: int = 0
    
    # Cuts by type
    cuts_tt: int = 0
    cuts_killer: int = 0
    cuts_history: int = 0
    cuts_capture: int = 0
    cuts_null: int = 0
    cuts_futility: int = 0
    cuts_other: int = 0
    
    # Aspiration
    asp_fail_low: int = 0
    asp_fail_high: int = 0
    asp_researches: int = 0
    
    # LMR
    lmr_applied: int = 0
    lmr_researched: int = 0
    
    # Null move
    null_tried: int = 0
    null_success: int = 0
    
    # Move ordering
    cutoff_indices: List[int]
    first_move_cuts: int = 0
    total_beta_cuts: int = 0
    
    # Per-depth tracking
    depth_scores: List[float]
    depth_nodes: List[int]
    depth_times: List[float]
    pv_changes: int = 0
    
    # Timing
    time_order: float = 0.0
    time_eval: float = 0.0
    time_qsearch: float = 0.0
    time_alpha_beta: float = 0.0
```

### Step 2: Add Stats to _SearchState
- Replace scattered counters with SearchStats instance
- Track everything in one place

### Step 3: Instrument Alpha-Beta
- Track cut types (TT/killer/history/capture/etc)
- Track cutoff move indices
- Time each phase

### Step 4: Instrument Root Search
- Track root move ordering
- Time per root move
- Track aspiration failures

### Step 5: Add Reporting Methods
```python
def print_comprehensive_summary(stats, depth, score, pv, time, budget):
    # One-line summary
    # Multi-line detailed breakdown
    # Composite metrics
```

### Step 6: Diagnose Node Inflation
- Compare move ordering scores between engines
- Check if TT is working properly
- Verify history/killer heuristics

### Step 7: Increase Budgets
- Adjust base_time_limit multiplier
- Increase max_time_limit
- Test depth improvements

---

## Expected Outcomes

1. **Visibility**: Every move shows comprehensive stats
2. **Diagnosis**: Clear view of where nodes are being spent
3. **Tuning**: Data-driven parameter adjustments
4. **Performance**: Reduced node inflation through better ordering
5. **Depth**: Deeper searches with increased budgets

---

## Testing Approach

1. Run self-play with new instrumentation
2. Compare metrics between pvsengine and engine.py
3. Identify specific differences causing node inflation
4. Adjust parameters based on data
5. Verify improvements

---

*Next: Implement SearchStats class and integrate into _SearchState*
