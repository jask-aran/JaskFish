# PVS Engine Improvement Plan

## 1. Context
- Recent analysis of Sunfish’s architecture (`agent_plans/sunfish_engine_analysis.md`) highlighted how a compact engine derives strength from incremental evaluation and aggressive pruning rather than elaborate piece-square tables (PSTs).
- Self-play logs (e.g., `self_play_traces/3_selfplay.txt`) show PVS and HS entering nearly identical knight-first openings due to shared heuristics and deterministic search.

## 2. Findings Summary
- Our evaluator combines fixed PSTs with additional terms (material, bishop pair, passed pawns, mobility, king safety). This already exceeds Sunfish’s PST-only scoring, so swapping tables would require retuning every auxiliary heuristic to avoid double-counting.
- The deterministic move ordering coupled with mirrored PSTs explains repetitive openings more than PST quality itself; we see rooted convergence even when both sides run different engine wrappers.
- Sunfish’s PST layout embeds material and uses a 10×12 index (`agent_plans/sunfish_engine_analysis.md`, sections 2.2 and 7). Direct transplantation would break evaluation scaling inside PVS.

## 3. Planned Improvements
1. **Search Efficiency Profiling**  
   - Instrument node/time breakdowns around move ordering and pruning hotspots. Goal: verify where depth stalls versus Sunfish’s MTD-bi approach (see section 3 of the Sunfish document).
2. **Incremental Evaluation Optimization**  
   - Investigate caching PST deltas and piece counts per move to cut recomputation, inspired by Sunfish’s `Position.value`; avoids PST retunes but improves throughput.
3. **Opening Diversity Enhancements**  
   - Introduce light randomness or heuristic tie-breakers when scores fall within a narrow margin (e.g., ±5 cp) to prevent mirrored knight development without touching PST baselines.
4. **Quiescence/LMR Adjustments**  
   - Re-evaluate quiescence delta margins and late-move reductions to free search time for deeper principal lines; record impact relative to Sunfish’s intrinsic capture threshold (section 3.2).

## 4. PST Strategy
- Maintain current tables while profiling their contribution; if future tuning is warranted, derive adjustments through data-driven methods (e.g., self-play gradient tuning) rather than importing Sunfish weights.
- Any PST experiments must subtract base material and align indices before comparison; reference section 7 of the Sunfish analysis for transformation requirements.

## 5. Next Actions
- Set up benchmark runs (`pytest` plus targeted search profiling with `pytest -S tests/test_pvsengine_pvsearch.py`) before and after search tweaks.
- Draft instrumentation tasks for incremental evaluation and tie-breaker randomness, then circulate for review.
