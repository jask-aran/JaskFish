# Sunfish Engine Architecture & Algorithm Study

## 1. Why Sunfish Matters
- `sunfish.py` compresses a full-strength amateur chess engine into **~500 LOC** while remaining tactically sharp. Its strength stems from a disciplined architecture: incremental evaluation, color-agnostic state handling, and a minimal-but-stacked searcher.
- The engine embraces “do one thing well” principles—every subsystem feeds the search loop with just enough information, no more. This yields high practical strength with tiny code footprint and memory use.

## 2. Core Architectural Patterns

### 2.1 Unified Board Perspective
- **Single-side representation:** After every play, `Position.rotate` mirrors and swap-cases the board so the next player always appears as white (`sunfish.py:193`).  
  ```python
  return Position(
      self.board[::-1].swapcase(), -self.score, self.bc, self.wc,
      119 - self.ep if self.ep and not nullmove else 0,
      119 - self.kp if self.kp and not nullmove else 0,
  )
  ```
- **0x88-inspired padding:** The board is stuffed into a 120-character string with sentinel rows/columns (`sunfish.py:84`). Sliding beyond the real board hits whitespace, terminating rays without extra bounds checks.
- **Implication for us:** Flip-the-board logic slashes branching (no “if side == black”); we can copy this to simplify evaluation and move generation across our engines.

### 2.2 Incremental Evaluation & PST Layout
- **Piece-square tables** are pre-padded to align with the 120-square board. Each table includes outer zeros, making lookups direct array indexes (`sunfish.py:74`).
- `Position.value` computes delta evaluation using PST differences and special-case bonuses in ~10 lines (`sunfish.py:235`). Captures, promotions, en-passant, castle rook travel, and king-in-check detection all fold into this delta before rotating the position.
- Promotions reuse the same PST tables: promotion target adds `pst[prom][j] - pst["P"][j]` (`sunfish.py:252`).
- **Lesson:** Re-evaluate incremental score updates; Sunfish proves we can avoid full-board scans if scoring tables align with board layout.

### 2.3 Move Generator Design
- **Rays in a single loop:** `Position.gen_moves` enumerates all moves with nested loops over direction deltas (`sunfish.py:153`). Sliding pieces keep stepping until friendly pieces or whitespace (padded border) break the loop.
- **Pawn rules folded tightly:**  
  - Double advance guard: `if d == N + N and (i < A1 + N or self.board[i + N] != ".")` to enforce rank and vacancy (`sunfish.py:168`).  
  - En-passant capture gating uses both `ep` and `kp` to prevent illegal castle-passed en-passants (`sunfish.py:170`).  
  - Promotion yields four `Move` objects inline (`sunfish.py:178`).
- **Castling as rook slides:** After a rook slide touches the king, extra pseudo-moves slide the rook adjacent, rather than a special-case king move (`sunfish.py:188`).
- **Takeaway:** Combine move-gen and legality/ordering heuristics. We could port castling-as-rook-slide and inline pawn logic to streamline our generator.

## 3. Search Stack Anatomy

### 3.1 Transposition Table Layout
- Transposition cache keyed by `(Position, depth, can_null)` ensures null-move and regular probes don’t collide (`sunfish.py:296`). The stored entry uses a lightweight `Entry(lower, upper)`.
- Move table `tp_move[pos]` keeps the last PV move; it doubles as a killer heuristic store.

### 3.2 `bound` – Fail-soft Negamax with Enhancements
- **Fail-soft windowing:** `bound` seeks to confirm `s* >= gamma` or build a tighter upper bound otherwise (`sunfish.py:274`).
- **Null-move pruning:** If depth > 2 and `abs(pos.score) < 500`, the engine tries skipping a move by rotating with `nullmove=True` (`sunfish.py:312`). The score is inverted and reduced depth-wise (`depth - 3`) to keep stability.
- **History repetition** detection uses a `history` set shared across recursive calls; repeat positions return draw scores (`sunfish.py:303`).
- **Internal Iterative Deepening (IID):** On a miss, `bound` re-enters itself with depth -3 to recover a killer move (`sunfish.py:333`).
- **Move ordering:** Inline generator yields (value, move) tuples sorted descending by `pos.value` (`sunfish.py:348`). This reuses the incremental eval with zero additional heuristics.
- **Quiescence / intrinsic cutoff:** `val_lower = QS - depth * QS_A` grows harsher as depth shrinks, ensuring leaf nodes only explore high-impact captures/promotions (`sunfish.py:336`).
- **Futility pruning:** When depth <= 1 and intrinsic score can’t reach `gamma`, the code shortcuts and caps high scores at `MATE_UPPER` for mate-in-one detection (`sunfish.py:356`).
- **Mate/stalemate disambiguation:** Encountering `-MATE_UPPER` triggers a secondary search of the flipped position at depth 0 (`sunfish.py:394`) to distinguish stalemates from actual mates.

### 3.3 Top-level MTD-bi Loop
- `Searcher.search` runs iterative deepening with a binary search on scores: `gamma` starts at 0, while `[lower, upper]` converge via repeated `bound` calls (`sunfish.py:424`).  
  ```python
  while lower < upper - EVAL_ROUGHNESS:
      score = self.bound(history[-1], gamma, depth, can_null=False)
      ...
      gamma = (lower + upper + 1) // 2
  ```
- Each iteration yields `(depth, gamma, score, move)`; the driver prints info as soon as a fail-high confirms the PV move (`sunfish.py:487`).
- **Strategic angle:** MTD-bi works because pruning is stable and move ordering strong; otherwise the binary search could trash performance. Sunfish’s incremental eval plus null move keeps it tight.

## 4. UCI & Time Management
- The engine supports standard `uci`, `isready`, `position`, and `go` commands (`sunfish.py:461`). Input moves are converted via simple `parse/render` helpers that respect the flipped board perspective (`sunfish.py:439`, `sunfish.py:445`).
- Time allocation: `think = min(wtime / 40 + winc, wtime / 2 - 1)`—roughly 2.5% of remaining clock with increment guard (`sunfish.py:480`).
- Search loop clones a fresh `Searcher` each `go` command to keep history tables local (`sunfish.py:487`), avoiding stale TT interference between moves.

## 5. Lessons for JaskFish
1. **Adopt a unified perspective layer**  
   - Porting a rotate-and-swapcase approach can simplify evaluation and reduce duplication in `pvsengine.py`. It also makes null-move and repetition checks more straightforward.
2. **Incremental PST deltas as primary eval**  
   - Introduce pre-padded PST tables and move-level delta evaluation to slash recomputation costs, especially in our principal variation search where nodes skyrocket.
3. **Low-cost heuristics > feature sprawl**  
   - Sunfish’s val-threshold quiescence and futility heuristics deliver nearly the same gains as advanced history heuristics. Consider experimenting with similar linear thresholds in our quiescence to stabilize leaf evaluation.
4. **Revisit null-move and IID implementations**  
   - The combination `depth > 2` + bounded score ensures null-move stays safe even in simplified code. Our engine can mirror this gating to balance tactical strength and search breadth.
5. **Binary-search driver simplicity**  
   - MTD-bi is only ~15 lines here but yields strong move confidence. We can prototype an MTD-bi wrapper around our existing alpha-beta to test whether the tighter convergence benefits our evaluation style.

## 6. Suggested Next Steps
- Prototype a sandbox branch to evaluate rotate/swapcase position handling inside `pvsengine.Position`, ensuring UCI integration remains stable.
- Build new PST tables (or reuse Sunfish’s as a baseline) and wire incremental `value(move)` logic, verifying via existing regression suites (`pytest` plus `pytest -S tests/test_pvsengine_pvsearch.py`).
- Experiment with intrinsic capture thresholds in quiescence while logging node counts and search stability to compare against current heuristics.

## 7. Piece-Square Table Observations
- Sunfish’s PSTs already include the underlying material value for each square; the padding step adds `piece[k]` before writing the final arrays (`sunfish.py:74`). Transplanting them verbatim into engines that *also* add material (e.g., `pvsengine.py:46`, `pvsengine.py:1576`) would double-count every piece.
- The tables are tuned for a 10×12 board index—entries map directly to the padded board after rotation—so importing them would require reprojecting onto python-chess’s 0–63 indices and mirroring rows for black.
- Our engines share identical PSTs, which explains repeated knight-to-the-center openings in deterministic searches. Introducing variety therefore needs either a search diversification tweak or targeted PST retuning rather than wholesale borrowing from Sunfish.
