• I dug through the most recent artefacts you mentioned:

  - Game snapshot gamestates/chess_game_2025-10-14 23:05:25.json shows the match starting from the
    usual initial FEN and ending at 8/5k1p/2RR4/2p2p2/2K5/4PP2/7P/8 w - - 0 46. White (the Python
    PVS) finishes two rooks up with a completely converted rook ending.
  - Self‑play log self_play_traces/16_selfplay.txt records the full dialogue. For black (cPVS) you
    repeatedly see 2–3 M NPS, depth 8–9 in under a second, and yet—the key point—the evaluation
    swings from “+1 pawn” early on to –12 pawns by the finish. Worse, there are several stretches
    where the C engine announces score cp 30000 with an empty PV and zero nodes for depths 1…8 before
    a deeper iteration finally falls back to a sane negative score.

  Here’s what the trace reveals, move by move:

  ### Critical moments in the game

  1. Moves 20–30 (after 27…Qxe5).
     The C engine has just bagged the e5 pawn and its PST-heavy evaluator loves the look of doubled
     rooks on the second rank: the evaluation it prints is ~+1.2 pawns, even though material is
     level and white has the long-term pair of bishops. The Python engine immediately challenges the
     rooks (31 Rd4 etc.). Because the C engine’s evaluator consists of material + PST only, there’s
     no penalty for the rooks being overextended or for black’s king safety. That inflated score
     persuades search to keep forcing the rooks forward instead of consolidating.
  2. Move 34…Re7 and onward: the trapped rooks plan.
     Black doubles on the second rank (…Ra6, …Rf6, …Rb6+, …Rb2). PST gives rooks on the 7th a fat
     bonus, but there is no static-exchange evaluation or mobility term to warn that the rooks are
     vulnerable. Python plays 43 Rc5 and 45 Rxa5. After 45…Rxa2+ 47 Kb4 Rxc2 the C engine is still
     convinced it’s fine (trace shows +0.3) because it has three connected pawns and still no king
     danger penalties.
  3. Move 52…c2c3 (the exchange blunder).
     That’s the first outright collapse. With rooks on e8 and c2, the engine plays c2c3 (capturing
     the pawn on c3). The trace around line 1124 shows it returning bestmove c2c3 even though the
     depth-9 score immediately afterwards is already –2.35. It made the move anyway because every
     alternative it considered still looked unpleasant given its simplistic horizon: the search
     is missing a sanity check such as SEE or a “don’t sacrifice an exchange unless you’re clearly
     compensating.”
  4. Moves 70…b5?! and 74…Bxc8.
     After the exchange sac, black still has bishop and pawns versus two rooks. Instead of targeting
     white’s weak king, the engine plays 70…b5. That lets white combine 71 Bxc8 b4+ 72 Kb2 Bxc8
     75 Ra8 g5 77 Rxc8+, and from there white sweeps up the remaining material. Again, PST ratings in
     the evaluator prefer pushing the passed pawn (the table gives a nice bonus for b5/b4) and there
     is no counterbalancing term for losing the last rook or for allowing the white rooks to invade
     uncontested.
  5. The bogus +30000 announcements.
     Each time you see info depth N score cp 30000 time 0 nodes 0 pv, the search is hitting a
     transposition-table entry flagged as an EXACT mate score from a shallower depth. Because the
     TT stores bare negamax scores without normalising for ply, hauling that value back to the root
     makes the engine think it has a mate in zero until a full-width iteration finally disproves
     it. That wastes time on bogus PVs and skews move ordering towards lines it already “knows” are
     winning—even though they aren’t. It also explains why the search sometimes keeps the previous PV
     (empty or stale) for the depth summaries.

  ### Root causes

  Putting it all together, the C engine loses not because it is slow—it’s blazing fast—but because:

  1. Evaluation is far too primitive.
     It’s pure material plus hard-coded PSTs. There’s no mobility, king safety, pawn structure,
     passer scores, or endgame scaling. Rooks on the 7th score +50 even when they’re trapped; a king
     walking into checks is unpenalised; long-term weaknesses and exchange sacrifices are invisible.
     The Python PVS, on the other hand, has a full tuned evaluation with mobility, king rings, passed
     pawns, etc. So the Python engine squeezes small, sound advantages while the C engine chases
     phantom “+120 cp” positions created by PST bonuses.
  2. No static exchange evaluation (SEE) or capture heuristics.
     Moves like c2c3 should be rejected outright because the recapture Kxc3 is immediate and
     crushing. Without SEE, the move generator happily feeds them to the search, and because the eval
     doesn’t punish the resulting material deficit strongly enough until after the fact, the search
     commits to them.
  3. Transposition-table scores aren’t mate-distance adjusted.
     When a cutoff writes best_value = 30000 - ply, it should be stored as a mate distance. On
     retrieval, that needs re-normalising so that depth-1 and depth-9 nodes interpret it correctly.
     The current code stores raw negamax scores; when you probe the table from a different ply you
     get a malformed +30000 that makes the root think the position is won. That’s why you repeatedly
     see entire depth iterations “completed” with zero nodes and PV “(empty)”. The fix is to add
     mate-distance normalisation (store as value += ply, recover with value -= ply) or clamp on
     retrieval.
  4. Search lacks modern selective heuristics.
     There’s no null move pruning, no late-move reductions, no history scoring beyond a simple MVV/
     a full quiescence with SEE pruning, so it sees those resources much earlier even at the same
     depth.
  5. Time management is greedy but unsophisticated.
     The C engine burns the full movetime on every turn and only stops when the budget expires. When
     stop ratio, aspiration windows) lets it spend time where it matters.

  ### What to do next

  - Rework the evaluator so it at least includes mobility, king safety, pawn structure, and basic
    endgame scaling. Even lightweight terms (king safety via attack maps, passed-pawn bonuses, rook
    mobility penalties) will stop the engine from preferring trapped rooks.
  - Add SEE (or at least MVV/LVA with capture pruning) before quiescence to stop the engine playing
    losing exchanges like c2c3.
  - Normalise mate scores in the transposition table (store/retrieve with ply adjustments). That will
    remove the repeated score cp 30000 artefacts and stop the search from believing in phantom mates.
  - Introduce basic pruning/reduction heuristics (null move, LMR, history/killer tables, checks
    in quiescence). That will let the C engine use its speed to look at the right lines instead of
    ploughing every branch equally.
  - Improve time management (e.g. bail out early when depth iteration fails low, or when the TT tells
    you the root score hasn’t improved). Right now it uses the entire 4 s even when no new info is
    generated.

  Until those structural issues are addressed, simply being faster won’t help—the evaluator is
  telling the search that obviously bad plans are great, and the TT is feeding in “mate” scores that
  don’t exist. The Python engine, with its richer heuristics, keeps collecting the real advantages
  and converts them methodically, which is exactly what shows up in both the trace and the final
  game state.