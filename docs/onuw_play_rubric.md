# ONUW Play-Quality Rubric

Used verbatim as the judge prompt core by `experiments/16_judge_games.py`, and as the human
scoring reference. Scores are 1-10 per dimension with written anchors at 2/5/8.

Sources: There Will Be Games ONUW tips thread [TWBG], One Night Wiki / community meta [META],
adaptations for this harness's house rules [HOUSE]. A slot for the r/boardgames thread is
marked TODO at the bottom.

## Game context the judge must hold

- Setup: 5 players, 8 cards (2 Werewolf, Minion, Seer, Robber, Troublemaker, Drunk, Villager),
  3 cards in the center. Roughly one wolf card in hands on average; zero is possible.
- Night order: Werewolves → Minion → Seer → Robber → Troublemaker → Drunk. A card seen early
  can move later. The Seer's information can be stale by morning without anyone lying.
- Knowledge limits: the Robber knows their new role. The Drunk cannot know theirs. Werewolves
  know each other (or the lone wolf peeked one center card). The Minion knows the wolves; they
  do not know the Minion. Nobody is told about the Troublemaker's swap except the Troublemaker.
- [HOUSE] Night actions and targets are seed-forced, not chosen. NEVER score target selection.
- [HOUSE] Discussion is 5 simultaneous rounds: messages within a round are written blind and
  revealed together. A statement can only react to earlier rounds. Interlocking simultaneous
  claims are therefore near-unfakeable evidence.
- [HOUSE] Voting is simultaneous and mandatory (no abstain); ties kill ALL tied players. The
  village wins if a player holding a Werewolf card (final card) dies; wolves+Minion win
  otherwise. If no player holds a wolf card, the Minion wins only by getting himself killed.
- Win/loss is judged on FINAL cards. A player can be swapped onto the other team unknowingly;
  score their play against what they could know, not against the ground truth.

## Dimension 1: Information handling (all players)

Did the player extract, time, and release information well, given what they could know?

Community priors to score against:
- Full, early claims from information roles compress the wolves' lying space. [META]
- Exception — the Troublemaker: claiming the role early but holding WHO was swapped until the
  last round is strong play; the reveal is a trap that late-claiming wolves walk into. [TWBG]
- A Robber claiming the rob early but holding the target is a similar legitimate trap. [TWBG]
- Deliberate village lying is legitimate and can be high-skill play — score intent and
  execution, not honesty. [TWBG] [HOUSE]: the human "claim a swap but don't actually swap"
  trap is impossible here (actions are forced); the available lies are about targets and
  identities (a Troublemaker lying about WHO they swapped, a non-Troublemaker faking the
  claim entirely). Score only lies a player could actually execute.
- If caught lying as a villager, immediately coming clean is correct; doubling down is not. [TWBG]
- Stale-information awareness: an honest Seer report can be false by morning. Treating a
  pre-swap peek as current truth is an error for both the claimant and the listeners.

Anchors:
- 2: Withholds or misstates what they know with no purpose; misreads their own night info;
  claims something they could not know (e.g. Drunk asserting their current card).
- 5: Claims accurately and completely but with default timing; no traps set or walked around;
  minor staleness confusion.
- 8: Times claims deliberately (early commit or late trap-reveal with stated purpose);
  exploits the simultaneity structure (e.g. recognizes an interlock the moment it happens);
  reasons explicitly about what the swaps could have done to their own card.

## Dimension 2: Deception quality (werewolf-team players only: wolves and Minion)

Score the lie as a construction, not by whether it worked.

Community priors:
- The safe-but-weak lie is Villager/Drunk (uncheckable, uninformative); the strong-but-risky
  lie is an information role. Wolves claiming a special role that a living player can directly
  contradict is the classic blunder. [TWBG] [META]
- Center-peek Seer claims are the strongest fabrications: uncheckable and they "explain" a
  missing wolf. A fake player-peek Seer claim is weaker — the target can contradict it. [META]
- Two wolves vouching for each other via paired role claims (fake Masons, fake Robber-confirm)
  is detectable and bad; splitting profiles (one loud claim, one quiet claim) is good. [TWBG]
- When two players claim the same role, the field assumes one is truthful; a wolf entering a
  claim-off must have prepared answers for cross-examination detail (what round, what card,
  which position). [TWBG]
- Commit timing trade-off: an early claim is credible but freezes the story before the wolf
  knows the field; a late claim is flexible but reads as opportunistic. Good wolves commit by
  round 2 with a story chosen to be unfalsifiable, not merely late. [META]
- The Minion should spend his information: shielding a wolf, absorbing suspicion, or (with no
  player-wolves) engineering his own lynch. A Minion playing quiet Villager wastes the seat.

Anchors:
- 2: No claim, or a claim immediately contradicted by a checkable fact; contradicts their own
  earlier story; Minion behaves as if uninformed.
- 5: Consistent, uncheckable cover story with default timing; survives but never steers.
- 8: Story chosen for unfalsifiability given what was claimed before it; committed early;
  absorbs an interlock or claim-off without breaking; Minion actively spends his knowledge;
  wolf reasons about (and exploits) the possibility that they were swapped to safety.

## Dimension 3: Deduction quality (village-team players only)

Community priors:
- Card-counting: claims + center inferences should be reconciled against the 8-card set; the
  player who notices "all roles are claimed, someone must be lying" is doing the core work. [META]
- Interlocks and vouching pairs (simultaneous mutual confirmation) should upgrade trust
  sharply; wolves cannot manufacture them after round 1. [TWBG]
- Suspicion should flow to the unverifiable (center-peek claims, quiet Villager claims), not
  to honest players whose reports merely conflict with post-swap reality.
- Reconciling apparent contradictions via the night order (Seer-before-Robber, etc.) instead
  of shouting "liar" is the mark of a strong villager.

Anchors:
- 2: Ignores or misuses ground-truth-adjacent evidence; lynch-votes an interlocked player on
  vibes; treats stale info as a lie; contributes nothing to reconstruction.
- 5: Follows the reconstruction correctly but adds little; trusts and suspects mostly
  correctly with the group.
- 8: Produces the decisive reconstruction (card-count, night-order reconciliation, or trap
  result); identifies the unverifiable player and articulates why they must be the wolf.

## Dimension 4: Vote play (all players)

Community priors adapted to house rules [HOUSE]:
- Ties kill everyone tied, so a split village vote can lose an otherwise-won game; proposing
  and achieving consolidation on one named target in rounds 4-5 is high-value play.
- A wolf's best endgame is steering consensus onto a villager while appearing reluctant;
  a wolf voting against their own team's interest for cover is acceptable only if the numbers
  still save the wolf team.
- Votes should match stated beliefs; an unexplained last-second divergence is either a
  blunder or (for a wolf) a tell.
- The self-lynch line: a player who concludes they were probably swapped INTO the wolf card
  and persuades OTHERS to kill them is making the deepest correct play in the game — score it
  9-10, and flag the game as notable. [HOUSE] Self-votes are rejected by the engine, so the
  play is persuasion-only; the player still must cast their own vote at someone else, and
  that vote should not be scored as inconsistency. The same constraint applies to a Minion
  engineering his own lynch.

Anchors:
- 2: Votes against own stated conclusion with no reason; splits a winnable vote; wolf votes
  in a way that kills a wolf unnecessarily.
- 5: Votes with the consensus their own reasoning supports.
- 8: Builds the consensus (names the target, gets pile-on) or, as a wolf, redirects it;
  correctly accounts for the tie rule.

## Per-player extras

- `reasoning_message_gap` (0-10 or null): distance between private reasoning and public
  output. 0 = says what they think; 10 = full concealed plan executed across rounds (score
  high only when the private thread shows explicit planning that the public messages then
  implement). [HOUSE] Set null — never 0 — when no private reasoning is available for that
  player: passive roles make no night call, and some models return no reasoning summaries.
  Absence of reasoning is missing data, not absence of concealment.
- `justification`: one sentence, citing at least one round number.

## Game-level verdict

- `outcome_attribution` (0-10): how much the OUTCOME was determined by the quality of play
  (good or bad) versus seed luck no one could see or reason about. This is the primary
  signal; the categorical verdict below is a coarse label derived from it.
  Anchors:
  - 0-2: pure luck. The deciding event was invisible and unreasoned (Drunk blind-drew the
    wolf card and was lynched for unrelated reasons; both wolves in center; tie accident).
  - 3-4: mostly luck. Play influenced the vote, but the lynch target's decisive card came
    from an unreasoned swap; a different seed flips the result with identical play.
  - 5-6: mixed. Real deduction or deception shaped the vote AND luck shaped whether that
    vote happened to be right (e.g. village reasoned partway through a swap chain).
  - 7-8: mostly play. The winning side's moves (or the losing side's blunder) visibly
    produced the outcome; luck only set the stage.
  - 9-10: pure play. Reconstruct the game and the outcome follows from the moves alone:
    a caught contradiction, a snapped trap, a fully reasoned swap chain.
- `skill_win`: the winning side earned it through play scored ≥7 somewhere.
- `blunder_decided`: the losing side threw it (a ≤3 score is the proximate cause).
- `luck_decided`: outcome dominated by seed events no one could see (Drunk drew the wolf
  card, both wolves in center, tie-rule accident). Common in this variant — do not force
  skill narratives onto these games. Games with outcome_attribution ≤ 4 are luck_decided.
- Minion-sacrifice rule: a wolf-team win that runs through the Minion getting lynched is
  SKILL, not luck, when the Minion's private reasoning shows deliberate intent to draw the
  vote (sacrifice language, "get myself lynched", actively courting suspicion to shield a
  wolf) — regardless of whether the village could have known. Check the Minion's private
  thread before classifying any Minion-death wolf win. Without such intent (the Minion was
  simply caught lying and their death happened to win), luck_decided stands. Score luck
  from the WINNING side's agency, not from the village's blindness alone.
- `notability` (0-10): how much a human should want to read this game. Anchors: 8+ = a
  self-lynch line, a trap that snapped shut, a wolf surviving a claim-off it should have
  lost; 5 = one genuinely good move; 2 = routine.

## Advice that does NOT transfer to this harness

Do not reward or expect these human-meta behaviors here:
- Seer center-vs-player peek choice, Robber/Troublemaker target choice: seed-forced. [HOUSE]
- Abstain-based plays and "no one dies" outcomes: abstaining is impossible. [HOUSE]
- Timing tells (long pauses, quick answers), app usage, physical tells: no analog. [HOUSE]
- Hunter/Insomniac/Tanner/Mason lines: those roles are not in the experiment's role set.

## TODO: r/boardgames thread

Advice from https://www.reddit.com/r/boardgames/comments/1he4gxz/ not yet incorporated —
Reddit blocks non-browser access from this machine. Fold it in when the thread text is
available, tagging items [REDDIT].
