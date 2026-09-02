# Stage 0.5 kill-test: results

Ran per `RESEARCH_PLAN.md` Stage 0.5. Question: does an EFE-driven control
loop produce measurably different/better act/gather/escalate decisions
than a heuristic and a learned router — on the same observations — and
does it match a hand-derived Bayes-optimal (value-of-information)
controller without hand-deriving the utility math per domain?

Code: `src/aif_orchestrator/kill_test/`. Reproduce:
`.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test`
Raw output: `results/stage0_5_kill_test_results.json`.

## Setup

Tiny synthetic environment (`env.py`): hidden task state ∈
`{solvable (p=0.5), needs_info (p=0.35), needs_human (p=0.15)}`, one noisy
confidence observation per step (low/medium/high, overlapping
distributions by state), 3 actions (`act_now`, `gather_info`, `escalate`),
up to 3 gather steps before a forced terminal choice. Reward: +1 success /
-1 failure / +0.5 correct escalation / -0.7 unnecessary escalation / -0.1
per gather. Full parameters in `env.py`.

5 controllers, 3000 held-out episodes each (seeds disjoint from training):

- **random** — uniform over valid actions (floor baseline)
- **heuristic** — fixed confidence thresholds (`if low: escalate; if high: act; else: gather`), the production-style rule this project is trying to replace
- **learned_router** — tabular Q-learning, 8000 training episodes, features = (current confidence bin, gathers used so far)
- **voi_decision_theoretic** — hand-derived exact Bayesian belief update + expected-utility computation (one-step value-of-information lookahead) over the *true* generative model — the strong baseline
- **efe_active_inference** — pymdp Expected Free Energy over the same true generative model, re-planned from the running belief at every decision

EFE and VOI are given the identical generative model on purpose: the
question isn't "does active inference know more," it's "does the EFE
*mechanism* reproduce optimal decision-theoretic behavior automatically."

## Results (3000 episodes, 95% CI)

| Controller | avg reward | 95% CI | success | unnecessary escalation | silent failure | avg gathers |
|---|---|---|---|---|---|---|
| random | -0.167 | [-0.197, -0.137] | 0.317 | 0.432 | 0.173 | 0.48 |
| heuristic | 0.374 | [0.345, 0.403] | 0.587 | 0.154 | 0.132 | 0.36 |
| learned_router | 0.432 | [0.407, 0.456] | 0.737 | 0.042 | 0.102 | 2.33 |
| **voi_decision_theoretic** | **0.487** | **[0.463, 0.510]** | 0.757 | 0.019 | 0.100 | 2.19 |
| **efe_active_inference** | **0.482** | **[0.458, 0.505]** | 0.757 | 0.019 | 0.100 | 2.24 |

## Reading

1. **EFE ≈ VOI, and the CIs heavily overlap.** This is the headline
   finding, and it's the sharpened claim from the plan's Stage 0.5
   revision: EFE reproduces near-Bayes-optimal value-of-information
   behavior **without hand-deriving the utility formulas** — the same
   generative model spec (A/B/C/D matrices) that defines the problem is
   all EFE needs; VOI required hand-writing the expected-utility
   recursion. That's the actual mechanism-level claim worth making, not
   "EFE is better than VOI."

2. **EFE/VOI clearly beat the learned router, and this is a real gap, not
   noise** — EFE's CI `[0.458, 0.505]` and the router's CI
   `[0.407, 0.456]` don't overlap. The gap shows up concentrated in
   unnecessary escalations (EFE/VOI: 0.019 vs. router: 0.042 — more than
   2x) rather than raw success rate (0.757 vs. 0.737, close).

3. **EFE/VOI clearly beat the heuristic** — no CI overlap, and by a wide
   margin (unnecessary escalation 0.019 vs. 0.154, ~8x).

4. **Important caveat — the router baseline here is memory-limited, and
   that's not a fully fair fight.** `learned_router`'s features are only
   *(current confidence bin, gathers used)* — it does not integrate the
   full history of confidence readings across gather steps the way
   VOI/EFE's running Bayesian belief does. Some of the router's gap is
   plausibly a state-representation handicap, not an inherent limit of
   learned routers. **This must be fixed before the finding is reported
   in Stage 3** — give the router baseline an equivalent running
   belief/summary feature (or make it recurrent) so the comparison is
   between control *mechanisms*, not between stateful-vs-stateless
   feature sets. Flagged in `RESEARCH_PLAN.md` §5 risks.

## Decision: proceed to Stage 1

The kill-test's condition for proceeding (RESEARCH_PLAN.md Stage 0.5)
was: does EFE produce measurably different/better decisions than a
router/heuristic, and is that difference real rather than noise? Yes on
both counts, with one fix required in Stage 2 (give the router baseline
belief-state parity) before the finding can be trusted at benchmark
scale. Proceeding to Stage 1 (LangGraph EFE control node).
