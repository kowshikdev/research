# Stage 2 baseline results

Heuristic / learned-router / VOI / ReAct baselines, ported from the
Stage 0.5 kill-test's toy versions to the real decision-POMDP
(`docs/decision-pomdp.md`), evaluated against EFE. See
`src/aif_orchestrator/baselines/` and RESEARCH_PLAN.md Stage 2.

## Setup

Each controller shares `EFEControlNode`'s interface (`decide(observation,
valid_policies=None) -> Decision`) and is evaluated the same way: 3000
held-out episodes of `graph.mock_agent_step` (the existing free,
deterministic-per-seed stand-in agent step), 30% of episodes seeded with
a `forced-bad` task id (mock_agent_step deterministically returns
repeated tool errors/low confidence for these — the only ground-truth
signal this environment exposes), the rest a normal, stochastically-
resolvable trajectory. Reward is `belief.step_reward`: the real
C-weighted observation payoff (`efe_controller.C_WEIGHTS` — the same
preference EFE itself optimizes for) plus a ground-truth-keyed
outcome adjustment, paid only on a genuine terminal turn (`continue` /
`escalate_to_human` chosen, or turns exhausted); every other turn pays a
flat cost. Full derivation and the two bugs this caught (reward-hacking
via never terminating; zero training signal for correct-vs-unnecessary
escalation) are documented in `baselines/belief.py`.

Run: `.venv/Scripts/python -m aif_orchestrator.baselines.run_stage2_eval`
(raw output: `results/stage2_baselines_results.json`). Each controller is
also verified running inside the actual LangGraph scaffold (not just
this fast loop), including the real `escalate_to_human` → `interrupt()`
pause: `python -m aif_orchestrator.graph --stage2`.

## Results (3000 episodes each)

| Controller | avg reward (95% CI) | correct escalation | unnecessary escalation | normal resolved | forced stop | avg steps |
|---|---|---|---|---|---|---|
| heuristic | 2.525 [2.425, 2.625] | 1.000 | 0.000 | 0.992 | 0.005 | 1.99 |
| learned_router | 2.450 [2.351, 2.548] | 1.000 | 0.000 | 1.000 | 0.000 | 1.91 |
| voi_decision_theoretic | 1.870 [1.752, 1.989] | 1.000 | 0.000 | 0.842 | 0.110 | 3.43 |
| react | 1.864 [1.728, 2.001] | 0.000 | 0.000 | 0.992 | 0.306 | 3.19 |
| efe_active_inference | 1.564 [1.468, 1.659] | 1.000 | 0.000 | 0.682 | 0.223 | 3.13 |

## Findings

**React ties VOI on raw reward (1.864 vs 1.870, CIs overlap almost
entirely) despite having zero escalation capability** (`correct_
escalation_rate=0.000` — by construction, ReAct never asks for help,
see `baselines/react.py`). It ties because it resolves normal tasks
efficiently (0.992) and pays no gather-info cost dawdling; its damage is
invisible to average reward and shows up only in the escalation-specific
metrics. This is a direct, concrete illustration of the exact concern
RESEARCH_PLAN.md §3.4 raises about why HiL-Bench's Ask-F1 (not raw
success rate) is the right metric for selective escalation — a
"floor baseline" can look statistically competitive on an aggregate
reward while being qualitatively broken at the one thing this whole
project is about calibrating.

**Heuristic and the belief-parity-fixed learned_router are the top two
performers here**, both with perfect or near-perfect escalation
calibration. This is not the Stage 0.5 kill-test's finding (there, EFE
and VOI beat a memory-limited router on unnecessary-escalation rate) and
that's expected — see the caveat below.

**EFE underperforms VOI and heuristic here** (1.564 vs 1.870/2.525,
normal_resolved=0.682 — the worst of the five, forced_stop=0.223 —
second-worst), the opposite of the Stage 0.5 kill-test's result (EFE ≈
VOI there, both beating the router). Do not read this as a regression or
as evidence against EFE's approach — see the model-mismatch caveat
below, which explains why this comparison isn't actually testing the
same thing Stage 0.5 tested.

## Important caveat: this is not a fair like-for-like test the way Stage 0.5 was

The Stage 0.5 kill-test's environment (`kill_test/env.py`) **was**
literally the same generative model VOI and EFE both reasoned over —
that's what made "EFE ≈ VOI, both beat the router" a meaningful
Bayes-optimality result. `graph.mock_agent_step` is not that: it's a
small hand-scripted stochastic function (see its own docstring — "exists
purely to exercise the graph, not to model anything real"), not a
sample from `efe_controller.py`'s `OBS_DISTS`/`B_TRANSITIONS`. Every
controller here (heuristic excepted, which doesn't consult a belief)
carries some degree of **model mismatch** between what it assumes
(EFE's own hand-specified, explicitly-not-calibrated placeholder
matrices — see `efe_controller.py`'s docstring) and what
`mock_agent_step` actually does. What Stage 2 actually validates:

1. All five controllers share one interface and are genuinely pluggable
   into the same LangGraph scaffold (`graph.build_graph(control_step=...)`,
   `graph.run_stage2_demo()`), including the real `interrupt()` pause.
2. The belief-state-parity fix the kill-test flagged as necessary
   (`docs/stage0.5-kill-test-results.md`) is implemented and the router
   now correctly learns to escalate (an earlier version, without
   ground-truth-keyed reward, never learned to — see `belief.py`).
3. The evaluation harness and reward design are debugged against a caught
   reward-hacking failure mode.

What it does **not** validate: EFE's actual decision quality under a
correctly-specified model at this scale, or a fair repeat of Stage
0.5's Bayes-optimality claim. That requires either (a) a real
generative-model-matched simulator for the 4-state/6-policy decision-
POMDP (not yet built), or (b) Stage 3's real benchmark data. EFE's A/B/C/E/D
values remain uncalibrated placeholders (per `efe_controller.py` and
RESEARCH_PLAN.md Stage 1) — this result is one more data point motivating
that calibration work, not a finding to report as-is in the thesis.

**Decision:** proceed to OPA `policy_gate` wiring, then Stage 3. If a
model-matched Stage 2 comparison is wanted before Stage 3, it needs a
purpose-built simulator over `efe_controller.py`'s own matrices (treating
them as ground truth for that comparison only) — not yet done, noted in
`context/TODOS.md`.
