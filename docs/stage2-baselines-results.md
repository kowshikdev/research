# Stage 2 baseline results

Heuristic / learned-router / VOI / ReAct baselines, ported from the
Stage 0.5 kill-test's toy versions to the real decision-POMDP
(`docs/decision-pomdp.md`), evaluated against EFE. See
`src/aif_orchestrator/baselines/` and RESEARCH_PLAN.md Stage 2.

This document has two comparisons. **Part 1** (`run_stage2_eval.py`)
runs everything against `graph.mock_agent_step`, a hand-scripted
stand-in — it validates the plumbing and caught two real bugs, but is
NOT a fair test of decision quality (see its caveat). **Part 2**
(`run_model_matched_eval.py`) closes that gap: every controller runs
against a simulator that samples from `efe_controller.py`'s own
generative model — the real analogue of what `kill_test/env.py` did for
Stage 0.5 — and is the comparison that actually matters. Read Part 2 if
you only have time for one.

## Part 1: mock_agent_step comparison (plumbing sanity check)

### Setup

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

### Results (3000 episodes each)

| Controller | avg reward (95% CI) | correct escalation | unnecessary escalation | normal resolved | forced stop | avg steps |
|---|---|---|---|---|---|---|
| heuristic | 2.525 [2.425, 2.625] | 1.000 | 0.000 | 0.992 | 0.005 | 1.99 |
| learned_router | 2.450 [2.351, 2.548] | 1.000 | 0.000 | 1.000 | 0.000 | 1.91 |
| voi_decision_theoretic | 1.870 [1.752, 1.989] | 1.000 | 0.000 | 0.842 | 0.110 | 3.43 |
| react | 1.864 [1.728, 2.001] | 0.000 | 0.000 | 0.992 | 0.306 | 3.19 |
| efe_active_inference | 1.564 [1.468, 1.659] | 1.000 | 0.000 | 0.682 | 0.223 | 3.13 |

### Findings

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

### Caveat: this is not a fair like-for-like test the way Stage 0.5 was

The Stage 0.5 kill-test's environment (`kill_test/env.py`) **was**
literally the same generative model VOI and EFE both reasoned over —
that's what made "EFE ≈ VOI, both beat the router" a meaningful
Bayes-optimality result. `graph.mock_agent_step` is not that: it's a
small hand-scripted stochastic function (see its own docstring — "exists
purely to exercise the graph, not to model anything real"), not a
sample from `efe_controller.py`'s `OBS_DISTS`/`B_TRANSITIONS`. Every
controller here (heuristic excepted, which doesn't consult a belief)
carries some degree of **model mismatch** between what it assumes
and what `mock_agent_step` actually does. What Part 1 validates: all
five controllers share one interface and are genuinely pluggable into
the real LangGraph scaffold (incl. the real `interrupt()` pause), the
belief-state-parity fix works, and the reward-hacking bug it caught is
fixed. What it does **not** validate: decision quality under a
correctly-specified model. That's Part 2.

## Part 2: model-matched comparison (the one that actually answers Stage 0.5's question at real scale)

### Setup

Same reward design and same five controllers, but the environment is
now `model_env.ModelMatchedEnv`: it samples the true `task_state` from
`efe_controller.D_PRIOR`, samples each observation modality from
`efe_controller.OBS_DISTS[true_state]`, and transitions the true state
via `efe_controller.B_TRANSITIONS[policy][true_state]` — i.e. it treats
EFE's own generative model as ground truth, exactly as VOI and EFE both
assume it is. `learned_router` is retrained against this same
environment (`router.model_matched_source_factory`) — training against
the mock and evaluating against this would just be testing transfer,
not closing the gap. Ground truth for the escalation metrics is the
env's actual sampled `task_state` (`needs_human`/`likely_to_fail` count
as genuinely unresolvable), not the coarse `forced-bad` task-id proxy
Part 1 used.

The router needed far more training here than Part 1's default (10k
episodes): `ModelMatchedEnv`'s joint observation space (144 bins) ×
belief buckets (8) is much richer than `mock_agent_step`'s handful of
scripted combos, so it's trained for 100k episodes
(`run_model_matched_eval.py`) to get reasonable state coverage
(~1000/1152 possible keys).

Run: `.venv/Scripts/python -m aif_orchestrator.baselines.run_model_matched_eval`
(raw output: `results/model_matched_baselines_results.json`).

### Results (3000 episodes each)

| Controller | avg reward (95% CI) | correct escalation | unnecessary escalation | resolvable resolved | forced stop | avg steps |
|---|---|---|---|---|---|---|
| heuristic | 2.999 [2.918, 3.079] | 0.915 | 0.067 | 0.924 | 0.016 | 1.67 |
| learned_router | 3.393 [3.327, 3.460] | 0.255 | 0.009 | 0.967 | 0.075 | 1.97 |
| voi_decision_theoretic | 3.142 [3.072, 3.212] | 0.766 | 0.002 | 0.891 | 0.115 | 2.20 |
| react | 3.183 [3.105, 3.261] | 0.000 | 0.000 | 0.983 | 0.113 | 2.01 |
| efe_active_inference | 2.804 [2.723, 2.885] | 0.951 | 0.062 | 0.860 | 0.073 | 1.87 |

> **Reproducibility scope (added after a cross-environment re-run).**
> Four of these five rows reproduce **bit-exactly** on a different
> machine and Python version (heuristic, VOI, ReAct, EFE). The
> `learned_router` row does not: a Linux/Python 3.11 re-run gives
> avg reward 3.434 and correct escalation 0.236 against the 3.393 /
> 0.255 above. The router is the only controller with trained state,
> and its training is bit-deterministic *within* an environment — so
> quote its exact figures together with the environment that produced
> them, and expect ~1% drift on reward across machines. No qualitative
> conclusion below changes: the router still has by far the worst
> correct-escalation rate and the highest raw reward either way. Full
> detail: `known-issues-and-gotchas.md` #12.

### Findings

**EFE has the best correct-escalation rate of all five (0.951, beating
even heuristic's 0.915) but the lowest average reward.** It's the most
conservative escalator — it catches almost every genuinely unresolvable
task, at the cost of throughput on the resolvable ones
(`resolvable_resolved=0.860`, the worst of the five). This is a
real, model-matched result (unlike Part 1's), and it's a defensible
trade for a system whose whole point is catching cases a human should
see — but it means **EFE does not simply dominate VOI here the way
Stage 0.5 found at toy scale.**

**VOI is the best-calibrated controller on precision**
(`unnecessary_escalation=0.002`, the lowest) while still catching most
real cases (`correct_escalation=0.766`), and it beats EFE on reward.
This is the headline divergence from Stage 0.5: there, EFE ≈ VOI with
overlapping CIs on a 3-state/3-action toy model. Here, at the real
4-state/6-policy scale, **they diverge** — EFE trades reward for
escalation recall, VOI is the more balanced of the two. Both remain
clearly better-calibrated than the router or ReAct on
`unnecessary_escalation` (0.002 and 0.062 vs. the router's 0.009 sounds
close, but see below — the router's low unnecessary-escalation rate
here is a symptom of a different problem, not good calibration).

**The learned router has the highest reward (3.393) but the worst
correct-escalation rate (0.255) — worse than not-yet-fully-trained
Part-1 numbers, not better, despite 100k training episodes and solid
state coverage.** This replicated across three training seeds during
development (`correct_escalation` between 0.26 and 0.37 across seeds
at 100k episodes) — not noise. The mechanism: genuinely unresolvable
tasks are a minority of the training distribution (`D_PRIOR`'s
`needs_human + likely_to_fail` mass is 0.15, and only ~2-3% of eval
episodes end in one), and Q-learning optimizes *aggregate* reward over
that distribution — so a policy that quietly accepts the occasional
large `belief.OUTCOME_ADJUSTMENT` silent-failure penalty (-2.0) in
exchange for resolving the common case fast maximizes average reward
better than one that reliably escalates. **This is a genuine, reward-
maximizing-under-class-imbalance failure mode distinct from the
belief-state-parity issue Stage 0.5 fixed** — belief parity gave the
router access to the same *information* EFE/VOI have; it says nothing
about whether reward-maximizing training will actually use that
information to catch a rare, high-stakes event. Reporting this as-is
rather than re-tuning the router's reward until the number looks
better: it's a real property of this baseline design, and arguably the
more interesting finding than a well-calibrated router would have been.

**ReAct again looks statistically strong on raw reward (3.183, second
overall) despite zero escalation ability by construction.** Now even
more strikingly than Part 1, since it's competitive with VOI
(`3.183` vs `3.142`, CIs close) — the same point Part 1 made, confirmed
under the fair, model-matched comparison this time.

### What this closes and what's still open

This closes the gap Part 1 flagged: it's a genuine repeat of Stage
0.5's methodology (controllers reasoning over the same true generative
model they're evaluated against) at the real decision-POMDP's scale.
Unlike Stage 0.5, EFE and VOI **do not** converge to statistically
indistinguishable behavior here — they trade off differently on
escalation recall vs. reward, and the learned router (even with belief
parity and heavy training) has a real, mechanistically-understood
calibration weakness Stage 0.5's simpler setup didn't surface. All of
this is still against EFE's uncalibrated placeholder A/B/C/E/D values
(`efe_controller.py`) — Stage 3 calibration against real agent
trajectories could shift where EFE lands on the reward/recall trade-off
shown here. Treat this as a real, informative result for the thesis
(more so than Part 1), not as a final word on EFE vs. VOI.
