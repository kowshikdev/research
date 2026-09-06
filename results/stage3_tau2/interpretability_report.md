# Stage 5 interpretability: EFE on the real Stage 3 sweep

Source: `results/stage3_tau2/efe_decision_log.jsonl` (2163 decisions,
`efe_active_inference` controller only — baselines zero the epistemic
term by construction, so an analysis over their logs would be vacuous).
Produced by a dedicated EFE-only re-run across retail/airline/telecom
with `EFE_DECISION_LOG_PATH` set (see `efe_agent.py`'s
`_log_decision_if_enabled`).

**This supersedes an earlier version of this report** (278 decisions,
one per task) that ran before two real upstream bugs were found and
fixed (`docs/known-issues-and-gotchas.md` #15): `policy_gate` was stuck
at the fail-safe `"needs_review"` because this environment had no `opa`
CLI installed, and the cold-start skip only covered the literal first
message, so EFE was making its one and only real decision before ever
attempting a tool call. Both fixed, then EFE was re-run across all 3
domains. Escalation counts dropped sharply on retail (114→10) and
airline (50→10), and real reward improved on retail (0.053→0.281, EFE
went from the worst controller to beating heuristic); telecom's drop
was smaller (114→63) and its reward fell slightly (0.175→0.096) —
flagged as an unexplained open question, not smoothed over.

Computed via `analysis.interpretability.analyze_decision_log`.

| metric | before the fix (278 decisions) | after the fix (2163 decisions) |
|---|---|---|
| n_decisions | 278 | 2163 |
| policy_counts | `{"escalate_to_human": 278}` | `{"continue": 2053, "escalate_to_human": 84, "call_tool": 16, "gather_info": 10}` |
| mean_epistemic_of_chosen | 0.436 | 0.030 |
| mean_pragmatic_of_chosen | -4.747 | -3.989 |
| epistemic_share | 0.084 | 0.0075 |
| **decisions_driven_by_epistemic** | **0 / 278** | **0 / 2163** |
| epistemic_by_confidence_band | `{"confident": 0.435, "moderate": 0.503}` | `{"confident": 0.016, "moderate": 0.343, "uncertain": 0.440}` |

`mean_turns_per_task` and per-task `escalation_rate` as
`analyze_decision_log` would compute them from this log are **not**
reported here: a separate bug (also #15) meant the synthetic task ids
logged for this run collided across the three separate per-domain
processes (fixed with `uuid.uuid4()`, but too late to fix this run's
already-recorded ids). Use `results/stage3_tau2/summary.json`'s
`n_simulations`/`escalations` fields for real per-domain escalation
rates instead — those come from tau2's own objects, unaffected by the
id collision.

## Reading this result

The picture changed completely on the surface — EFE went from
escalating on literally every task (100%, one decision per task) to
mostly continuing normally (95% of 2163 decisions are `continue`), with
real multi-turn tool-using behavior now visible for the first time.

But the interpretability answer is not just unchanged, it's **more
decisive**: `decisions_driven_by_epistemic` is still 0, now out of 2163
real decisions instead of 278, and `epistemic_share` actually *fell*
further (0.75% vs. 8.4% before). This rules out the concern that the
original 0/278 finding was an artifact of the broken single-shot-escalate
pattern — with real agentic behavior unlocked, across an order of
magnitude more decisions, the epistemic term still never wins a
decision that pragmatic value wouldn't have won on its own. Even in the
`"uncertain"` confidence band (belief < 0.5, `epistemic_by_confidence_band`
= 0.440 — the highest of the three bands, exactly where information-seeking
should matter most if it mattered anywhere), it's still never the
deciding factor.

See `ROADMAP.md`'s Milestone 3 for what this means for the thesis.
