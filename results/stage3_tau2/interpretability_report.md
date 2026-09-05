# Stage 5 interpretability: EFE on the real Stage 3 sweep

Source: `results/stage3_tau2/efe_decision_log.jsonl` (278 decisions,
`efe_active_inference` controller only — baselines zero the epistemic
term by construction, so an analysis over their logs would be vacuous).
Produced by a dedicated EFE-only re-run across retail/airline/telecom
with `EFE_DECISION_LOG_PATH` set (see `efe_agent.py`'s
`_log_decision_if_enabled`); reward/escalation numbers from that re-run
match the primary sweep within normal run-to-run variance.

Computed via `analysis.interpretability.analyze_decision_log`.

| metric | value |
|---|---|
| n_decisions | 278 |
| has_epistemic_decomposition | True |
| policy_counts | `{"escalate_to_human": 278}` |
| mean_turns_per_task | 1.0 |
| escalation_rate (per-task) | 1.0 |
| mean_epistemic_of_chosen | 0.436 |
| mean_pragmatic_of_chosen | -4.747 |
| epistemic_share | 0.084 |
| **decisions_driven_by_epistemic** | **0 / 278** |
| epistemic_by_confidence_band | `{"confident": 0.435, "moderate": 0.503}` (no "uncertain" band present) |

## Reading this result

EFE chose `escalate_to_human` on every one of the 278 real decisions —
one decision per task, at the very first turn. The epistemic term is not
zero (mean 0.436, ~8.4% share of the decision signal) but never the
deciding factor: the pragmatic term, though strongly negative for the
chosen policy (-4.747), was still the least-bad option among the
alternatives on every decision, so goal-seeking alone would have picked
identically.

This matches the earlier mock-agent pilot (0/8 driven by the epistemic
term, epistemic share ~0.065) in kind, now confirmed at full scale
(278 decisions, three real domains, genuinely varied tasks) rather than
a small scripted sample. See `ROADMAP.md`'s Milestone 3 for what this
means for the thesis and the two live hypotheses (mis-calibrated
generative model vs. genuinely low-ambiguity task class) Milestone 4's
calibration work would help distinguish.
