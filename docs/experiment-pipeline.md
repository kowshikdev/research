# Experiment pipeline

The research stages this codebase implements, and their current status.
`RESEARCH_PLAN.md` is the authoritative narrative (thesis statement,
literature framing, risks); this document is the visual map of how the
stages chain together and what each one produced. `context/TODOS.md` is
the actionable checklist; this is the "how do the pieces connect" view.

## Pipeline

```mermaid
flowchart TD
    S0["Stage 0<br/>pymdp sanity check<br/>scripts/tmaze_sanity_check.py"]
    S05["Stage 0.5<br/>kill-test: does EFE beat<br/>a router or hand-derived VOI?<br/>kill_test/"]
    S1["Stage 1<br/>EFE control node + LangGraph<br/>efe_controller.py, graph.py"]
    S1C["Stage 1c<br/>real tool-calling LLM agent<br/>llm_agent.py"]
    S2["Stage 2<br/>4 baselines vs EFE<br/>baselines/"]
    S2M["Stage 2 Part 2<br/>model-matched comparison<br/>(the real result)"]
    OPA["OPA policy_gate wiring<br/>opa_policy.py"]
    S3W["Stage 3 wiring<br/>tau2-bench cloned + pinned<br/>tau2_integration/"]
    S3E["Stage 3 sweep<br/>the actual benchmark run<br/>NOT YET RUN"]
    S4["Stage 4<br/>GAIA validation subset<br/>optional"]
    S5["Stage 5<br/>interpretability analysis<br/>(logging already in place)"]
    S6["Stage 6<br/>writing"]

    S0 -->|engine confirmed working| S05
    S05 -->|positive result: EFE ≈ VOI,<br/>both beat router/heuristic| S1
    S1 --> S1C
    S1C --> S2
    S1C --> OPA
    S2 --> S2M
    S2M -->|real finding: EFE and VOI<br/>DIVERGE at real scale| S3W
    OPA --> S3W
    S3W -->|cost estimator built,<br/>scope decision pending| S3E
    S3E -.optional, time-permitting.-> S4
    S3E --> S5
    S5 --> S6

    style S0 fill:#d8f0d8,stroke:#373
    style S05 fill:#d8f0d8,stroke:#373
    style S1 fill:#d8f0d8,stroke:#373
    style S1C fill:#d8f0d8,stroke:#373
    style S2 fill:#d8f0d8,stroke:#373
    style S2M fill:#d8f0d8,stroke:#373
    style OPA fill:#d8f0d8,stroke:#373
    style S3W fill:#d8f0d8,stroke:#373
    style S3E fill:#fff3cf,stroke:#a80
    style S4 fill:#e8e8e8,stroke:#888
    style S5 fill:#e8e8e8,stroke:#888
    style S6 fill:#e8e8e8,stroke:#888
```

Green = done and verified. Yellow = wiring done, execution pending (a
scope/budget decision, not a coding task). Gray = not started.

## Why the pipeline bends the way it does

The shape here isn't a fixed plan executed top-to-bottom — it's the
result of two explicit gate decisions along the way, both driven by
external review rather than the original schedule:

1. **Stage 0.5 was inserted after Stage 0**, not originally planned. An
   external review of the initial plan asked the sharpest possible
   question — *"what does EFE give you that a good learned router
   doesn't?"* — and the response was to build a cheap, 1-2 week kill-test
   *before* committing to the full LangGraph/benchmark build, not after.
   See `RESEARCH_PLAN.md` §4a and [`stage0.5-kill-test-results.md`](stage0.5-kill-test-results.md).
2. **Stage 2 got a second pass (Part 2)** after the first pass's own
   result exposed its own weakness: comparing every controller against
   `mock_agent_step` validated plumbing but wasn't a real repeat of Stage
   0.5's Bayes-optimality methodology. `model_env.ModelMatchedEnv` closed
   that gap, and the result changed materially — see
   [`stage2-baselines-results.md`](stage2-baselines-results.md) and
   [`baselines-design.md`](baselines-design.md).

Both are the same discipline applied twice: don't trust a result until
the comparison is actually fair, and say so explicitly when an earlier
result wasn't.

## What each stage actually produced

| Stage | Deliverable | Where |
|---|---|---|
| 0 | Working pymdp engine, confirmed to reproduce the classic epistemic-value effect | `scripts/tmaze_sanity_check.py`, `results/stage0_tmaze_sanity_check.json` |
| 0.5 | EFE ≈ hand-derived VOI (statistically indistinguishable), both beat heuristic/router with real significance | `docs/stage0.5-kill-test-results.md`, `results/stage0_5_kill_test_results.json` |
| 1 | EFE control node over the real decision-POMDP; LangGraph wiring incl. working `interrupt()` human-review pause | `docs/efe-control-node-design.md`, `docs/langgraph-integration.md` |
| 1c | Real tool-calling LLM agent step (demo-scale) | `docs/observation-derivation.md` |
| 2 (Part 1) | Plumbing check — all 5 controllers pluggable into the same LangGraph scaffold | `docs/stage2-baselines-results.md` Part 1 (caveated: not a fair decision-quality test) |
| 2 (Part 2) | **The real result**: EFE and VOI diverge at real scale (unlike Stage 0.5's toy scale) — EFE wins escalation recall, VOI wins precision/reward; router has a real class-imbalance weakness | `docs/stage2-baselines-results.md` Part 2, `docs/baselines-design.md` |
| — | Real OPA policy_gate (not hardcoded `allow`) | `docs/observation-derivation.md` |
| 3 (wiring) | tau2-bench cloned/pinned; all 5 agents registered and smoke-tested; 2 real integration bugs caught and fixed; cost estimator built | `docs/tau2-bench-integration.md`, `scripts/estimate_stage3_cost.py` |
| 3 (sweep) | **Not yet run** — pending scope/budget sign-off | `context/TODOS.md` |
| 4-6 | Not started | `RESEARCH_PLAN.md` |

## See also

- [`known-issues-and-gotchas.md`](known-issues-and-gotchas.md) — every
  real bug caught across every stage above, in one place
- `RESEARCH_PLAN.md` — the full narrative, risks, and pivot triggers
- `context/HANDOFF.md` / `context/TODOS.md` — the actionable resume point
