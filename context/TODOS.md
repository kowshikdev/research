# TODOs — pick up here

Read `context/HANDOFF.md` first for full context. This file is just the
actionable checklist.

## Done (no API key needed — all verified working in this repo)

- [x] Stage 0: pymdp installed, T-maze sanity check passes (`scripts/tmaze_sanity_check.py`)
- [x] Stage 0.5: kill-test — EFE matches hand-derived VOI, beats heuristic/router with real statistical significance (`src/aif_orchestrator/kill_test/`, `docs/stage0.5-kill-test-results.md`)
- [x] Stage 1a: EFE control node generalized to the real frozen decision-POMDP (4 states, 4 obs modalities, 6 policies) — `src/aif_orchestrator/efe_controller.py`. Sanity-tested on 3 hand-picked scenarios (clean success / ambiguous / clearly stuck), all produce the expected policy.
- [x] Stage 1b: LangGraph wiring — `src/aif_orchestrator/graph.py`. Proven end-to-end with a mock (non-LLM) agent step:
  - Multi-turn belief tracking works (`task-A` demo: gather_info → call_tool → continue as belief concentrates)
  - `escalate_to_human` genuinely pauses the graph via `interrupt()` + `MemorySaver` checkpointer, and resuming with `Command(resume=...)` correctly injects human feedback (`task-B` demo)
  - Per-decision epistemic/pragmatic value logged to `results/stage1_decision_log.jsonl` (placeholder for the real OTel GenAI instrumentation — see below)

## BLOCKED — needs API credentials (this environment has none configured)

- [ ] **Stage 1c: swap `mock_agent_step` in `graph.py` for a real tool-calling LLM agent step.** This is the single next concrete task. `mock_agent_step()` in `src/aif_orchestrator/graph.py` currently returns synthetic, hand-scripted observations. Replace it with a real LLM call (Claude via `anthropic` SDK, or whatever you're using) that:
  1. Takes the current task context, makes a tool call or produces an answer
  2. Derives the 4 observation modalities (`tool_result`, `confidence`, `policy_gate`, `retrieval_quality`) from what actually happened — this derivation logic doesn't exist yet and needs real design (e.g., confidence could come from self-consistency sampling or a verifier prompt; policy_gate needs an actual OPA policy, not just a hardcoded "allow")
  3. Everything downstream (`efe_control_step`, routing, `interrupt()`) should work unchanged — it doesn't care where observations come from
- [ ] Wire an actual OPA (Open Policy Agent) instance for the `policy_gate` observation modality — currently hardcoded to `allow` in the mock agent. Needs OPA running somewhere + real policies to evaluate against.
- [ ] Replace the plain JSONL decision log with real OTel GenAI semantic-convention spans (`gen_ai.agent` / `invoke_agent`) — noted in `RESEARCH_PLAN.md` §3.2 as still-experimental conventions; needs an actual OTel collector/backend to send spans to, which is an infra decision, not just code.

## Not started (Stage 2+ from RESEARCH_PLAN.md)

- [ ] **Stage 2 baselines**, ported from the kill-test's toy versions to the real decision-POMDP + real LLM agent:
  - Heuristic threshold (straightforward port)
  - Learned router — **must fix the belief-state-parity issue flagged in the kill-test** (`docs/stage0.5-kill-test-results.md`): give it a running belief/summary feature, not just the latest observation, or the Stage 0.5 finding won't hold at benchmark scale
  - VOI/decision-theoretic controller — at real-agent scale this needs an LLM-estimated `P(success | context)` and hand-coded cost function, not the kill-test's exact closed-form version (the true generative model isn't known here)
  - Plain ReAct (no explicit control-loop reasoning)
- [ ] **Stage 3 benchmark integration** — needs API keys AND external repos:
  - Clone `sierra-research/tau2-bench`, pin a commit, record the grader version
  - Get HiL-Bench (`arXiv:2604.09408`) — check for a public code/data release
  - Run all 4 conditions (EFE, heuristic, router, ReAct) on both
  - This is the most API-cost-intensive stage — budget for it before starting
- [ ] Stage 4: GAIA validation subset (optional, only if Stage 3 finishes early)
- [ ] Stage 5: interpretability analysis — the epistemic/pragmatic decomposition logging already exists (`efe_controller.py`'s `Decision.epistemic_value` / `.pragmatic_value`), so this is mostly analysis + writeup once Stage 3 data exists
- [ ] Stage 6: writing

## First thing to do when you resume locally with API access

1. `cd` into the repo, `python3 -m venv .venv && .venv/bin/pip install -e .`
2. Run `.venv/bin/python -m aif_orchestrator.graph` to confirm the mock-agent demo still works in your environment
3. Start on Stage 1c above — that's the actual unblock point
