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

## Done (Stage 1c, resumed locally with API credentials — see below)

- [x] **Stage 1c: swap `mock_agent_step` for a real tool-calling LLM agent step.** `src/aif_orchestrator/llm_agent.py` (new) + `graph.py`'s `llm_agent_step`. Uses an OpenAI-compatible client (OpenRouter, via `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` in `.env`), a fake deterministic `lookup_order` tool (real LLM tool-call decisions, scripted backend — same role a benchmark harness environment plays), and a verifier-prompt call for `confidence` (the cheaper of the two options `context/HANDOFF.md` flagged; self-consistency sampling was the alternative). `policy_gate` still hardcoded `allow` — OPA wiring is still its own TODO below. EFE's chosen policy is fed back into the next turn as a steering message (`llm_agent.POLICY_STEER`) so `retry`/`call_tool`/`gather_info` actually change what the agent does next, not just re-ask the same question. Verified end-to-end: `.venv/Scripts/python -m aif_orchestrator.graph --llm` — order-1001 task resolves in 1 turn (`continue`), order-9999 task genuinely escalates and pauses via `interrupt()`, same as the mock demo's Task A/B.
  - **Gotcha hit and fixed:** the model this session's `.env` pointed at (`stealth/ox-alpha` on OpenRouter) was retired mid-session; swapped to `z-ai/glm-5.3-flash`. That model makes reasoning mandatory (can't disable via `reasoning: {enabled: false}` — 400s) and burns the full `max_tokens` budget on hidden reasoning tokens if uncapped, returning empty `content` and making calls balloon. Fixed with `extra_body={"reasoning": {"max_tokens": N}}` plus a matching overall `max_tokens` headroom on every call in `llm_agent.py`. If you swap models again, check whether this still applies.

## BLOCKED — needs API credentials (this environment has none configured)

- [ ] Wire an actual OPA (Open Policy Agent) instance for the `policy_gate` observation modality — currently hardcoded to `allow` in both the mock and real agent steps. Needs OPA running somewhere + real policies to evaluate against.
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

## Environment setup (done in this repo already, kept here for reproducibility)

1. `.venv` created with `python -m venv .venv && .venv/Scripts/pip install -e .` (Windows paths — use `.venv/bin/...` on Linux/Mac)
2. `.env` at repo root holds `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL` (OpenRouter). `python-dotenv` and `openai` are now in `pyproject.toml`'s dependencies.
3. Mock demo: `.venv/Scripts/python -m aif_orchestrator.graph`. Real-LLM demo: `.venv/Scripts/python -m aif_orchestrator.graph --llm`.

## Next up

Stage 1c is done — the actual next unblock point is either the OPA `policy_gate` wiring above, or starting Stage 2 baselines below.
