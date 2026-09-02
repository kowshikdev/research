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

## Done (Stage 2 baselines)

- [x] **Stage 2: heuristic / learned_router / VOI / ReAct baselines**, ported from the kill-test's toy versions to the real 4-state/6-policy decision-POMDP — `src/aif_orchestrator/baselines/`. All four share `EFEControlNode`'s interface and are pluggable into the real LangGraph scaffold via `graph.build_graph(control_step=make_control_step(<cls>))` (verified end-to-end incl. the real `interrupt()` pause: `python -m aif_orchestrator.graph --stage2`), plus a fast statistical comparison against `mock_agent_step` (`python -m aif_orchestrator.baselines.run_stage2_eval`, results + write-up in `docs/stage2-baselines-results.md`).
  - `heuristic.py`: straightforward threshold port, as planned.
  - `router.py`: **belief-state-parity fix applied** (Stage 0.5's flagged gap) — Q-learning features are (belief bucket, observation), not just the latest observation.
  - `voi.py`: **diverges from the original TODO here** ("LLM-estimated P(success|context)") — implemented instead as a hand-derived one-step Bayesian expected-utility lookahead over the *same assumed observation model EFE itself uses* (`efe_controller.OBS_DISTS`/`B_TRANSITIONS`), with hand-coded step/escalate costs. Reasoning: the true generative model still isn't known at this stage either way, and reusing EFE's own assumed model (not a separately invented one) is the fairer "same observations, same assumptions, different math" comparison this stage needs — see `belief.py`'s docstring. An LLM-estimated version remains a real option for Stage 3, where actual agent trajectories would let it estimate something the current placeholder model can't.
  - `react.py`: fixed reactive floor baseline (call_tool until success, no escalation), as planned.
  - **Two real bugs caught and fixed during this stage** (details in `belief.py`'s docstring): (1) rewarding every turn's observation quality let the router learn to never choose `continue` at all (free reward for stalling); (2) with no ground-truth-keyed reward term, nothing taught the router to prefer `escalate_to_human` specifically over `continue` when a task was genuinely unresolvable. Both fixed via `belief.step_reward` (terminal-only payoff + an outcome adjustment keyed off ground truth).

- [x] **Model-matched Stage 2 comparison — closes the "not a fair test like Stage 0.5" gap the first Stage 2 pass left open.** `src/aif_orchestrator/baselines/model_env.py` (new): a simulator sampling from `efe_controller.py`'s own `D_PRIOR`/`OBS_DISTS`/`B_TRANSITIONS` as ground truth — the real analogue of `kill_test/env.py`. `router.py`'s `train()` was refactored to take a pluggable `source_factory` so the router can be retrained against this env too (`model_matched_source_factory`), not just evaluated on it after training on the mismatched mock (that would test transfer, not close the gap). Run: `python -m aif_orchestrator.baselines.run_model_matched_eval`. Full results + findings: `docs/stage2-baselines-results.md` Part 2.
  - **Real finding, not just plumbing this time:** EFE and VOI do NOT converge to statistically indistinguishable behavior at this scale, unlike Stage 0.5's toy model. EFE has the best correct-escalation rate (0.951) but lowest reward/throughput; VOI is the best-calibrated on precision (0.002 unnecessary-escalation) with good recall (0.766) and higher reward than EFE. **They diverge and trade off differently — this is the real Stage 2 result, and Part 1's numbers should not be cited over it.**
  - **The learned router, even with belief parity and 100k training episodes (up from 10k — the model-matched env's observation space needed much more coverage), has a real, mechanistically-understood calibration weakness**: it has the highest raw reward (3.393) but the worst correct-escalation rate (0.255), because genuinely unresolvable tasks are a training-distribution minority and Q-learning optimizes aggregate reward, not rare-event recall. This is DIFFERENT from the belief-parity issue (which was about information access, not what training does with that information) — worth remembering if anyone proposes "just give the router more training" as a fix; more of the same training regime won't fix a class-imbalance problem.
  - React ties/beats VOI on raw reward again here (3.183, second overall) despite zero escalation ability — confirms Part 1's finding under the fair comparison.

## Done (OPA policy_gate wiring)

- [x] **Real OPA instance for `policy_gate`** — `policies/policy_gate.rego` + `src/aif_orchestrator/opa_policy.py`. Shells out to `opa eval` per turn (no long-running server needed — a single CLI call per decision, evaluated via stdin-input) against the local `opa` binary. The policy is deliberately small given the current toolset (`llm_agent.py` only has one read-only `lookup_order` tool, nothing irreversible yet to genuinely deny): `deny` after 3+ repeated lookups of the same order_id (a real retry-loop circuit breaker), `needs_review` on a tool error, `allow` otherwise. Wired into `llm_agent.real_agent_step` only — `mock_agent_step` stays hardcoded `allow` since it doesn't track real tool-call history to check a retry policy against (a free non-LLM stand-in, per its own docstring). `opa_policy.evaluate_policy_gate` fails safe to `"needs_review"` (not `"allow"`) if OPA isn't installed or the policy errors. Verified end-to-end via `python -m aif_orchestrator.graph --llm`.
  - Extend `policies/policy_gate.rego` (not the Python wrapper) for richer policies once `llm_agent.py` grows tools worth actually denying (e.g. a cancel/refund action).

## Not started

- [ ] Replace the plain JSONL decision log with real OTel GenAI semantic-convention spans (`gen_ai.agent` / `invoke_agent`) — noted in `RESEARCH_PLAN.md` §3.2 as still-experimental conventions; needs an actual OTel collector/backend to send spans to, which is an infra decision, not just code.

## Not started (Stage 3+ from RESEARCH_PLAN.md)

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

Stage 1c, Stage 2 (both parts), and OPA `policy_gate` wiring are all done. Next: Stage 3 (real τ²-bench/HiL-Bench benchmark integration) — needs API keys AND external repos, budget for it before starting.
