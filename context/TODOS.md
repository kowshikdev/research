# TODOs — pick up here

Read `context/HANDOFF.md` first for full context, and `docs/` (start with
[`docs/architecture-overview.md`](../docs/architecture-overview.md)) for
how everything fits together. This file is the actionable checklist.

## Done — verified, no LLM API access needed

- [x] **Stage 0**: pymdp installed, T-maze sanity check passes (`scripts/tmaze_sanity_check.py`)
- [x] **Stage 0.5**: kill-test — EFE matches hand-derived VOI, beats heuristic/router with real statistical significance (`src/aif_orchestrator/kill_test/`, `docs/stage0.5-kill-test-results.md`)
- [x] **Stage 1a**: EFE control node over the real frozen decision-POMDP (`src/aif_orchestrator/efe_controller.py`) — see `docs/efe-control-node-design.md` for the design, including a real bug caught and fixed (missing unnecessary-action cost, made EFE indifferent between `continue`/`gather_info` when already solved)
- [x] **Stage 1b**: LangGraph wiring (`src/aif_orchestrator/graph.py`) — multi-turn belief tracking, `escalate_to_human` → real `interrupt()`/checkpointer pause, verified end-to-end. See `docs/langgraph-integration.md`.
- [x] **Stage 2**: all 4 baselines (heuristic/learned_router/VOI/ReAct, `src/aif_orchestrator/baselines/`) ported to the real decision-POMDP, pluggable into the same LangGraph scaffold via `make_control_step()`. Two real reward-shaping bugs caught and fixed along the way. See `docs/baselines-design.md`.
- [x] **Stage 2 Part 2**: model-matched comparison (`run_model_matched_eval.py`) — the real repeat of Stage 0.5's methodology at real scale. **Real finding**: EFE and VOI diverge here (unlike Stage 0.5's toy scale) — EFE wins escalation recall (0.951), VOI wins precision (0.002 unnecessary-escalation)/reward. Read `docs/stage2-baselines-results.md` Part 2, not Part 1.
- [x] **OPA `policy_gate` wiring** — real `opa eval` against `policies/policy_gate.rego`, not a hardcoded `allow`. See `docs/observation-derivation.md`.
- [x] **tau2 agent wrappers for all 4 baselines** (`src/aif_orchestrator/tau2_integration/baseline_agents.py`) — `HeuristicAgent`/`RouterAgent`/`VOIAgent`/`ReActAgent`, each just `ControlNodeAgent` with `control_node_cls` swapped, registered in `register.py` under `heuristic_agent`/`router_agent`/`voi_agent`/`react_agent`. **Correction**: an earlier version of this file said these "would need... their own tau2 agent wrappers, not yet built — only EFE has one" — that was stale by the time it was written; they exist and `run_stage3_eval.py` already targets all 5 by default. Not yet exercised in a real sweep (see below), but the wrapper code itself is done.
- [x] **Cost/budget estimator for Stage 3** (`scripts/estimate_stage3_cost.py`, built this pass) — parametrized, documents its own assumptions, runs with no network/tau2 install needed (best-effort real task-count loading if tau2 *is* installed). Default scenario (all domains, 1 trial, all 5 agents): **~$5 total** at current DeepSeek V4 Flash pricing on OpenRouter — re-run with corrected `--avg-turns`/`--escalation-rate`/`--policy-tokens` once a real pilot exists before treating this as final.
- [x] **Router training-source gap for Stage 3** (found and fixed this pass) — `register.py` now pre-trains `LearnedRouterControlNode` against `model_matched_source_factory` (100,000 episodes, ~2.5s, verified in isolation) before registering any agents. Without this, `RouterAgent` would have lazily self-trained against the mismatched mock env on first use inside a real tau2 sweep — the same "not a fair test" issue Stage 2 Part 1 already flagged, silently reintroduced. See `docs/known-issues-and-gotchas.md` #10. **Needs a local re-run of the smoke test to confirm this doesn't break anything through the full tau2 import chain**, which this cloud session can't reach (see below).
- [x] **8 new architecture/design docs** (this pass) — `docs/architecture-overview.md`, `efe-control-node-design.md`, `langgraph-integration.md`, `baselines-design.md`, `tau2-bench-integration.md`, `observation-derivation.md`, `experiment-pipeline.md`, `known-issues-and-gotchas.md`. Start with `architecture-overview.md`.

## Done — needs real LLM API access, verified in an earlier local session

- [x] **Stage 1c**: real tool-calling LLM agent step (`src/aif_orchestrator/llm_agent.py`, `graph.py`'s `llm_agent_step`). Verified: `.venv/Scripts/python -m aif_orchestrator.graph --llm`.
- [x] **Stage 3 wiring**: tau2-bench cloned and pinned (commit `a2c0247`, tag `tau2==1.0.1`), `ControlNodeAgent` built generically (all 5 agents share it), registered, smoke-tested against a real `mock`-domain task (reward 1.0). Two real integration bugs caught and fixed (turn-0 cold start, `escalate_to_human` re-triggering — see `docs/known-issues-and-gotchas.md` #8-9). See `docs/tau2-bench-integration.md`.

## Environment constraint discovered this pass — read before assuming something is blocked on credentials alone

This cloud sandbox's network egress is locked to a fixed allowlist
(Anthropic's own API, PyPI/npm/crates.io/Go-proxy) — `openrouter.ai` is
**not** on it, and a call to it fails at the proxy layer
(`403 Forbidden` on the CONNECT) regardless of API key correctness. This
is architectural, not a missing-credential problem — see
`docs/known-issues-and-gotchas.md` #11. Any LLM-dependent verification
(Stage 1c, Stage 2 baseline demos against a live model, Stage 3 smoke
test or sweep, and the router fix's full-chain verification above) has
to happen locally, where network access is open.

An OpenRouter key **was** provided during this pass and is stored in
`.env` (gitignored, never committed, never appeared in any command
output) with `LLM_MODEL=deepseek/deepseek-v4-flash` (verified as the
real, current, cheap slug — $0.098/$0.196 per MTok input/output) — ready
to use the moment this repo runs somewhere with open network access.

## Not started

- [ ] **The actual Stage 3 evaluation sweep** — the real cost center. Run `scripts/estimate_stage3_cost.py` first, get explicit scope sign-off (which domains, how many trials), THEN:
  ```bash
  .venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_smoke   # confirm the router fix didn't break anything
  .venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_eval --domain retail --num-trials 1   # one domain first
  ```
  `run_stage3_eval.py` saves incrementally (`results/stage3_tau2/summary.json`, gitignored except that summary file) and supports `auto_resume` — a re-run picks up an existing save rather than re-paying for completed simulations.
- [ ] Once a small real pilot exists, correct `scripts/estimate_stage3_cost.py`'s placeholder assumptions (`--avg-turns`, `--escalation-rate`, `--policy-tokens`) from real observed numbers, and re-estimate before committing to the full sweep.
- [ ] Replace the plain JSONL decision log with real OTel GenAI semantic-convention spans (`gen_ai.agent`/`invoke_agent`) — still experimental conventions (`RESEARCH_PLAN.md` §3.2); needs an actual OTel collector/backend, an infra decision, not just code.
- [ ] HiL-Bench (`arXiv:2604.09408`) — not yet checked for a public code/data release.
- [ ] Stage 4: GAIA validation subset (optional, only if Stage 3 finishes early).
- [ ] Stage 5: interpretability analysis — the epistemic/pragmatic decomposition logging already exists and has from day one (`efe_controller.py`'s `Decision.epistemic_value`/`.pragmatic_value`, logged by both `graph.py` and `tau2_integration/`), so this is mostly analysis + writeup once Stage 3 sweep data exists.
- [ ] Stage 6: writing.

## Environment setup (for a fresh local session)

```bash
python3 -m venv .venv && .venv/bin/pip install -e .   # (.venv/Scripts/... on Windows)
```

`.env` at repo root needs `LLM_API_KEY` / `LLM_MODEL` / `LLM_BASE_URL`
(OpenRouter by default; `.env` already exists in this checkout with a
working key + `deepseek/deepseek-v4-flash` if this session's work was
pulled in). `python-dotenv` and `openai` are in `pyproject.toml`'s
dependencies already.

tau2-bench (for Stage 3 only): clone `sierra-research/tau2-bench` into
`external/tau2-bench` (gitignored), pin commit `a2c0247`, `uv sync`
inside it (its own venv, Python 3.12), then `pip install -e
external/tau2-bench` into *this* project's venv too — plus the
`audioop-lts` backport if this venv is Python 3.13+ (stdlib dropped
`audioop`, which tau2's voice-module import chain needs even unused).
`run_stage3_eval.py`/`run_stage3_smoke.py` map `.env`'s `LLM_API_KEY` to
`OPENROUTER_API_KEY` automatically (tau2's LiteLLM backend reads that
name, not ours).

## Quick verification (no network needed)

```bash
.venv/bin/python scripts/tmaze_sanity_check.py                       # Stage 0
.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test          # Stage 0.5 (~1 min)
.venv/bin/python -m aif_orchestrator.graph                             # Stage 1 (mock agent)
.venv/bin/python -m aif_orchestrator.graph --stage2                     # all 5 controllers through the real graph
.venv/bin/python -m aif_orchestrator.baselines.run_model_matched_eval    # Stage 2 Part 2 (~1-2 min)
.venv/bin/python scripts/estimate_stage3_cost.py --domain all             # Stage 3 cost estimate
```

## Next up

Everything network-independent is done, including the router-training
fix and the cost estimator. The one remaining decision is the same one
flagged before this pass, now with a concrete number attached:
**scope/budget for the real Stage 3 sweep** — the estimator says ~$5 for
the full default scenario (all domains, 1 trial, all 5 agents) at
current DeepSeek V4 Flash pricing, but that rests on placeholder
assumptions about average turns/escalation rate/policy size (see the
estimator's own docstring) that only a real local pilot can correct.
First step locally: re-run the smoke test (re-verifies the router fix
through the full tau2 import chain this cloud session can't reach), then
a small real pilot to correct the estimator's assumptions, then decide
scope for the full sweep.
