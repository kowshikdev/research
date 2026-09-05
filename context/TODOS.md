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
- [x] **9 architecture/design docs** — `docs/architecture-overview.md`, `efe-control-node-design.md`, `langgraph-integration.md`, `baselines-design.md`, `tau2-bench-integration.md`, `observation-derivation.md`, `experiment-pipeline.md`, `known-issues-and-gotchas.md`, `testing-and-pipeline.md`. Start with `architecture-overview.md`.
- [x] **Test suite** (`tests/`, 135 tests, ~2s, no network) — unit + integration coverage of the engine, all 5 controllers' interface conformance, graph routing and the real `interrupt()`/resume flow, OPA fail-safe behavior, the kill-test's reproducibility, the analysis layer, and the cost estimator. Includes a **regression test pinned to each of the 8 silent-failure bugs** in `docs/known-issues-and-gotchas.md` — that archive shows this project's characteristic bug is a plausible wrong decision, not a crash, so regressions are the main defense. Run: `pytest -q`.
- [x] **Analysis layer** (`src/aif_orchestrator/analysis/`) — the part that turns logged artifacts into reported numbers, which did not exist before:
  - `decision_log.py` — reader for the JSONL decision logs; raises on malformed/incomplete records rather than silently analyzing a subset.
  - `interpretability.py` — **Stage 5's deliverable**. Answers the question the epistemic/pragmatic decomposition was logged for since Stage 1: *did the epistemic term ever actually change a decision?* Explicitly flags the uncomfortable case (`decisions_driven_by_epistemic == 0`, meaning a purely goal-seeking controller would have chosen identically) instead of leaving it for a reader to notice, and refuses to report a 0.0 epistemic share for baselines that have no epistemic term by construction.
  - `stage3_report.py` — turns a tau2 sweep into the per-controller comparison table. The `summary.json` path is fully supported (our own schema); the raw-dump enrichment path is **written against an inferred schema and has never seen real tau2 output** — its precision/recall numbers are not citable until validated on the first real sweep (it says so in its docstring, its output, and its tests).
- [x] **End-to-end pipeline runner** (`scripts/run_full_pipeline.py`) — runs every offline stage in order, fails loudly on the first broken one, writes `results/pipeline_run.json` with per-stage status and headline numbers. Verified: all 7 stages pass, and the headline numbers reproduce the documented Stage 2 Part 2 result exactly.
- [x] **CI** (`.github/workflows/ci.yml`) — tests on Python 3.11 + 3.13 with an explicit wrong-`pymdp`-package check, `--quick` pipeline on every PR, and a nightly full pipeline that uploads `results/` (the runs that actually reproduce the cited numbers).
- [x] **`ROADMAP.md`** — the medium-horizon plan between `RESEARCH_PLAN.md` and this file: milestones, decision gates (including the null-result and "epistemic term never mattered" branches), what's deferred and why.
- [x] **Consistent controller naming** — `EFEControlNode` had no `name` attribute while all four baselines did, so decision logs labelled it `"EFEControlNode"` against the baselines' `"heuristic"`/`"voi_decision_theoretic"`. Now `name = "efe_active_inference"`, matching the key `run_model_matched_eval.py` already reports it under. Found by a test, not by reading.
- [x] **The actual Stage 3 evaluation sweep — done.** All 5 controllers × all 3 domains (retail/airline/telecom, 15 combinations) evaluated against real tau2-bench via Vertex AI (`gemini-2.5-flash`, GCP service-account credentials, headless — `vertex_auth.py`), not OpenRouter (the network constraint below was worked around this way, not resolved on OpenRouter's side). Results: `results/stage3_tau2/summary.json`. A real escalation-rate contamination bug was found mid-sweep and fixed in two layers (schema exclusion + `_strip_disallowed_transfer` post-generation filter in `efe_agent.py`) — see `docs/known-issues-and-gotchas.md` #13; retail and airline were re-run under the fix, telecom was run for the first time already fixed. One anomaly (`retail/voi_agent`'s escalation count swinging 19→113 between the original and re-run sweeps) is flagged as an open unknown, investigated and ruled out as caused by the fix itself but not otherwise explained.

## Done — needs real LLM API access, verified in an earlier local session

- [x] **Stage 1c**: real tool-calling LLM agent step (`src/aif_orchestrator/llm_agent.py`, `graph.py`'s `llm_agent_step`). Verified: `.venv/Scripts/python -m aif_orchestrator.graph --llm`.
- [x] **Stage 3 wiring**: tau2-bench cloned and pinned (commit `a2c0247`, tag `tau2==1.0.1`), `ControlNodeAgent` built generically (all 5 agents share it), registered, smoke-tested against a real `mock`-domain task (reward 1.0). Two real integration bugs caught and fixed (turn-0 cold start, `escalate_to_human` re-triggering — see `docs/known-issues-and-gotchas.md` #8-9). See `docs/tau2-bench-integration.md`.

## Environment constraint discovered earlier — resolved via Vertex, not OpenRouter

This cloud sandbox's network egress is locked to a fixed allowlist
(Anthropic's own API, PyPI/npm/crates.io/Go-proxy) — `openrouter.ai` is
**not** on it, and a call to it fails at the proxy layer
(`403 Forbidden` on the CONNECT) regardless of API key correctness. This
is architectural, not a missing-credential problem — see
`docs/known-issues-and-gotchas.md` #11. **Resolution actually used:**
Vertex AI's own API host is reachable from this sandbox, so the Stage 3
sweep ran there instead (`gemini-2.5-flash` via GCP service-account
credentials, `vertex_auth.py`) — OpenRouter itself was never made
reachable, this just routed around it. The `.env` OpenRouter key
mentioned in an earlier version of this file was never used for the
real sweep.

## Not started
- [ ] **Check the epistemic term against the real Stage 3 sweep data, not just the mock-agent log.** Running the interpretability analysis over the repo's existing mock-agent log showed **0/8 EFE decisions driven by the epistemic term** (epistemic share ~0.065) — but that was a tiny, non-real-task sample (the mock agent's scripted observations are unusually unambiguous, the regime where information gain has least to offer), so it proved nothing yet. Now that the real sweep exists (`results/stage3_tau2/summary.json`, all 3 domains), this is the actual test of `ROADMAP.md`'s Milestone 3 gate. Command: `python -m aif_orchestrator.analysis.stage3_report` for the sweep table; for the epistemic question, load the decision log and call `analyze_decision_log` (see `analysis/interpretability.py`).
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

Two commands cover everything:

```bash
.venv/bin/pip install -e ".[dev]"                 # dev extra adds pytest
.venv/bin/pytest -q                                # 135 tests, ~2s
.venv/bin/python scripts/run_full_pipeline.py       # every offline stage end-to-end
```

`--quick` skips the slow evaluations; `--only stage1 stage2-part2` runs a
subset; results land in `results/pipeline_run.json`. Individual stages
still have their own entry points if you want one in isolation
(`scripts/tmaze_sanity_check.py`, `-m aif_orchestrator.graph [--stage2]`,
`-m aif_orchestrator.kill_test.run_kill_test`,
`-m aif_orchestrator.baselines.run_model_matched_eval`,
`scripts/estimate_stage3_cost.py`).

After a sweep exists, the analysis entry points are:

```bash
.venv/bin/python -m aif_orchestrator.analysis.stage3_report            # comparison table
.venv/bin/python -m aif_orchestrator.analysis.stage3_report --raw       # + per-task enrichment (unvalidated)
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
