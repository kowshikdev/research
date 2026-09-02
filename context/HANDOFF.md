# Handoff — read this first

**Update:** Stage 1c (the real tool-calling LLM agent step), Stage 2
(the four baselines, plus the model-matched comparison that actually
answers Stage 0.5's question at real scale), OPA `policy_gate` wiring,
and Stage 3's *wiring* (tau2-bench cloned + `EFEAgent` built and
smoke-tested — but not the actual evaluation sweep, which is the real
cost center and an open scope/budget decision) are now done — see
items 6-10 below and `context/TODOS.md`. The rest of this file is the
original cloud-session handoff (kept for history); read items 6-10
first, then the rest for background on what they build on.

This file plus `context/TODOS.md` should let a fresh session resume
without re-deriving anything. Everything referenced below is committed to
`claude/active-inference-llm-orchestration-1vt5po`.

## The project, in one paragraph

Reframe an LLM agent orchestrator's control-loop decisions (retry / gather
more info / escalate to a human / hand off) as Expected Free Energy (EFE)
minimization over a small, fixed decision-POMDP, instead of hand-tuned
heuristic thresholds. Full thesis plan: `RESEARCH_PLAN.md`.

## What's real and working right now

1. **`docs/decision-pomdp.md`** — the frozen decision space (4 hidden
   task-state values, 4 observation modalities, 6 policies). This is the
   spec everything else implements.

2. **`src/aif_orchestrator/kill_test/`** — a small synthetic experiment
   (not the real system) that answered the make-or-break question before
   any of the real build started: *does EFE actually do anything a good
   learned router or hand-derived Bayesian decision-theory controller
   doesn't?* Result: **yes** — EFE statistically matches a hand-derived
   Bayes-optimal controller (95% CIs overlap almost entirely) and beats a
   trained Q-learning router with a real, non-overlapping-CI gap. Full
   writeup: `docs/stage0.5-kill-test-results.md`. This is the thing that
   justified spending more time on the idea at all — read it before
   deciding to change direction.

3. **`src/aif_orchestrator/efe_controller.py`** — the real EFE control
   node, implementing `docs/decision-pomdp.md`'s actual schema (not the
   kill-test's toy 3-state version). Takes an `Observation` (4 modalities),
   maintains a belief over task state across turns, returns a `Decision`
   with the chosen policy plus the epistemic/pragmatic value decomposition
   per candidate policy. **The A/B/C/E/D values in this file are
   hand-specified placeholders** matching the semantic descriptions in
   `docs/decision-pomdp.md`, not calibrated against real data — that's
   Stage 3 work, once there's real agent trajectories to calibrate
   against.

4. **`src/aif_orchestrator/graph.py`** — wires the EFE control node into
   an actual LangGraph graph, with a **mock (non-LLM) agent step**
   standing in for a real tool-calling agent. Proven working:
   - Multi-turn belief tracking (`run_demo()`'s Task A: belief goes from
     uncertain to 99.6% `task_solvable_now` over 3 turns, decisions
     shift from `gather_info` → `call_tool` → `continue` as it does)
   - `escalate_to_human` genuinely pauses the graph via `interrupt()` +
     `MemorySaver` checkpointer — this isn't simulated, `app.get_state()`
     really shows the graph paused mid-execution, and
     `Command(resume=...)` really resumes it with injected human
     feedback (Task B demo)
   - Run it yourself: `.venv/bin/python -m aif_orchestrator.graph`

5. **Structured decision logging** — every EFE decision is written to
   `results/stage1_decision_log.jsonl` with the full epistemic/pragmatic
   breakdown per candidate policy. This is a placeholder for real OTel
   GenAI span instrumentation (see TODOs) but the actual data being
   logged is already what Stage 5's interpretability analysis needs.

6. **`src/aif_orchestrator/llm_agent.py`** — the real tool-calling LLM
   agent step (Stage 1c), wired in as `graph.py`'s `llm_agent_step`.
   Talks to an OpenAI-compatible endpoint (OpenRouter by default; reads
   `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` from `.env` via
   `python-dotenv`, both now in `pyproject.toml`). A fixed, deterministic
   `lookup_order` tool stands in for a real backend (the LLM's decisions
   about whether/how to call it are real, not scripted — same role a
   benchmark harness environment plays). `confidence` is derived via a
   separate verifier-prompt call (the cheaper of the two options this
   file originally flagged, vs. self-consistency sampling). `policy_gate`
   is still hardcoded `allow` — OPA wiring is unchanged, still a TODO.
   EFE's chosen policy is fed back into the next turn as a steering
   message (`llm_agent.POLICY_STEER`) — without this, a non-terminal
   policy like `gather_info` just re-asked the model the same question
   against unchanged messages and it repeated itself for `max_turns`.
   Verified end-to-end: `.venv/Scripts/python -m aif_orchestrator.graph --llm`
   — order 1001 (exists) resolves to `continue` in one turn; order 9999
   (doesn't exist) genuinely escalates and pauses via `interrupt()`, same
   as the mock demo's Task B.

   **Gotcha for whoever touches the model config next:** this session's
   `.env` initially pointed at `stealth/ox-alpha` on OpenRouter, which
   was retired mid-session (404, redirected to `z-ai/glm-5.3-flash`).
   That model makes reasoning mandatory — `reasoning: {enabled: false}`
   is rejected with a 400 — and without a cap it burns the entire
   `max_tokens` budget on hidden reasoning tokens, returning empty
   `content` and making every call slow. Fixed by passing
   `extra_body={"reasoning": {"max_tokens": N}}` plus a larger overall
   `max_tokens` on every call in `llm_agent.py`. If the model changes
   again, check whether this still applies before assuming a hang is a
   network issue.

7. **`src/aif_orchestrator/baselines/`** — the four Stage 2 baselines
   (heuristic, learned_router, VOI, ReAct), ported from the toy
   kill-test controllers to the real decision-POMDP, all sharing
   `EFEControlNode`'s interface. `graph.py`'s `make_control_step()`
   generalizes what used to be the hardcoded `efe_control_step`, so any
   of these plug into the real LangGraph scaffold the same way
   `llm_agent_step` plugs in for the agent step
   (`graph.build_graph(control_step=make_control_step(<cls>))`,
   demoed by `graph.run_stage2_demo()` / `python -m aif_orchestrator.graph --stage2`
   — includes verifying the real `interrupt()` pause for each
   controller). A fast statistical comparison (3000 episodes each
   against `mock_agent_step`) is `baselines/run_stage2_eval.py`; **this
   is a plumbing sanity check, not the real result** — `mock_agent_step`
   isn't drawn from the same generative model EFE/VOI assume, so it
   doesn't repeat Stage 0.5's Bayes-optimality methodology. Read
   `docs/stage2-baselines-results.md` Part 1 for that caveat in full.

   The router's belief-state-parity fix the kill-test flagged is
   applied (features are (belief bucket, observation), not just the
   latest observation). VOI's implementation diverges from the original
   TODO sketch ("LLM-estimated P(success|context)") — see
   `baselines/voi.py`'s docstring for the reasoning.

   **Two real bugs were caught and fixed while building this — worth
   knowing about if you touch `belief.py`/`router.py` again:** (1)
   rewarding every turn's observation quality (not just terminal turns)
   let the router learn to never choose `continue` at all, since
   re-observing a good state was pure upside; (2) even after fixing
   that, nothing in the reward differentiated `continue` from
   `escalate_to_human` at the same observation, so the router never
   learned to prefer escalating specifically. Both fixed in
   `belief.step_reward` (terminal-only real payoff + a ground-truth-
   keyed outcome adjustment). If you build a Stage 3 reward function,
   check whether either failure mode applies there too.

8. **`baselines/model_env.py` + `run_model_matched_eval.py`** — closes
   the Part-1 gap: a simulator that samples the true `task_state` from
   `efe_controller.D_PRIOR`, observations from `OBS_DISTS[true_state]`,
   and transitions via `B_TRANSITIONS[policy][true_state]` — i.e. the
   real analogue of `kill_test/env.py`, treating EFE's own generative
   model as ground truth for the eval. `router.py`'s `train()` was
   refactored to take a pluggable `source_factory` so the router trains
   against this same environment (`model_matched_source_factory`), not
   just the mismatched mock. **This is the result to actually cite** —
   full table and analysis in `docs/stage2-baselines-results.md` Part 2.

   **Read this before assuming Stage 0.5's "EFE ≈ VOI" finding still
   holds:** it doesn't, at this scale. EFE has the best correct-
   escalation rate (0.951) but the lowest reward/throughput; VOI has the
   best precision (0.002 unnecessary-escalation) and beats EFE on
   reward. They trade off differently, not identically, at the real
   4-state/6-policy scale. Also: the learned router, even fully
   belief-parity-fixed and trained for 100k episodes, has the *worst*
   correct-escalation rate (0.255) of the five despite the *highest*
   reward — a distinct, reward-maximizing-under-class-imbalance failure
   mode (genuinely unresolvable tasks are a training-distribution
   minority), not a belief-access problem. Don't conflate the two if
   you're deciding whether the router needs more training or a
   different reward design.

9. **`policies/policy_gate.rego` + `src/aif_orchestrator/opa_policy.py`**
   — real OPA evaluation for `policy_gate`, replacing the hardcoded
   `allow` (`opa eval` shelled out per turn, no server, ~10s timeout,
   fails safe to `"needs_review"` not `"allow"` if OPA errors or isn't
   installed). Wired into `llm_agent.real_agent_step` only —
   `mock_agent_step` stays hardcoded `allow` since it has no real
   tool-call history to check a retry policy against. The policy itself
   is small on purpose: `llm_agent.py`'s only tool is a read-only
   `lookup_order`, so there's nothing irreversible yet worth a real
   `deny` for beyond a retry-loop circuit breaker (3+ repeated lookups
   of the same order_id → `deny`; a tool error → `needs_review`).
   Extend the `.rego` file, not the Python wrapper, once there's a tool
   worth denying. Verified: `allow` and `needs_review` both observed in
   the real `--llm` demo; `deny` unit-verified directly (it needs 3+
   identical retries, which the demo tasks don't happen to trigger).

10. **`src/aif_orchestrator/tau2_integration/`** — Stage 3's real
    benchmark wiring. `external/tau2-bench` (gitignored, cloned and
    pinned at commit `a2c0247` / `tau2==1.0.1`; upstream is now branded
    τ³-bench but it's the same `sierra-research/tau2-bench` repo/lineage
    RESEARCH_PLAN.md calls τ²-bench) is installed both in its own venv
    (`uv sync`) and editable into *this* project's venv
    (`pip install -e external/tau2-bench` + the `audioop-lts` backport —
    tau2's voice-module import chain needs stdlib `audioop`, which
    Python 3.13 dropped; our venv is 3.13, tau2's own is 3.12 where it
    still works). `efe_agent.py` implements `EFEAgent`, a real tau2
    `HalfDuplexAgent` wrapping `EFEControlNode` the same way
    `llm_agent.py` wraps the order-lookup demo; `register.py` registers
    it as a community agent (`registry.register_agent_factory`, called
    from our own code — the vendored repo is never edited). **New in
    this integration**: `escalate_to_human` now calls a real tool
    (`transfer_to_human_agents`, present in every core tau2 domain) —
    tau2's own evaluator can grade whether that call was actually
    correct for a task, unlike the LangGraph demo's `interrupt()` pause,
    which just proves the mechanism works without external grading.

    **Smoke-tested only** (`run_stage3_smoke.py`, one `mock`-domain
    task, reward 1.0) — the real evaluation sweep (retail/airline/
    telecom, EFE vs. baselines, multiple trials) has NOT run. That's
    the actual cost center RESEARCH_PLAN.md's Stage 3 section warns
    about; the smoke test cost about as much as one ordinary eval task.
    Get explicit scope/budget sign-off before running a real sweep.
    Two gotchas if you touch this: tau2 (LiteLLM) reads
    `OPENROUTER_API_KEY`, not our own `.env`'s `LLM_API_KEY` — mapped
    in `run_stage3_smoke.py`, do the same in any new entry point; and
    only EFE has a tau2 agent wrapper so far — the four Stage 2
    baselines would each need their own before a real EFE-vs-baselines
    comparison on tau2-bench is possible.

## What's NOT done

The actual Stage 3 evaluation sweep (see item 10) — the wiring is done,
the sweep is a scope/budget decision, not a coding task. HiL-Bench
(`arXiv:2604.09408`) also hasn't been checked for a public code/data
release yet.

Full breakdown of what's done vs. not-started: `context/TODOS.md`.

## One thing worth deciding early when you resume

**How to derive `confidence` from a real agent, if building anything
beyond the current demo tool** (`policy_gate` is now real, see item 9
above). `confidence` currently uses a verifier-prompt call
(`llm_agent.py`); self-consistency sampling was the other option, not
used (real cost, multiple LLM calls).

## How to verify everything in this handoff is real, not claimed

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/tmaze_sanity_check.py                        # Stage 0
.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test           # Stage 0.5 (~2 min)
.venv/bin/python -m aif_orchestrator.graph                             # Stage 1 (mock agent)
.venv/bin/python -m aif_orchestrator.graph --llm                        # Stage 1c (real LLM, needs .env)
.venv/bin/python -m aif_orchestrator.graph --stage2                      # Stage 2, all 5 controllers in the real graph
.venv/bin/python -m aif_orchestrator.baselines.run_stage2_eval           # Stage 2 Part 1: mock-based (~5 min)
.venv/bin/python -m aif_orchestrator.baselines.run_model_matched_eval    # Stage 2 Part 2: model-matched, the real result (~5 min)
```

All seven should run clean and reproduce the numbers/behavior described
above and in `RESEARCH_PLAN.md`. `--llm` needs `LLM_API_KEY` /
`LLM_MODEL` / `LLM_BASE_URL` set in `.env` at the repo root, and the
`opa` CLI installed and on `PATH` for the `policy_gate` evaluation
(`opa_policy.py` fails safe to `"needs_review"` if it's missing, so
`--llm` still runs without it, just without a real policy check); the
rest need no credentials or extra binaries.
