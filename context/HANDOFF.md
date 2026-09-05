# Handoff — read this first

**Latest update (this pass): the Stage 3 sweep is done.** Items 10-13
below describe the state *before* the sweep ran (wiring done, sweep not
started, this cloud sandbox can't reach any LLM provider) — that
constraint was resolved via Vertex AI (gemini-2.5-flash, GCP
service-account credentials work headless; `vertex_auth.py`), not by
finding a way to reach OpenRouter. All 5 controllers are now evaluated
against real tau2-bench across all three domains (retail/airline/
telecom — `results/stage3_tau2/summary.json`, 15 domain/agent
combinations).

Along the way, a real escalation-rate contamination bug was found and
fixed: the underlying LLM could call `transfer_to_human_agents` on any
turn regardless of the control node's decision — first because the tool
was included in the schema on non-escalate turns, and then (a deeper
layer, found by re-checking after the first fix) because
gemini-2.5-flash via Vertex's OpenAI-compatible passthrough didn't
strictly confine its output to the declared schema even once the tool
was excluded from it. Closing it needed two independent layers: schema
exclusion AND a post-generation filter (`_strip_disallowed_transfer` in
`efe_agent.py`). Verified via `react_agent` (whose control node can
never select `escalate_to_human` by construction) showing 0 transfer
calls across all three domains post-fix, vs. 24/50 on airline before it.
Full writeup: `docs/known-issues-and-gotchas.md` #13. Retail and airline
were re-run under the fix; telecom was run for the first time already
under it, so it has no contaminated history.

One anomaly is flagged but not fully explained: `retail/voi_agent`'s
escalation count swung from 19 (original sweep) to 113 (re-run) — by far
the largest change of any re-run controller. Investigated and ruled out
as caused by the fix itself (VOI's own decision logic and the OPA
policy_gate wiring are both unchanged between the two runs); most likely
explanation is run-to-run sampling variance in the live, real
conversations, but that's not confirmed. Documented as an open unknown,
not silently accepted — see `docs/known-issues-and-gotchas.md`'s Open
Unknowns section.

Prior "Latest update" (kept below, still accurate as history): corrected
a stale claim in the previous version of this file (item 10 below said
the four Stage 2 baselines "would need their own tau2 agent wrappers,
not yet built" — they were already built by the time that was written;
corrected here), found and fixed a real gap in how the Stage 3 router
baseline gets trained (item 11), built the Stage 3 cost/budget estimator
TODOS.md was waiting on (item 12), discovered a hard environment
constraint this cloud session can't work around (item 13 — since
resolved via Vertex, see above), and wrote 8 new architecture/design
docs under `docs/` (item 14). Read items 10-14 for that history; the
rest of this file is prior history, kept because it's still accurate.

This file plus `context/TODOS.md` should let a fresh session resume
without re-deriving anything. Everything referenced below is committed to
`claude/active-inference-llm-orchestration-1vt5po`. **Start with
[`docs/architecture-overview.md`](../docs/architecture-overview.md)** for
the system map before diving into any individual file.

## The project, in one paragraph

Reframe an LLM agent orchestrator's control-loop decisions (retry / gather
more info / escalate to a human / hand off) as Expected Free Energy (EFE)
minimization over a small, fixed decision-POMDP, instead of hand-tuned
heuristic thresholds. Full thesis plan: `RESEARCH_PLAN.md`.

## What's real and working right now

1. **`docs/decision-pomdp.md`** — the frozen decision space (4 hidden
   task-state values, 4 observation modalities, 6 policies). This is the
   spec everything else implements. Design rationale (why these numbers,
   why this structure): `docs/efe-control-node-design.md`.

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
   Stage 3+ work, once there's real agent trajectories to calibrate
   against. Full design writeup, including a real bug caught and fixed
   here (EFE was indifferent between `continue` and `gather_info` when
   the task was already solved — no cost signal for unnecessary action):
   `docs/efe-control-node-design.md`.

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
   Full node/edge diagram and pluggability details: `docs/langgraph-integration.md`.

5. **Structured decision logging** — every EFE decision is written to
   `results/stage1_decision_log.jsonl` with the full epistemic/pragmatic
   breakdown per candidate policy. This is a placeholder for real OTel
   GenAI span instrumentation (see TODOs) but the actual data being
   logged is already what Stage 5's interpretability analysis needs.

6. **`src/aif_orchestrator/llm_agent.py`** — the real tool-calling LLM
   agent step (Stage 1c), wired in as `graph.py`'s `llm_agent_step`.
   Talks to an OpenAI-compatible endpoint (OpenRouter by default; reads
   `LLM_API_KEY`/`LLM_MODEL`/`LLM_BASE_URL` from `.env` via
   `python-dotenv`, both in `pyproject.toml`). A fixed, deterministic
   `lookup_order` tool stands in for a real backend (the LLM's decisions
   about whether/how to call it are real, not scripted — same role a
   benchmark harness environment plays). `confidence` is derived via a
   separate verifier-prompt call (the cheaper of the two options
   originally flagged, vs. self-consistency sampling). `policy_gate` is
   now a real OPA evaluation (item 9). EFE's chosen policy is fed back
   into the next turn as a steering message (`llm_agent.POLICY_STEER`) —
   without this, a non-terminal policy like `gather_info` just re-asked
   the model the same question against unchanged messages and it
   repeated itself for `max_turns`. Verified end-to-end:
   `.venv/Scripts/python -m aif_orchestrator.graph --llm` — order 1001
   (exists) resolves to `continue` in one turn; order 9999 (doesn't
   exist) genuinely escalates and pauses via `interrupt()`, same as the
   mock demo's Task B. Derivation details for all 4 modalities across all
   3 environments (mock/llm_agent/tau2): `docs/observation-derivation.md`.

   **Gotcha for whoever touches the model config next:** an earlier
   session's `.env` initially pointed at `stealth/ox-alpha` on
   OpenRouter, which was retired mid-session (404, redirected to
   `z-ai/glm-5.3-flash`). That model makes reasoning mandatory —
   `reasoning: {enabled: false}` is rejected with a 400 — and without a
   cap it burns the entire `max_tokens` budget on hidden reasoning
   tokens, returning empty `content` and making every call slow. Fixed
   by passing `extra_body={"reasoning": {"max_tokens": N}}` plus a
   larger overall `max_tokens` on every call. **This pass's `.env` now
   points at `deepseek/deepseek-v4-flash`** (see item 13) — check
   whether the mandatory-reasoning gotcha still applies before assuming a
   hang is a network issue; full list of every gotcha like this one:
   `docs/known-issues-and-gotchas.md`.

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
   `docs/stage2-baselines-results.md` Part 1 for that caveat in full, and
   `docs/baselines-design.md` for the shared-interface design and a
   controller-by-controller mechanism comparison table.

   The router's belief-state-parity fix the kill-test flagged is
   applied (features are `(belief bucket, observation)`, not just the
   latest observation). VOI's implementation diverges from the original
   TODO sketch ("LLM-estimated P(success|context)") — see
   `baselines/voi.py`'s docstring, or `docs/baselines-design.md`, for the
   reasoning.

   **Two real bugs were caught and fixed while building this — worth
   knowing about if you touch `belief.py`/`router.py` again:** (1)
   rewarding every turn's observation quality (not just terminal turns)
   let the router learn to never choose `continue` at all, since
   re-observing a good state was pure upside; (2) even after fixing
   that, nothing in the reward differentiated `continue` from
   `escalate_to_human` at the same observation, so the router never
   learned to prefer escalating specifically. Both fixed in
   `belief.step_reward` (terminal-only real payoff + a ground-truth-
   keyed outcome adjustment). Full narrative in
   `docs/baselines-design.md` and `docs/known-issues-and-gotchas.md`
   (#5-6).

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
   installed). Wired into `llm_agent.real_agent_step` and, since the tau2
   integration landed, `tau2_integration/efe_agent.py`'s
   `ControlNodeAgent` too — `mock_agent_step` stays hardcoded `allow`
   since it has no real tool-call history to check a retry policy
   against. The policy itself is small on purpose: a retry-loop circuit
   breaker (3+ repeated identical tool calls → `deny`) and a review flag
   on tool errors. Extend the `.rego` file, not the Python wrapper, once
   there's a tool worth denying more specifically (e.g. tau2's
   refund/cancel actions). Full detail: `docs/observation-derivation.md`.

10. **`src/aif_orchestrator/tau2_integration/`** — Stage 3's real
    benchmark wiring, **and this is more complete than a previous version
    of this file said.** `external/tau2-bench` (gitignored, cloned and
    pinned at commit `a2c0247` / `tau2==1.0.1`; upstream is now branded
    τ³-bench but it's the same `sierra-research/tau2-bench` repo/lineage
    RESEARCH_PLAN.md calls τ²-bench) is installed both in its own venv
    (`uv sync`) and editable into *this* project's venv. `efe_agent.py`
    implements `ControlNodeAgent`, a **generic** real tau2
    `HalfDuplexAgent` wrapper — not EFE-specific despite the file name —
    and `EFEAgent` is just that class with `control_node_cls =
    EFEControlNode`. `baseline_agents.py` does the exact same thing for
    all four Stage 2 baselines (`HeuristicAgent`/`RouterAgent`/`VOIAgent`/
    `ReActAgent`), and `register.py` registers all five
    (`efe_agent`/`heuristic_agent`/`router_agent`/`voi_agent`/`react_agent`).
    **Correction: an earlier version of this file said the baselines
    "would need their own tau2 agent wrappers, not yet built — only EFE
    has one" — that was already stale when it was written; they exist,
    and `run_stage3_eval.py`'s `AGENTS` list already targets all 5 by
    default.** Full turn-by-turn sequence diagram and the two real
    integration bugs caught here (turn-0 cold start misread as
    `needs_human` evidence; `escalate_to_human` re-triggering up to 7
    times per conversation before a terminal-flag fix):
    `docs/tau2-bench-integration.md`.

    `escalate_to_human` now calls a real tool (`transfer_to_human_agents`,
    present in every core tau2 domain) — tau2's own evaluator can grade
    whether that call was actually correct for a task, unlike the
    LangGraph demo's `interrupt()` pause, which just proves the mechanism
    works without external grading.

    **Smoke-tested only** (`run_stage3_smoke.py`, one `mock`-domain
    task, reward 1.0) — the real evaluation sweep (retail/airline/
    telecom, all 5 agents, multiple trials) has NOT run. That's the
    actual cost center RESEARCH_PLAN.md's Stage 3 section warns about.
    Get explicit scope/budget sign-off before running a real sweep — see
    item 12 below, the estimator now exists for exactly this. Gotcha if
    you touch this: tau2 (LiteLLM) reads `OPENROUTER_API_KEY`, not our
    own `.env`'s `LLM_API_KEY` — mapped automatically in both
    `run_stage3_smoke.py` and `run_stage3_eval.py`.

11. **Router training-source gap for Stage 3 — found and fixed this
    pass.** `LearnedRouterControlNode` self-trains lazily on first
    construction (a class-level `_q` cache) using its own default —
    10,000 episodes against `graph.mock_agent_step`, the exact
    mismatched training regime item 7/8 above already established as
    "not a fair test". `RouterAgent`'s factory never overrode this, so a
    real tau2 sweep would have silently used the weaker Q-table instead
    of the model-matched one item 8 established as fairer — no error,
    no warning, just a worse baseline result nobody would have noticed
    without comparing Q-table sizes. Fixed: `register.py` now explicitly
    pre-trains against `model_matched_source_factory` (100,000 episodes)
    *before* registering any agents, verified in isolation (2.5s, no
    errors, 1021 states learned) — **not yet re-verified through the
    full tau2 import chain**, which this cloud session can't reach (item
    13). Run `run_stage3_smoke.py` locally first thing to confirm this
    didn't break anything. Full writeup: `docs/known-issues-and-gotchas.md` #10.

12. **`scripts/estimate_stage3_cost.py` — built this pass**, closing the
    exact gap TODOS.md was waiting on ("get explicit sign-off on
    scope/budget before running this, not just before starting Stage 3
    generally"). Parametrized cost estimator, no network or tau2 install
    needed to run (best-effort real task-count loading if tau2 *is*
    available locally, else a documented placeholder split). Every
    assumption (average turns/task, escalation rate, domain-policy token
    size) is a labeled placeholder, not a measurement — the script says
    so in its own output. Default scenario (all 3 domains, 1 trial, all
    5 agents) estimates **~$5 total** at current `deepseek/deepseek-v4-flash`
    pricing. Correct the placeholders from a real small pilot run before
    treating this as final for a large-scale decision.

13. **Hard environment constraint discovered this pass: this cloud
    sandbox cannot reach OpenRouter (or most third-party hosts) at all.**
    An OpenRouter key was provided and stored in `.env`
    (`LLM_MODEL=deepseek/deepseek-v4-flash` — verified as the real,
    current, cheap slug: $0.098/$0.196 per MTok in/out) — but every
    attempt to actually call it fails at the network proxy layer
    (`httpcore2.ProxyError: 403 Forbidden` on the CONNECT to
    `openrouter.ai:443`). Checked the proxy's own status endpoint: egress
    is locked to a fixed allowlist (Anthropic's own API, PyPI, npm,
    crates.io, the Go module proxy) — this is an organization network
    policy, not a missing-credential problem, and not something fixable
    from inside the sandbox. **Anything needing a real LLM call has to
    run locally** (or in any environment with open network egress) —
    this includes re-verifying item 11's router fix through the full
    tau2 import chain, and obviously the actual Stage 3 sweep itself.
    Full detail: `docs/known-issues-and-gotchas.md` #11.

15. **Test suite, analysis layer, pipeline runner, CI, and roadmap —
    built after the docs pass.** These closed the remaining end-to-end
    gaps: there were no tests at all, no CI, and nothing that read back
    the decision logs the pipeline had been writing since Stage 1.
    - **`tests/` (135 tests, ~2s, no network)** — engine, all five
      controllers' interface conformance (parametrized, so "they're
      interchangeable" is checked rather than asserted in prose), graph
      routing and the real `interrupt()`/resume flow, OPA fail-safe
      behavior, kill-test reproducibility, the analysis layer, and the
      cost estimator. Every silent-failure bug in
      `docs/known-issues-and-gotchas.md` has a regression test pinned to
      it — that archive shows the characteristic failure here is a
      plausible wrong decision, not a crash.
    - **`src/aif_orchestrator/analysis/`** — `decision_log.py` (reader
      that raises rather than silently analyzing a subset),
      `interpretability.py` (**Stage 5's deliverable** — answers whether
      the epistemic term ever actually changed a decision, and flags the
      case where it never did), `stage3_report.py` (sweep → comparison
      table; the raw-dump path is explicitly unvalidated against real
      tau2 output).
    - **`scripts/run_full_pipeline.py`** — every offline stage in one
      command, writing `results/pipeline_run.json`. Verified: 7/7 stages
      pass and the headline numbers reproduce the documented Stage 2
      Part 2 result exactly.
    - **`.github/workflows/ci.yml`** — tests on 3.11/3.13 with a
      wrong-`pymdp`-package check, quick pipeline per PR, full pipeline
      nightly with `results/` uploaded.
    - **`ROADMAP.md`** — the medium-horizon plan with its decision gates,
      including the two that could redirect the thesis (a null result on
      Stage 3, and the epistemic term turning out never to matter).
    - One real inconsistency fixed along the way: `EFEControlNode` had no
      `name` attribute while every baseline did, so decision logs
      labelled it `"EFEControlNode"` against `"heuristic"` etc. A test
      caught it.

14. **8 architecture/design docs, `docs/` — built in the docs pass**
    (a 9th, `testing-and-pipeline.md`, came with the test suite above).
    `architecture-overview.md` (start here — the system map),
    `efe-control-node-design.md`, `langgraph-integration.md`,
    `baselines-design.md`, `tau2-bench-integration.md`,
    `observation-derivation.md`, `experiment-pipeline.md` (the research
    stages as a diagram, cross-referenced to what each one produced), and
    `known-issues-and-gotchas.md` (every real bug across every stage,
    consolidated in one place, with a "lesson" for each). Each has at
    least one mermaid diagram matching the actual code structure, not a
    simplified stand-in for it. These sit alongside the 3 pre-existing
    docs (`decision-pomdp.md`, `stage0.5-kill-test-results.md`,
    `stage2-baselines-results.md`) — 11 files in `docs/` total.

## What's NOT done

The actual Stage 3 evaluation sweep (item 10/12) — the wiring is done,
the cost estimator exists, the sweep itself is a scope/budget decision
now backed by a real number, not a coding task. HiL-Bench
(`arXiv:2604.09408`) also hasn't been checked for a public code/data
release yet. OTel GenAI span instrumentation is still a plain-JSONL
placeholder (needs an actual collector/backend — an infra decision).

Full breakdown of what's done vs. not-started: `context/TODOS.md`.

## Two things worth deciding early when you resume locally

1. **Run `run_stage3_smoke.py` first**, before anything else — it
   re-verifies item 11's router-training fix through the actual tau2
   import chain, which this cloud session could not do.
2. **Correct `scripts/estimate_stage3_cost.py`'s placeholder assumptions**
   from a small real pilot (a handful of tasks, one domain) before using
   its estimate to sign off on the full sweep's scope/budget.

## How to verify everything in this handoff is real, not claimed

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/tmaze_sanity_check.py                        # Stage 0
.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test           # Stage 0.5 (~1 min)
.venv/bin/python -m aif_orchestrator.graph                             # Stage 1 (mock agent)
.venv/bin/python -m aif_orchestrator.graph --llm                        # Stage 1c (real LLM, needs .env + network)
.venv/bin/python -m aif_orchestrator.graph --stage2                      # Stage 2, all 5 controllers in the real graph
.venv/bin/python -m aif_orchestrator.baselines.run_stage2_eval           # Stage 2 Part 1: mock-based (~5 min)
.venv/bin/python -m aif_orchestrator.baselines.run_model_matched_eval    # Stage 2 Part 2: model-matched, the real result (~1-2 min)
.venv/bin/python scripts/estimate_stage3_cost.py --domain all             # Stage 3 cost estimate, no network needed
.venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_smoke # Stage 3 smoke test (needs tau2-bench installed + network)
```

Everything except `--llm` and the tau2 smoke test runs with no
credentials, no extra binaries, and no network — those two need `.env`
set, the `opa` CLI on `PATH` (optional — fails safe if missing), and
(for the smoke test) `external/tau2-bench` installed per
`context/TODOS.md`'s environment setup section, plus open network
egress this cloud session doesn't have (item 13).
