# Handoff — read this first

**Update:** Stage 1c (the real tool-calling LLM agent step) is now done
— see the new item 6 below and `context/TODOS.md`. The rest of this file
is the original cloud-session handoff (kept for history); read item 6
first, then the rest for background on what it builds on.

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

## What's NOT done

With Stage 1c done, the next piece of work is wiring a real OPA (Open
Policy Agent) instance for the `policy_gate` observation modality — it's
hardcoded to `allow` in both `mock_agent_step` and `llm_agent_step`.
After that, Stage 2 (heuristic/router/VOI/ReAct baselines against the
same LangGraph scaffold) is the next unstarted stage.

Full breakdown of what's done vs. not-started: `context/TODOS.md`.

## Two things worth deciding early when you resume

1. **How to derive `confidence` and `policy_gate` from a real agent.**
   These aren't free — `confidence` needs either self-consistency
   sampling (multiple LLM calls, real cost) or a verifier prompt;
   `policy_gate` needs an actual OPA instance with real policies, not the
   mock's hardcoded `allow`. Neither exists yet; both are design
   decisions, not just wiring.
2. **Router belief-state parity** (flagged in the kill-test results,
   `docs/stage0.5-kill-test-results.md`) — when you build the real
   learned-router baseline in Stage 2, give it a running belief/summary
   feature, not just the latest observation. The kill-test's finding that
   EFE beats the router may partly be a feature-parity artifact if this
   isn't fixed, and that would undermine the Stage 3 results if caught
   late.

## How to verify everything in this handoff is real, not claimed

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python scripts/tmaze_sanity_check.py              # Stage 0
.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test # Stage 0.5 (~2 min)
.venv/bin/python -m aif_orchestrator.graph                   # Stage 1 (mock agent)
.venv/bin/python -m aif_orchestrator.graph --llm              # Stage 1c (real LLM, needs .env)
```

All four should run clean and reproduce the numbers/behavior described
above and in `RESEARCH_PLAN.md`. The last one needs `LLM_API_KEY` /
`LLM_MODEL` / `LLM_BASE_URL` set in `.env` at the repo root.
