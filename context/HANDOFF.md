# Handoff — read this first

This session (cloud, no API keys) got as far as it could go without LLM
API credentials. This file plus `context/TODOS.md` should let a fresh
session (yours, local, with API keys attached) resume without
re-deriving anything. Everything referenced below is committed to
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

## What's NOT done, and why this session stopped here

The next piece of work is: **replace `mock_agent_step()` in `graph.py`
with a real tool-calling LLM agent step.** That needs LLM API credentials
this cloud environment doesn't have (checked: no `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY`, only `ANTHROPIC_BASE_URL` which is Claude Code's own
internal routing, not something reusable for an independent agent's API
calls). Per your instruction, this session stopped here rather than
guessing at credentials or building further speculative scaffolding
around an untested integration point.

Full breakdown of what's done vs. blocked vs. not-started:
`context/TODOS.md`.

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
.venv/bin/python -m aif_orchestrator.graph                   # Stage 1
```

All three should run clean and reproduce the numbers/behavior described
above and in `RESEARCH_PLAN.md`.
