# research

Active Inference / Expected-Free-Energy control for LLM multi-agent
orchestration. See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full
thesis plan, resource inventory, and stage-by-stage timeline.

## Status

Stage 1 partially complete, blocked on LLM API credentials. **Resuming
locally with API access? Start at [`context/HANDOFF.md`](context/HANDOFF.md)
and [`context/TODOS.md`](context/TODOS.md)** — they say exactly what's
done, what's blocked, and what to do first.

Summary: decision-POMDP schema frozen (`docs/decision-pomdp.md`); pymdp
engine sanity check passing; Stage 0.5 kill-test complete with a positive
result (`docs/stage0.5-kill-test-results.md`); the real EFE control node
and LangGraph wiring (including a working `escalate_to_human` →
`interrupt()` human-review pause) are built and proven against a mock
agent (`src/aif_orchestrator/`).

## Quick verification

```bash
.venv/bin/python scripts/tmaze_sanity_check.py               # Stage 0
.venv/bin/python -m aif_orchestrator.kill_test.run_kill_test  # Stage 0.5 (~2 min)
.venv/bin/python -m aif_orchestrator.graph                    # Stage 1 (mock agent)
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Note: install `inferactively-pymdp`, not the plain `pymdp` PyPI package
(name collision with an unrelated project) — see `pyproject.toml`.
