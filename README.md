# research

Active Inference / Expected-Free-Energy control for LLM multi-agent
orchestration. See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full
thesis plan, resource inventory, and stage-by-stage timeline, and
[`docs/architecture-overview.md`](docs/architecture-overview.md) for the
system architecture (start there — every other `docs/*.md` file zooms
into one part of it).

## Status

Stages 0, 0.5, 1, 2, and Stage 3's wiring are done. The Stage 3 real
benchmark sweep (the actual cost center) hasn't run yet — it's a
scope/budget decision, backed by a cost estimator
(`scripts/estimate_stage3_cost.py`), not a coding task. **Resuming with
LLM API access? Start at [`context/HANDOFF.md`](context/HANDOFF.md) and
[`context/TODOS.md`](context/TODOS.md)** — they say exactly what's done,
what's blocked, and what to do first.

## Docs

| File | Covers |
|---|---|
| [`docs/architecture-overview.md`](docs/architecture-overview.md) | System map — start here |
| [`docs/decision-pomdp.md`](docs/decision-pomdp.md) | The frozen decision space every layer implements |
| [`docs/efe-control-node-design.md`](docs/efe-control-node-design.md) | The EFE engine itself |
| [`docs/langgraph-integration.md`](docs/langgraph-integration.md) | The LangGraph demo/orchestration layer |
| [`docs/baselines-design.md`](docs/baselines-design.md) | The 4 non-EFE controllers, and why each exists |
| [`docs/tau2-bench-integration.md`](docs/tau2-bench-integration.md) | The real benchmark wiring |
| [`docs/observation-derivation.md`](docs/observation-derivation.md) | How the 4 observation modalities get computed, across all 3 environments |
| [`docs/experiment-pipeline.md`](docs/experiment-pipeline.md) | The research stages, as a diagram, with status |
| [`docs/testing-and-pipeline.md`](docs/testing-and-pipeline.md) | How this is verified, and what "verified" doesn't cover |
| [`docs/known-issues-and-gotchas.md`](docs/known-issues-and-gotchas.md) | Every real bug caught, consolidated, with a lesson for each |
| [`docs/stage0.5-kill-test-results.md`](docs/stage0.5-kill-test-results.md) / [`docs/stage2-baselines-results.md`](docs/stage2-baselines-results.md) | Actual experiment results |

Plans: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) (thesis narrative) ·
[`ROADMAP.md`](ROADMAP.md) (what happens next, and the gates that could
stop it) · [`context/TODOS.md`](context/TODOS.md) (short-horizon
checklist).

## Quick verification

Two commands cover everything reproducible offline:

```bash
.venv/bin/pytest -q                              # 135 tests, seconds
.venv/bin/python scripts/run_full_pipeline.py     # every offline stage, end to end
```

`run_full_pipeline.py --quick` skips the slow evaluations; `--only
stage1 stage2-part2` runs a subset. It writes `results/pipeline_run.json`
with per-stage pass/fail and the headline numbers.

The real-LLM demo (`graph.py --llm`) and the tau2-bench integration need
`.env` set and open network access — see `context/TODOS.md`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Note: install `inferactively-pymdp`, not the plain `pymdp` PyPI package
(name collision with an unrelated project) — see `pyproject.toml`.
