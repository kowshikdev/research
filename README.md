# research

Active Inference / Expected-Free-Energy control for LLM multi-agent
orchestration. See [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) for the full
thesis plan, resource inventory, and stage-by-stage timeline.

## Status

Stage 0 (setup): decision-POMDP schema frozen in
[`docs/decision-pomdp.md`](docs/decision-pomdp.md); pymdp engine sanity
check passing.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Note: install `inferactively-pymdp`, not the plain `pymdp` PyPI package
(name collision with an unrelated project) — see `pyproject.toml`.

## Sanity check

```bash
.venv/bin/python scripts/tmaze_sanity_check.py
```

Confirms the active-inference engine correctly favors an
information-seeking action when outcome-relevant state is uncertain
(the same epistemic-value mechanism the real EFE control node needs).
Results are written to `results/stage0_tmaze_sanity_check.json`.
