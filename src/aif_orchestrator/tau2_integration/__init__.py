"""Stage 3: tau2-bench integration (RESEARCH_PLAN.md Stage 3,
context/TODOS.md). Talks to external/tau2-bench (pinned commit a2c0247,
tau2==1.0.1) via its public Python API (`tau2.run.run_domain` /
`run_single_task`) rather than the CLI, and registers `EFEAgent` as a
community agent factory (`register()`) instead of editing the vendored
repo -- see `efe_agent.py`.
"""
from .register import register

__all__ = ["register"]
