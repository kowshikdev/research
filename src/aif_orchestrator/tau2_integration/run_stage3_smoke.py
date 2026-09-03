"""Cheap end-to-end smoke test for all five control-node agents against
tau2-bench's `mock` domain (the domain that exists specifically for
exactly this purpose -- see src/tau2/agent/README.md's "Understanding
the Environment"). Not the real Stage 3 evaluation sweep
(RESEARCH_PLAN.md Stage 3, run_stage3_eval.py) -- this just proves the
wiring works for every agent before spending real budget on
retail/airline/telecom.

Run: .venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_smoke
"""
import os

from dotenv import load_dotenv

load_dotenv()

from . import register  # noqa: E402 -- registers agents before any tau2.run call
from .vertex_auth import start_token_refresher  # noqa: E402

register()

from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.run import get_tasks, run_single_task  # noqa: E402

AGENTS = ["efe_agent", "heuristic_agent", "router_agent", "voi_agent", "react_agent"]


def main():
    model = "google/gemini-2.5-flash"
    vertex_kwargs = dict(start_token_refresher())
    tasks = get_tasks("mock", num_tasks=1)
    task = tasks[0]

    for agent in AGENTS:
        config = TextRunConfig(
            domain="mock", agent=agent, user="user_simulator",
            llm_agent=model, llm_args_agent=dict(vertex_kwargs),
            # cap the user simulator's own call -- see run_stage3_eval.py
            llm_user=model, llm_args_user={"max_tokens": 1000, **vertex_kwargs},
            max_steps=20,
        )
        print(f"=== {agent} on task {task.id} ===")
        sim = run_single_task(config, task, seed=0)
        reward = sim.reward_info.reward if sim.reward_info else "N/A"
        print(f"  termination={sim.termination_reason}  reward={reward}  messages={len(sim.messages or [])}")


if __name__ == "__main__":
    main()
