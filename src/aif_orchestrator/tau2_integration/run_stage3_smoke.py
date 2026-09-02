"""Cheap end-to-end smoke test for EFEAgent against tau2-bench's `mock`
domain (the domain that exists specifically for exactly this purpose --
see src/tau2/agent/README.md's "Understanding the Environment"). Not
the real Stage 3 evaluation sweep (RESEARCH_PLAN.md Stage 3) -- this
just proves the wiring works before spending real budget on
retail/airline/telecom.

Run: .venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_smoke
"""
import os

from dotenv import load_dotenv

load_dotenv()
# tau2-bench (LiteLLM) reads OPENROUTER_API_KEY; our own .env uses
# LLM_API_KEY (llm_agent.py's own naming) -- map one to the other rather
# than keeping two separate secrets for the same key.
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))

from . import register  # noqa: E402 -- registers "efe_agent" before any tau2.run call

register()

from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.run import get_tasks, run_single_task  # noqa: E402


def main():
    model = f"openrouter/{os.environ['LLM_MODEL']}"
    config = TextRunConfig(
        domain="mock",
        agent="efe_agent",
        user="user_simulator",
        llm_agent=model,
        llm_args_agent={},
        llm_user=model,
        llm_args_user={},
        max_steps=20,
    )
    tasks = get_tasks("mock", num_tasks=1)
    print(f"Running task {tasks[0].id} with efe_agent ({model})...")
    sim = run_single_task(config, tasks[0], seed=0)

    print(f"\nTermination: {sim.termination_reason}")
    print(f"Reward: {sim.reward_info.reward if sim.reward_info else 'N/A'}")
    print(f"Messages: {len(sim.messages)}")
    print(f"Agent cost: ${sim.agent_cost or 0:.4f}  User cost: ${sim.user_cost or 0:.4f}")


if __name__ == "__main__":
    main()
