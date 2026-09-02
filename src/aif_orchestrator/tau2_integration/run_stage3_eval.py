"""Stage 3 primary evaluation (RESEARCH_PLAN.md Stage 3): all five
control-node agents (EFE + the four Stage 2 baselines) against
tau2-bench's real domains (retail/airline/telecom, "base" task split --
the standard evaluation split, per the upstream README: "If you are
evaluating an agent (not training), use the base task split").

Run one domain at a time by default (`--domain retail`), not all three
at once -- this is the real cost center RESEARCH_PLAN.md's Stage 3
section warns about (278 tasks x 5 agents = ~1390 simulations across
all three domains), so checking one domain's results before committing
to the rest is the responsible default. Pass `--domain all` to run all
three in one process.

Run: .venv/Scripts/python -m aif_orchestrator.tau2_integration.run_stage3_eval --domain retail
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# tau2-bench (LiteLLM) reads OPENROUTER_API_KEY; our own .env uses
# LLM_API_KEY (llm_agent.py's own naming).
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))

from . import register  # noqa: E402

register()

from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.run import run_domain  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "stage3_tau2"
DOMAINS = ["retail", "airline", "telecom"]
AGENTS = ["efe_agent", "heuristic_agent", "router_agent", "voi_agent", "react_agent"]


def run_one(domain: str, agent: str, model: str, num_trials: int, max_concurrency: int) -> dict:
    save_to = RESULTS_DIR / f"{domain}__{agent}.json"
    config = TextRunConfig(
        domain=domain, agent=agent, user="user_simulator",
        llm_agent=model, llm_args_agent={}, llm_user=model, llm_args_user={},
        task_split_name="base", num_trials=num_trials,
        max_steps=30, max_concurrency=max_concurrency,
        save_to=str(save_to),
    )
    print(f"=== {domain} / {agent} ({num_trials} trial(s), max_concurrency={max_concurrency}) ===")
    t0 = time.time()
    results = run_domain(config)
    elapsed = time.time() - t0

    rewards = [s.reward_info.reward for s in results.simulations if s.reward_info]
    escalations = sum(
        1 for s in results.simulations
        for m in (s.messages or [])
        if getattr(m, "tool_calls", None) and any(tc.name == "transfer_to_human_agents" for tc in m.tool_calls)
    )
    n = len(rewards)
    avg_reward = sum(rewards) / n if n else None
    metrics = {
        "n_simulations": n,
        "avg_reward": avg_reward,
        "pass_1": sum(1 for r in rewards if r == 1.0) / n if n else None,
        "escalations": escalations,
        "elapsed_sec": elapsed,
    }
    print(f"  n={n}  avg_reward={avg_reward}  pass^1={metrics['pass_1']}  "
          f"escalations={escalations}  ({elapsed:.1f}s)")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="retail")
    parser.add_argument("--agents", nargs="+", default=AGENTS, choices=AGENTS)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=8)
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]
    model = f"openrouter/{os.environ['LLM_MODEL']}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS_DIR / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for domain in domains:
        for agent in args.agents:
            key = f"{domain}/{agent}"
            summary[key] = run_one(domain, agent, model, args.num_trials, args.max_concurrency)
            summary_path.write_text(json.dumps(summary, indent=2))  # incremental save

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
