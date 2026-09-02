"""Stage 2 baseline comparison: heuristic / learned_router / VOI / ReAct
vs. EFE, all four sharing the same interface and the same observation
model, evaluated on the same free environment (graph.mock_agent_step)
used to train the router -- see RESEARCH_PLAN.md Stage 2 and
context/TODOS.md. This is the statistical-comparison half of the Stage 2
deliverable; the other half (all controllers running inside the actual
LangGraph scaffold, incl. against the real LLM agent) is
`graph.build_graph(control_step=make_control_step(<cls>))`, exercised in
`graph.run_stage2_demo()`.

mock_agent_step has no hidden "true state" the way kill_test's env did
(docs/decision-pomdp.md's task_state is EFE's own belief, not something
the mock scripts against) -- the only ground truth mock_agent_step
exposes is its task_id convention: any id starting with "forced-bad"
deterministically produces a bad trajectory (repeated tool errors), any
other id stochastically improves over a few turns. Escalation-quality
metrics are keyed off that split, the direct analogue of kill-test's
correct-vs-unnecessary-escalation-rate split against a true hidden state.

Run: .venv/bin/python -m aif_orchestrator.baselines.run_stage2_eval
"""
import json
import random
import time
from pathlib import Path

from ..efe_controller import EFEControlNode, Observation
from ..graph import mock_agent_step
from .belief import step_reward
from .heuristic import HeuristicControlNode
from .react import ReActControlNode
from .router import LearnedRouterControlNode
from .voi import VOIControlNode

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"
NUM_EVAL_EPISODES = 3000
EVAL_SEED_START = 500_000  # disjoint from the router's training seed range (1000+)
MAX_TURNS = 5
FORCED_BAD_FRAC = 0.3


def run_episode(controller, task_id, max_turns=MAX_TURNS):
    """Mirrors graph.route_from_decision's actual termination rule: only
    `continue`/`escalate_to_human` are genuine terminal choices; running
    out of turns while still on an info-seeking policy is a distinct
    `forced_stop` outcome (graph.py's own "forced stop -- real system
    would escalate here too" branch), not the same thing as the
    controller having deliberately picked escalate_to_human."""
    controller.reset()
    forced_bad = task_id.startswith("forced-bad")
    state = {"task_id": task_id}
    total_reward = 0.0
    outcome = "forced_stop"
    for turn in range(max_turns):
        state["turn"] = turn
        observation = Observation(**mock_agent_step(state)["observation"])
        decision = controller.decide(observation)
        is_terminal = decision.policy in ("continue", "escalate_to_human") or turn == max_turns - 1
        total_reward += step_reward(observation, is_terminal, policy=decision.policy, forced_bad=forced_bad)
        steps = turn + 1
        if decision.policy in ("continue", "escalate_to_human"):
            outcome = decision.policy
            break
        if turn == max_turns - 1:
            break
        state["last_policy"] = decision.policy
    return {
        "task_id": task_id,
        "forced_bad": forced_bad,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def evaluate(controller, num_episodes, seed_start):
    rng = random.Random(seed_start)
    episodes = []
    for i in range(num_episodes):
        eid = seed_start + i
        task_id = f"forced-bad-eval-{eid}" if rng.random() < FORCED_BAD_FRAC else f"eval-{eid}"
        episodes.append(run_episode(controller, task_id))

    n = len(episodes)
    rewards = [e["total_reward"] for e in episodes]
    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / (n - 1)
    std = variance ** 0.5
    ci95 = 1.96 * std / (n ** 0.5)

    bad = [e for e in episodes if e["forced_bad"]]
    normal = [e for e in episodes if not e["forced_bad"]]
    escalate_on_bad = sum(1 for e in bad if e["outcome"] == "escalate_to_human")
    escalate_on_normal = sum(1 for e in normal if e["outcome"] == "escalate_to_human")
    continue_on_normal = sum(1 for e in normal if e["outcome"] == "continue")
    forced_stop_rate = sum(1 for e in episodes if e["outcome"] == "forced_stop") / n

    return {
        "n_episodes": n,
        "avg_reward": mean_reward,
        "reward_ci95_halfwidth": ci95,
        "n_forced_bad": len(bad),
        "correct_escalation_rate": escalate_on_bad / len(bad) if bad else None,
        "unnecessary_escalation_rate": escalate_on_normal / len(normal) if normal else None,
        "normal_resolved_rate": continue_on_normal / len(normal) if normal else None,
        "forced_stop_rate": forced_stop_rate,
        "avg_steps": sum(e["steps"] for e in episodes) / n,
    }


def main():
    results = {}

    print("Training learned router (10000 episodes, tabular Q-learning, belief-state features)...")
    t0 = time.time()
    router = LearnedRouterControlNode()
    print(f"  done in {time.time() - t0:.1f}s, {len(LearnedRouterControlNode._q)} (belief_bucket, obs) states learned")

    controllers = {
        "heuristic": HeuristicControlNode(),
        "learned_router": router,
        "voi_decision_theoretic": VOIControlNode(),
        "react": ReActControlNode(),
        "efe_active_inference": EFEControlNode(),
    }

    for name, controller in controllers.items():
        print(f"Evaluating {name} on {NUM_EVAL_EPISODES} held-out episodes...")
        t0 = time.time()
        metrics = evaluate(controller, NUM_EVAL_EPISODES, EVAL_SEED_START)
        metrics["wall_time_sec"] = time.time() - t0
        results[name] = metrics
        print(f"  avg_reward={metrics['avg_reward']:.4f} (95% CI +/-{metrics['reward_ci95_halfwidth']:.4f})  "
              f"correct_escalation={metrics['correct_escalation_rate']:.3f}  "
              f"unnecessary_escalation={metrics['unnecessary_escalation_rate']:.3f}  "
              f"normal_resolved={metrics['normal_resolved_rate']:.3f}  "
              f"forced_stop={metrics['forced_stop_rate']:.3f}  "
              f"avg_steps={metrics['avg_steps']:.2f}  "
              f"({metrics['wall_time_sec']:.1f}s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "stage2_baselines_results.json"
    out_path.write_text(json.dumps({"num_eval_episodes": NUM_EVAL_EPISODES, "results": results}, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
