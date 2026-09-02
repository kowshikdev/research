"""Model-matched Stage 2 comparison -- the genuine repeat of the Stage
0.5 kill-test's Bayes-optimality methodology at the real decision-POMDP's
scale, closing the gap `docs/stage2-baselines-results.md` flags in the
original mock_agent_step-based Stage 2 run: every controller here is
evaluated (and the router trained) against `model_env.ModelMatchedEnv`,
which samples from `efe_controller.py`'s own D_PRIOR/OBS_DISTS/
B_TRANSITIONS -- the same generative model EFE and VOI both assume, not
a hand-scripted stand-in. Reward/ground-truth is identical in structure
to `run_stage2_eval.py` (`belief.step_reward`, keyed off whether the
env's true state was genuinely unresolvable) -- only the environment
generating observations and true state changed.

Run: .venv/Scripts/python -m aif_orchestrator.baselines.run_model_matched_eval
"""
import json
import time
from pathlib import Path

from ..efe_controller import EFEControlNode
from .belief import step_reward
from .heuristic import HeuristicControlNode
from .model_env import ModelMatchedEnv
from .react import ReActControlNode
from .router import LearnedRouterControlNode, model_matched_source_factory
from .voi import VOIControlNode

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"
NUM_EVAL_EPISODES = 3000
EVAL_SEED_START = 500_000  # disjoint from the router's training seed range
MAX_TURNS = 5
UNRESOLVABLE_STATES = ("needs_human", "likely_to_fail")


def run_episode(controller, seed, max_turns=MAX_TURNS):
    controller.reset()
    env = ModelMatchedEnv(seed=seed)
    observation = env.reset()
    total_reward = 0.0
    outcome = "forced_stop"
    steps = max_turns
    for turn in range(max_turns):
        forced_bad = env.state in UNRESOLVABLE_STATES
        decision = controller.decide(observation)
        is_terminal = decision.policy in ("continue", "escalate_to_human") or turn == max_turns - 1
        total_reward += step_reward(observation, is_terminal, policy=decision.policy, forced_bad=forced_bad)
        if decision.policy in ("continue", "escalate_to_human"):
            outcome = decision.policy
            steps = turn + 1
            break
        if turn == max_turns - 1:
            break
        observation = env.step(decision.policy)
    return {
        "final_true_state": env.state,
        "unresolvable": env.state in UNRESOLVABLE_STATES,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }


def evaluate(controller, num_episodes, seed_start):
    episodes = [run_episode(controller, seed_start + i) for i in range(num_episodes)]

    n = len(episodes)
    rewards = [e["total_reward"] for e in episodes]
    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / (n - 1)
    std = variance ** 0.5
    ci95 = 1.96 * std / (n ** 0.5)

    # Ground truth here is the env's *initial* true state's resolvability
    # class isn't tracked separately from its (possibly transitioned)
    # final state -- episodes ending via continue/escalate score against
    # whatever state produced the observation that triggered that choice
    # (run_episode's forced_bad at decision time), so bucket by the
    # final recorded true state as a proxy for episode difficulty.
    unresolvable = [e for e in episodes if e["unresolvable"]]
    resolvable = [e for e in episodes if not e["unresolvable"]]
    escalate_on_unresolvable = sum(1 for e in unresolvable if e["outcome"] == "escalate_to_human")
    escalate_on_resolvable = sum(1 for e in resolvable if e["outcome"] == "escalate_to_human")
    continue_on_resolvable = sum(1 for e in resolvable if e["outcome"] == "continue")
    forced_stop_rate = sum(1 for e in episodes if e["outcome"] == "forced_stop") / n

    return {
        "n_episodes": n,
        "avg_reward": mean_reward,
        "reward_ci95_halfwidth": ci95,
        "n_unresolvable_final_state": len(unresolvable),
        "correct_escalation_rate": escalate_on_unresolvable / len(unresolvable) if unresolvable else None,
        "unnecessary_escalation_rate": escalate_on_resolvable / len(resolvable) if resolvable else None,
        "resolvable_resolved_rate": continue_on_resolvable / len(resolvable) if resolvable else None,
        "forced_stop_rate": forced_stop_rate,
        "avg_steps": sum(e["steps"] for e in episodes) / n,
    }


def main():
    results = {}

    # ModelMatchedEnv's observation space is far richer than
    # mock_agent_step's handful of scripted combos (144 joint observation
    # bins x 8 belief buckets = up to ~1150 (belief, obs) keys, vs. ~8 for
    # the mock-based Stage 2 run) -- 10000 episodes (the mock-tuned
    # default) undertrains it badly; 100000 gets solid state coverage.
    print("Training learned router against ModelMatchedEnv (100000 episodes)...")
    t0 = time.time()
    LearnedRouterControlNode.train(
        source_factory=model_matched_source_factory,
        num_episodes=100_000, epsilon_start=0.4, epsilon_end=0.02,
    )
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
        print(f"Evaluating {name} on {NUM_EVAL_EPISODES} held-out episodes (model-matched env)...")
        t0 = time.time()
        metrics = evaluate(controller, NUM_EVAL_EPISODES, EVAL_SEED_START)
        metrics["wall_time_sec"] = time.time() - t0
        results[name] = metrics
        print(f"  avg_reward={metrics['avg_reward']:.4f} (95% CI +/-{metrics['reward_ci95_halfwidth']:.4f})  "
              f"correct_escalation={metrics['correct_escalation_rate']:.3f}  "
              f"unnecessary_escalation={metrics['unnecessary_escalation_rate']:.3f}  "
              f"resolvable_resolved={metrics['resolvable_resolved_rate']:.3f}  "
              f"forced_stop={metrics['forced_stop_rate']:.3f}  "
              f"avg_steps={metrics['avg_steps']:.2f}  "
              f"({metrics['wall_time_sec']:.1f}s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "model_matched_baselines_results.json"
    out_path.write_text(json.dumps({"num_eval_episodes": NUM_EVAL_EPISODES, "results": results}, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
