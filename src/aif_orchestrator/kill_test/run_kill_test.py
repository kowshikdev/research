"""Stage 0.5 kill-test: does EFE produce measurably different/better
act/gather/escalate decisions than a heuristic, a learned router, or an
explicit Bayesian value-of-information controller, given the same noisy
observations? See RESEARCH_PLAN.md Stage 0.5.

Run: .venv/bin/python -m aif_orchestrator.kill_test.run_kill_test
"""
import json
import time
from pathlib import Path

from .env import TinyEscalationEnv, STATES
from .controllers import (
    RandomController, HeuristicController, LearnedRouterController,
    VOIController, EFEController,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"
NUM_EVAL_EPISODES = 3000
EVAL_SEED_START = 500_000  # disjoint from the router's training seed range


def run_episode(controller, env):
    controller.start_episode()
    obs = env.reset()
    true_state = env.state
    steps = 0
    total_reward = 0.0
    while True:
        valid = env.valid_actions()
        action = controller.decide(obs, env.gathers_used, valid)
        result = env.step(action)
        total_reward += result.reward
        steps += 1
        if result.done:
            return {
                "true_state": true_state,
                "outcome": result.outcome,
                "gathers_used": result.gathers_used,
                "steps": steps,
                "total_reward": total_reward,
            }
        obs = result.obs_confidence


def evaluate(controller, num_episodes, seed_start):
    episodes = [run_episode(controller, TinyEscalationEnv(seed=seed_start + i))
                for i in range(num_episodes)]

    n = len(episodes)
    rewards = [e["total_reward"] for e in episodes]
    mean_reward = sum(rewards) / n
    variance = sum((r - mean_reward) ** 2 for r in rewards) / (n - 1)
    std = variance ** 0.5
    se = std / (n ** 0.5)
    ci95 = 1.96 * se
    success = sum(1 for e in episodes if e["outcome"] == "success")
    failure = sum(1 for e in episodes if e["outcome"] == "failure")
    escalate_correct = sum(1 for e in episodes if e["outcome"] == "escalate_correct")
    escalate_wrong = sum(1 for e in episodes if e["outcome"] == "escalate_wrong")

    # "Unnecessary intervention": escalated when the task didn't need a human.
    # "Missed intervention" / silent failure: acted now on a task that
    # actually needed a human, and it failed.
    silent_failure_needs_human = sum(
        1 for e in episodes if e["outcome"] == "failure" and e["true_state"] == "needs_human"
    )

    return {
        "n_episodes": n,
        "success_rate": success / n,
        "silent_failure_rate": failure / n,
        "correct_escalation_rate": escalate_correct / n,
        "unnecessary_escalation_rate": escalate_wrong / n,
        "silent_failure_on_needs_human_rate": silent_failure_needs_human / n,
        "avg_gathers": sum(e["gathers_used"] for e in episodes) / n,
        "avg_steps": sum(e["steps"] for e in episodes) / n,
        "avg_reward": mean_reward,
        "reward_std": std,
        "reward_ci95_halfwidth": ci95,
        "reward_ci95_low": mean_reward - ci95,
        "reward_ci95_high": mean_reward + ci95,
    }


def main():
    results = {}

    print("Training learned router (8000 episodes, tabular Q-learning)...")
    router = LearnedRouterController(seed=7)
    t0 = time.time()
    router.train(TinyEscalationEnv, num_episodes=8000, seed=1_000_000)
    print(f"  done in {time.time() - t0:.1f}s, {len(router.q)} (confidence, gathers) states learned")

    controllers = {
        "random": RandomController(seed=1),
        "heuristic": HeuristicController(),
        "learned_router": router,
        "voi_decision_theoretic": VOIController(),
        "efe_active_inference": EFEController(),
    }

    for name, controller in controllers.items():
        print(f"Evaluating {name} on {NUM_EVAL_EPISODES} held-out episodes...")
        t0 = time.time()
        metrics = evaluate(controller, NUM_EVAL_EPISODES, EVAL_SEED_START)
        metrics["wall_time_sec"] = time.time() - t0
        results[name] = metrics
        print(f"  avg_reward={metrics['avg_reward']:.4f} (95% CI ±{metrics['reward_ci95_halfwidth']:.4f})  "
              f"success={metrics['success_rate']:.3f}  "
              f"unnecessary_escalation={metrics['unnecessary_escalation_rate']:.3f}  "
              f"silent_failure={metrics['silent_failure_rate']:.3f}  "
              f"avg_gathers={metrics['avg_gathers']:.2f}  "
              f"({metrics['wall_time_sec']:.1f}s)")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "stage0_5_kill_test_results.json"
    out_path.write_text(json.dumps({
        "num_eval_episodes": NUM_EVAL_EPISODES,
        "state_prior": dict(zip(STATES, [0.5, 0.35, 0.15])),
        "results": results,
    }, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
