"""Learned-router baseline (RESEARCH_PLAN.md §3.6.2), ported from
kill_test.controllers.LearnedRouterController's tabular Q-learning to
the real decision-POMDP, with the belief-state-parity fix the Stage 0.5
kill-test flagged as necessary (docs/stage0.5-kill-test-results.md,
context/TODOS.md): state features are (a coarse bucket of a running
belief summary, the latest raw observation), not just the latest
observation alone. Without this, any gap to EFE/VOI is arguably a
feature-engineering artifact rather than a finding about control
mechanisms -- see RESEARCH_PLAN.md §5.

Trains by default against `graph.mock_agent_step` (not through the full
LangGraph app) -- a free, deterministic-per-seed stand-in that reacts to
whatever `last_policy` was chosen, regardless of which controller
produced it. `train()`'s `source_factory` is pluggable, though: pass
`model_matched_source_factory` (below) to train against
`model_env.ModelMatchedEnv` instead -- the same generative model
EFE/VOI assume, for the model-matched Stage 2 comparison
(`run_model_matched_eval.py`) that closes the gap
`docs/stage2-baselines-results.md` flags in the mock-based one. Reward
is `belief.step_reward`: the real C-weighted observation payoff only on
a terminal turn (continue/escalate chosen, or turns exhausted), a flat
cost otherwise -- see belief.py's docstring for why paying the full
observation reward on every turn (an earlier version of this) taught
the router to never choose `continue` at all.
"""
import random

from .. import efe_controller as efe
from ..graph import mock_agent_step
from .belief import bayes_update, step_reward, uniform_decision
from .model_env import ModelMatchedEnv

TERMINAL_POLICIES = ("continue", "escalate_to_human")


def _obs_key(observation):
    return tuple(observation.as_indices())


def _belief_bucket(belief):
    best = max(range(len(belief)), key=lambda i: belief[i])
    confident = 1 if belief[best] >= 0.6 else 0
    return (best, confident)


class _MockAgentEpisodeSource:
    """Adapter so training can drive graph.mock_agent_step through the
    same reset()/step(policy)/forced_bad interface as ModelMatchedEnv."""

    def __init__(self, rng, ep):
        forced_bad = rng.random() < 0.3
        # ~30% forced-bad trajectories so the router actually sees enough
        # escalate-worthy states to learn from (mock_agent_step
        # deterministically returns error/low-confidence observations for
        # any task_id starting with "forced-bad").
        self.forced_bad = forced_bad
        self._state = {"task_id": f"forced-bad-train-{ep}" if forced_bad else f"train-{ep}"}
        self._turn = 0

    def reset(self):
        self._state["turn"] = self._turn
        return efe.Observation(**mock_agent_step(self._state)["observation"])

    def step(self, policy):
        self._state["last_policy"] = policy
        self._turn += 1
        self._state["turn"] = self._turn
        return efe.Observation(**mock_agent_step(self._state)["observation"])


class _ModelMatchedEpisodeSource:
    """Adapter over ModelMatchedEnv exposing the same interface -- ground
    truth (`forced_bad`) comes from the env's actual sampled true state,
    not a task-id convention."""

    def __init__(self, rng, ep):
        self._env = ModelMatchedEnv(seed=rng.randint(0, 2**31 - 1))

    @property
    def forced_bad(self):
        return self._env.state in ("needs_human", "likely_to_fail")

    def reset(self):
        return self._env.reset()

    def step(self, policy):
        return self._env.step(policy)


def model_matched_source_factory(rng, ep):
    return _ModelMatchedEpisodeSource(rng, ep)


class LearnedRouterControlNode:
    name = "learned_router"

    _q = None  # trained once, shared across instances/episodes (class-level cache)

    def __init__(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)
        if LearnedRouterControlNode._q is None:
            LearnedRouterControlNode.train()

    def reset(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    @classmethod
    def _get_q(cls, key):
        return cls._q.setdefault(key, {p: 0.0 for p in efe.POLICIES})

    @classmethod
    def train(cls, num_episodes=10000, max_turns=5, alpha=0.1, gamma=0.9,
              epsilon_start=0.3, epsilon_end=0.02, seed=1000, source_factory=_MockAgentEpisodeSource):
        cls._q = {}
        rng = random.Random(seed)
        for ep in range(num_episodes):
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / num_episodes)
            source = source_factory(rng, ep)
            observation = source.reset()
            belief = list(efe.D_PRIOR)
            prev_key = prev_policy = prev_reward = None

            for turn in range(max_turns):
                belief = bayes_update(belief, observation)
                key = (_belief_bucket(belief), _obs_key(observation))
                qvals = cls._get_q(key)

                if rng.random() < epsilon:
                    policy = rng.choice(efe.POLICIES)
                else:
                    policy = max(qvals, key=qvals.get)
                is_terminal = policy in TERMINAL_POLICIES or turn == max_turns - 1
                # step_reward: real payoff only on a terminal turn (else a
                # flat cost), plus a ground-truth-keyed outcome
                # adjustment for continue/escalate so the router actually
                # has a reason to prefer escalating on forced-bad
                # trajectories -- see belief.py's docstring for why both
                # pieces are needed (an earlier version without either
                # never learned to choose `continue`, and separately
                # never learned to escalate specifically).
                reward = step_reward(observation, is_terminal, policy=policy, forced_bad=source.forced_bad)

                if prev_key is not None:
                    target = prev_reward + gamma * max(qvals.values())
                    prev_q = cls._get_q(prev_key)
                    prev_q[prev_policy] += alpha * (target - prev_q[prev_policy])

                if is_terminal:
                    q = cls._get_q(key)
                    q[policy] += alpha * (reward - q[policy])
                    break

                observation = source.step(policy)
                prev_key, prev_policy, prev_reward = key, policy, reward

    def decide(self, observation, valid_policies=None) -> efe.Decision:
        valid_policies = valid_policies or list(efe.POLICIES)
        self.belief = bayes_update(self.belief, observation)
        key = (_belief_bucket(self.belief), _obs_key(observation))
        qvals = self._get_q(key)
        valid_q = {p: qvals[p] for p in valid_policies}
        chosen = max(valid_q, key=valid_q.get)

        decision = uniform_decision(chosen, self.belief)
        decision.pragmatic_value = dict(qvals)
        return decision
