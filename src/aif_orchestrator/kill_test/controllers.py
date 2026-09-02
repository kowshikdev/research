"""Four controllers for the Stage 0.5 kill-test, all deciding over the
same three actions from the same observation each step:

  - HeuristicController: fixed confidence thresholds (production-style
    `if confidence < tau: escalate()`).
  - LearnedRouterController: tabular Q-learning trained from experience
    on the same (confidence, gathers_used) features -- the practical
    "train a classifier/bandit on the same observations" baseline.
  - VOIController: explicit Bayesian belief update + hand-derived
    expected-utility computation (exact value-of-information), using the
    *true* generative model. This is the strong baseline: if EFE can't
    match this, the active-inference formalism isn't earning its keep.
  - EFEController: pymdp Expected Free Energy over the same true
    generative model, re-planned fresh from the current belief at every
    decision (mirrors VOIController's one-step-lookahead structure so
    the comparison is fair).

See RESEARCH_PLAN.md Stage 0.5 and docs/decision-pomdp.md.
"""
import itertools
import random

import numpy as np

from .env import (
    ACTIONS, CONF_BINS, CONF_DIST, STATES, STATE_PRIOR,
    P_SUCCESS, GATHER_RESOLVE_PROB, MAX_GATHERS,
    R_SUCCESS, R_FAILURE, R_ESCALATE_CORRECT, R_ESCALATE_WRONG, R_GATHER_COST,
)


class RandomController:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def start_episode(self):
        pass

    def decide(self, obs_confidence, gathers_used, valid_actions):
        return self.rng.choice(valid_actions)


class HeuristicController:
    """Production-style fixed threshold: act on high confidence, escalate
    on low confidence, gather on the ambiguous middle band."""

    name = "heuristic"

    def start_episode(self):
        pass

    def decide(self, obs_confidence, gathers_used, valid_actions):
        if obs_confidence == "high" and "act_now" in valid_actions:
            return "act_now"
        if obs_confidence == "low" and "escalate" in valid_actions:
            return "escalate"
        if "gather_info" in valid_actions:
            return "gather_info"
        # gathers exhausted and still ambiguous -- fall back to acting
        return "act_now" if "act_now" in valid_actions else "escalate"


class LearnedRouterController:
    """Tabular Q-learning over (confidence_bin, gathers_used) -> action,
    trained purely from environment reward feedback (not supervised on
    an oracle policy) -- the fair "learned from the same observations"
    comparison point."""

    name = "learned_router"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.q = {}

    def _key(self, conf_idx, gathers_used):
        return (conf_idx, gathers_used)

    def _get_q(self, key):
        return self.q.setdefault(key, [0.0, 0.0, 0.0])

    def train(self, env_cls, num_episodes=8000, alpha=0.1, gamma=0.95,
              epsilon_start=0.3, epsilon_end=0.02, seed=1000):
        env = env_cls(seed=seed)
        for ep in range(num_episodes):
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / num_episodes)
            obs = env.reset()
            done = False
            while not done:
                conf_idx = CONF_BINS.index(obs)
                gathers = env.gathers_used
                valid = env.valid_actions()
                valid_idx = [ACTIONS.index(a) for a in valid]
                qvals = self._get_q(self._key(conf_idx, gathers))
                if self.rng.random() < epsilon:
                    a_idx = self.rng.choice(valid_idx)
                else:
                    a_idx = max(valid_idx, key=lambda i: qvals[i])
                result = env.step(ACTIONS[a_idx])
                if not result.done:
                    next_conf_idx = CONF_BINS.index(result.obs_confidence)
                    next_valid_idx = [ACTIONS.index(a) for a in env.valid_actions()]
                    next_q = self._get_q(self._key(next_conf_idx, result.gathers_used))
                    target = result.reward + gamma * max(next_q[i] for i in next_valid_idx)
                else:
                    target = result.reward
                qvals[a_idx] += alpha * (target - qvals[a_idx])
                obs = result.obs_confidence
                done = result.done

    def start_episode(self):
        pass

    def decide(self, obs_confidence, gathers_used, valid_actions):
        conf_idx = CONF_BINS.index(obs_confidence)
        qvals = self._get_q(self._key(conf_idx, gathers_used))
        valid_idx = [ACTIONS.index(a) for a in valid_actions]
        best = max(valid_idx, key=lambda i: qvals[i])
        return ACTIONS[best]


class VOIController:
    """Exact Bayesian decision theory over the true generative model:
    maintain a belief over {solvable, needs_info, needs_human}, and at
    each step compute the expected utility of act_now / escalate now
    versus gather_info (one-step value-of-information lookahead: the
    expected best terminal payoff after gathering, minus its cost)."""

    name = "voi_decision_theoretic"

    def __init__(self):
        self.belief = None

    def start_episode(self):
        self.belief = list(STATE_PRIOR)

    def _bayes_update(self, belief, obs_confidence):
        conf_idx = CONF_BINS.index(obs_confidence)
        unnorm = [belief[i] * CONF_DIST[STATES[i]][conf_idx] for i in range(3)]
        z = sum(unnorm)
        return [u / z for u in unnorm] if z > 0 else list(belief)

    def _eu_act_now(self, belief):
        return sum(
            belief[i] * (P_SUCCESS[STATES[i]] * R_SUCCESS + (1 - P_SUCCESS[STATES[i]]) * R_FAILURE)
            for i in range(3)
        )

    def _eu_escalate(self, belief):
        b_human = belief[STATES.index("needs_human")]
        return b_human * R_ESCALATE_CORRECT + (1 - b_human) * R_ESCALATE_WRONG

    def _eu_gather(self, belief):
        # Transition: needs_info -> solvable w.p. GATHER_RESOLVE_PROB
        i_info, i_solv, i_human = STATES.index("needs_info"), STATES.index("solvable"), STATES.index("needs_human")
        post_transition = [0.0, 0.0, 0.0]
        post_transition[i_solv] += belief[i_solv] + belief[i_info] * GATHER_RESOLVE_PROB
        post_transition[i_info] += belief[i_info] * (1 - GATHER_RESOLVE_PROB)
        post_transition[i_human] += belief[i_human]

        expected_value = 0.0
        for conf_idx, conf_bin in enumerate(CONF_BINS):
            p_obs = sum(post_transition[i] * CONF_DIST[STATES[i]][conf_idx] for i in range(3))
            if p_obs <= 0:
                continue
            updated = self._bayes_update(post_transition, conf_bin)
            best_next = max(self._eu_act_now(updated), self._eu_escalate(updated))
            expected_value += p_obs * best_next
        return R_GATHER_COST + expected_value

    def decide(self, obs_confidence, gathers_used, valid_actions):
        self.belief = self._bayes_update(self.belief, obs_confidence)
        scores = {}
        if "act_now" in valid_actions:
            scores["act_now"] = self._eu_act_now(self.belief)
        if "escalate" in valid_actions:
            scores["escalate"] = self._eu_escalate(self.belief)
        if "gather_info" in valid_actions:
            scores["gather_info"] = self._eu_gather(self.belief)
        return max(scores, key=scores.get)


class EFEController:
    """pymdp Expected Free Energy over the same true generative model as
    VOIController. Re-planned fresh from the current belief at every
    decision (a fresh small Agent per decision, seeded with the running
    belief as its prior D) rather than relying on pymdp's own multi-step
    internal state tracking -- this keeps the comparison structurally
    identical to VOIController's one-step lookahead and avoids subtle
    bugs in cross-timestep belief propagation for a 1-2 week kill-test."""

    name = "efe_active_inference"

    # "done" absorbs any further lookahead steps after a terminal action so
    # a policy can't be re-rewarded for "re-terminating" within the
    # lookahead window (an earlier version of this without a `done` phase
    # let e.g. ['escalate','escalate'] collect the terminal-reward
    # observation twice, structurally biasing EFE toward premature
    # termination regardless of whether gathering more info would help --
    # see RESEARCH_PLAN.md Stage 0.5 notes).
    PHASES = ["pre_decision", "post_act_now", "post_escalate", "done"]
    OUTCOMES = ["no_outcome", "success", "failure", "escalate_correct", "escalate_wrong"]

    def __init__(self, policy_len=2):
        self.policy_len = policy_len
        self.belief = None
        self._policies = self._build_synchronized_policies(policy_len)

    def _build_synchronized_policies(self, policy_len):
        # Both hidden-state factors (task_state, phase) are driven by the
        # *same* action each step -- pymdp's default policy enumeration
        # would treat them as independently controllable (9x9 combos for
        # policy_len=2), so we build the synchronized policy list by hand.
        policies = []
        for action_seq in itertools.product(range(3), repeat=policy_len):
            policies.append(np.array([[a, a] for a in action_seq]))
        return policies

    def _build_agent(self):
        from pymdp.legacy.agent import Agent
        from pymdp.legacy import utils

        num_phases = len(self.PHASES)
        num_states = [3, num_phases]  # task_state, phase
        num_obs = [3, 5]              # confidence, outcome
        num_controls = [3, 3]

        A = utils.obj_array_zeros([[o] + num_states for o in num_obs])
        for s_idx, s in enumerate(STATES):
            for phase_idx in range(num_phases):
                A[0][:, s_idx, phase_idx] = CONF_DIST[s]

        for s_idx, s in enumerate(STATES):
            A[1][0, s_idx, 0] = 1.0  # pre_decision -> always no_outcome
            A[1][1, s_idx, 1] = P_SUCCESS[s]       # post_act_now -> success
            A[1][2, s_idx, 1] = 1 - P_SUCCESS[s]   # post_act_now -> failure
            if s == "needs_human":
                A[1][3, s_idx, 2] = 1.0  # post_escalate -> escalate_correct
            else:
                A[1][4, s_idx, 2] = 1.0  # post_escalate -> escalate_wrong
            A[1][0, s_idx, 3] = 1.0  # done -> always no_outcome (already resolved)

        B = utils.obj_array_zeros([[n, n, c] for n, c in zip(num_states, num_controls)])
        # task_state transitions (factor 0): only gather_info (action 1) can move
        # needs_info -> solvable; act_now/escalate leave state unchanged.
        i_solv, i_info, i_human = STATES.index("solvable"), STATES.index("needs_info"), STATES.index("needs_human")
        for a in range(3):
            if a == 1:  # gather_info
                B[0][i_solv, i_solv, a] = 1.0
                B[0][i_solv, i_info, a] = GATHER_RESOLVE_PROB
                B[0][i_info, i_info, a] = 1 - GATHER_RESOLVE_PROB
                B[0][i_human, i_human, a] = 1.0
            else:
                B[0][:, :, a] = np.eye(3)
        # phase transitions (factor 1): depends on the CURRENT phase, not
        # just the action -- once a terminal action has been taken
        # (phase != pre_decision), any further step goes to `done` rather
        # than re-entering post_act_now/post_escalate, so a policy can only
        # collect the terminal-reward observation once.
        for current_phase in range(num_phases):
            for a in range(3):
                if current_phase == 0:  # pre_decision
                    next_phase = {0: 1, 1: 0, 2: 2}[a]  # act_now/gather/escalate
                else:
                    next_phase = 3  # done
                B[1][next_phase, current_phase, a] = 1.0

        C = utils.obj_array_zeros(num_obs)
        C[0][:] = 0.0  # confidence obs: purely informative, no direct preference
        C[1][self.OUTCOMES.index("no_outcome")] = -0.3
        C[1][self.OUTCOMES.index("success")] = 3.0
        C[1][self.OUTCOMES.index("failure")] = -3.0
        C[1][self.OUTCOMES.index("escalate_correct")] = 1.0
        C[1][self.OUTCOMES.index("escalate_wrong")] = -2.0

        D = utils.obj_array_zeros(num_states)
        D[0] = np.array(self.belief)
        D[1] = utils.onehot(0, num_phases)  # always start a decision at pre_decision phase

        return Agent(A=A, B=B, C=C, D=D, policies=self._policies, num_controls=num_controls)

    def start_episode(self):
        self.belief = list(STATE_PRIOR)

    def decide(self, obs_confidence, gathers_used, valid_actions):
        agent = self._build_agent()
        conf_idx = CONF_BINS.index(obs_confidence)
        qs = agent.infer_states([conf_idx, 0])  # 0 = no_outcome (haven't acted yet this step)
        self.belief = [float(x) for x in qs[0]]
        q_pi, _efe = agent.infer_policies()

        action_marginals = {0: 0.0, 1: 0.0, 2: 0.0}
        for i, policy in enumerate(self._policies):
            first_action = int(policy[0, 0])
            action_marginals[first_action] += q_pi[i]

        valid_idx = [ACTIONS.index(a) for a in valid_actions]
        best = max(valid_idx, key=lambda i: action_marginals[i])
        return ACTIONS[best]
