"""The real EFE control node: implements the frozen decision-POMDP from
docs/decision-pomdp.md (4 hidden task-state values, 4 observation
modalities, 6 policies) as a reusable component -- not test-only code,
unlike src/aif_orchestrator/kill_test/.

Structurally simpler than the kill-test controllers: every observation
modality here is observed every orchestrator turn regardless of which
policy was last taken (per decision-pomdp.md's "observations are derived
once per orchestrator turn"), so there's no terminal-only observation
channel and therefore no need for the kill-test's "phase" state trick
(which existed only to stop a terminal action's reward observation from
being double-counted across a multi-step lookahead -- see
docs/stage0.5-kill-test-results.md). Planning horizon is 1 step: the real
system gets fresh observations every turn, so this replans from scratch
each time rather than committing to a multi-step plan in advance -- also
per decision-pomdp.md's "Mapping to the LangGraph agent loop".

The A/B/C/E/D values below are hand-specified defaults matching the
semantic descriptions in docs/decision-pomdp.md, NOT calibrated against
real data -- that calibration is Stage 3 work. Treat every probability
here as a placeholder whose only job right now is to be structurally
reasonable, not correct.
"""
from dataclasses import dataclass

import numpy as np

TASK_STATES = ["task_solvable_now", "needs_more_info", "needs_human", "likely_to_fail"]
POLICIES = ["continue", "retry", "call_tool", "gather_info", "escalate_to_human", "hand_off_to_agent"]

TOOL_RESULT_BINS = ["success", "error", "partial", "no_tool_called"]
CONFIDENCE_BINS = ["low", "medium", "high"]
POLICY_GATE_BINS = ["allow", "deny", "needs_review"]
RETRIEVAL_QUALITY_BINS = ["poor", "adequate", "good", "n/a"]

OBS_MODALITIES = ["tool_result", "confidence", "policy_gate", "retrieval_quality"]
OBS_BINS = [TOOL_RESULT_BINS, CONFIDENCE_BINS, POLICY_GATE_BINS, RETRIEVAL_QUALITY_BINS]

# P(observation bin | task_state), per modality. Rows sum to 1.
TOOL_RESULT_DIST = {
    "task_solvable_now": [0.85, 0.05, 0.05, 0.05],
    "needs_more_info":   [0.10, 0.15, 0.55, 0.20],
    "needs_human":       [0.05, 0.30, 0.15, 0.50],
    "likely_to_fail":    [0.05, 0.55, 0.30, 0.10],
}
CONFIDENCE_DIST = {
    "task_solvable_now": [0.05, 0.15, 0.80],
    "needs_more_info":   [0.30, 0.50, 0.20],
    "needs_human":       [0.55, 0.30, 0.15],
    "likely_to_fail":    [0.65, 0.25, 0.10],
}
POLICY_GATE_DIST = {
    "task_solvable_now": [0.90, 0.02, 0.08],
    "needs_more_info":   [0.70, 0.05, 0.25],
    "needs_human":       [0.30, 0.20, 0.50],
    "likely_to_fail":    [0.50, 0.35, 0.15],
}
RETRIEVAL_QUALITY_DIST = {
    "task_solvable_now": [0.05, 0.15, 0.10, 0.70],
    "needs_more_info":   [0.20, 0.30, 0.10, 0.40],
    "needs_human":       [0.25, 0.10, 0.05, 0.60],
    "likely_to_fail":    [0.30, 0.10, 0.05, 0.55],
}
OBS_DISTS = [TOOL_RESULT_DIST, CONFIDENCE_DIST, POLICY_GATE_DIST, RETRIEVAL_QUALITY_DIST]

# Preferences (C): log-preference weight per observation bin. Neutral (0)
# on confidence -- it's purely informative, matching the T-maze/kill-test
# pattern where a cue observation has no preference of its own, only
# epistemic value.
C_WEIGHTS = [
    [3.0, -2.0, -0.5, -0.3],   # tool_result: success / error / partial / no_tool_called
    [0.0, 0.0, 0.0],            # confidence: neutral
    [0.2, -1.0, -0.5],          # policy_gate: allow / deny / needs_review
    [-0.2, 0.3, 0.6, 0.0],      # retrieval_quality: poor / adequate / good / n/a
]

# Habits (E): prior weight over policies before EFE -- a mild, principled
# (and tunable, per RESEARCH_PLAN.md Sec 5 ablations) bias against
# reaching for escalation/hand-off by default, rather than a hard
# threshold rule.
E_WEIGHTS = {
    "continue": 1.0, "retry": 1.0, "call_tool": 1.0, "gather_info": 1.0,
    "escalate_to_human": 0.5, "hand_off_to_agent": 0.6,
}

# B: P(next task_state | task_state, policy). Each policy's resolving
# power reflects its semantic role in docs/decision-pomdp.md -- e.g.
# gather_info resolves needs_more_info most reliably; escalate_to_human
# resolves needs_human most reliably and does nothing useful otherwise.
def _identity_row(state):
    return {s: (1.0 if s == state else 0.0) for s in TASK_STATES}


B_TRANSITIONS = {
    "continue": {s: _identity_row(s) for s in TASK_STATES},
    # retry/call_tool/gather_info each carry a small, increasing risk of
    # nudging an *already-solvable* task toward needs_more_info -- an
    # unnecessary extra step can surface confusing or spurious signals
    # (a real failure mode) -- so EFE has an actual reason to prefer
    # `continue` when the task is already on track, instead of being
    # indifferent among all four non-escalation policies. `continue`
    # alone carries zero risk.
    "retry": {
        "task_solvable_now": {"task_solvable_now": 0.97, "needs_more_info": 0.03, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_more_info": {"task_solvable_now": 0.2, "needs_more_info": 0.8, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_human": _identity_row("needs_human"),
        "likely_to_fail": {"task_solvable_now": 0.1, "needs_more_info": 0.0, "needs_human": 0.0, "likely_to_fail": 0.9},
    },
    "call_tool": {
        "task_solvable_now": {"task_solvable_now": 0.95, "needs_more_info": 0.05, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_more_info": {"task_solvable_now": 0.5, "needs_more_info": 0.5, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_human": _identity_row("needs_human"),
        "likely_to_fail": {"task_solvable_now": 0.15, "needs_more_info": 0.0, "needs_human": 0.0, "likely_to_fail": 0.85},
    },
    "gather_info": {
        "task_solvable_now": {"task_solvable_now": 0.90, "needs_more_info": 0.10, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_more_info": {"task_solvable_now": 0.65, "needs_more_info": 0.35, "needs_human": 0.0, "likely_to_fail": 0.0},
        "needs_human": {"task_solvable_now": 0.0, "needs_more_info": 0.15, "needs_human": 0.85, "likely_to_fail": 0.0},
        "likely_to_fail": {"task_solvable_now": 0.1, "needs_more_info": 0.0, "needs_human": 0.0, "likely_to_fail": 0.9},
    },
    "escalate_to_human": {
        "task_solvable_now": _identity_row("task_solvable_now"),
        "needs_more_info": _identity_row("needs_more_info"),
        "needs_human": {"task_solvable_now": 0.9, "needs_more_info": 0.0, "needs_human": 0.1, "likely_to_fail": 0.0},
        "likely_to_fail": {"task_solvable_now": 0.3, "needs_more_info": 0.0, "needs_human": 0.0, "likely_to_fail": 0.7},
    },
    "hand_off_to_agent": {
        "task_solvable_now": _identity_row("task_solvable_now"),
        "needs_more_info": _identity_row("needs_more_info"),
        "needs_human": {"task_solvable_now": 0.2, "needs_more_info": 0.0, "needs_human": 0.8, "likely_to_fail": 0.0},
        "likely_to_fail": {"task_solvable_now": 0.55, "needs_more_info": 0.0, "needs_human": 0.0, "likely_to_fail": 0.45},
    },
}

D_PRIOR = [0.6, 0.25, 0.10, 0.05]  # [task_solvable_now, needs_more_info, needs_human, likely_to_fail]


@dataclass
class Observation:
    tool_result: str
    confidence: str
    policy_gate: str
    retrieval_quality: str

    def as_indices(self):
        return [
            TOOL_RESULT_BINS.index(self.tool_result),
            CONFIDENCE_BINS.index(self.confidence),
            POLICY_GATE_BINS.index(self.policy_gate),
            RETRIEVAL_QUALITY_BINS.index(self.retrieval_quality),
        ]


@dataclass
class Decision:
    policy: str
    belief: dict  # task_state -> posterior probability
    action_marginals: dict  # policy -> marginal probability under EFE
    epistemic_value: dict  # policy -> states_info_gain component
    pragmatic_value: dict  # policy -> utility component


class EFEControlNode:
    """Reusable EFE control node over the frozen decision-POMDP. One
    instance persists belief across turns of a single task/conversation;
    call `decide()` once per orchestrator turn."""

    def __init__(self, prior=None):
        self.belief = list(prior) if prior is not None else list(D_PRIOR)

    def reset(self, prior=None):
        self.belief = list(prior) if prior is not None else list(D_PRIOR)

    def _build_agent(self):
        from pymdp.legacy.agent import Agent
        from pymdp.legacy import utils

        num_states = [4]
        num_obs = [len(bins) for bins in OBS_BINS]
        num_controls = [len(POLICIES)]

        A = utils.obj_array_zeros([[o] + num_states for o in num_obs])
        for m, dist in enumerate(OBS_DISTS):
            for s_idx, s in enumerate(TASK_STATES):
                A[m][:, s_idx] = dist[s]

        B = utils.obj_array_zeros([[4, 4, len(POLICIES)]])
        for p_idx, policy in enumerate(POLICIES):
            for s_idx, s in enumerate(TASK_STATES):
                row = B_TRANSITIONS[policy][s]
                B[0][:, s_idx, p_idx] = [row[s2] for s2 in TASK_STATES]

        C = utils.obj_array_zeros(num_obs)
        for m, weights in enumerate(C_WEIGHTS):
            C[m][:] = weights

        E = np.array([E_WEIGHTS[p] for p in POLICIES], dtype=float)
        E = E / E.sum()

        D = utils.obj_array_zeros(num_states)
        D[0] = np.array(self.belief)

        return Agent(A=A, B=B, C=C, D=D, E=E, num_controls=num_controls, policy_len=1)

    def decide(self, observation: Observation, valid_policies=None) -> Decision:
        valid_policies = valid_policies or list(POLICIES)
        agent = self._build_agent()
        qs = agent.infer_states(observation.as_indices())
        self.belief = [float(x) for x in qs[0]]
        q_pi, efe = agent.infer_policies()

        # policy_len=1 -> agent.policies has one row per policy, each a
        # single-timestep [policy_idx] control.
        action_marginals = {POLICIES[int(p[0, 0])]: float(q_pi[i]) for i, p in enumerate(agent.policies)}
        efe_by_policy = {POLICIES[int(p[0, 0])]: float(efe[i]) for i, p in enumerate(agent.policies)}

        valid_marginals = {p: v for p, v in action_marginals.items() if p in valid_policies}
        chosen = max(valid_marginals, key=valid_marginals.get)

        # Decompose EFE's two terms for logging/interpretability (Stage 5):
        # epistemic value = expected information gain about task_state;
        # pragmatic value = expected log-preference of predicted obs.
        # pymdp's Agent doesn't expose these split out of the box for the
        # legacy API, so we recompute them directly for transparency.
        epistemic, pragmatic = self._decompose_efe(agent)

        return Decision(
            policy=chosen,
            belief=dict(zip(TASK_STATES, self.belief)),
            action_marginals=action_marginals,
            epistemic_value={POLICIES[int(p[0, 0])]: epistemic[i] for i, p in enumerate(agent.policies)},
            pragmatic_value={POLICIES[int(p[0, 0])]: pragmatic[i] for i, p in enumerate(agent.policies)},
        )

    def _decompose_efe(self, agent):
        from pymdp.legacy import control
        qs = agent.qs
        epistemic, pragmatic = [], []
        for policy in agent.policies:
            qs_pi = control.get_expected_states(qs, agent.B, policy)
            qo_pi = control.get_expected_obs(qs_pi, agent.A)
            info_gain = sum(
                control.calc_states_info_gain(agent.A, [qs_pi_t]) for qs_pi_t in qs_pi
            )
            utility = sum(control.calc_expected_utility([qo_pi_t], agent.C) for qo_pi_t in qo_pi)
            epistemic.append(float(info_gain))
            pragmatic.append(float(utility))
        return epistemic, pragmatic
