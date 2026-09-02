"""Heuristic-threshold baseline (RESEARCH_PLAN.md §3.6.1): the
"production-style" `escalate <=> p_success < tau OR policy_gate == deny`
pattern, ported from kill_test.controllers.HeuristicController's
confidence-threshold logic to the real 4-modality observation.

Confidence is used directly as the threshold signal (no separate
p_success estimate) -- exactly like the kill-test's HeuristicController,
and exactly what a real production system would do with a binned
confidence score. tool_result and policy_gate override it: a hard
`deny` always escalates, and a clean `success` always continues,
regardless of confidence -- mirroring "act on high confidence, escalate
on low confidence, gather in the ambiguous middle" but with the two
extra observation modalities this decision-POMDP has that the kill-test
env didn't.
"""
from .. import efe_controller as efe
from .belief import bayes_update, uniform_decision


class HeuristicControlNode:
    name = "heuristic"

    def __init__(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def reset(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def decide(self, observation, valid_policies=None) -> efe.Decision:
        valid_policies = valid_policies or list(efe.POLICIES)
        # Belief isn't used for the decision (a real heuristic threshold
        # acts on the latest signal only) -- tracked purely so the
        # decision log's belief column is comparable across controllers.
        self.belief = bayes_update(self.belief, observation)

        if observation.policy_gate == "deny":
            policy = "escalate_to_human"
        elif observation.tool_result == "success" and observation.confidence in ("medium", "high"):
            policy = "continue"
        elif observation.confidence == "low":
            policy = "escalate_to_human" if observation.tool_result == "error" else "gather_info"
        elif observation.tool_result == "error":
            policy = "retry"
        elif observation.tool_result in ("partial", "no_tool_called"):
            policy = "gather_info"
        else:
            policy = "call_tool"

        if policy not in valid_policies:
            policy = "escalate_to_human" if "escalate_to_human" in valid_policies else valid_policies[0]

        return uniform_decision(policy, self.belief)
