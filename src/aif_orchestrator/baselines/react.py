"""Plain ReAct floor baseline (RESEARCH_PLAN.md §3.6.4): no explicit
control-loop reasoning at all -- just react to the latest tool result.
No calibrated escalation, no retry/gather distinction, nothing tuned;
this is the floor every other controller needs to beat.
"""
from .. import efe_controller as efe
from .belief import bayes_update, uniform_decision


class ReActControlNode:
    name = "react"

    def __init__(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def reset(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def decide(self, observation, valid_policies=None) -> efe.Decision:
        valid_policies = valid_policies or list(efe.POLICIES)
        # Belief tracked for decision-log comparability only -- ReAct
        # doesn't consult it, by definition of "no control-loop reasoning".
        self.belief = bayes_update(self.belief, observation)

        policy = "continue" if observation.tool_result == "success" else "call_tool"
        if policy not in valid_policies:
            policy = valid_policies[0]

        return uniform_decision(policy, self.belief)
