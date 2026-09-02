"""Value-of-information / decision-theoretic baseline (RESEARCH_PLAN.md
§3.6.3), ported from kill_test.controllers.VOIController's hand-derived
Bayesian expected-utility lookahead to the real decision-POMDP.

Deliberately has NO epistemic/information-gain term and NO active-
inference formalism -- just explicit Bayes belief updates plus expected-
utility maximization over the same observation model EFE uses
(efe_controller.OBS_DISTS/B_TRANSITIONS/C_WEIGHTS -- see belief.py's
docstring on why reusing EFE's own assumed model, not a separately
validated "true" one, is the fair comparison at this stage). This is the
sharpest baseline: per the kill-test result (docs/stage0.5-kill-test-
results.md), if EFE can't beat this, the active-inference framing isn't
earning anything beyond what hand-derived decision theory already gets.

For the four "terminal-ish" policies (continue / escalate_to_human /
hand_off_to_agent -- the ones that exit the agent_step loop per
graph.py's route_from_decision) the value is just the expected C-weighted
utility of the resulting belief. For the three info-seeking policies
(retry / call_tool / gather_info -- they loop back for a fresh
observation), the value is a genuine one-step value-of-information
computation: transition the belief, enumerate the joint observation this
decision-POMDP actually produces each turn (all 4 modalities together,
144 joint bins -- small enough to do exactly, no sampling), Bayes-update
per observation, and take the expected best terminal value -- mirroring
kill_test's `_eu_gather` but over the real observation space.
"""
import itertools

from .. import efe_controller as efe
from .belief import bayes_update

# Hand-coded per-step cost for the info-seeking / hand-off policies --
# the same role kill_test's R_GATHER_COST played (an extra turn costs
# real time/tokens beyond whatever the resulting observation is worth).
STEP_COST = {"retry": -0.2, "call_tool": -0.15, "gather_info": -0.25, "hand_off_to_agent": -0.35}

# escalate_to_human's B_TRANSITIONS (efe_controller.py) models escalation
# as *resolving* needs_human/likely_to_fail (0.9 / 0.3 back to
# task_solvable_now) -- correctly, since asking a human usually does fix
# things -- which makes its one-step expected utility look strong even
# when only a small belief mass needs it. EFE offsets that with its
# E_WEIGHTS habit prior (a mechanism specific to active inference, not
# available here); VOI has no habits, so it needs the explicit "cost of
# human time" RESEARCH_PLAN.md §2 documents as a preference but which
# isn't actually encoded anywhere in efe_controller.C_WEIGHTS (C only
# covers observation bins, not a per-policy cost). Hand-coding it here
# is exactly the "explicit decision theory, not derived from the
# generative-model spec" contrast §1 draws between VOI and EFE.
ESCALATE_COST = -0.5
HANDOFF_COST = -0.35  # routing to a different agent: a bigger disruption than one more tool call

_JOINT_BIN_IDX = list(itertools.product(*[range(len(bins)) for bins in efe.OBS_BINS]))


def _transition(policy, belief):
    post = [0.0] * len(efe.TASK_STATES)
    for s_idx, s in enumerate(efe.TASK_STATES):
        row = efe.B_TRANSITIONS[policy][s]
        for s2_idx, s2 in enumerate(efe.TASK_STATES):
            post[s2_idx] += belief[s_idx] * row[s2]
    return post


def _eu_terminal(belief):
    """Expected C-weighted utility of the observation this belief would
    produce if observed right now -- same additive-across-modalities
    structure efe_controller.py's own C matrix uses."""
    total = 0.0
    for m, dist in enumerate(efe.OBS_DISTS):
        for s_idx, s in enumerate(efe.TASK_STATES):
            for b_idx, p_bin in enumerate(dist[s]):
                total += belief[s_idx] * p_bin * efe.C_WEIGHTS[m][b_idx]
    return total


def _enumerate_joint_observations(belief):
    """(p_obs, updated_belief) for every joint 4-modality observation bin
    reachable from `belief`, computed exactly (144 combinations)."""
    results = []
    for idx_tuple in _JOINT_BIN_IDX:
        unnorm = [0.0] * len(efe.TASK_STATES)
        for s_idx, s in enumerate(efe.TASK_STATES):
            p = belief[s_idx]
            for m, dist in enumerate(efe.OBS_DISTS):
                p *= dist[s][idx_tuple[m]]
            unnorm[s_idx] = p
        p_obs = sum(unnorm)
        if p_obs > 0:
            results.append((p_obs, [u / p_obs for u in unnorm]))
    return results


class VOIControlNode:
    name = "voi_decision_theoretic"

    def __init__(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def reset(self, prior=None):
        self.belief = list(prior) if prior is not None else list(efe.D_PRIOR)

    def _best_terminal_value(self, belief):
        """max(continue, escalate) at a hypothetical belief -- what the
        lookahead in _eu_info_seeking compares against, so the recursive
        step is judged by the same escalate cost as a direct choice
        would be, not an uncosted version of it."""
        eu_continue = _eu_terminal(_transition("continue", belief))
        eu_escalate = _eu_terminal(_transition("escalate_to_human", belief)) + ESCALATE_COST
        return max(eu_continue, eu_escalate)

    def _eu_info_seeking(self, policy, belief):
        post = _transition(policy, belief)
        expected = sum(
            p_obs * self._best_terminal_value(updated)
            for p_obs, updated in _enumerate_joint_observations(post)
        )
        return STEP_COST[policy] + expected

    def decide(self, observation, valid_policies=None) -> efe.Decision:
        valid_policies = valid_policies or list(efe.POLICIES)
        self.belief = bayes_update(self.belief, observation)

        # continue / escalate_to_human are the only policies that actually
        # exit the agent_step loop in graph.py's route_from_decision --
        # everything else (including hand_off_to_agent, which has no real
        # multi-agent routing implemented yet) loops back for a fresh
        # observation, so it gets the genuine info-seeking lookahead too.
        scores = {}
        for policy in valid_policies:
            if policy == "continue":
                scores[policy] = _eu_terminal(_transition(policy, self.belief))
            elif policy == "escalate_to_human":
                scores[policy] = _eu_terminal(_transition(policy, self.belief)) + ESCALATE_COST
            else:
                scores[policy] = self._eu_info_seeking(policy, self.belief)

        chosen = max(scores, key=scores.get)
        full_scores = {p: scores.get(p, 0.0) for p in efe.POLICIES}
        return efe.Decision(
            policy=chosen,
            belief=dict(zip(efe.TASK_STATES, self.belief)),
            action_marginals={p: (1.0 if p == chosen else 0.0) for p in efe.POLICIES},
            epistemic_value={p: 0.0 for p in efe.POLICIES},  # no info-gain term, by design
            pragmatic_value=full_scores,
        )
