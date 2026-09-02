"""Shared belief update for the non-EFE baselines.

Reuses efe_controller.OBS_DISTS -- the same per-modality likelihoods
EFE's own A matrix encodes -- as the assumed observation model. This is
NOT a claim of a validated "true" generative model (efe_controller.py's
own docstring already flags those distributions as hand-specified
placeholders); it's the same assumption EFE itself runs on, which is
what "same observations, same assumed model" fairness requires for a
baseline comparison at this stage. Stage 3's real-agent-scale run is
where "the true generative model isn't known" actually bites (see
context/TODOS.md) -- these baselines don't get to cheat with more
information than EFE has.
"""
from .. import efe_controller as efe


def bayes_update(belief, observation):
    idxs = observation.as_indices()
    unnorm = []
    for s_idx, s in enumerate(efe.TASK_STATES):
        p = belief[s_idx]
        for m, dist in enumerate(efe.OBS_DISTS):
            p *= dist[s][idxs[m]]
        unnorm.append(p)
    z = sum(unnorm)
    return [u / z for u in unnorm] if z > 0 else list(belief)


def observation_reward(observation):
    """The C-weighted preference EFE itself optimizes for
    (efe_controller.C_WEIGHTS) -- judging a baseline by EFE's own
    preference definition, not a separately invented metric."""
    idxs = observation.as_indices()
    return sum(efe.C_WEIGHTS[m][b] for m, b in enumerate(idxs))


# Paying observation_reward on every turn -- including non-terminal ones
# -- rewards a controller for merely observing something good, not for
# actually resolving the task, and an agent that keeps re-observing a
# good-looking state forever collects that reward indefinitely (verified
# empirically: an early version of the learned router converged to never
# choosing `continue` at all, since every extra turn was pure upside).
# kill_test's env avoided this by construction -- R_GATHER_COST during
# info-seeking, real payoff (R_SUCCESS/R_FAILURE/...) only at a genuine
# terminal action (kill_test/env.py). step_reward reproduces that: the
# real observation payoff only counts when this turn is terminal
# (continue/escalate chosen, or turns ran out); every other turn pays a
# flat cost regardless of how good that turn's observation looked.
NONTERMINAL_STEP_COST = -0.3

# observation_reward alone scores the OBSERVATION, not whether continue
# vs escalate_to_human was the right call given it -- confirmed
# empirically: with only observation_reward, an early version of the
# learned router never learned to prefer escalating on forced-bad tasks
# at all (nothing in the reward differed between the two choices at the
# same observation). kill_test's env fixed this with ground-truth-keyed
# R_ESCALATE_CORRECT/R_ESCALATE_WRONG/R_SUCCESS/R_FAILURE
# (kill_test/env.py); the direct analogue here is `forced_bad` -- the
# same task_id convention mock_agent_step itself already uses to decide
# whether a trajectory is genuinely unresolvable. This bonus/penalty is
# for the EVALUATION HARNESS and the router's TRAINING signal only --
# never fed to a controller's own decide(), same as kill_test's env
# knew the true state but controllers only ever saw observations.
OUTCOME_ADJUSTMENT = {
    ("continue", False): 1.0,     # correctly resolved a genuinely solvable task
    ("continue", True): -2.0,     # silent failure: claimed done on an unresolvable task
    ("escalate_to_human", True): 1.0,     # correctly escalated
    ("escalate_to_human", False): -1.0,   # unnecessary escalation (wasted human time)
}


def step_reward(observation, is_terminal, policy=None, forced_bad=False):
    if not is_terminal:
        return NONTERMINAL_STEP_COST
    reward = observation_reward(observation)
    reward += OUTCOME_ADJUSTMENT.get((policy, forced_bad), 0.0)
    return reward


def uniform_decision(policy, belief):
    """Build an efe_controller.Decision for a baseline that doesn't
    produce EFE's epistemic/pragmatic decomposition -- zero those fields
    rather than omitting them, so the decision log schema stays uniform
    across controllers."""
    zeros = {p: 0.0 for p in efe.POLICIES}
    return efe.Decision(
        policy=policy,
        belief=dict(zip(efe.TASK_STATES, belief)),
        action_marginals={p: (1.0 if p == policy else 0.0) for p in efe.POLICIES},
        epistemic_value=dict(zeros),
        pragmatic_value=dict(zeros),
    )
