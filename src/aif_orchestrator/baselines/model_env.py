"""Model-matched simulator for the real decision-POMDP -- closes the gap
`docs/stage2-baselines-results.md` flagged: the original Stage 2
comparison ran every controller against `graph.mock_agent_step`, a small
hand-scripted stand-in, NOT a sample from the same generative model
EFE/VOI actually assume. That's fine for proving the plumbing/interfaces
work (which it did, and it caught two real reward-design bugs -- see
belief.py), but it is not a repeat of the Stage 0.5 kill-test's
Bayes-optimality claim, because the kill-test's env WAS literally the
same true generative model VOI/EFE reasoned over.

This environment is that: it samples its TRUE task_state from
`efe_controller.D_PRIOR`, samples each observation modality
independently from `efe_controller.OBS_DISTS[true_state]` (matching the
A-matrix's structure -- modalities are conditionally independent given
state, no cross-modality correlation is encoded), and transitions the
TRUE state via `efe_controller.B_TRANSITIONS[policy][true_state]` when a
non-terminal policy is taken. It treats efe_controller.py's own
(explicitly hand-specified, not calibrated -- see that file's docstring)
matrices as ground truth, exactly as VOI and EFE both assume. Any gap
this reveals between controllers is now genuinely about decision
mechanism, not about who happens to model `mock_agent_step`'s scripted
quirks better -- the same guarantee kill_test/env.py gave Stage 0.5.
"""
import random

from .. import efe_controller as efe


def _sample(probs, labels, rng):
    r = rng.random()
    cum = 0.0
    for p, label in zip(probs, labels):
        cum += p
        if r <= cum:
            return label
    return labels[-1]


class ModelMatchedEnv:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = None
        self.done = True

    def reset(self):
        self.state = _sample(efe.D_PRIOR, efe.TASK_STATES, self.rng)
        self.done = False
        return self._observe()

    def _observe(self):
        bins = [
            _sample(dist[self.state], labels, self.rng)
            for dist, labels in zip(efe.OBS_DISTS, efe.OBS_BINS)
        ]
        return efe.Observation(
            tool_result=bins[0], confidence=bins[1],
            policy_gate=bins[2], retrieval_quality=bins[3],
        )

    def step(self, policy):
        """Transition the TRUE state per B_TRANSITIONS[policy], return the
        observation this produces for the next turn. Only meaningful for
        non-terminal policies -- callers should stop once `continue` /
        `escalate_to_human` is chosen, same as graph.py's
        route_from_decision."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        row = efe.B_TRANSITIONS[policy][self.state]
        self.state = _sample([row[s] for s in efe.TASK_STATES], efe.TASK_STATES, self.rng)
        return self._observe()
