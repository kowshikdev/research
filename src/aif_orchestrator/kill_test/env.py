"""Tiny synthetic escalation environment for the Stage 0.5 kill-test.

A deliberately small version of the decision-POMDP in
docs/decision-pomdp.md (3 states x 3 actions x 1 noisy observation
channel), chosen so that:

  (a) a hand-derived Bayes-optimal / value-of-information (VOI) controller
      is tractable to write by hand and can be trusted as a genuine
      "best possible under this model" baseline, and
  (b) pymdp's EFE engine can be compared against that same true-optimal
      baseline on the exact same generative model, not just against a
      production-style heuristic.

This is not a stand-in for the real orchestrator -- it exists purely to
answer one question cheaply, before committing to the full LangGraph +
benchmark integration: does active inference produce meaningfully
different (better-calibrated) act/gather/escalate decisions than a
learned router or an explicit decision-theoretic controller, given the
*same* observations? See RESEARCH_PLAN.md Stage 0.5.
"""
from dataclasses import dataclass
import random

STATES = ["solvable", "needs_info", "needs_human"]
STATE_PRIOR = [0.5, 0.35, 0.15]

CONF_BINS = ["low", "medium", "high"]
# P(confidence bin | true state) -- deliberately noisy/overlapping, like a
# real self-consistency or verifier score would be, not a clean signal.
CONF_DIST = {
    "solvable":    [0.05, 0.15, 0.80],
    "needs_info":  [0.25, 0.50, 0.25],
    "needs_human": [0.65, 0.25, 0.10],
}

# P(act_now succeeds | true state) -- acting without enough info sometimes
# gets lucky, and even a "needs_human" task is occasionally solvable by
# chance, mirroring real agent behavior.
P_SUCCESS = {"solvable": 0.92, "needs_info": 0.35, "needs_human": 0.08}

# Gathering info can resolve genuine ambiguity but can't manufacture
# information a human uniquely holds.
GATHER_RESOLVE_PROB = 0.6  # needs_info -> solvable, after one gather_info step
MAX_GATHERS = 3

R_SUCCESS = 1.0
R_FAILURE = -1.0
R_ESCALATE_CORRECT = 0.5
R_ESCALATE_WRONG = -0.7
R_GATHER_COST = -0.1

ACTIONS = ["act_now", "gather_info", "escalate"]


def sample_categorical(probs, rng):
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


@dataclass
class StepResult:
    obs_confidence: str
    outcome: str | None  # None while the episode continues
    reward: float
    done: bool
    gathers_used: int


class TinyEscalationEnv:
    """One episode = one task instance with a fixed (but possibly
    resolvable) hidden state. The agent picks one action per step until a
    terminal action (act_now / escalate) or MAX_GATHERS is exhausted
    (forced terminal choice)."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.state = None
        self.gathers_used = 0
        self.done = True

    def reset(self):
        self.state = STATES[sample_categorical(STATE_PRIOR, self.rng)]
        self.gathers_used = 0
        self.done = False
        return self._observe()

    def _observe(self):
        conf_idx = sample_categorical(CONF_DIST[self.state], self.rng)
        return CONF_BINS[conf_idx]

    def valid_actions(self):
        if self.gathers_used >= MAX_GATHERS:
            return ["act_now", "escalate"]
        return list(ACTIONS)

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        assert action in self.valid_actions(), f"{action!r} not valid (gathers_used={self.gathers_used})"

        if action == "act_now":
            success = self.rng.random() < P_SUCCESS[self.state]
            outcome = "success" if success else "failure"
            reward = R_SUCCESS if success else R_FAILURE
            self.done = True
            return StepResult(self._observe(), outcome, reward, True, self.gathers_used)

        if action == "escalate":
            correct = self.state == "needs_human"
            outcome = "escalate_correct" if correct else "escalate_wrong"
            reward = R_ESCALATE_CORRECT if correct else R_ESCALATE_WRONG
            self.done = True
            return StepResult(self._observe(), outcome, reward, True, self.gathers_used)

        # gather_info
        if self.state == "needs_info" and self.rng.random() < GATHER_RESOLVE_PROB:
            self.state = "solvable"
        self.gathers_used += 1
        return StepResult(self._observe(), None, R_GATHER_COST, False, self.gathers_used)
