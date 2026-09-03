"""Tests for the Stage 0.5 kill-test (src/aif_orchestrator/kill_test/).

This code is historical -- it answered the go/no-go question before the
real build started and isn't imported by anything above it -- but its
result (docs/stage0.5-kill-test-results.md) is cited as the reason the
whole project continued, so it needs to stay reproducible. A silently
broken kill-test would mean a cited result nobody can regenerate.

Includes a regression test for known-issues #2 (the EFE lookahead
double-counting bug), which originally looked like a genuine
EFE-vs-VOI finding before being traced to a modelling error.
"""
import pytest

from aif_orchestrator.kill_test import controllers as kt_controllers
from aif_orchestrator.kill_test.env import (
    ACTIONS,
    CONF_BINS,
    MAX_GATHERS,
    STATE_PRIOR,
    STATES,
    TinyEscalationEnv,
)


def test_env_distributions_are_normalized():
    from aif_orchestrator.kill_test.env import CONF_DIST

    assert sum(STATE_PRIOR) == pytest.approx(1.0)
    for state, probs in CONF_DIST.items():
        assert sum(probs) == pytest.approx(1.0), f"CONF_DIST[{state}]"


def test_env_is_deterministic_per_seed():
    a, b = TinyEscalationEnv(seed=11), TinyEscalationEnv(seed=11)
    assert a.reset() == b.reset()
    assert a.state == b.state
    assert a.step("gather_info").obs_confidence == b.step("gather_info").obs_confidence


def test_terminal_actions_end_the_episode():
    for action in ("act_now", "escalate"):
        env = TinyEscalationEnv(seed=3)
        env.reset()
        result = env.step(action)
        assert result.done is True
        assert result.outcome is not None
        with pytest.raises(RuntimeError):
            env.step("act_now")


def test_gather_is_withdrawn_once_the_budget_is_spent():
    env = TinyEscalationEnv(seed=5)
    env.reset()
    for _ in range(MAX_GATHERS):
        assert "gather_info" in env.valid_actions()
        env.step("gather_info")
    assert "gather_info" not in env.valid_actions()
    assert set(env.valid_actions()) == {"act_now", "escalate"}


def test_escalation_is_scored_against_the_hidden_state():
    """The environment knows ground truth; controllers only ever see
    observations. That asymmetry is the whole point of the setup."""
    env = TinyEscalationEnv(seed=1)
    for _ in range(30):
        env.reset()
        true_state = env.state
        result = env.step("escalate")
        expected = "escalate_correct" if true_state == "needs_human" else "escalate_wrong"
        assert result.outcome == expected


@pytest.mark.parametrize("controller_name", ["heuristic", "voi", "efe", "random"])
def test_kill_test_controllers_return_valid_actions(controller_name):
    factory = {
        "heuristic": kt_controllers.HeuristicController,
        "voi": kt_controllers.VOIController,
        "efe": kt_controllers.EFEController,
        "random": lambda: kt_controllers.RandomController(seed=0),
    }[controller_name]
    controller = factory()
    controller.start_episode()

    env = TinyEscalationEnv(seed=9)
    obs = env.reset()
    for _ in range(MAX_GATHERS + 1):
        action = controller.decide(obs, env.gathers_used, env.valid_actions())
        assert action in env.valid_actions()
        result = env.step(action)
        if result.done:
            break
        obs = result.obs_confidence


def test_efe_and_voi_agree_on_a_clearly_uncertain_belief():
    """Regression: known-issues #2. Before the absorbing `done` phase was
    added, EFE double-counted terminal rewards and chose `escalate` here
    while the hand-derived Bayes-optimal VOI controller chose
    `gather_info` -- the divergence that looked like a finding and was
    actually a bug."""
    efe = kt_controllers.EFEController()
    voi = kt_controllers.VOIController()
    efe.start_episode()
    voi.start_episode()

    valid = list(ACTIONS)
    assert efe.decide("low", 0, valid) == voi.decide("low", 0, valid) == "gather_info"


def test_efe_phase_model_absorbs_repeated_terminal_actions():
    """The structural fix behind the test above: a terminal action can
    only collect its outcome observation once, no matter how many
    lookahead steps follow it."""
    controller = kt_controllers.EFEController()
    assert "done" in controller.PHASES
    controller.start_episode()
    agent = controller._build_agent()
    # From any already-terminal phase, every action must lead to `done`.
    done_idx = controller.PHASES.index("done")
    for phase_idx in range(1, len(controller.PHASES)):
        for action in range(len(ACTIONS)):
            assert agent.B[1][done_idx, phase_idx, action] == 1.0


def test_confidence_bins_are_the_ones_controllers_expect():
    assert CONF_BINS == ["low", "medium", "high"]
    assert set(STATES) == {"solvable", "needs_info", "needs_human"}
