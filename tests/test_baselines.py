"""Tests for the four Stage 2 baselines and the shared belief/reward
plumbing (src/aif_orchestrator/baselines/).

The interface-conformance tests here are load-bearing: docs/architecture-overview.md
claims all five controllers are drop-in interchangeable, and graph.py /
tau2_integration/ both depend on that being literally true. These tests
are what make that claim checkable rather than aspirational.

Also includes regression tests for known-issues #5, #6 and #10.
"""
import pytest

from aif_orchestrator import efe_controller as efe
from aif_orchestrator.efe_controller import Observation
from aif_orchestrator.baselines import belief as belief_mod
from aif_orchestrator.baselines.model_env import ModelMatchedEnv
from aif_orchestrator.baselines.router import (
    LearnedRouterControlNode,
    model_matched_source_factory,
)
from aif_orchestrator.baselines.voi import VOIControlNode

OBSERVATION_CASES = [
    dict(tool_result="success", confidence="high", policy_gate="allow", retrieval_quality="n/a"),
    dict(tool_result="partial", confidence="medium", policy_gate="needs_review", retrieval_quality="adequate"),
    dict(tool_result="error", confidence="low", policy_gate="deny", retrieval_quality="poor"),
    dict(tool_result="no_tool_called", confidence="medium", policy_gate="allow", retrieval_quality="n/a"),
]


# --- interface conformance (all five controllers) --------------------------

def test_controller_implements_the_shared_interface(controller_cls):
    node = controller_cls()
    assert hasattr(node, "decide") and hasattr(node, "reset")
    assert hasattr(controller_cls, "name") or controller_cls.__name__


@pytest.mark.parametrize("obs_kwargs", OBSERVATION_CASES)
def test_controller_returns_a_wellformed_decision(controller_cls, obs_kwargs):
    """Every controller must return a Decision with the SAME shape, even
    the ones with no epistemic term -- baselines/belief.uniform_decision
    zeroes those fields rather than omitting them so the decision-log
    schema stays uniform across controllers."""
    decision = controller_cls().decide(Observation(**obs_kwargs))

    assert decision.policy in efe.POLICIES
    assert set(decision.belief) == set(efe.TASK_STATES)
    assert sum(decision.belief.values()) == pytest.approx(1.0, abs=1e-6)
    for field in (decision.action_marginals, decision.epistemic_value, decision.pragmatic_value):
        assert set(field) == set(efe.POLICIES)


def test_controller_honors_valid_policies(controller_cls):
    allowed = ["continue", "gather_info"]
    for obs_kwargs in OBSERVATION_CASES:
        decision = controller_cls().decide(Observation(**obs_kwargs), valid_policies=allowed)
        assert decision.policy in allowed


def test_controller_accepts_and_uses_a_prior(controller_cls):
    prior = [0.1, 0.1, 0.7, 0.1]
    node = controller_cls(prior=prior)
    assert node.belief == pytest.approx(prior)
    node.reset()
    assert node.belief == pytest.approx(list(efe.D_PRIOR))


def test_only_efe_produces_a_nonzero_epistemic_term(controller_cls):
    """The epistemic/information-gain term is what's actually under test
    in this project -- if a baseline started reporting one, the
    comparison would no longer be measuring what it claims to."""
    decision = controller_cls().decide(Observation(
        tool_result="partial", confidence="medium",
        policy_gate="allow", retrieval_quality="adequate",
    ))
    has_epistemic = any(v != 0.0 for v in decision.epistemic_value.values())
    assert has_epistemic == (controller_cls is efe.EFEControlNode)


# --- reward shaping (known-issues #5 and #6) --------------------------------

def test_nonterminal_turns_do_not_pay_the_observation_reward():
    """Regression: known-issues #5. Paying observation_reward on every
    turn let the router farm reward by never terminating."""
    good = Observation(tool_result="success", confidence="high",
                       policy_gate="allow", retrieval_quality="good")
    nonterminal = belief_mod.step_reward(good, is_terminal=False, policy="gather_info")
    assert nonterminal == belief_mod.NONTERMINAL_STEP_COST
    assert nonterminal < 0, "a non-terminal turn must cost, not pay"
    assert nonterminal < belief_mod.step_reward(good, is_terminal=True, policy="continue")


def test_escalation_is_rewarded_only_when_the_task_was_unresolvable():
    """Regression: known-issues #6. With observation_reward alone,
    `continue` and `escalate_to_human` scored identically at the same
    observation, so nothing ever taught the router to escalate."""
    obs = Observation(tool_result="error", confidence="low",
                      policy_gate="needs_review", retrieval_quality="poor")

    correct_escalation = belief_mod.step_reward(obs, True, policy="escalate_to_human", forced_bad=True)
    wasted_escalation = belief_mod.step_reward(obs, True, policy="escalate_to_human", forced_bad=False)
    silent_failure = belief_mod.step_reward(obs, True, policy="continue", forced_bad=True)
    clean_resolve = belief_mod.step_reward(obs, True, policy="continue", forced_bad=False)

    assert correct_escalation > wasted_escalation
    assert clean_resolve > silent_failure
    # The two choices must differ at the SAME observation -- that's the bug.
    assert correct_escalation != silent_failure


def test_bayes_update_is_a_proper_distribution():
    b = list(efe.D_PRIOR)
    for obs_kwargs in OBSERVATION_CASES:
        b = belief_mod.bayes_update(b, Observation(**obs_kwargs))
        assert sum(b) == pytest.approx(1.0, abs=1e-9)
        assert all(0.0 <= p <= 1.0 for p in b)


# --- VOI-specific -----------------------------------------------------------

def test_voi_info_seeking_lookahead_enumerates_the_full_joint_observation_space():
    """VOI's one-step lookahead is exact, not sampled -- 4 modalities of
    4/3/3/4 bins = 144 joint outcomes. A silent truncation here would
    quietly turn the 'sharpest baseline' into an approximation."""
    from aif_orchestrator.baselines import voi as voi_mod

    assert len(voi_mod._JOINT_BIN_IDX) == 4 * 3 * 3 * 4 == 144
    enumerated = voi_mod._enumerate_joint_observations(list(efe.D_PRIOR))
    assert sum(p for p, _ in enumerated) == pytest.approx(1.0, abs=1e-6)


def test_voi_prefers_escalation_when_belief_says_a_human_is_needed():
    node = VOIControlNode(prior=[0.02, 0.03, 0.90, 0.05])
    decision = node.decide(Observation(
        tool_result="error", confidence="low",
        policy_gate="needs_review", retrieval_quality="poor",
    ))
    assert decision.policy == "escalate_to_human"


# --- router -----------------------------------------------------------------

def test_router_features_include_belief_not_just_the_latest_observation():
    """Regression: the belief-state-parity fix Stage 0.5 flagged. If the
    Q-table were keyed on the observation alone, any EFE-vs-router gap
    would be a feature-engineering artifact rather than a finding about
    control mechanisms (RESEARCH_PLAN.md Sec 5)."""
    from aif_orchestrator.baselines import router as router_mod

    confident = router_mod._belief_bucket([0.95, 0.02, 0.02, 0.01])
    uncertain = router_mod._belief_bucket([0.30, 0.30, 0.20, 0.20])
    assert confident != uncertain, "belief bucket must distinguish confident from uncertain"

    LearnedRouterControlNode()  # ensure trained
    key = next(iter(LearnedRouterControlNode._q))
    assert isinstance(key, tuple) and len(key) == 2, (
        "router Q-table key must be (belief_bucket, observation), not observation alone"
    )


def test_router_training_is_deterministic_within_an_environment():
    """The router is the only controller with trained state, and its
    exact numbers are known to differ across machines/Python versions
    (known-issues #12) -- so within-environment determinism is the
    property that must actually hold. If this breaks, a re-run stops
    reproducing its own results, which is a real bug rather than an
    environment difference.

    Uses a small episode count: the property under test is
    reproducibility, not table quality.
    """
    import json

    def train_and_snapshot():
        LearnedRouterControlNode._q = None
        LearnedRouterControlNode.train(
            source_factory=model_matched_source_factory,
            num_episodes=3000, epsilon_start=0.4, epsilon_end=0.02,
        )
        return json.dumps(
            {str(k): v for k, v in sorted(LearnedRouterControlNode._q.items(), key=lambda kv: str(kv[0]))},
            sort_keys=True,
        )

    first = train_and_snapshot()
    second = train_and_snapshot()
    LearnedRouterControlNode._q = None
    assert first == second, "router training is not reproducible within a single environment"


def test_router_can_be_retrained_against_the_model_matched_env():
    """Regression: known-issues #10. tau2_integration/register.py depends
    on exactly this call working to avoid silently falling back to the
    mock-trained router in a real sweep."""
    LearnedRouterControlNode.train(
        source_factory=model_matched_source_factory,
        num_episodes=2000, epsilon_start=0.4, epsilon_end=0.02,
    )
    assert LearnedRouterControlNode._q, "training produced an empty Q-table"
    decision = LearnedRouterControlNode().decide(Observation(**OBSERVATION_CASES[0]))
    assert decision.policy in efe.POLICIES
    LearnedRouterControlNode._q = None  # don't leak this table into other tests


# --- model-matched environment ---------------------------------------------

def test_model_matched_env_is_deterministic_per_seed():
    a = ModelMatchedEnv(seed=42)
    b = ModelMatchedEnv(seed=42)
    obs_a, obs_b = a.reset(), b.reset()
    assert a.state == b.state
    assert obs_a.as_indices() == obs_b.as_indices()
    assert a.step("gather_info").as_indices() == b.step("gather_info").as_indices()


def test_model_matched_env_samples_valid_states_and_bins():
    env = ModelMatchedEnv(seed=7)
    for _ in range(50):
        obs = env.reset()
        assert env.state in efe.TASK_STATES
        for idx, bins in zip(obs.as_indices(), efe.OBS_BINS):
            assert 0 <= idx < len(bins)


def test_model_matched_env_rejects_stepping_a_finished_episode():
    env = ModelMatchedEnv(seed=1)
    with pytest.raises(RuntimeError):
        env.step("continue")
