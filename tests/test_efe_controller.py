"""Tests for the EFE engine itself (src/aif_orchestrator/efe_controller.py).

Includes regression tests for two real bugs documented in
docs/known-issues-and-gotchas.md -- #1 (wrong pymdp package) and #3 (no
cost signal for unnecessary action). Both were silent failures: the code
ran fine and produced *a* decision, just the wrong one. Regression tests
are the only thing that stops that class of bug from coming back
unnoticed.
"""
import math

import pytest

from aif_orchestrator import efe_controller as efe
from aif_orchestrator.efe_controller import EFEControlNode, Observation


def test_correct_pymdp_package_is_installed():
    """Regression: known-issues #1. `pip install pymdp` silently installs
    an unrelated MDP toolkit that also imports as `pymdp` -- the failure
    mode is an AttributeError deep in a run, not at install time."""
    from pymdp.legacy.agent import Agent  # noqa: F401
    from pymdp.legacy import utils

    assert hasattr(utils, "obj_array_zeros"), (
        "wrong pymdp package -- install 'inferactively-pymdp', not 'pymdp'"
    )


def test_generative_model_is_normalized():
    """Every observation/transition distribution must sum to 1. A typo in
    a hand-specified probability row is easy to make and produces
    silently skewed inference rather than an error."""
    for modality, dist in enumerate(efe.OBS_DISTS):
        for state, probs in dist.items():
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-9), (
                f"OBS_DISTS[{modality}][{state}] sums to {sum(probs)}"
            )

    for policy, rows in efe.B_TRANSITIONS.items():
        for state, row in rows.items():
            total = sum(row[s] for s in efe.TASK_STATES)
            assert math.isclose(total, 1.0, abs_tol=1e-9), (
                f"B_TRANSITIONS[{policy}][{state}] sums to {total}"
            )

    assert math.isclose(sum(efe.D_PRIOR), 1.0, abs_tol=1e-9)


def test_schema_sizes_match_frozen_decision_pomdp():
    """docs/decision-pomdp.md freezes these sizes and says changing them
    needs an explicit written justification. This test is the tripwire."""
    assert len(efe.TASK_STATES) == 4
    assert len(efe.POLICIES) == 6
    assert len(efe.OBS_BINS) == 4
    assert [len(b) for b in efe.OBS_BINS] == [4, 3, 3, 4]
    assert len(efe.C_WEIGHTS) == len(efe.OBS_BINS)
    for weights, bins in zip(efe.C_WEIGHTS, efe.OBS_BINS):
        assert len(weights) == len(bins)


def test_belief_update_matches_hand_computed_bayes():
    """pymdp's infer_states should reproduce a plain Bayes update against
    the same A matrix. Verified by hand for a single-modality-dominant
    case so a change in pymdp's inference defaults doesn't slip past."""
    node = EFEControlNode()
    obs = Observation(
        tool_result="error", confidence="low",
        policy_gate="needs_review", retrieval_quality="poor",
    )
    decision = node.decide(obs)

    idxs = obs.as_indices()
    unnorm = []
    for s_idx, s in enumerate(efe.TASK_STATES):
        p = efe.D_PRIOR[s_idx]
        for m, dist in enumerate(efe.OBS_DISTS):
            p *= dist[s][idxs[m]]
        unnorm.append(p)
    z = sum(unnorm)
    expected = [u / z for u in unnorm]

    for state, want in zip(efe.TASK_STATES, expected):
        assert decision.belief[state] == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize(
    "obs_kwargs, expected_policy",
    [
        (dict(tool_result="success", confidence="high", policy_gate="allow",
              retrieval_quality="n/a"), "continue"),
        (dict(tool_result="partial", confidence="medium", policy_gate="needs_review",
              retrieval_quality="adequate"), "gather_info"),
        (dict(tool_result="error", confidence="low", policy_gate="needs_review",
              retrieval_quality="poor"), "escalate_to_human"),
    ],
)
def test_decides_sensibly_on_the_three_canonical_scenarios(obs_kwargs, expected_policy):
    """The three hand-checked scenarios the control node was originally
    validated against: clean success -> continue, ambiguous -> gather,
    clearly stuck -> escalate."""
    node = EFEControlNode()
    assert node.decide(Observation(**obs_kwargs)).policy == expected_policy


def test_unnecessary_action_carries_a_cost():
    """Regression: known-issues #3. B_TRANSITIONS originally left
    task_solvable_now unchanged under EVERY policy, so with belief
    concentrated there, EFE was numerically indifferent between
    `continue` and `gather_info` (marginals ~0.197 each) -- there was no
    signal preferring 'do nothing' over 'do something pointless'."""
    solvable = efe.TASK_STATES.index("task_solvable_now")

    # `continue` must be the unique zero-risk policy from a solved state.
    assert efe.B_TRANSITIONS["continue"]["task_solvable_now"]["task_solvable_now"] == 1.0
    for policy in ("retry", "call_tool", "gather_info"):
        row = efe.B_TRANSITIONS[policy]["task_solvable_now"]
        assert row["task_solvable_now"] < 1.0, (
            f"{policy} has no cost from an already-solved state -- EFE will be "
            "indifferent between it and `continue` (known-issues #3)"
        )

    # And end-to-end: from a near-certain solved belief, `continue` wins.
    node = EFEControlNode(prior=[0.97, 0.02, 0.005, 0.005])
    decision = node.decide(Observation(
        tool_result="success", confidence="high",
        policy_gate="allow", retrieval_quality="n/a",
    ))
    assert decision.policy == "continue"
    assert decision.action_marginals["continue"] > decision.action_marginals["gather_info"]
    assert decision.belief["task_solvable_now"] > 0.9
    assert solvable == 0  # ordering assumption the prior above relies on


def test_decision_carries_full_epistemic_pragmatic_decomposition():
    """Stage 5's interpretability analysis reads these; they're logged
    from day one specifically so that stage isn't a retrofit."""
    node = EFEControlNode()
    decision = node.decide(Observation(
        tool_result="partial", confidence="medium",
        policy_gate="allow", retrieval_quality="adequate",
    ))

    for field in (decision.action_marginals, decision.epistemic_value, decision.pragmatic_value):
        assert set(field) == set(efe.POLICIES)
        assert all(isinstance(v, float) for v in field.values())

    assert sum(decision.action_marginals.values()) == pytest.approx(1.0, abs=1e-6)
    # Info-seeking must have strictly positive expected information gain
    # when belief is genuinely uncertain -- a zero here means the
    # epistemic term isn't being computed at all.
    assert decision.epistemic_value["gather_info"] > 0.0


def test_gather_info_has_more_epistemic_value_than_continue_when_uncertain():
    node = EFEControlNode(prior=[0.25, 0.25, 0.25, 0.25])
    decision = node.decide(Observation(
        tool_result="partial", confidence="medium",
        policy_gate="allow", retrieval_quality="adequate",
    ))
    assert decision.epistemic_value["gather_info"] > decision.epistemic_value["continue"]


def test_valid_policies_restriction_is_honored():
    """The runtime must be able to take a policy off the table (e.g. no
    escalation path available) without the controller ignoring it."""
    node = EFEControlNode()
    allowed = ["continue", "retry"]
    decision = node.decide(
        Observation(tool_result="error", confidence="low",
                    policy_gate="needs_review", retrieval_quality="poor"),
        valid_policies=allowed,
    )
    assert decision.policy in allowed


def test_belief_persists_across_turns_and_concentrates():
    """Belief is threaded by the caller across turns; repeated consistent
    evidence should sharpen it rather than reset each decision."""
    node = EFEControlNode()
    obs = Observation(tool_result="success", confidence="high",
                      policy_gate="allow", retrieval_quality="good")
    first = node.decide(obs).belief["task_solvable_now"]
    second = node.decide(obs).belief["task_solvable_now"]
    assert second > first
    assert second > 0.9


def test_reset_restores_the_prior():
    node = EFEControlNode()
    node.decide(Observation(tool_result="error", confidence="low",
                            policy_gate="deny", retrieval_quality="poor"))
    node.reset()
    assert node.belief == pytest.approx(list(efe.D_PRIOR))
