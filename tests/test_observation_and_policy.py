"""Tests for observation derivation and the OPA policy gate
(src/aif_orchestrator/opa_policy.py, policies/policy_gate.rego).

The fail-safe test matters most: opa_policy deliberately falls back to
"needs_review", NOT "allow", when OPA is missing or errors. A broken
policy setup must make the system more conservative, never silently
permissive -- and "we defaulted to allow when the policy engine was
down" is exactly the kind of thing that only gets noticed after it
matters.
"""
import shutil

import pytest

from aif_orchestrator import opa_policy
from aif_orchestrator.efe_controller import POLICY_GATE_BINS, Observation, OBS_BINS

OPA_INSTALLED = shutil.which("opa") is not None
requires_opa = pytest.mark.skipif(not OPA_INSTALLED, reason="opa CLI not installed")


def test_policy_file_exists_and_is_the_single_source_of_policy():
    """The .rego file is meant to be the place policy lives -- if it
    went missing, evaluate_policy_gate would silently fail-safe to
    needs_review on every turn instead of erroring loudly."""
    assert opa_policy.POLICY_PATH.exists(), f"missing policy file: {opa_policy.POLICY_PATH}"
    text = opa_policy.POLICY_PATH.read_text()
    assert "package aif.policy_gate" in text
    assert opa_policy.QUERY.startswith("data.aif.policy_gate")


def test_fails_safe_to_needs_review_when_opa_is_unavailable(monkeypatch):
    """Simulates OPA not being installed -- must NOT return 'allow'."""
    def _boom(*args, **kwargs):
        raise FileNotFoundError("opa not found")

    monkeypatch.setattr(opa_policy.subprocess, "run", _boom)
    assert opa_policy.evaluate_policy_gate({"last_tool_result": "success"}) == "needs_review"


def test_fails_safe_on_nonzero_exit_and_on_garbage_output(monkeypatch):
    class _Result:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    monkeypatch.setattr(opa_policy.subprocess, "run",
                        lambda *a, **k: _Result(1, ""))
    assert opa_policy.evaluate_policy_gate({}) == "needs_review"

    monkeypatch.setattr(opa_policy.subprocess, "run",
                        lambda *a, **k: _Result(0, "not json"))
    assert opa_policy.evaluate_policy_gate({}) == "needs_review"


def test_rejects_a_decision_outside_the_declared_bins(monkeypatch):
    """A policy edit that returns an unexpected string must not leak an
    invalid observation bin into the POMDP -- Observation.as_indices()
    would raise deep inside a run instead."""
    class _Result:
        returncode = 0
        stdout = '{"result":[{"expressions":[{"value":"sudo-allow"}]}]}'

    monkeypatch.setattr(opa_policy.subprocess, "run", lambda *a, **k: _Result())
    assert opa_policy.evaluate_policy_gate({}) == "needs_review"


def test_returns_a_valid_bin_for_any_input(monkeypatch):
    class _Result:
        returncode = 0
        stdout = '{"result":[{"expressions":[{"value":"deny"}]}]}'

    monkeypatch.setattr(opa_policy.subprocess, "run", lambda *a, **k: _Result())
    assert opa_policy.evaluate_policy_gate({"same_tool_call_count": 5}) in POLICY_GATE_BINS


@requires_opa
@pytest.mark.parametrize(
    "input_doc, expected",
    [
        ({"same_tool_call_count": 0, "last_tool_result": "success"}, "allow"),
        ({"same_tool_call_count": 1, "last_tool_result": "error"}, "needs_review"),
        ({"same_tool_call_count": 3, "last_tool_result": "success"}, "deny"),
        # the retry-loop circuit breaker outranks the error flag
        ({"same_tool_call_count": 4, "last_tool_result": "error"}, "deny"),
    ],
)
def test_real_opa_policy_decisions(input_doc, expected):
    """Runs the actual .rego policy through the real `opa` binary when
    it's available (skipped otherwise, so a fresh clone stays green)."""
    assert opa_policy.evaluate_policy_gate(input_doc) == expected


# --- observation schema -----------------------------------------------------

def test_observation_indices_match_the_declared_bins():
    obs = Observation(tool_result="partial", confidence="low",
                      policy_gate="deny", retrieval_quality="good")
    indices = obs.as_indices()
    assert len(indices) == len(OBS_BINS)
    for idx, bins in zip(indices, OBS_BINS):
        assert 0 <= idx < len(bins)
    assert indices == [2, 0, 1, 2]


def test_observation_rejects_an_unknown_bin():
    """An unknown bin must fail loudly at the boundary, not silently map
    to index 0 and quietly corrupt every downstream belief update."""
    with pytest.raises(ValueError):
        Observation(tool_result="exploded", confidence="high",
                    policy_gate="allow", retrieval_quality="n/a").as_indices()


def test_mock_agent_observations_are_all_schema_valid():
    """Every observation the mock agent can emit must be a legal
    observation -- otherwise a rare branch blows up mid-run."""
    from aif_orchestrator.graph import mock_agent_step

    for task_id in ("task-A", "forced-bad-A"):
        for turn in range(4):
            for last_policy in (None, "gather_info", "call_tool", "retry", "continue"):
                state = {"task_id": task_id, "turn": turn}
                if last_policy:
                    state["last_policy"] = last_policy
                obs = Observation(**mock_agent_step(state)["observation"])
                assert len(obs.as_indices()) == 4
