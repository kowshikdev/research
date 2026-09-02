"""Tests for the Stage 3 cost estimator (scripts/estimate_stage3_cost.py).

This script's output is meant to back a real spending decision, so the
things worth testing are the ones that would make it lie: token
accounting that ignores an input, cost that doesn't scale with the sweep
size, or call shapes that drift out of sync with what efe_agent.py
actually requests.
"""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "estimate_stage3_cost.py"


def _load():
    spec = importlib.util.spec_from_file_location("estimate_stage3_cost", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def estimator():
    return _load()


def test_call_shapes_match_the_max_tokens_the_agent_actually_requests(estimator):
    """If efe_agent.py's max_tokens change, this estimate silently drifts
    from reality -- so pin the numbers the estimator assumes against the
    values in the agent source."""
    agent_src = (Path(__file__).resolve().parent.parent
                 / "src/aif_orchestrator/tau2_integration/efe_agent.py").read_text()

    confidence = estimator.CALL_SHAPES["confidence_verifier"]
    response = estimator.CALL_SHAPES["agent_response"]

    assert f"max_tokens={confidence['max_tokens']}" in agent_src
    assert f"max_tokens={response['max_tokens']}" in agent_src
    assert f'"max_tokens": {confidence["reasoning_max_tokens"]}' in agent_src
    assert f'"max_tokens": {response["reasoning_max_tokens"]}' in agent_src


def test_confidence_call_does_not_carry_the_domain_policy(estimator):
    """_derive_confidence builds its own minimal message list rather than
    resending the (multi-thousand-token) policy -- the estimate must
    reflect that or it overstates cost substantially."""
    assert estimator.CALL_SHAPES["confidence_verifier"]["includes_policy"] is False
    assert estimator.CALL_SHAPES["agent_response"]["includes_policy"] is True


def test_tokens_scale_with_turns_and_policy_size(estimator):
    base_in, base_out = estimator.estimate_tokens_per_task(4, 0.15, 2, 3000)
    more_turns_in, more_turns_out = estimator.estimate_tokens_per_task(8, 0.15, 2, 3000)
    bigger_policy_in, _ = estimator.estimate_tokens_per_task(4, 0.15, 2, 6000)

    assert more_turns_in > base_in and more_turns_out > base_out
    assert bigger_policy_in > base_in, "policy tokens must affect the input estimate"


def test_escalation_rate_affects_the_estimate(estimator):
    never_in, _ = estimator.estimate_tokens_per_task(4, 0.0, 2, 3000)
    always_in, _ = estimator.estimate_tokens_per_task(4, 1.0, 2, 3000)
    assert always_in > never_in


def test_zero_turn_task_still_costs_the_first_turn(estimator):
    """Turn 0 always happens (efe_agent.py's turn-0 special case), so the
    floor is one agent call plus its user-simulator counterpart, not zero."""
    in_tokens, out_tokens = estimator.estimate_tokens_per_task(1, 0.0, 0, 3000)
    assert in_tokens > 0 and out_tokens > 0


def test_pricing_constants_are_positive_and_output_costs_more(estimator):
    assert estimator.PRICE_PER_MTOK_INPUT > 0
    assert estimator.PRICE_PER_MTOK_OUTPUT > estimator.PRICE_PER_MTOK_INPUT


def test_placeholder_task_counts_match_the_documented_total(estimator):
    """run_stage3_eval.py's docstring cites ~278 tasks across the three
    domains; the placeholder split must not drift from that."""
    assert sum(estimator.DEFAULT_TASKS_PER_DOMAIN.values()) == 278
    assert set(estimator.DEFAULT_TASKS_PER_DOMAIN) == set(estimator.DOMAINS)


def test_agents_list_matches_the_registered_tau2_agents(estimator):
    register_src = (Path(__file__).resolve().parent.parent
                    / "src/aif_orchestrator/tau2_integration/register.py").read_text()
    for agent in estimator.AGENTS:
        assert f'"{agent}"' in register_src, f"{agent} is estimated but not registered"


def test_real_task_count_loader_degrades_cleanly_without_tau2(estimator):
    """Must return None (not a partial dict) when tau2 isn't installed,
    so callers fall back cleanly instead of mixing real and placeholder
    counts in one estimate."""
    counts = estimator.try_load_real_task_counts()
    assert counts is None or set(counts) == set(estimator.DOMAINS)
