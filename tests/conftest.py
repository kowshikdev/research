"""Shared fixtures. Kept deliberately small -- most tests here construct
their own inputs, because a test whose setup is hidden three files away
is worse than one that repeats two lines.
"""
import pytest

from aif_orchestrator.efe_controller import EFEControlNode, Observation
from aif_orchestrator.baselines.heuristic import HeuristicControlNode
from aif_orchestrator.baselines.react import ReActControlNode
from aif_orchestrator.baselines.router import LearnedRouterControlNode
from aif_orchestrator.baselines.voi import VOIControlNode

# The 5 controllers under test, by the name they're logged/reported under.
ALL_CONTROLLER_CLASSES = {
    "efe_active_inference": EFEControlNode,
    "heuristic": HeuristicControlNode,
    "learned_router": LearnedRouterControlNode,
    "voi_decision_theoretic": VOIControlNode,
    "react": ReActControlNode,
}


@pytest.fixture(params=sorted(ALL_CONTROLLER_CLASSES), ids=sorted(ALL_CONTROLLER_CLASSES))
def controller_cls(request):
    """Every controller, one test run each -- the interface-conformance
    tests in test_baselines.py rely on this to prove all five really are
    interchangeable, rather than asserting it in prose."""
    return ALL_CONTROLLER_CLASSES[request.param]


@pytest.fixture
def clean_observation():
    return Observation(
        tool_result="success", confidence="high",
        policy_gate="allow", retrieval_quality="n/a",
    )


@pytest.fixture
def ambiguous_observation():
    return Observation(
        tool_result="partial", confidence="medium",
        policy_gate="needs_review", retrieval_quality="adequate",
    )


@pytest.fixture
def stuck_observation():
    return Observation(
        tool_result="error", confidence="low",
        policy_gate="needs_review", retrieval_quality="poor",
    )
