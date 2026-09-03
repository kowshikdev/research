"""Tests for the LangGraph orchestration layer (src/aif_orchestrator/graph.py).

The interrupt/resume test is the important one here: `escalate_to_human`
genuinely pausing the graph is the single piece of infrastructure the
whole project leans on (docs/langgraph-integration.md), and "the graph
really stopped" is only checkable by inspecting live graph state, not by
reading the code.

All of these run against mock_agent_step -- no LLM, no network, no cost.
"""
import json

import pytest
from langgraph.types import Command

from aif_orchestrator import graph as graph_mod
from aif_orchestrator.efe_controller import POLICIES, EFEControlNode


def _invoke(app, task_id, thread_id, max_turns=5):
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(
        {"task_id": task_id, "turn": 0, "max_turns": max_turns, "decision_trace": []},
        config=config,
    )
    return result, config


# --- routing ---------------------------------------------------------------

@pytest.mark.parametrize(
    "policy, turn, max_turns, expected",
    [
        ("escalate_to_human", 0, 5, "human_review"),
        ("continue", 0, 5, "finish"),
        ("gather_info", 0, 5, "agent_step"),
        ("retry", 0, 5, "agent_step"),
        ("call_tool", 0, 5, "agent_step"),
        ("hand_off_to_agent", 0, 5, "agent_step"),
        # turn budget exhausted: loop back would overrun, so stop instead
        ("gather_info", 4, 5, "finish"),
        # escalation still wins over the turn budget -- a human is needed
        # regardless of how many turns are left
        ("escalate_to_human", 4, 5, "human_review"),
    ],
)
def test_route_from_decision(policy, turn, max_turns, expected):
    state = {"last_policy": policy, "turn": turn, "max_turns": max_turns}
    assert graph_mod.route_from_decision(state) == expected


def test_every_policy_has_a_defined_route():
    """A new policy added to the schema without a routing rule would
    silently fall through to the agent_step loop."""
    for policy in POLICIES:
        state = {"last_policy": policy, "turn": 0, "max_turns": 5}
        assert graph_mod.route_from_decision(state) in {"human_review", "finish", "agent_step"}


# --- end-to-end graph behavior ---------------------------------------------

def test_normal_task_resolves_without_escalation(tmp_path):
    app = graph_mod.build_graph(
        control_step=graph_mod.make_control_step(log_path=tmp_path / "log.jsonl")
    )
    result, config = _invoke(app, "task-A", "t-normal")

    assert result.get("done") is True
    assert app.get_state(config).next == (), "graph should have finished, not paused"
    assert result["decision_trace"], "no decisions were recorded"
    assert result["decision_trace"][-1]["chosen_policy"] == "continue"


def test_escalation_genuinely_pauses_the_graph_and_resumes(tmp_path):
    """The core Stage 1 claim. `state.next` being non-empty is the real,
    checkable proof the run is suspended mid-execution rather than
    finished -- not a simulated pause."""
    app = graph_mod.build_graph(
        control_step=graph_mod.make_control_step(log_path=tmp_path / "log.jsonl")
    )
    result, config = _invoke(app, "forced-bad-B", "t-escalate")

    state = app.get_state(config)
    assert state.next, "graph did not pause on escalate_to_human"
    assert result.get("done") is not True

    payload = state.tasks[0].interrupts[0].value
    assert payload["task_id"] == "forced-bad-B"
    assert set(payload["belief"]) == {
        "task_solvable_now", "needs_more_info", "needs_human", "likely_to_fail",
    }

    resumed = app.invoke(Command(resume="Approved by a human."), config=config)
    assert resumed.get("done") is True
    assert resumed["human_feedback"] == "Approved by a human."
    assert app.get_state(config).next == ()


def test_belief_is_threaded_across_turns(tmp_path):
    """Belief must accumulate across turns rather than resetting each
    decision. Driven by a deterministic ambiguous-then-consistent agent
    step -- the stochastic mock's own trajectory can legitimately move
    belief in either direction, which would make a direction assertion
    flaky rather than meaningful."""
    observations = iter([
        dict(tool_result="partial", confidence="medium", policy_gate="allow", retrieval_quality="adequate"),
        dict(tool_result="success", confidence="high", policy_gate="allow", retrieval_quality="good"),
        dict(tool_result="success", confidence="high", policy_gate="allow", retrieval_quality="good"),
    ])
    last_seen = {}

    def scripted_agent_step(state):
        last_seen["obs"] = next(observations, last_seen.get("obs"))
        return {"observation": last_seen["obs"]}

    app = graph_mod.build_graph(
        agent_step=scripted_agent_step,
        control_step=graph_mod.make_control_step(log_path=tmp_path / "log.jsonl"),
    )
    result, _ = _invoke(app, "task-multi", "t-belief")

    trace = result["decision_trace"]
    assert len(trace) >= 2, "expected more than one turn before resolving"
    first = trace[0]["belief"]["task_solvable_now"]
    last = trace[-1]["belief"]["task_solvable_now"]
    assert last > first, "belief did not accumulate across turns"


def test_turn_budget_is_enforced(tmp_path):
    """A task that never resolves must stop at max_turns rather than
    looping forever."""
    app = graph_mod.build_graph(
        agent_step=lambda state: {
            "observation": dict(tool_result="partial", confidence="medium",
                                policy_gate="allow", retrieval_quality="adequate")
        },
        control_step=graph_mod.make_control_step(log_path=tmp_path / "log.jsonl"),
    )
    result, config = _invoke(app, "never-resolves", "t-budget", max_turns=3)
    assert len(result["decision_trace"]) <= 3
    assert result.get("done") is True or app.get_state(config).next


# --- pluggability ----------------------------------------------------------

def test_every_controller_runs_in_the_real_graph(controller_cls, tmp_path):
    """docs/baselines-design.md claims all five are drop-in replacements
    inside the real LangGraph scaffold; this runs each one through it."""
    app = graph_mod.build_graph(
        control_step=graph_mod.make_control_step(controller_cls, log_path=tmp_path / "log.jsonl")
    )
    result, _ = _invoke(app, "task-A", f"t-{controller_cls.__name__}")
    assert result["decision_trace"]
    for record in result["decision_trace"]:
        assert record["chosen_policy"] in POLICIES


def test_control_step_logs_the_full_decision_record(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    app = graph_mod.build_graph(control_step=graph_mod.make_control_step(log_path=log_path))
    _invoke(app, "task-A", "t-log")

    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    assert records
    for record in records:
        assert set(record) >= {
            "task_id", "turn", "controller", "observation", "chosen_policy",
            "belief", "action_marginals", "epistemic_value", "pragmatic_value",
        }
        assert record["controller"] == EFEControlNode.name


def test_decision_log_is_append_only(tmp_path):
    """Documented behavior (docs/langgraph-integration.md): re-running
    grows the log rather than clobbering history."""
    log_path = tmp_path / "decisions.jsonl"
    app = graph_mod.build_graph(control_step=graph_mod.make_control_step(log_path=log_path))
    _invoke(app, "task-A", "t-append-1")
    first = len(log_path.read_text().splitlines())
    _invoke(app, "task-A", "t-append-2")
    assert len(log_path.read_text().splitlines()) > first


# --- mock agent ------------------------------------------------------------

def test_mock_agent_is_deterministic_per_task_and_turn():
    state = {"task_id": "task-X", "turn": 1, "last_policy": "gather_info"}
    assert graph_mod.mock_agent_step(dict(state)) == graph_mod.mock_agent_step(dict(state))


def test_forced_bad_tasks_always_produce_a_stuck_observation():
    for turn in range(4):
        obs = graph_mod.mock_agent_step({"task_id": "forced-bad-x", "turn": turn})["observation"]
        assert obs["tool_result"] == "error"
        assert obs["confidence"] == "low"


def test_mock_agent_turn_zero_is_not_read_as_stuck():
    """Regression for the same class of bug as known-issues #8: turn 0
    must not look like evidence a human is needed."""
    obs = graph_mod.mock_agent_step({"task_id": "task-A", "turn": 0})["observation"]
    assert obs["tool_result"] != "no_tool_called"
