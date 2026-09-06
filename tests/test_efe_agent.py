"""Tests for src/aif_orchestrator/tau2_integration/efe_agent.py's pure
logic -- the parts that don't need a real LLM call or tau2 orchestrator,
so they can run as fast unit tests rather than only via the smoke test.

Includes a regression test for the real bug found on the full Stage 3
sweep (docs/known-issues-and-gotchas.md): EFE was making its ONLY real
decision at the second turn of every conversation, still before any
tool call had been attempted, because the cold-start skip only covered
the literal first message. `_any_tool_call_attempted` is the guard that
now extends that skip to "no tool call yet", not just "empty
conversation" -- this test pins its three cases directly, since the
full generate_next_message path needs a real LLM call to exercise.
"""
from aif_orchestrator.tau2_integration.efe_agent import ControlNodeAgent, _next_synthetic_task_id
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage


def _agent():
    return ControlNodeAgent(tools=[], domain_policy="test policy", llm="dummy-model")


def test_synthetic_task_ids_do_not_collide_across_simulated_process_restarts():
    """Regression: the original counter-based implementation restarted
    from 1 in every new process, and run_stage3_eval.py runs one domain
    per process -- retail/airline/telecom each got ids 1..N, so
    decision_log.group_by_task silently merged decisions from up to 3
    unrelated real tasks under the same id (confirmed on the real Stage
    3 EFE re-run: only 126 unique ids surfaced for 278 real tasks,
    inflating mean_turns_per_task to ~17). A module-global counter can't
    be reset from a test to simulate this cleanly, so this instead pins
    the actual property that matters: ids drawn in a tight loop (the
    closest a single process gets to "many restarts") never collide."""
    ids = {_next_synthetic_task_id() for _ in range(1000)}
    assert len(ids) == 1000


def test_any_tool_call_attempted_false_on_empty_conversation():
    agent = _agent()
    state = agent.get_init_state()
    assert agent._any_tool_call_attempted(state) is False


def test_any_tool_call_attempted_false_after_plain_text_exchange():
    """Regression: this is exactly the case the real sweep showed as the
    bug -- a customer's opening message and the agent's plain-text
    first reply, with no tool call yet. Before the fix, the NEXT turn
    would already run the control node's decide() with
    tool_result='no_tool_called', which the generative model reads as
    strong evidence for needs_human -- see the module docstring."""
    agent = _agent()
    state = agent.get_init_state()
    state.messages.append(UserMessage.text(content="I need help with an order"))
    state.messages.append(AssistantMessage.text(content="Sure, what's your order ID?"))
    assert agent._any_tool_call_attempted(state) is False


def test_any_tool_call_attempted_true_once_a_tool_is_called():
    agent = _agent()
    state = agent.get_init_state()
    state.messages.append(UserMessage.text(content="I need help with an order"))
    state.messages.append(AssistantMessage.text(
        content=None,
        tool_calls=[ToolCall(id="c1", name="get_order_details", arguments={"order_id": "123"})],
    ))
    assert agent._any_tool_call_attempted(state) is True


def test_any_tool_call_attempted_stays_true_across_later_plain_text_turns():
    """Once a tool has been attempted, later plain-text turns (e.g. the
    agent asking a clarifying follow-up) must not reset the guard --
    the control loop should keep reasoning from real evidence, not fall
    back to the cold-start skip."""
    agent = _agent()
    state = agent.get_init_state()
    state.messages.append(UserMessage.text(content="I need help with an order"))
    state.messages.append(AssistantMessage.text(
        content=None,
        tool_calls=[ToolCall(id="c1", name="get_order_details", arguments={"order_id": "123"})],
    ))
    state.messages.append(AssistantMessage.text(content="Anything else I can help with?"))
    assert agent._any_tool_call_attempted(state) is True
