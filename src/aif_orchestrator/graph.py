"""LangGraph wiring for the EFE control node -- the actual Stage 1
deliverable from RESEARCH_PLAN.md.

Uses a MOCK agent-step node (no real LLM/tool calls) so the control-loop
plumbing -- observation derivation, EFE decision, routing, and the
escalate_to_human interrupt()/checkpointer pause -- can be proven correct
without API credentials. Swapping the mock agent step for a real
tool-calling LLM agent is the next piece of work and is documented,
not implemented, in context/TODOS.md -- it needs API keys this
environment doesn't have.

Run: .venv/bin/python -m aif_orchestrator.graph
"""
import json
import random
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from .efe_controller import EFEControlNode, Observation, POLICIES

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
DECISION_LOG_PATH = RESULTS_DIR / "stage1_decision_log.jsonl"
STAGE2_DECISION_LOG_PATH = RESULTS_DIR / "stage2_decision_log.jsonl"


class OrchestratorState(TypedDict, total=False):
    task_id: str
    turn: int
    max_turns: int
    observation: dict          # last observation dict (tool_result/confidence/policy_gate/retrieval_quality)
    belief: dict                # task_state belief, carried across turns
    last_policy: str
    done: bool
    human_feedback: str
    decision_trace: list        # accumulated per-turn decision records for this task
    task_prompt: str            # real-agent only: the user task seeding the conversation
    messages: list               # real-agent only: chat history carried across turns


def mock_agent_step(state: OrchestratorState) -> OrchestratorState:
    """Stand-in for a real tool-calling LLM step. Deterministically drifts
    toward `success` over a few turns, seeded by task_id + turn so runs
    are reproducible -- exists purely to exercise the graph, not to model
    anything real."""
    rng = random.Random(f"{state['task_id']}-{state['turn']}")
    turn = state["turn"]

    if state["task_id"].startswith("forced-bad"):
        # deterministic bad trajectory, for reliably demonstrating the
        # escalate_to_human -> interrupt() flow rather than depending on
        # the stochastic mock agent getting unlucky.
        return {"observation": dict(tool_result="error", confidence="low", policy_gate="needs_review", retrieval_quality="poor")}

    if turn == 0:
        # first turn: task starts ambiguous
        obs = dict(tool_result="partial", confidence="medium", policy_gate="allow", retrieval_quality="n/a")
    elif state.get("last_policy") in ("gather_info", "call_tool", "retry"):
        # info-seeking actions tend to resolve things over a couple of turns
        if rng.random() < 0.7:
            obs = dict(tool_result="success", confidence="high", policy_gate="allow", retrieval_quality="good")
        else:
            obs = dict(tool_result="partial", confidence="medium", policy_gate="allow", retrieval_quality="adequate")
    else:
        obs = dict(tool_result="success", confidence="high", policy_gate="allow", retrieval_quality="n/a")

    return {"observation": obs}


def llm_agent_step(state: OrchestratorState) -> OrchestratorState:
    """Real tool-calling LLM agent step -- Stage 1c. Same contract as
    mock_agent_step (returns an `observation` dict), plus persists chat
    history across turns via `messages`."""
    from .llm_agent import POLICY_STEER, real_agent_step, seed_messages

    messages = state.get("messages") or seed_messages(state["task_prompt"])
    steer = POLICY_STEER.get(state.get("last_policy"))
    if steer:
        messages = messages + [{"role": "user", "content": steer}]
    messages, observation = real_agent_step(messages)
    return {"observation": observation, "messages": messages}


def make_control_step(control_node_cls=EFEControlNode, log_path=DECISION_LOG_PATH):
    """Builds a control-step node for any controller sharing
    EFEControlNode's interface (__init__(prior=None), decide(observation,
    valid_policies=None) -> Decision) -- Stage 2 baselines
    (src/aif_orchestrator/baselines/) plug in here the same way
    llm_agent_step swaps in for mock_agent_step."""

    def control_step(state: OrchestratorState) -> OrchestratorState:
        node = control_node_cls(prior=list(state["belief"].values()) if state.get("belief") else None)
        obs = Observation(**state["observation"])
        decision = node.decide(obs)

        record = {
            "task_id": state["task_id"],
            "turn": state["turn"],
            "controller": getattr(control_node_cls, "name", control_node_cls.__name__),
            "observation": state["observation"],
            "chosen_policy": decision.policy,
            "belief": decision.belief,
            "action_marginals": decision.action_marginals,
            "epistemic_value": decision.epistemic_value,
            "pragmatic_value": decision.pragmatic_value,
        }
        trace = state.get("decision_trace", []) + [record]

        RESULTS_DIR.mkdir(exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "belief": decision.belief,
            "last_policy": decision.policy,
            "decision_trace": trace,
        }

    return control_step


efe_control_step = make_control_step()


def route_from_decision(state: OrchestratorState) -> str:
    policy = state["last_policy"]
    if policy == "escalate_to_human":
        return "human_review"
    if policy == "continue":
        return "finish"
    if state["turn"] + 1 >= state["max_turns"]:
        return "finish"  # forced stop -- real system would escalate here too
    return "agent_step"


def human_review_node(state: OrchestratorState) -> OrchestratorState:
    """The one node that pauses execution for a real human via
    interrupt() + the graph's checkpointer -- this is the concrete proof
    that escalate_to_human isn't just a label, it actually stops the
    graph and waits."""
    feedback = interrupt({
        "reason": "EFE control node chose escalate_to_human",
        "task_id": state["task_id"],
        "turn": state["turn"],
        "belief": state["belief"],
    })
    return {"human_feedback": feedback, "done": True}


def finish_node(state: OrchestratorState) -> OrchestratorState:
    return {"done": True}


def increment_turn(state: OrchestratorState) -> OrchestratorState:
    return {"turn": state["turn"] + 1}


def build_graph(agent_step=mock_agent_step, control_step=efe_control_step):
    graph = StateGraph(OrchestratorState)
    graph.add_node("agent_step", agent_step)
    graph.add_node("efe_control", control_step)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finish", finish_node)
    graph.add_node("bump_turn", increment_turn)

    graph.add_edge(START, "agent_step")
    graph.add_edge("agent_step", "efe_control")
    graph.add_conditional_edges("efe_control", route_from_decision, {
        "human_review": "human_review",
        "finish": "finish",
        "agent_step": "bump_turn",
    })
    graph.add_edge("bump_turn", "agent_step")
    graph.add_edge("human_review", END)
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=MemorySaver())


def run_demo():
    app = build_graph()

    print("=== Task A: expected to resolve without escalation ===")
    config_a = {"configurable": {"thread_id": "task-A"}}
    result_a = app.invoke(
        {"task_id": "task-A", "turn": 0, "max_turns": 5, "decision_trace": []},
        config=config_a,
    )
    for rec in result_a["decision_trace"]:
        print(f"  turn {rec['turn']}: obs={rec['observation']} -> {rec['chosen_policy']}")
    print(f"  final belief: {result_a['belief']}")
    print(f"  done={result_a.get('done', False)}")
    print()

    print("=== Task B: forced toward escalation (repeated failures) ===")
    config_b = {"configurable": {"thread_id": "task-B"}}

    result_b = app.invoke(
        {"task_id": "forced-bad-B", "turn": 0, "max_turns": 5, "decision_trace": []},
        config=config_b,
    )
    for rec in result_b["decision_trace"]:
        print(f"  turn {rec['turn']}: obs={rec['observation']} -> {rec['chosen_policy']}")

    state_b = app.get_state(config_b)
    if state_b.next:
        print("  GRAPH PAUSED (interrupt) -- awaiting human review.")
        print(f"  interrupt payload: {state_b.tasks[0].interrupts[0].value}")
        print("  Resuming with human feedback...")
        result_b_resumed = app.invoke(Command(resume="Approved: task requires manual data correction."), config=config_b)
        print(f"  resumed state: done={result_b_resumed.get('done')}, human_feedback={result_b_resumed.get('human_feedback')!r}")
    else:
        print(f"  done={result_b.get('done', False)} (did not escalate this run -- mock agent is stochastic)")

    print(f"\nFull decision log: {DECISION_LOG_PATH}")


def run_llm_demo():
    """Same shape as run_demo() but with a real tool-calling LLM agent
    step instead of the mock -- Stage 1c."""
    app = build_graph(agent_step=llm_agent_step)

    print("=== Task A (real LLM): order that exists, should resolve cleanly ===")
    config_a = {"configurable": {"thread_id": "llm-task-A"}}
    result_a = app.invoke(
        {
            "task_id": "llm-task-A",
            "task_prompt": "What's the status of order 1001?",
            "turn": 0,
            "max_turns": 5,
            "decision_trace": [],
        },
        config=config_a,
    )
    for rec in result_a["decision_trace"]:
        print(f"  turn {rec['turn']}: obs={rec['observation']} -> {rec['chosen_policy']}")
    print(f"  final belief: {result_a['belief']}")
    print(f"  done={result_a.get('done', False)}")
    print()

    print("=== Task B (real LLM): nonexistent order, expect escalation ===")
    config_b = {"configurable": {"thread_id": "llm-task-B"}}
    result_b = app.invoke(
        {
            "task_id": "llm-task-B",
            "task_prompt": "What's the status of order 9999?",
            "turn": 0,
            "max_turns": 5,
            "decision_trace": [],
        },
        config=config_b,
    )
    for rec in result_b["decision_trace"]:
        print(f"  turn {rec['turn']}: obs={rec['observation']} -> {rec['chosen_policy']}")

    state_b = app.get_state(config_b)
    if state_b.next:
        print("  GRAPH PAUSED (interrupt) -- awaiting human review.")
        print(f"  interrupt payload: {state_b.tasks[0].interrupts[0].value}")
        print("  Resuming with human feedback...")
        result_b_resumed = app.invoke(Command(resume="Approved: order not found, ask customer to double-check the ID."), config=config_b)
        print(f"  resumed state: done={result_b_resumed.get('done')}, human_feedback={result_b_resumed.get('human_feedback')!r}")
    else:
        print(f"  done={result_b.get('done', False)} (did not escalate this run)")

    print(f"\nFull decision log: {DECISION_LOG_PATH}")


def run_stage2_demo():
    """Proves each Stage 2 baseline is genuinely pluggable into the same
    LangGraph scaffold EFE runs in -- not just comparable in the fast
    mock_agent_step-driven loop `baselines/run_stage2_eval.py` uses for
    the statistical comparison. Runs Task A/B (mock agent) once per
    controller."""
    from .baselines.heuristic import HeuristicControlNode
    from .baselines.react import ReActControlNode
    from .baselines.router import LearnedRouterControlNode
    from .baselines.voi import VOIControlNode

    controllers = {
        "efe": EFEControlNode,
        "heuristic": HeuristicControlNode,
        "learned_router": LearnedRouterControlNode,
        "voi": VOIControlNode,
        "react": ReActControlNode,
    }

    for name, cls in controllers.items():
        print(f"=== controller={name} ===")
        app = build_graph(control_step=make_control_step(cls, log_path=STAGE2_DECISION_LOG_PATH))

        for label, task_id in [("Task A", f"stage2-{name}-A"), ("Task B", f"forced-bad-stage2-{name}-B")]:
            config = {"configurable": {"thread_id": f"{name}-{label}"}}
            result = app.invoke(
                {"task_id": task_id, "turn": 0, "max_turns": 5, "decision_trace": []},
                config=config,
            )
            policies = [rec["chosen_policy"] for rec in result["decision_trace"]]
            state = app.get_state(config)
            paused = bool(state.next)
            print(f"  {label} ({task_id}): policies={policies} paused_for_human={paused}")
            if paused:
                app.invoke(Command(resume="Approved."), config=config)
        print()

    print(f"Full decision log: {STAGE2_DECISION_LOG_PATH}")


if __name__ == "__main__":
    import sys

    if "--stage2" in sys.argv:
        run_stage2_demo()
    elif "--llm" in sys.argv:
        run_llm_demo()
    else:
        run_demo()
