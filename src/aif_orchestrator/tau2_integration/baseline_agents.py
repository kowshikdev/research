"""tau2 agent wrappers for the Stage 2 baselines (src/aif_orchestrator/
baselines/), for the real EFE-vs-baselines comparison on tau2-bench
(RESEARCH_PLAN.md Stage 3) -- each is just `ControlNodeAgent`
(efe_agent.py) with `control_node_cls` swapped, since all five
controllers share the same interface.
"""
from typing import Generic

from ..baselines.heuristic import HeuristicControlNode
from ..baselines.react import ReActControlNode
from ..baselines.router import LearnedRouterControlNode
from ..baselines.voi import VOIControlNode
from .efe_agent import ControlNodeAgent, ControlNodeAgentStateType


class HeuristicAgent(ControlNodeAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]):
    control_node_cls = HeuristicControlNode


class RouterAgent(ControlNodeAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]):
    control_node_cls = LearnedRouterControlNode


class VOIAgent(ControlNodeAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]):
    control_node_cls = VOIControlNode


class ReActAgent(ControlNodeAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]):
    control_node_cls = ReActControlNode


def _make_factory(agent_cls):
    def factory(tools, domain_policy, **kwargs):
        return agent_cls(
            tools=tools, domain_policy=domain_policy,
            llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
        )
    return factory


create_heuristic_agent = _make_factory(HeuristicAgent)
create_router_agent = _make_factory(RouterAgent)
create_voi_agent = _make_factory(VOIAgent)
create_react_agent = _make_factory(ReActAgent)
