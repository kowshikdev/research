"""Registers all five Stage 2/3 control-node agents with tau2's global
registry as "community" agents (src/tau2/agent/README.md's second
registration path) -- calling `registry.register_agent_factory` from
our own code instead of editing external/tau2-bench/src/tau2/registry.py.
Call `register()` once before any `tau2.run.*` call.
"""
from .baseline_agents import (
    create_heuristic_agent,
    create_react_agent,
    create_router_agent,
    create_voi_agent,
)
from .efe_agent import create_efe_agent

_registered = False


def register():
    global _registered
    if _registered:
        return
    from tau2.registry import registry

    registry.register_agent_factory(create_efe_agent, "efe_agent")
    registry.register_agent_factory(create_heuristic_agent, "heuristic_agent")
    registry.register_agent_factory(create_router_agent, "router_agent")
    registry.register_agent_factory(create_voi_agent, "voi_agent")
    registry.register_agent_factory(create_react_agent, "react_agent")
    _registered = True
