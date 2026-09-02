"""Registers EFEAgent with tau2's global registry as a "community"
agent (src/tau2/agent/README.md's second registration path) -- calling
`registry.register_agent_factory` from our own code instead of editing
external/tau2-bench/src/tau2/registry.py. Call `register()` once before
any `tau2.run.*` call.
"""
from .efe_agent import create_efe_agent

_registered = False


def register():
    global _registered
    if _registered:
        return
    from tau2.registry import registry

    registry.register_agent_factory(create_efe_agent, "efe_agent")
    _registered = True
