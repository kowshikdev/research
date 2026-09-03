"""Registers all five Stage 2/3 control-node agents with tau2's global
registry as "community" agents (src/tau2/agent/README.md's second
registration path) -- calling `registry.register_agent_factory` from
our own code instead of editing external/tau2-bench/src/tau2/registry.py.
Call `register()` once before any `tau2.run.*` call.
"""
from ..baselines.router import LearnedRouterControlNode, model_matched_source_factory
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

    # Without this, RouterAgent's underlying LearnedRouterControlNode
    # lazily self-trains on first use with its class default --
    # 10000 episodes against graph.mock_agent_step, the SAME mismatched
    # stand-in docs/stage2-baselines-results.md Part 1 already flagged as
    # "not a fair test of decision quality". Stage 2 Part 2 fixed exactly
    # this for the offline comparison by training against
    # ModelMatchedEnv instead (100000 episodes -- the observation space is
    # far richer than the mock's handful of scripted combos and undertrains
    # badly at 10000); pre-training here the same way before any
    # RouterAgent is constructed carries that same fix into the real tau2
    # sweep, which otherwise would have silently regressed to the
    # mock-trained router with no error or warning. Real tau2 dynamics are
    # still not what either option trains against -- see
    # docs/stage2-baselines-results.md and docs/known-issues-and-gotchas.md
    # for why "model-matched" is the best currently-available choice, not
    # a claim of being calibrated to the real benchmark. Cheap regardless
    # (~2-3s, no LLM calls) -- always pay it rather than risk the silent
    # mock-trained fallback.
    if LearnedRouterControlNode._q is None:
        LearnedRouterControlNode.train(
            source_factory=model_matched_source_factory,
            num_episodes=100_000, epsilon_start=0.4, epsilon_end=0.02,
        )

    registry.register_agent_factory(create_efe_agent, "efe_agent")
    registry.register_agent_factory(create_heuristic_agent, "heuristic_agent")
    registry.register_agent_factory(create_router_agent, "router_agent")
    registry.register_agent_factory(create_voi_agent, "voi_agent")
    registry.register_agent_factory(create_react_agent, "react_agent")
    _registered = True
