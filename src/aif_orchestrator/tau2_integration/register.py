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

    _patch_nl_assertions_judge_model()
    _registered = True


def _patch_nl_assertions_judge_model() -> None:
    """tau2's NL_ASSERTION reward grading (part of every task's reward,
    not something a run config can opt out of) is hardcoded to
    "gpt-4.1-2025-04-14" via tau2.config.DEFAULT_LLM_NL_ASSERTIONS /
    DEFAULT_LLM_NL_ASSERTIONS_ARGS, which tau2.evaluator.
    evaluator_nl_assertions imports by value at module load time (`from
    tau2.config import DEFAULT_LLM_NL_ASSERTIONS` binds a NEW name in
    that module's own namespace, decoupled from tau2.config's) -- so
    patching tau2.config's copy after import does nothing, this module's
    copy must be patched directly. This call needs real OpenAI
    credentials we never configured, and was the actual source of every
    "Missing credentials" failure chased across the OpenRouter/Groq/
    Vertex provider switches -- none of those were ever this call's
    fault; the agent/user model config was correct the whole time.
    Point it at whichever provider vertex_auth.py has already resolved.
    """
    import re

    import tau2.evaluator.evaluator_nl_assertions as nl_eval
    from .vertex_auth import start_token_refresher

    vertex_kwargs = start_token_refresher()
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = "google/gemini-2.5-flash"
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {"temperature": 0.0, **vertex_kwargs}

    # The judge's caller does json.loads(assistant_message.content)
    # directly with no error handling, but Gemini (unlike gpt-4.1, the
    # default this was written against) wraps its JSON answer in a
    # ```json ... ``` markdown fence even when explicitly asked for raw
    # JSON, breaking that parse outright (confirmed: reproduced the
    # exact response). generate is imported by value into this module
    # (same decoupling as DEFAULT_LLM_NL_ASSERTIONS above), so patch it
    # here to strip a wrapping fence before returning.
    _orig_generate = nl_eval.generate
    _fence_re = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)
    # Gemini also occasionally emits a trailing comma before a closing
    # `}`/`]` (valid in JSON5, not in strict JSON) -- confirmed: a real
    # task hit "Expecting property name enclosed in double quotes" at
    # the exact position of such a comma. Strip it too.
    _trailing_comma_re = re.compile(r",(\s*[}\]])")

    def _generate_stripped(*args, **kwargs):
        from litellm.exceptions import AuthenticationError as LiteLLMAuthenticationError

        from .vertex_auth import force_refresh, overlay_live_kwargs
        kwargs = overlay_live_kwargs(kwargs)
        try:
            resp = _orig_generate(*args, **kwargs)
        except LiteLLMAuthenticationError:
            # See efe_agent.py's generate wrapper for the confirmed
            # symptom/reasoning: a transient Vertex-side/OAuth-refresh
            # hiccup, not a genuinely bad token -- one forced re-fetch
            # usually clears it.
            force_refresh()
            resp = _orig_generate(*args, **overlay_live_kwargs(kwargs))
        if resp.content:
            content = resp.content.strip()
            m = _fence_re.match(content)
            if m:
                content = m.group(1)
            resp.content = _trailing_comma_re.sub(r"\1", content)
        return resp

    nl_eval.generate = _generate_stripped

    # Belt-and-suspenders: the fence/comma cleanup above doesn't cover
    # every way Gemini can produce syntactically invalid JSON (confirmed
    # empirically -- the exact same task succeeded on one run and hit a
    # *different* JSONDecodeError position on another, so this is
    # content-dependent LLM output variance, not a single fixed pattern
    # worth regexing around). json.loads(assistant_message.content)
    # happens inside the evaluator, not something our generate patch
    # can reach -- wrap the classmethod itself instead: on a parse
    # failure, just re-run the whole judge call (a fresh generation is
    # very likely to come back parseable) rather than let one grader
    # glitch fail the entire task's reward.
    import json

    _orig_evaluate = nl_eval.NLAssertionsEvaluator.evaluate_nl_assertions.__func__

    def _evaluate_impl(cls, trajectory, nl_assertions):
        for attempt in range(3):
            try:
                return _orig_evaluate(cls, trajectory, nl_assertions)
            except json.JSONDecodeError:
                if attempt == 2:
                    raise

    nl_eval.NLAssertionsEvaluator.evaluate_nl_assertions = classmethod(_evaluate_impl)

    _patch_user_simulator_live_token()


def _patch_user_simulator_live_token() -> None:
    """Same staleness problem as the judge model, on the user-simulator
    side: tau2.user.user_simulator imports `generate` by value too, and
    its self.llm_args is deepcopied once per UserSimulator construction
    from a TextRunConfig.llm_args_user that was frozen at the start of
    a whole domain/agent run -- so its api_key can go stale mid-run the
    same way the agent side's did (see efe_agent.py's generate wrapper
    for the confirmed symptom: consecutive real tasks hitting 401
    ACCESS_TOKEN_TYPE_UNSUPPORTED well before token expiry, because the
    frozen config copy was stale, not each token individually bad).
    """
    import tau2.user.user_simulator as user_sim
    from .vertex_auth import force_refresh, overlay_live_kwargs

    _orig_generate = user_sim.generate

    def _generate_live(*args, **kwargs):
        from litellm.exceptions import AuthenticationError as LiteLLMAuthenticationError

        try:
            return _orig_generate(*args, **overlay_live_kwargs(kwargs))
        except LiteLLMAuthenticationError:
            force_refresh()
            return _orig_generate(*args, **overlay_live_kwargs(kwargs))

    user_sim.generate = _generate_live
