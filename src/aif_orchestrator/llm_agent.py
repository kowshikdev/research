"""Real tool-calling LLM agent step -- Stage 1c (context/TODOS.md), the
unblock point after mock_agent_step in graph.py.

Talks to an OpenAI-compatible endpoint (LLM_API_KEY / LLM_MODEL /
LLM_BASE_URL from .env; OpenRouter by default). The order-lookup tool's
backend is a small fixed dict, deterministic like a benchmark harness
environment (tau2-bench etc.) -- what's real here is the LLM genuinely
deciding whether to call the tool, what to answer, and how confident it
is; nothing about its choices is scripted.

Confidence derivation uses the verifier-prompt approach flagged as an
open design decision in context/HANDOFF.md (the other option,
self-consistency sampling, costs N extra calls instead of 1) -- a second,
cheap LLM call rates the transcript so far as low/medium/high.

policy_gate is a real OPA evaluation (opa_policy.py, policy against
policies/policy_gate.rego) -- not an LLM call, a deterministic policy
check against the same-order-lookup retry count and the last tool
result. mock_agent_step still hardcodes "allow": it's a free, non-LLM
stand-in that doesn't track real tool-call history to check a retry
policy against (see its own docstring), so wiring OPA there wouldn't be
checking anything real.
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from .opa_policy import evaluate_policy_gate

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"], timeout=30.0)
    return _client


def _model() -> str:
    return os.environ["LLM_MODEL"]


# Fixed fake order backend. 9999 never exists -- used to reliably exercise
# the escalate_to_human path, the same role "forced-bad" task ids played
# for mock_agent_step.
ORDER_DB = {
    "1001": {"status": "shipped", "eta": "2 days"},
    "1002": {"status": "delayed", "eta": "unknown, backorder"},
}

LOOKUP_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_order",
        "description": "Look up a customer order by ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
}

SYSTEM_PROMPT = (
    "You are a customer support agent. Use the lookup_order tool to check "
    "order status before answering. When you have enough information, "
    "reply in plain text (no tool call) with the final answer for the "
    "customer. If the order can't be found, say so plainly -- don't guess."
)


def _lookup_order(order_id: str) -> dict:
    if order_id in ORDER_DB:
        return {"found": True, **ORDER_DB[order_id]}
    return {"found": False, "error": f"no order with id {order_id}"}


def _count_prior_lookups(messages: list[dict], order_id: str) -> int:
    """How many times `order_id` has already been looked up in this
    conversation -- input to the OPA retry-loop circuit breaker
    (policies/policy_gate.rego)."""
    count = 0
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            if fn.get("name") != "lookup_order":
                continue
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                continue
            if args.get("order_id") == order_id:
                count += 1
    return count


def seed_messages(user_task: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]


# EFE's chosen policy has to actually change what the agent does next turn,
# or a non-terminal policy just re-asks the same question against
# unchanged messages and the model repeats itself. Steering text for the
# policies that route back to agent_step (graph.py route_from_decision).
POLICY_STEER = {
    "retry": "That didn't fully resolve it. Retry: reformulate your last tool call or answer.",
    "call_tool": "Use the lookup_order tool now to resolve this before answering.",
    "gather_info": "You don't have enough information yet. Ask the customer a clarifying "
    "question, or try the tool again with a corrected order ID, before finalizing an answer.",
}


def _derive_confidence(messages: list[dict]) -> str:
    transcript = "\n".join(
        f"{m['role']}: {m.get('content') or m.get('tool_calls')}" for m in messages[-4:]
    )
    resp = _get_client().chat.completions.create(
        model=_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Rate confidence this support task is on track to resolve "
                    "correctly. Reply with exactly one word: low, medium, or high."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        # z-ai/glm-5.3-flash (this endpoint) makes reasoning mandatory --
        # it can't be turned off, only capped, or it burns the whole
        # max_tokens budget on hidden reasoning and returns empty content.
        max_tokens=150,
        extra_body={"reasoning": {"max_tokens": 100}},
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    for bin_ in ("low", "medium", "high"):
        if bin_ in text:
            return bin_
    return "medium"


def real_agent_step(messages: list[dict]) -> tuple[list[dict], dict]:
    """One real LLM turn. Returns (updated_messages, observation_dict) --
    matches the 4 modalities efe_controller.Observation expects."""
    resp = _get_client().chat.completions.create(
        model=_model(),
        messages=messages,
        tools=[LOOKUP_TOOL],
        max_tokens=600,
        extra_body={"reasoning": {"max_tokens": 300}},
    )
    msg = resp.choices[0].message
    tool_calls = [tc.model_dump() for tc in msg.tool_calls] if msg.tool_calls else None
    messages = messages + [{"role": "assistant", "content": msg.content, "tool_calls": tool_calls}]

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        args = json.loads(tc.function.arguments or "{}")
        order_id = args.get("order_id", "")
        result = _lookup_order(order_id)
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        if result["found"]:
            tool_result, retrieval_quality = "success", "good"
        else:
            tool_result, retrieval_quality = "error", "poor"
        same_order_lookup_count = _count_prior_lookups(messages, order_id)
    else:
        tool_result, retrieval_quality = "no_tool_called", "n/a"
        same_order_lookup_count = 0

    observation = dict(
        tool_result=tool_result,
        confidence=_derive_confidence(messages),
        policy_gate=evaluate_policy_gate({
            "last_tool_result": tool_result,
            "same_order_lookup_count": same_order_lookup_count,
        }),
        retrieval_quality=retrieval_quality,
    )
    return messages, observation
