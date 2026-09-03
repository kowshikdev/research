"""Stage 3: any control node (EFE, or one of the Stage 2 baselines)
wrapping a real tool-calling LLM agent inside tau2-bench
(external/tau2-bench, pinned at a2c0247 / v1.0.1), plugged in the same
way llm_agent.py wraps the order-lookup demo -- see docs/decision-pomdp.md
and context/TODOS.md.

tau2's turn-based contract (`HalfDuplexAgent.generate_next_message`,
src/tau2/agent/README.md) splits generation and tool execution across
two calls: the agent proposes a tool call, the orchestrator executes it,
and the RESULT arrives as the `message` argument on the *next* call. So
observation derivation happens at the START of each turn (from the
incoming ToolMessage/MultiToolMessage/UserMessage -- i.e. what the
*previous* turn's action produced), before generating this turn's
response -- mirroring how llm_agent.py derives its observation only
after a tool call resolves.

`escalate_to_human` is not just a steering hint here, unlike the other
five policies -- every core domain (mock, retail, airline, telecom) has
a real `transfer_to_human_agents(summary)` tool, so choosing that
policy makes the agent genuinely call it, a real, benchmark-scored
action (tau2's evaluation criteria can check whether transfer was the
correct/incorrect call for a task) -- this is the first point in the
project where "escalate" is graded by an external benchmark rather than
just pausing a LangGraph demo.

`ControlNodeAgent` is generic over which controller drives the decision
-- Stage 2's five controllers (EFEControlNode + the four baselines in
baselines/) all share the same `__init__(prior=None)` /
`decide(observation, valid_policies=None) -> Decision` /
`reset(prior=None)` interface, so the same tau2 agent wrapper works for
all of them (baseline_agents.py registers factories for the other
four); only EFEAgent's factory/name are kept here for backward
compatibility with the Stage 3 smoke test.
"""
import uuid
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from litellm.exceptions import AuthenticationError as LiteLLMAuthenticationError
from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate as _tau2_generate

from ..efe_controller import D_PRIOR, EFEControlNode, Observation
from ..opa_policy import evaluate_policy_gate

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def generate(*, messages: list, **kwargs) -> "AssistantMessage":
    """Wraps tau2's own `generate` with a retry for a deterministic Groq
    gpt-oss-120b quirk: it sometimes emits `null` for an optional string
    tool argument, which Groq's strict schema validator rejects outright
    (litellm.BadRequestError, code "tool_use_failed") -- not a transient
    error, so tau2/litellm's own retry logic (which targets 5xx/rate
    limits) never fires for it and the exception propagates uncaught.
    One retry with an explicit nudge, then fall back to a plain-text
    message rather than crash the whole simulation over one bad call.

    Also overlays the live Vertex token on every call (vertex_auth.py)
    -- kwargs threaded through here originated from self.llm_args,
    itself deepcopied from a TextRunConfig frozen once per domain/agent
    run, so it can carry a stale api_key for a token refreshed since
    that agent was constructed (confirmed: real sweep tasks hit 401
    ACCESS_TOKEN_TYPE_UNSUPPORTED well before the token's nominal
    lifetime, on a stretch of consecutive tasks -- i.e. the config's
    frozen copy going stale, not each token being individually bad).
    """
    from .vertex_auth import force_refresh, overlay_live_kwargs

    kwargs = overlay_live_kwargs(kwargs)
    try:
        try:
            return _tau2_generate(messages=messages, **kwargs)
        except LiteLLMAuthenticationError:
            # Observed: a burst of consecutive real tasks all failed
            # with 401 while the same gcloud-issued token, tested
            # moments later via a direct curl call, worked fine -- a
            # transient Vertex-side/OAuth-refresh hiccup rather than a
            # genuinely bad token. One immediate forced re-fetch (not
            # waiting for the background thread's interval) usually
            # clears it; if not, let it propagate to tau2's task-level
            # retry.
            force_refresh()
            kwargs = overlay_live_kwargs(kwargs)
            return _tau2_generate(messages=messages, **kwargs)
    except LiteLLMBadRequestError as e:
        if "tool_use_failed" not in str(e):
            raise
        nudge = SystemMessage(
            role="system",
            content=(
                "Your previous tool call was rejected because it passed null "
                "for an optional parameter. Omit optional parameters you don't "
                "need entirely -- never pass null for them."
            ),
        )
        try:
            return _tau2_generate(messages=messages + [nudge], **kwargs)
        except LiteLLMBadRequestError:
            return AssistantMessage.text(
                content="Could you confirm the details for your request? I want to make sure I get this right."
            )


def _ensure_non_empty(message: "AssistantMessage") -> "AssistantMessage":
    """Some models (observed: Groq gpt-oss-120b) occasionally spend the
    whole max_tokens budget on hidden reasoning and return a message
    with neither content nor tool_calls, which tau2 rejects outright."""
    if message.content or message.is_tool_call():
        return message
    return AssistantMessage.text(content="Could you tell me more about what you need help with?")

TRANSFER_TOOL_NAME = "transfer_to_human_agents"

# Mirrors llm_agent.POLICY_STEER -- the chosen policy has to actually
# change what the agent does next turn, or a non-terminal policy just
# re-prompts against unchanged context and the model repeats itself.
# Injected as an EPHEMERAL extra system message for this turn's
# generation only -- never appended to state.messages, so the official
# transcript tau2 evaluates and shows the user simulator stays clean.
POLICY_STEER = {
    "retry": "That didn't fully resolve it. Retry: reformulate your last tool call, "
    "or rephrase your message to the customer.",
    "call_tool": "Use one of the available tools now to make progress on this request "
    "before responding to the customer.",
    "gather_info": "You don't have enough information yet. Ask the customer a clarifying "
    "question, or try a different tool call, before finalizing an answer.",
    "hand_off_to_agent": "Consider whether a different specialized flow or tool better "
    "fits this request before proceeding.",
}


class ControlNodeAgentState(BaseModel):
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]
    belief: list[float]
    last_policy: Optional[str] = None
    decision_trace: list[dict] = []
    escalated: bool = False


ControlNodeAgentStateType = TypeVar("ControlNodeAgentStateType", bound="ControlNodeAgentState")


def _tool_call_key(name: str, arguments: dict) -> tuple:
    return (name, tuple(sorted(arguments.items())))


class ControlNodeAgent(
    LLMConfigMixin, HalfDuplexAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]
):
    """Half-duplex tau2 agent whose turn-to-turn control (continue / retry
    / call_tool / gather_info / escalate_to_human / hand_off_to_agent) is
    chosen by `control_node_cls` instead of being left entirely to the LLM."""

    control_node_cls = EFEControlNode  # overridden by baseline_agents.py subclasses

    def __init__(self, tools: list[Tool], domain_policy: str, llm: str, llm_args: Optional[dict] = None):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self._transfer_tool = next((t for t in tools if t.name == TRANSFER_TOOL_NAME), None)

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(domain_policy=self.domain_policy, agent_instruction=AGENT_INSTRUCTION)

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> ControlNodeAgentStateType:
        return ControlNodeAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history or [],
            belief=list(D_PRIOR),
        )

    def _derive_observation(self, incoming: ValidAgentInputMessage, state: ControlNodeAgentStateType) -> Observation:
        if isinstance(incoming, (ToolMessage, MultiToolMessage)):
            tool_messages = incoming.tool_messages if isinstance(incoming, MultiToolMessage) else [incoming]
            had_error = any(m.error for m in tool_messages)
            tool_result = "error" if had_error else "success"
            # retrieval_quality per docs/decision-pomdp.md: relevance of
            # retrieved context "when gather_info was last taken" --
            # meaningless for other policies, so n/a otherwise.
            retrieval_quality = ("poor" if had_error else "good") if state.last_policy == "gather_info" else "n/a"
            same_tool_call_count = self._count_repeated_last_call(state)
        else:
            tool_result, retrieval_quality = "no_tool_called", "n/a"
            same_tool_call_count = 0

        policy_gate = evaluate_policy_gate({
            "last_tool_result": tool_result,
            "same_tool_call_count": same_tool_call_count,
        })
        confidence = self._derive_confidence(state)
        return Observation(
            tool_result=tool_result, confidence=confidence,
            policy_gate=policy_gate, retrieval_quality=retrieval_quality,
        )

    def _count_repeated_last_call(self, state: ControlNodeAgentStateType) -> int:
        last_call = None
        for m in reversed(state.messages):
            if isinstance(m, AssistantMessage) and m.is_tool_call():
                last_call = m.tool_calls[0]
                break
        if last_call is None:
            return 0
        key = _tool_call_key(last_call.name, last_call.arguments)
        count = 0
        for m in state.messages:
            if isinstance(m, AssistantMessage) and m.is_tool_call():
                for tc in m.tool_calls:
                    if _tool_call_key(tc.name, tc.arguments) == key:
                        count += 1
        return count

    def _derive_confidence(self, state: ControlNodeAgentStateType) -> str:
        transcript = "\n".join(
            f"{getattr(m, 'role', type(m).__name__)}: {getattr(m, 'content', None) or getattr(m, 'tool_calls', None)}"
            for m in state.messages[-4:]
        )
        resp = generate(
            model=self.llm,
            messages=[
                SystemMessage(
                    role="system",
                    content=(
                        "Rate confidence this customer service task is on track to resolve "
                        "correctly. Reply with exactly one word: low, medium, or high."
                    ),
                ),
                UserMessage.text(content=transcript or "(no transcript yet)"),
            ],
            call_name="control_node_confidence",
            max_tokens=250,
            **self.llm_args,
        )
        text = (resp.content or "").strip().lower()
        for bin_ in ("low", "medium", "high"):
            if bin_ in text:
                return bin_
        return "medium"

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: ControlNodeAgentStateType
    ) -> tuple[AssistantMessage, ControlNodeAgentStateType]:
        # Turn 0 (the opening customer message, before the agent has done
        # anything): there's no real signal to reason about yet, and
        # _derive_observation would return tool_result="no_tool_called"
        # -- which the model treats as evidence FOR needs_human (0.50
        # probability in TOOL_RESULT_DIST), not "nothing has happened
        # yet". That misread the very first turn as already-stuck and
        # escalated immediately (verified: a real run against tau2's
        # mock domain did exactly this). mock_agent_step avoids the same
        # trap with its own turn==0 special case; here the fix is to
        # skip the control loop on a genuinely empty conversation and
        # just let the agent take its first natural action.
        if not state.messages and isinstance(message, UserMessage):
            state.messages.append(message)
            assistant_message = generate(
                model=self.llm, tools=self.tools, messages=state.system_messages + state.messages,
                call_name="control_node_agent_first_turn", max_tokens=1000,
                **self.llm_args,
            )
            assistant_message = _ensure_non_empty(assistant_message)
            state.messages.append(assistant_message)
            return assistant_message, state

        # Once escalate_to_human has genuinely fired, treat it as
        # terminal for this conversation -- unlike graph.py's
        # interrupt(), which actually pauses the LangGraph run,
        # transfer_to_human_agents returns a normal ToolMessage and tau2
        # just keeps calling generate_next_message. Without this check,
        # a control node whose belief/confidence stays poor after the
        # transfer (a live risk: tool_result="success" nudges belief
        # toward solvable, but a noisy confidence-verifier call can pull
        # it right back) keeps re-escalating -- verified: a real retail
        # sweep produced conversations with 2-7 transfer_to_human_agents
        # calls each, almost all scoring reward 0.0. No tools are passed
        # on these turns so the model literally cannot call transfer (or
        # anything else) again.
        if state.escalated:
            if isinstance(message, MultiToolMessage):
                state.messages.extend(message.tool_messages)
            else:
                state.messages.append(message)
            assistant_message = generate(
                model=self.llm, messages=state.system_messages + state.messages,
                call_name="control_node_agent_post_escalation", max_tokens=1000,
                **self.llm_args,
            )
            if not assistant_message.content and not assistant_message.is_tool_call():
                # Some models (observed: Groq gpt-oss-120b) occasionally
                # spend the whole max_tokens budget on hidden reasoning
                # and return empty content here, which tau2 rejects
                # (AssistantMessage requires content or tool_calls). No
                # tools are offered on this turn by design (see above),
                # so a fixed closing line is a safe substitute -- this
                # path only fires after a transfer has already gone out.
                assistant_message = AssistantMessage.text(
                    content="Thank you -- I've transferred you to a human agent who can help further."
                )
            state.messages.append(assistant_message)
            return assistant_message, state

        observation = self._derive_observation(message, state)

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        node = self.control_node_cls(prior=state.belief)
        decision = node.decide(observation)
        state.belief = list(decision.belief.values())
        state.decision_trace.append({
            "chosen_policy": decision.policy,
            "observation": observation.__dict__,
            "belief": decision.belief,
        })

        if decision.policy == "escalate_to_human" and self._transfer_tool is not None:
            summary = self._build_transfer_summary(state)
            assistant_message = AssistantMessage.text(
                content=None,
                tool_calls=[ToolCall(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    name=TRANSFER_TOOL_NAME, arguments={"summary": summary},
                )],
            )
            state.escalated = True
        else:
            messages = state.system_messages + state.messages
            steer = POLICY_STEER.get(decision.policy)
            if steer:
                messages = messages + [SystemMessage(role="system", content=steer)]
            assistant_message = generate(
                model=self.llm, tools=self.tools, messages=messages,
                call_name="control_node_agent_response", max_tokens=1000,
                **self.llm_args,
            )
            assistant_message = _ensure_non_empty(assistant_message)

        state.messages.append(assistant_message)
        state.last_policy = decision.policy
        return assistant_message, state

    def _build_transfer_summary(self, state: ControlNodeAgentStateType) -> str:
        for m in reversed(state.messages):
            if isinstance(m, UserMessage) and m.content:
                return f"Escalating per {self.control_node_cls.__name__}: {m.content[:200]}"
        return f"Escalating per {self.control_node_cls.__name__}: unable to resolve automatically."


class EFEAgent(ControlNodeAgent[ControlNodeAgentStateType], Generic[ControlNodeAgentStateType]):
    control_node_cls = EFEControlNode


def create_efe_agent(tools, domain_policy, **kwargs):
    """Factory function for EFEAgent, registered under the name "efe_agent"
    (src/aif_orchestrator/tau2_integration/register.py) -- pass
    --agent efe_agent on the tau2 CLI, or "efe_agent" to run_domain/
    run_single_task's TextRunConfig(agent=...)."""
    return EFEAgent(
        tools=tools, domain_policy=domain_policy,
        llm=kwargs.get("llm"), llm_args=kwargs.get("llm_args"),
    )
