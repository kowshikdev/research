# Decision-POMDP schema (frozen)

This is the frozen decision space for the EFE control node, per
`RESEARCH_PLAN.md` §2. It defines the hidden states, observations,
and policies the active-inference engine reasons over — deliberately
small so Expected Free Energy stays tractable (exact enumeration over
policies × horizon), and deliberately **not** raw LLM token space.

Any change to this file requires a written justification in the PR/commit
message and a re-check that EFE enumeration is still cheap (see
"Tractability budget" below) — this is not a place to iterate casually.

## Hidden state factors (≤4)

One factor, four mutually-exclusive values — the orchestrator's belief
about what's actually going on with the current task:

| Value | Meaning |
|---|---|
| `task_solvable_now` | Agent has what it needs; the next tool call / answer should succeed |
| `needs_more_info` | Missing information exists and is plausibly retrievable (search, another tool call, RAG) |
| `needs_human` | Missing information is not retrievable by the agent itself (ambiguous spec, irreversible action, policy conflict) |
| `likely_to_fail` | Retrying or gathering more info is unlikely to change the outcome (e.g. repeated tool errors, contradictory constraints) |

## Observation modalities

Each is a small discrete/binned signal, not raw text:

| Modality | Source | Example bins |
|---|---|---|
| `tool_result` | Last tool call outcome | `success \| error \| partial \| no_tool_called` |
| `confidence` | Self-consistency / verifier score | `low \| medium \| high` (binned from a continuous score) |
| `policy_gate` | OPA verdict on the proposed next action | `allow \| deny \| needs_review` |
| `retrieval_quality` | Relevance/coverage of retrieved context, when `gather_info` was last taken | `poor \| adequate \| good \| n/a` |

## Policies / actions (≤6)

| Policy | Effect |
|---|---|
| `continue` | Proceed with the current plan / answer as-is |
| `retry` | Re-attempt the last step (e.g. reformulate a tool call) |
| `call_tool` | Invoke a specific tool to resolve `needs_more_info` |
| `gather_info` | Broader retrieval/search step before proceeding |
| `escalate_to_human` | Pause via LangGraph `interrupt()` + checkpointer for human review |
| `hand_off_to_agent` | Route to a different specialized agent in the multi-agent system |

## Preferences (C matrix) — starting point, sensitivity-tested in Stage 3

Encoded as log-probabilities over preferred observations, not hidden
states directly (standard active-inference convention):

- Strong preference for `tool_result = success`
- Mild preference against `policy_gate = needs_review` recurring (cost signal)
- Neutral-to-mild preference against reaching `escalate_to_human` repeatedly
  within one task (cost of human time), balanced against the strong
  preference for eventual `task_solvable_now` — this trade-off is exactly
  what the Stage 3 ablation on the preference prior interrogates.

## Tractability budget

- 4 state values × 4 observation modalities (each ≤4 bins) × 6 policies,
  planning horizon ≤3 steps ahead → EFE enumeration stays in the
  hundreds-of-evaluations range per decision, not exponential blowup.
  Recompute this budget before adding any state/observation/policy.

## Mapping to the LangGraph agent loop

- Observations are derived once per orchestrator turn, right after a tool
  call / retrieval step and before the next action is chosen.
- The EFE control node consumes the current belief state + observations,
  returns one policy, and the LangGraph graph dispatches accordingly.
- `escalate_to_human` is the only policy that triggers `interrupt()`.
