# Real OPA policy for the `policy_gate` observation modality
# (docs/decision-pomdp.md), replacing the hardcoded "allow" every agent
# step used until now. Evaluated once per orchestrator turn against the
# same runtime context the agent step already has -- no LLM call
# involved, this is a deterministic, auditable policy check, exactly
# the point of using OPA instead of another model call.
#
# Deliberately small and generic, since it has to work across both
# llm_agent.py's single-tool order-lookup demo and Stage 3's tau2-bench
# agent (arbitrary domain tools -- retail, airline, telecom): a
# retry-loop circuit breaker (deny after 3+ identical repeats of the
# same tool call, by name+arguments -- a real production concern: don't
# let an agent hammer the same call forever) and a review flag on tool
# errors. Extend this file, not the Python wrapper, when richer
# domain-specific policies are needed -- that's the whole point of
# externalizing policy from code.
package aif.policy_gate

default decision := "allow"

decision := "deny" if {
	input.same_tool_call_count >= 3
} else := "needs_review" if {
	input.last_tool_result == "error"
} else := "allow"
