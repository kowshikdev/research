# Real OPA policy for the `policy_gate` observation modality
# (docs/decision-pomdp.md), replacing the hardcoded "allow" every agent
# step used until now. Evaluated once per orchestrator turn against the
# same runtime context the agent step already has -- no LLM call
# involved, this is a deterministic, auditable policy check, exactly
# the point of using OPA instead of another model call.
#
# Kept deliberately small and honest about what it can check given the
# current llm_agent.py toolset (one read-only lookup_order tool, no
# irreversible actions yet): a real deny/needs_review policy needs
# something worth denying. This policy enforces a retry-loop circuit
# breaker (deny after repeated identical lookups -- a real production
# concern: don't let an agent hammer the same query forever) and flags
# tool errors for review. Extend this file, not the Python wrapper, when
# richer policies are needed -- that's the whole point of externalizing
# policy from code.
package aif.policy_gate

default decision := "allow"

decision := "deny" if {
	input.same_order_lookup_count >= 3
} else := "needs_review" if {
	input.last_tool_result == "error"
} else := "allow"
