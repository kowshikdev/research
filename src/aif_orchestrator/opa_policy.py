"""Real OPA (Open Policy Agent) wiring for the `policy_gate` observation
modality (docs/decision-pomdp.md) -- replaces the hardcoded "allow"
every agent step used until now (context/TODOS.md).

Shells out to `opa eval` per turn (no long-running server -- a single
CLI invocation per decision is cheap and keeps this stateless, matching
how the rest of the observation derivation works). The policy itself
lives in policies/policy_gate.rego, not here -- extend that file for
richer policies; this module only knows how to call OPA and validate
its output lands in efe_controller.POLICY_GATE_BINS.
"""
import json
import subprocess
from pathlib import Path

from .efe_controller import POLICY_GATE_BINS

POLICY_PATH = Path(__file__).resolve().parent.parent.parent / "policies" / "policy_gate.rego"
QUERY = "data.aif.policy_gate.decision"


def evaluate_policy_gate(input_doc: dict) -> str:
    """Runs the OPA policy against `input_doc`, returns one of
    POLICY_GATE_BINS. Falls back to "needs_review" (the conservative
    choice -- flag for a human rather than silently allow) if OPA isn't
    installed or the policy errors, rather than crashing the agent loop
    or silently defaulting to "allow"."""
    try:
        result = subprocess.run(
            ["opa", "eval", "--format=json", "--data", str(POLICY_PATH), "--stdin-input", QUERY],
            input=json.dumps(input_doc), capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return "needs_review"
        payload = json.loads(result.stdout)
        decision = payload["result"][0]["expressions"][0]["value"]
    except (FileNotFoundError, subprocess.TimeoutExpired, KeyError, IndexError, json.JSONDecodeError):
        return "needs_review"

    return decision if decision in POLICY_GATE_BINS else "needs_review"
