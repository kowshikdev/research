"""Stage 0 sanity check: confirm the pymdp engine runs and actually
produces epistemic-value-driven behavior, before wiring anything into
LangGraph.

Classic T-maze (using pymdp's own reference environment, not a hand-rolled
model -- an earlier hand-rolled version of this script had a generative-
model bug that produced tied EFE across actions, which is exactly the kind
of subtle error we don't want silently baked into the real EFE control
node later): the agent starts at a central location and must choose
whether to visit a "cue" location (pure information-seeking / epistemic
action, no reward) before committing to a reward arm. A correctly working
active-inference agent should prefer visiting the cue first when the
reward location is initially unknown, because doing so has positive
epistemic value even though it costs a timestep. That is the same
"is it worth gathering more info before committing" trade-off the EFE
control node in RESEARCH_PLAN.md needs to reproduce over the real
decision-POMDP in docs/decision-pomdp.md.

pymdp 1.0+ is JAX-native; this uses pymdp.legacy, which preserves the
classic numpy object-array API and ships the reference TMazeEnv. The real
EFE control node in Stage 1 should evaluate the JAX API instead, since
that's the actively-developed path (GPU/autodiff/vmap) referenced in
RESEARCH_PLAN.md Sec 3.1 -- legacy is used here only because it's the
simplest, best-documented path for a one-shot engine sanity check.

Run: .venv/bin/python scripts/tmaze_sanity_check.py
"""

import json
from pathlib import Path

import numpy as np
from pymdp.legacy import utils
from pymdp.legacy.agent import Agent
from pymdp.legacy.envs.tmaze import TMazeEnv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

env = TMazeEnv(reward_probs=[0.98, 0.02])

A = env.get_likelihood_dist()
B = env.get_transition_dist()

# Preferences: strong preference for the "reward" observation (modality 1,
# index 1), dispreference for "no reward" (index 2). Index 0 is "no
# reward info yet" (e.g. at center or at the cue).
C = utils.obj_array_zeros([4, 3, 2])
C[1][1] = 3.0
C[1][2] = -3.0

# D: agent knows it starts at the center location, but does NOT know the
# reward condition -- this uncertainty is what should make visiting the
# cue worth it (epistemic value).
D = utils.obj_array_uniform([4, 2])
D[0] = utils.onehot(0, 4)

agent = Agent(A=A, B=B, C=C, D=D, policy_len=2)

obs = env.reset()
qs = agent.infer_states(obs)
q_pi, efe = agent.infer_policies()

location_names = {0: "center", 1: "reward_arm_1", 2: "reward_arm_2", 3: "cue_location"}
first_actions = [int(p[0, 0]) for p in agent.policies]
best_policy_idx = int(np.argmax(q_pi))
chosen_first_action = first_actions[best_policy_idx]

print("Expected Free Energy per policy (lower = preferred):")
for i, e in enumerate(efe):
    print(f"  policy {i} (first action={location_names[first_actions[i]]}): "
          f"EFE={e:.3f}  q_pi={q_pi[i]:.3f}")

print()
print(f"Agent's preferred first action: {location_names[chosen_first_action]}")

RESULTS_DIR.mkdir(exist_ok=True)
result = {
    "check": "tmaze_epistemic_value_sanity_check",
    "chosen_first_action": location_names[chosen_first_action],
    "expected_first_action": "cue_location",
    "efe_per_policy": [float(e) for e in efe],
    "q_pi_per_policy": [float(q) for q in q_pi],
    "pass": chosen_first_action == 3,
}
(RESULTS_DIR / "stage0_tmaze_sanity_check.json").write_text(json.dumps(result, indent=2))
print(f"\nResults written to {RESULTS_DIR / 'stage0_tmaze_sanity_check.json'}")

assert chosen_first_action == 3, (
    "Expected the agent to prefer visiting the cue first (epistemic value "
    "of resolving reward-location uncertainty) -- got "
    f"{location_names[chosen_first_action]} instead. Engine or model setup is wrong."
)
print("\nPASS: engine correctly favors the information-seeking (epistemic) "
      "action when reward location is uncertain.")
