"""Stage 2 baselines (RESEARCH_PLAN.md §3.6 / §4, Stage 2), ported from
the toy kill-test controllers (kill_test/controllers.py) to the real
frozen decision-POMDP (docs/decision-pomdp.md, efe_controller.py).

Each baseline shares EFEControlNode's interface -- `__init__(prior=None)`
/ `decide(observation, valid_policies=None) -> Decision` (the same
dataclass efe_controller.py defines) -- so graph.py can swap any of them
into the control step the same way it swaps agent_step implementations.
"""
