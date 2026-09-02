"""Cost/budget estimator for the Stage 3 real tau2-bench sweep --
RESEARCH_PLAN.md flags this as "the real budget item" and
context/TODOS.md explicitly withholds running the sweep pending a scope
decision. This script is that decision's input: a dollar estimate BEFORE
spending anything, not a guess made after the fact.

Every constant below is a documented assumption, not a measurement --
this environment cannot reach tau2-bench (not installed, external/ is
gitignored) or OpenRouter (network policy) to measure real averages. Two
things this script does NOT know and should be corrected once a small
real run exists locally:
  1. Real per-domain task counts (falls back to the ~278-total figure
     already in run_stage3_eval.py's docstring, split evenly as a rough
     placeholder) -- pass --tasks-per-domain to override, or run this
     inside an environment with external/tau2-bench installed, where it
     will try to load the real counts automatically.
  2. Real average turns-per-task and escalation rate -- pass
     --avg-turns / --escalation-rate once a small pilot run (e.g.
     `run_stage3_eval.py --domain retail --num-trials 1` on a handful of
     tasks) gives real numbers; the defaults here are conservative
     placeholders, not measurements.

What IS known exactly (read from the actual code, not guessed):
  - The 3 LLM call shapes per turn and their max_tokens ceilings
    (efe_agent.py, run_stage3_eval.py) -- see CALL_SHAPES below.
  - Current pricing for the configured model, deepseek/deepseek-v4-flash
    on OpenRouter ($0.098 / $0.196 per MTok in/out as of this writing --
    verify against https://openrouter.ai/deepseek/deepseek-v4-flash/pricing
    before trusting this for a real budget decision; OpenRouter pricing
    is not pinned/versioned the way a model ID is).
  - domain-policy token size is a real unknown too (tau2's policy docs
    are resent in full on every non-confidence LLM call -- see
    docs/tau2-bench-integration.md's cost note) -- placeholder below,
    override with --policy-tokens once measured (trivial: count the
    actual policy string's tokens with any tokenizer once tau2-bench is
    installed locally).

Run: .venv/Scripts/python scripts/estimate_stage3_cost.py --domain all --agents all --num-trials 1
"""
import argparse

DOMAINS = ["retail", "airline", "telecom"]
AGENTS = ["efe_agent", "heuristic_agent", "router_agent", "voi_agent", "react_agent"]

# Placeholder until corrected from a real tau2-bench install (see module
# docstring) -- run_stage3_eval.py's own docstring cites "278 tasks"
# total across the three domains; split evenly since the real per-domain
# breakdown isn't available in this environment.
DEFAULT_TASKS_PER_DOMAIN = {"retail": 93, "airline": 93, "telecom": 92}  # sums to 278

# Pricing for the currently configured .env model (LLM_MODEL) --
# deepseek/deepseek-v4-flash on OpenRouter. $/MTok. RE-VERIFY before a
# real budget decision -- OpenRouter prices can change without a model-ID
# change, unlike Anthropic's versioned model IDs.
PRICE_PER_MTOK_INPUT = 0.098
PRICE_PER_MTOK_OUTPUT = 0.196

# Read directly from source, not estimated:
#   efe_agent.py _derive_confidence(): max_tokens=150, reasoning.max_tokens=100
#   efe_agent.py generate_next_message() agent-response generate(): max_tokens=600, reasoning.max_tokens=300
#   run_stage3_eval.py llm_args_user: max_tokens=1000, reasoning.max_tokens=300
# Actual usage is typically well below the ceiling -- UTILIZATION below
# estimates the fraction of max_tokens actually spent on output;
# reasoning tokens in particular often run near their cap for this model
# (see context/HANDOFF.md's gotcha on z-ai/glm-5.3-flash-style mandatory
# reasoning -- re-check whether this still applies if LLM_MODEL changes).
UTILIZATION = 0.5
CALL_SHAPES = {
    "confidence_verifier": {"max_tokens": 150, "reasoning_max_tokens": 100, "includes_policy": False},
    "agent_response": {"max_tokens": 600, "reasoning_max_tokens": 300, "includes_policy": True},
    "post_escalation_response": {"max_tokens": 600, "reasoning_max_tokens": 300, "includes_policy": True},
    "user_simulator": {"max_tokens": 1000, "reasoning_max_tokens": 300, "includes_policy": True},
}

# Placeholder -- override with --policy-tokens once measured against the
# real domain policy text (tau2's retail/airline/telecom policies are
# each several thousand tokens; this project has never measured them
# directly in an environment that had tau2-bench installed).
DEFAULT_POLICY_TOKENS = 3000

# Non-network-derivable assumptions -- override once a small pilot run
# gives real numbers (see module docstring point 2).
DEFAULT_AVG_TURNS_BEFORE_TERMINAL = 4  # turns of (confidence + agent_response) before continue/escalate/max_steps
DEFAULT_ESCALATION_RATE = 0.15
DEFAULT_POST_ESCALATION_TAIL_TURNS = 2  # extra post_escalation_response-only turns after a genuine escalation
MAX_STEPS = 30  # hard cap from run_stage3_eval.py's TextRunConfig


def try_load_real_task_counts():
    """Best-effort: if run where external/tau2-bench is actually
    installed, load real per-domain task counts instead of the
    placeholder split. Returns None (not a partial dict) if unavailable,
    so callers fall back cleanly rather than silently mixing real and
    placeholder counts."""
    try:
        from tau2.registry import registry  # noqa
        counts = {}
        for domain in DOMAINS:
            tasks = registry.get_tasks_loader(domain)()
            counts[domain] = len(tasks)
        return counts
    except Exception:
        return None


def estimate_tokens_per_task(avg_turns, escalation_rate, post_escalation_tail, policy_tokens):
    """Returns (input_tokens, output_tokens) expected per single task
    simulation (one agent, one trial), per the call-by-call turn
    structure in efe_agent.py -- see module docstring."""
    conf = CALL_SHAPES["confidence_verifier"]
    resp = CALL_SHAPES["agent_response"]
    tail = CALL_SHAPES["post_escalation_response"]
    user = CALL_SHAPES["user_simulator"]

    def call_tokens(shape, n_calls):
        out = n_calls * (shape["max_tokens"] + shape["reasoning_max_tokens"]) * UTILIZATION
        inp = n_calls * (policy_tokens if shape["includes_policy"] else 0)
        return inp, out

    # Turn 0: one agent_response call (no confidence check, per
    # efe_agent.py's turn-0 special case), plus the paired user_simulator turn.
    turns_after_first = max(avg_turns - 1, 0)

    total_in = total_out = 0.0
    for shape, n in [
        (resp, 1),                       # turn 0
        (conf, turns_after_first),       # confidence check each subsequent pre-terminal turn
        (resp, turns_after_first),       # agent response each subsequent pre-terminal turn
        (user, avg_turns),                # user-simulator turn paired with every agent turn above
    ]:
        i, o = call_tokens(shape, n)
        total_in += i
        total_out += o

    # Escalation branch: weighted by escalation_rate, adds a
    # confidence-only decision turn (no agent_response -- escalation
    # skips it) plus post-escalation tail turns (post_escalation_response
    # only, no confidence/control-node) and their paired user turns.
    esc_conf_i, esc_conf_o = call_tokens(conf, 1)
    esc_tail_i, esc_tail_o = call_tokens(tail, post_escalation_tail)
    esc_user_i, esc_user_o = call_tokens(user, 1 + post_escalation_tail)
    total_in += escalation_rate * (esc_conf_i + esc_tail_i + esc_user_i)
    total_out += escalation_rate * (esc_conf_o + esc_tail_o + esc_user_o)

    return total_in, total_out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="all")
    parser.add_argument("--agents", nargs="+", default=AGENTS, choices=AGENTS)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--tasks-per-domain", type=int, default=None,
                         help="Override the per-domain task count (else tries a real tau2-bench install, else the placeholder split).")
    parser.add_argument("--avg-turns", type=float, default=DEFAULT_AVG_TURNS_BEFORE_TERMINAL)
    parser.add_argument("--escalation-rate", type=float, default=DEFAULT_ESCALATION_RATE)
    parser.add_argument("--policy-tokens", type=int, default=DEFAULT_POLICY_TOKENS)
    parser.add_argument("--price-in", type=float, default=PRICE_PER_MTOK_INPUT, help="$ per MTok input")
    parser.add_argument("--price-out", type=float, default=PRICE_PER_MTOK_OUTPUT, help="$ per MTok output")
    args = parser.parse_args()

    domains = DOMAINS if args.domain == "all" else [args.domain]

    real_counts = try_load_real_task_counts()
    if args.tasks_per_domain is not None:
        task_counts = {d: args.tasks_per_domain for d in domains}
        source = "--tasks-per-domain override"
    elif real_counts is not None:
        task_counts = real_counts
        source = "real tau2-bench install"
    else:
        task_counts = DEFAULT_TASKS_PER_DOMAIN
        source = "placeholder split of the ~278-task figure (NOT measured -- see module docstring)"

    in_per_task, out_per_task = estimate_tokens_per_task(
        args.avg_turns, args.escalation_rate, DEFAULT_POST_ESCALATION_TAIL_TURNS, args.policy_tokens,
    )

    print(f"Task counts source: {source}")
    print(f"Assumptions: avg_turns={args.avg_turns}, escalation_rate={args.escalation_rate}, "
          f"policy_tokens={args.policy_tokens}, utilization={UTILIZATION} "
          "(all placeholders until corrected from a real pilot run -- see module docstring)")
    print(f"Per-task-simulation estimate: ~{in_per_task:,.0f} input tokens, ~{out_per_task:,.0f} output tokens\n")

    grand_total_cost = 0.0
    grand_total_sims = 0
    print(f"{'domain':<10} {'agent':<16} {'tasks':>6} {'trials':>7} {'sims':>6} {'est. cost':>12}")
    for domain in domains:
        n_tasks = task_counts.get(domain, DEFAULT_TASKS_PER_DOMAIN.get(domain, 90))
        for agent in args.agents:
            n_sims = n_tasks * args.num_trials
            cost = n_sims * (
                in_per_task / 1_000_000 * args.price_in + out_per_task / 1_000_000 * args.price_out
            )
            grand_total_cost += cost
            grand_total_sims += n_sims
            print(f"{domain:<10} {agent:<16} {n_tasks:>6} {args.num_trials:>7} {n_sims:>6} ${cost:>11,.2f}")

    print(f"\nTOTAL: {grand_total_sims} simulations, est. ${grand_total_cost:,.2f} "
          f"at ${args.price_in}/${args.price_out} per MTok in/out")
    print("\nThis is a planning estimate, not a quote -- re-run with --avg-turns/--escalation-rate/"
          "--policy-tokens set from a real small pilot before committing to the full sweep.")


if __name__ == "__main__":
    main()
