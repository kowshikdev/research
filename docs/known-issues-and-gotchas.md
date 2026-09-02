# Known issues and gotchas

Every real bug, environment trap, and open unknown caught while building
this project, consolidated in one place. Each one is documented at its
original site too (commit messages, docstrings, the relevant `docs/*.md`)
— this file exists so the next person touching this code doesn't have to
go spelunking through git history to know what to watch for. Ordered by
where in the pipeline each one bit.

## Summary table

| # | Stage | What broke | Symptom | Fix |
|---|---|---|---|---|
| 1 | 0 | Wrong PyPI package | `import pymdp` silently succeeds, wrong library | Install `inferactively-pymdp`, not `pymdp` |
| 2 | 0.5 | EFE lookahead double-counted terminal reward | Escalate-twice policies beat genuinely better ones | Added absorbing "done" phase state |
| 3 | 1a | Missing cost signal for unnecessary action | EFE indifferent between `continue` and `gather_info` when task already solved | Small risk of confusion added to non-`continue` policies |
| 4 | 1c | OpenRouter model retired mid-session; replacement burns budget on hidden reasoning | Empty `content`, slow/hanging calls | Cap `reasoning.max_tokens` explicitly on every call |
| 5 | 2 | Reward paid every turn, not just terminal | Router learns to never choose `continue` | Real payoff only on terminal turns |
| 6 | 2 | No reward signal distinguishing `continue` vs `escalate_to_human` | Router never learns to escalate | Ground-truth-keyed `OUTCOME_ADJUSTMENT` |
| 7 | 2 | Comparison ran against a hand-scripted mock, not a real generative model | Result wasn't a repeat of Stage 0.5's methodology | Built `model_env.ModelMatchedEnv`, added as Part 2 |
| 8 | 3 | Turn-0 `no_tool_called` misread as evidence for `needs_human` | Agent escalates immediately, every time | Skip control loop entirely on empty conversation |
| 9 | 3 | `transfer_to_human_agents` doesn't halt execution like `interrupt()` does | Agent re-escalates 2-7 times per conversation, reward≈0 | Terminal `escalated` flag; no tools offered after |
| 10 | 3 | Router would lazily train against the mismatched mock env inside tau2 | Silent regression to Stage 2 Part 1's "not a fair test" baseline | `register.py` pre-trains against `ModelMatchedEnv` first |
| 11 | environment | Cloud sandbox network policy blocks arbitrary third-party hosts | `httpx2.ProxyError: 403 Forbidden` reaching `openrouter.ai` | Not fixable from the sandbox — run LLM-dependent work locally |

## Detail

### 1. PyPI name collision — `pymdp` vs `inferactively-pymdp`

`pip install pymdp` installs an **unrelated** MDP toolkit by a different
author (`github.com/minqi/PyMDP`). The active-inference library from
`infer-actively/pymdp` is published as `inferactively-pymdp` — it still
`import pymdp` in code, so the wrong install doesn't even fail loudly; it
just silently gives you a different library with a different API.
Caught immediately at Stage 0 (`import pymdp; pymdp.__file__` pointed at
the wrong package's `__init__.py`, and `pymdp.agent.Agent` didn't exist).
`pyproject.toml` documents this inline.

### 2. Kill-test's EFE double-counting bug

`kill_test/controllers.py`'s `EFEController` originally let a policy like
`['escalate', 'escalate']` collect the terminal-reward observation
*twice* within a 2-step lookahead — because nothing stopped the phase
state from re-entering `post_escalate` on a second consecutive
`escalate` action. This structurally biased EFE toward premature
termination regardless of whether gathering more info would actually
help, and it looked at first like a genuine EFE-vs-VOI algorithmic
finding (EFE chose `escalate` where VOI chose `gather_info` on an
identical belief) before being traced to the construction bug. Fixed by
adding a 4th, absorbing `"done"` phase: once a terminal action fires,
every further lookahead step goes to `done`, which emits a neutral
`no_outcome` observation regardless of task state — a policy can only
collect the terminal-reward observation once. Full writeup:
[`stage0.5-kill-test-results.md`](stage0.5-kill-test-results.md).

**Lesson**: when a generative model lets a "terminal" action be evaluated
more than once inside a multi-step lookahead, verify it can't be
re-triggered for extra reward before trusting any surprising result.

### 3. `efe_controller.py`'s missing unnecessary-action cost

The real `B_TRANSITIONS` initially left `task_solvable_now` completely
unchanged under *every* policy — including `retry`/`call_tool`/
`gather_info`. With belief ~99% concentrated on `task_solvable_now`, EFE
was genuinely indifferent between `continue` and `gather_info` (action
marginals ~0.197 each), because every non-escalation policy predicted the
exact same next observation. Fixed by giving unnecessary info-seeking
actions a small, realistic risk of nudging `task_solvable_now` toward
`needs_more_info` (0.03/0.05/0.10 for retry/call_tool/gather_info) —
`continue` alone stays risk-free. See
[`efe-control-node-design.md`](efe-control-node-design.md).

**Lesson**: a generative model needs an explicit cost for *unnecessary*
action, not just resolving power for genuinely bad states — otherwise
"doing nothing" has no advantage over "doing something pointless".

### 4. OpenRouter model retirement + mandatory hidden reasoning

`.env` initially pointed `LLM_MODEL` at `stealth/ox-alpha`, which was
retired mid-session (404). The replacement, `z-ai/glm-5.3-flash`, makes
reasoning mandatory — `reasoning: {enabled: false}` is rejected with a
400 — and without an explicit cap, it burns the *entire* `max_tokens`
budget on hidden reasoning tokens, returning empty `content` and making
every call slow. Fixed by passing `extra_body={"reasoning": {"max_tokens": N}}`
plus a larger overall `max_tokens` on every call. **If `LLM_MODEL`
changes again, check whether this still applies before assuming a hang
is a network issue** — it looks exactly like one.

### 5-6. Router reward-shaping bugs (two, sequential)

See [`baselines-design.md`](baselines-design.md)'s "`belief.py`: the
shared plumbing, and two real bugs it fixed" for the full narrative.
Short version: paying the C-weighted observation reward on every turn
(not just terminal ones) taught the router to stall forever; fixing that
alone still left no signal distinguishing a correct `continue` from a
correct `escalate_to_human` at the same observation, so a second,
independent fix (ground-truth-keyed `OUTCOME_ADJUSTMENT`, evaluation-only,
never fed to any controller's `decide()`) was needed before the router
learned to escalate at all.

**Lesson**: "reward the good observation" and "reward the right decision
given that observation" are two different things a reward function needs
to encode separately — the first doesn't imply the second.

### 7. Stage 2 Part 1's methodology gap

Not a code bug, a research-design one: the first Stage 2 comparison
(`run_stage2_eval.py`) validated that all 5 controllers plug into the
same scaffold correctly, but scored them against `mock_agent_step` — a
hand-scripted stand-in, not a sample from the generative model EFE/VOI
actually assume. That's a legitimate plumbing check but **not** a repeat
of Stage 0.5's Bayes-optimality claim. `model_env.ModelMatchedEnv`
(Part 2) closed this, and the result changed materially: EFE and VOI,
statistically indistinguishable at Stage 0.5's toy scale, **diverge** at
this scale — trading off recall vs. precision differently. See
[`stage2-baselines-results.md`](stage2-baselines-results.md).

**Lesson**: a comparison that validates plumbing and a comparison that
validates a scientific claim are different things, even when they share
code — say explicitly which one a result is before it gets cited.

### 8. tau2 turn-0 cold start

Detailed in [`tau2-bench-integration.md`](tau2-bench-integration.md).
`tool_result="no_tool_called"` on an empty conversation reads as 50%
evidence for `needs_human` under `TOOL_RESULT_DIST` — not "nothing has
happened yet, no evidence either way". Verified: an early version
escalated on turn 0 against tau2's own mock domain, every time. Fixed by
skipping the control loop on a genuinely empty conversation, mirroring
`mock_agent_step`'s own turn-0 special case.

**Lesson**: "no signal yet" and "a specific bad signal" are different
observations — an observation model needs a way to represent the
former, or it will get read as the latter.

### 9. `escalate_to_human` re-triggering in tau2

Detailed in [`tau2-bench-integration.md`](tau2-bench-integration.md).
`transfer_to_human_agents` is a normal tool call in tau2's world, not a
LangGraph `interrupt()` — it doesn't stop anything. Verified against a
real retail sweep: conversations with 2-7 transfer calls each, almost all
scoring `reward=0.0`. Fixed with a terminal `state.escalated` flag; once
set, later turns get no tools at all, so the model cannot call transfer
again.

**Lesson**: "genuinely stops execution" (`interrupt()`) and "returns a
message that looks like a stop" (a tool call) are different guarantees —
verify which one a new environment actually gives you before assuming
escalation is self-limiting.

### 10. Router training-source gap for Stage 3 (caught and fixed in this pass)

`LearnedRouterControlNode` self-trains lazily on first construction if
its class-level `_q` cache is empty, using its own default
(`_MockAgentEpisodeSource`, 10,000 episodes) — the exact training regime
Stage 2 Part 1 already flagged as not a fair test. `RouterAgent`
(`tau2_integration/baseline_agents.py`) never overrode this, so a real
tau2 sweep would have silently used the weaker, mismatched Q-table
instead of the model-matched one Stage 2 Part 2 established as fairer.
Fixed: `register.py` now explicitly pre-trains against
`model_matched_source_factory` (100,000 episodes, ~2.5s, no LLM calls)
before registering any agents, so `RouterAgent`'s lazy-training branch
never fires with the weaker default. See
[`baselines-design.md`](baselines-design.md)'s "Training data matters as
much as the algorithm".

**Lesson**: a lazy-initialization cache is a sharp edge when the *default*
initialization path is known to be worse than an available alternative —
the cache needs to be warmed deliberately, not left to whichever caller
happens to construct the object first.

### 11. Cloud sandbox network policy blocks third-party APIs entirely

Not a code bug — a hard environment constraint. This project's cloud
session has network egress locked to a fixed allowlist (Anthropic's own
API, package registries: PyPI, npm, crates.io, the Go proxy). Attempting
a real API call to `openrouter.ai` from this environment fails at the
proxy layer (`403 Forbidden` on the `CONNECT`), regardless of whether the
API key or model configuration is correct. This isn't a missing-credential
problem; it's architectural. **Any work that needs a real LLM call (Stage
1c verification, Stage 2 baseline demos against a live model, the Stage 3
smoke test or sweep) has to run in an environment with open network
access** — a local machine, typically. `context/HANDOFF.md` and
`context/TODOS.md` are written assuming this split.

**Lesson**: verify what a sandboxed environment can actually reach before
debugging a credential — a correctly-configured key can still fail for
reasons that have nothing to do with the key.

## Open unknowns (not bugs, but flagged so they don't get mistaken for measurements)

- **`scripts/estimate_stage3_cost.py`'s assumptions** — average turns
  per task, escalation rate, and domain-policy token size are all
  documented placeholders, not measurements (this environment can't run
  a pilot to measure them — see #11). Correct them from a real small
  local run before trusting the estimate for a large-scale budget
  decision.
- **`efe_controller.py`'s A/B/C/D/E values** are hand-specified to be
  *structurally* reasonable, not empirically calibrated. Every result in
  this project to date is honest about testing *mechanisms* under a
  shared, admittedly-uncalibrated model — not testing against
  ground-truth-calibrated probabilities. Calibration is explicitly Stage
  3+ work, once real trajectories exist to calibrate against.
