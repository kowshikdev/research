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
| 12 | 2 | Learned router's exact numbers don't reproduce across machines | Committed reward 3.393 / correct-escalation 0.255; re-run elsewhere gives 3.434 / 0.236 | Not fixed — documented; within-environment determinism pinned by a test |

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

### 12. The learned router's exact numbers are environment-dependent

Re-running `run_model_matched_eval.py` in a different environment
(Linux/Python 3.11) than the one that produced the committed results
(Windows/Python 3.13) reproduces **four of five controllers bit-exactly**
— heuristic 2.9989, VOI 3.1421, ReAct 3.1830, EFE 2.8042 all identical —
but not the learned router: committed `avg_reward` 3.3934 /
`correct_escalation_rate` 0.2548 versus 3.4344 / 0.2355 on the re-run.

The router is the only controller with *trained* state, which localizes
it: everything else is a pure function of the (identical) generative
model and seed. Training is **bit-deterministic within an environment** —
training twice in-process yields an identical Q-table hash — so this is
not nondeterminism in the code; it's a cross-environment difference in
how the seeded RNG stream and the training trajectory interact.

**What this does and doesn't affect.** It does not change any
qualitative conclusion in `stage2-baselines-results.md`: the router still
has by far the worst correct-escalation rate of the five (0.24-0.26 vs
EFE's 0.95) and still the highest raw reward, which is the whole finding.
It does mean **exact router figures should be quoted with the
environment that produced them**, and a reviewer re-running on different
hardware should expect ~1% drift on reward and a few points on the
escalation rate.

`test_router_training_is_deterministic_within_an_environment` pins the
property that actually matters — that a re-run reproduces its own
results. An exact cross-machine equality test would fail for reasons
that aren't bugs.

**Lesson**: "reproducible" needs a scope. Pure-function results reproduce
anywhere; anything with a trained artifact reproduces within an
environment, and the distinction belongs next to the numbers rather than
in a reader's assumptions.

### 13. Escalation-rate contamination: the transfer tool was reachable on every turn, not just the control node's escalate branch

The Stage 3 airline sweep produced an impossible data point: `react_agent`
(`ReActControlNode`, which can never select `escalate_to_human` -- it isn't
one of that controller's reachable policies by construction) showed
24/50 escalations. Every controller's escalation count was, in fact,
counting something other than the control node's decision.

Root cause: `efe_agent.py`'s non-escalate `generate()` calls (the turn-0
branch and the main per-turn branch) passed the agent's *full* tool list,
`self.tools`, which includes `transfer_to_human_agents`. The underlying
LLM is free to call any tool in the list it's given -- so on any turn,
regardless of what the control node decided, the model could invoke the
transfer tool on its own initiative. The dedicated escalate branch (which
fires when `decision.policy == "escalate_to_human"`) builds the transfer
call directly without going through `generate()` at all, so it was never
the only path to an escalation -- it was just the only *intended* one.

This silently invalidated escalation-rate as a cross-controller metric
for every controller except insofar as its own control-node decisions
happened to coincide with turns where the model would have escalated
anyway. Reward numbers are unaffected (tau2 scores those independently
of how the transfer happened).

Fix: build a `self._tools_sans_transfer` list once in `__init__` (all
tools except `transfer_to_human_agents`) and pass that instead of
`self.tools` on both non-escalate `generate()` call sites. The transfer
tool is now reachable only from the one branch that's supposed to reach
it.

There is no way to recover clean escalation counts from the contaminated
runs after the fact -- tau2's saved simulation results don't retain a
per-turn record of which tools were offered. The retail and airline
domains were re-run under the corrected code; only the re-run counts are
valid for escalation-rate comparisons.

**Lesson**: when a control layer's decision is supposed to gate an LLM's
tool access, verify the gate at the tool-list level, not just at the
branch that's meant to be the only caller. An anomalous data point from a
controller whose action space provably excludes the observed behavior is
a stronger signal than it looks -- it means the measured behavior isn't
coming from the mechanism under test at all.

**Update -- schema exclusion alone was not sufficient.** A targeted
re-check (running `react_agent` against real airline tasks right after
deploying the above fix) reproduced the same failure mode on task 0:
`transfer_to_human_agents` was called with empty arguments even though
it had been excluded from that exact `generate()` call's `tools`
parameter. gemini-2.5-flash, called through Vertex's generic
OpenAI-compatible passthrough (`vertex_auth.py`), does not appear to
strictly confine its function calls to the declared schema the way
OpenAI's own API does -- `domain_policy`'s system-prompt text describes
`transfer_to_human_agents` by name and signature for every core tau2
domain, and the model can apparently still emit a matching call from
that description alone. Compounding it: tau2's environment executes any
tool call by name against the domain's *full* registered tool set,
independent of what was actually offered in the request that produced
it -- so a schema-excluded-but-still-emitted call still gets executed
and scored as a real transfer.

The robust fix needed a second, independent layer:
`_strip_disallowed_transfer()` in `efe_agent.py` filters
`transfer_to_human_agents` out of the assistant message itself after
`generate()` returns, on both non-escalate branches, regardless of what
was in the request schema. If that leaves the message with no content
and no other tool calls, it substitutes a safe placeholder rather than
letting an empty message fail tau2's validation. A follow-up run of the
same real-task check (react_agent, 6 airline tasks) confirmed 0 transfer
calls under this fix.

**Lesson, restated**: for a model reachable through a non-native
API-compatibility layer, don't assume the provider enforces "the model
can only call what's in the schema" -- verify it against a real
adversarial case (a controller whose policy space *cannot* produce the
behavior, run against the model that actually misbehaved), and filter
the output as well as the input if it doesn't hold.

### 14. `stage3_report.py`'s raw-dump reader assumed a flat file; tau2 writes a directory

`enrich_from_raw_dump` (used by `--raw` for escalation precision/recall)
was written against an *inferred* schema before any real sweep existed,
and its own docstring said so. Running it for the first time against
the real Stage 3 sweep raised `IsADirectoryError` immediately: tau2's
`TextRunConfig.save_to=".../<domain>__<agent>.json"` actually creates a
*directory* at that path containing `results.json`, not a flat file at
the path itself. Once pointed at the right file, every other assumption
in the reader held -- `reward_info.reward` and
`messages[].tool_calls[].name` matched exactly, 0 parse warnings across
all 15 domain/agent combinations.

**Lesson**: "written defensively, never run against real data" is
exactly the state where a structural assumption (file vs. directory)
is the likeliest failure, not the field-level schema the code was
actually worried about — the first real run is worth doing before
trusting the rest of the reader's caveats.

### 15. EFE's real 100%-escalation cause: two compounding upstream defects, not a calibration problem

The interpretability finding (#14's neighbor -- 0/278 decisions driven
by the epistemic term) came with a second question: *why* was EFE
escalating on literally every task? Direct inspection of the real
decision log's observation distribution answered it exactly:
`tool_result` was `"no_tool_called"` 100% of the time and `confidence`
was `"low"` 276/278 times -- EFE was making its ONLY real decision at
the very first evidence-bearing turn, before any tool had ever been
attempted.

Two compounding, fixable defects, both confirmed causally (not just
correlated) by re-running `EFEControlNode.decide()` directly against
the real observation combination:

1. **`policy_gate` was corrupted.** This sandbox had no `opa` CLI
   installed, so `evaluate_policy_gate()` failed safe to
   `"needs_review"` on every call -- but the real policy
   (`policies/policy_gate.rego`) says `default := "allow"` for this
   exact input (no tool error, no repeated calls). Fixed by installing
   opa (`go install github.com/open-policy-agent/opa@latest`, symlinked
   onto PATH -- GitHub release downloads are blocked by this sandbox's
   network policy, but the Go module proxy isn't). Not sufficient
   alone: with `policy_gate` corrected but everything else unchanged,
   belief in `needs_human` dropped from 82% to 45%, and EFE *still*
   chose `escalate_to_human` with 100% certainty.
2. **The cold-start skip only covered the literal first message.**
   `_derive_confidence`'s prompt ("is this on track to resolve
   correctly?") reasonably answers "low" at the very first real turn,
   before the agent has had any chance to make progress -- but the
   generative model reads "low confidence + no tool called" as strong
   evidence for `needs_human`, the same misread the original turn-0 fix
   was meant to prevent, just recurring one level deeper. Fixed by
   broadening the skip from "the conversation is empty" to "no tool
   call has been attempted yet" (`_any_tool_call_attempted`).

Both fixes together, re-run across all 3 domains: escalations dropped
114→10 (retail), 50→10 (airline), 114→63 (telecom) -- real and
consistent in direction everywhere, but a much weaker effect on
telecom, and telecom's reward fell slightly (0.175→0.096) rather than
improving like retail's did (0.053→0.281, EFE went from worst controller
to beating heuristic) and airline's reward staying flat. Not yet
explained why telecom responds differently -- flagged as an open
question below rather than assumed away.

**A third, smaller bug found while re-running this fix's own
verification:** the decision-logger's synthetic task ids were a plain
incrementing counter, which restarts from 1 in every new process --
and `run_stage3_eval.py` runs one domain per process. Retail/airline/
telecom each produced ids 1..N, so `decision_log.group_by_task` silently
merged decisions from up to 3 unrelated real tasks under the same id
(126 unique ids surfaced for 278 real tasks, inflating
`mean_turns_per_task` to an implausible ~17). Only the *task-grouped*
metrics (`mean_turns_per_task`, per-task `escalation_rate` as computed
by `analyze_decision_log`) were affected -- `decisions_driven_by_epistemic`,
`epistemic_share`, and `policy_counts` aggregate over the flat record
list, not per-task groups, so they were never wrong. Fixed with
`uuid.uuid4()`, which has no process-boundary assumption to violate.

**Lesson, twice over**: (1) a hand-specified generative model can look
broken when it's actually being fed corrupted or premature evidence --
verify the observation stream before touching the probabilities, same
lesson as #6 and #13 in this file, now three instances of the identical
mistake at different layers. (2) any "unique id" scheme needs to state
what scope it's unique *within* -- a counter is unique within a
process, not across the separate processes a real sweep actually runs.

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
- ~~**`retail/voi_agent`'s escalation rate jumped from 19/114 (original
  sweep) to 113/114 (the re-run under the transfer-tool fix, #13)**~~
  **RESOLVED by #15.** Re-run again under the opa-install + cold-start-skip
  fixes: escalations dropped to **2/113**. This confirms the anomaly was
  the same two upstream defects #15 diagnosed for EFE -- `policy_gate`
  corrupted to the fail-safe `"needs_review"` (this environment had no
  `opa` CLI) and the cold-start skip not covering "no tool attempted
  yet" -- not something specific to `VOIControlNode`'s own decision
  logic, and not unexplained sampling variance in the user-simulator.
  Worth noting for the record: this entry originally reasoned that opa's
  absence was "constant across both runs, not a new variable" and used
  that to rule it out as the explanation for the 19→113 *swing*. That
  reasoning was correct as far as it went (it doesn't explain a
  difference between two runs that both have the same defect) but
  missed the bigger point: a defect present in every compared run can
  still be the dominant cause of the *elevated baseline* both runs
  shared, relative to what turns out to be true once actually fixed.
  "Constant across my comparison" ruled out one hypothesis, not the
  category of hypothesis that included the real answer.
- **`telecom/efe_agent` responded much more weakly to #15's fix than
  retail/airline did, and its reward fell rather than improved.**
  Escalations dropped 114→10 on retail and 50→10 on airline (both
  under 20% of the original rate), but only 114→63 on telecom (55%
  still escalating) -- real and directionally consistent with the other
  two domains, so the fix's mechanism is confirmed general, but clearly
  weaker here. Reward moved the wrong way too: retail's reward nearly
  tripled (0.053→0.281, EFE went from the worst controller to beating
  heuristic) and airline's held flat, but telecom's fell (0.175→0.096).
  Not yet investigated why telecom differs -- candidates include a
  genuinely higher rate of telecom tasks that need early escalation
  once real evidence is available (plausible: telecom conversations
  may surface unresolvable account/plan issues faster than retail's
  order-lookup flows do), or a domain-specific issue in telecom's tool
  set/policy text not yet diagnosed. Left open rather than assumed
  away in either direction.
