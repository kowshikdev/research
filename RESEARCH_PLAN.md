# Active Inference & Predictive-Coding Control for Grounded LLM Multi-Agent Orchestration

Research/thesis plan. Status: **Stage 1 partially complete, blocked on
API credentials.** Stage 0.5 kill-test done (positive result, §4a). Stage
1's EFE control node and LangGraph wiring (incl. the `escalate_to_human`
→ `interrupt()` pause) are built and proven with a mock agent — see
[`context/HANDOFF.md`](context/HANDOFF.md) and
[`context/TODOS.md`](context/TODOS.md) for exactly what's done vs.
blocked and how to resume. The next step (swap the mock agent for a real
tool-calling LLM) needs LLM API credentials this environment doesn't have.

## 1. Thesis statement

Reframe the heuristic control-loop decisions of an LLM multi-agent orchestrator
(retry / gather-more-information / escalate-to-human / hand-off) as an
approximate **Expected Free Energy (EFE)** computation that trades epistemic
value (information gain) against pragmatic/goal value, and benchmark it
against heuristic and learned-router baselines on established open agent
benchmarks.

**Primary claim to test:** an EFE-driven control loop over a small, fixed
discrete decision space is more reliable and better-calibrated at
retry/escalate decisions than a tuned heuristic-threshold or learned-router
baseline, at acceptable added latency/cost — as measured on τ²-bench and
HiL-Bench.

**Sharpened secondary question (added after external review, see §4a):**
"EFE beats a heuristic" is a low bar — a well-tuned heuristic or a decision-
theoretic controller can already do reasonably well, and a skeptical
reviewer's real question is *what does EFE give you that a sufficiently
good learned router, or an explicitly hand-derived Bayesian
value-of-information (VOI) controller, doesn't?* The more defensible claim,
and the one the Stage 0.5 kill-test was designed to test, is: **EFE
reproduces near-Bayes-optimal value-of-information behavior automatically
from a generative-model spec, without hand-deriving the utility/VOI math
per decision type or domain.** That is a claim about engineering cost and
transfer, not raw performance, and it is the one the kill-test data
(§4a) actually supports.

A rigorous **null result** (EFE ≈ baseline on outcomes, wins only on
interpretability/calibration) is an accepted, publishable fallback outcome —
see §6.

## 2. Scope decision (fixed for the life of the project)

The single design choice that makes this tractable: the EFE computation runs
over a **small, frozen discrete decision-POMDP**, never over raw LLM token
space.

- **Hidden state factors (≤4):** `{task_solvable_now, needs_more_info,
  needs_human, likely_to_fail}`
- **Observations:** tool-call results, self-consistency/confidence signal,
  validator/OPA policy-gate verdict, retrieval-quality signal
- **Policies (≤6):** `{continue, retry, call_tool, gather_info,
  escalate_to_human, hand_off_to_agent}`
- **Preferences (C):** task success, low escalation rate, low cost/latency —
  encoded as prior preferences, sensitivity-tested (§5, Stage 3)

Do not let this grow mid-project — that is what keeps EFE evaluation
tractable (classical AIF is exponential in horizon × state-space size).

## 3. What we need — full resource inventory

### 3.1 Inference engines (active inference / free energy)

| Tool | Role | Verified state (Sep 2026) |
|---|---|---|
| [`pymdp`](https://github.com/infer-actively/pymdp) ([paper](https://arxiv.org/abs/2201.03904)) | Primary EFE engine — discrete POMDP active inference | v1.0.0 rebuilt on JAX (GPU/TPU, autodiff, JIT, `vmap` batching); v1.0.1 released 2026-04-28. Actively maintained. **PyPI install gotcha (confirmed Stage 0):** the package literally named `pymdp` on PyPI is an unrelated MDP toolkit by a different author — install `inferactively-pymdp` instead (still `import pymdp` in code). |
| [`RxInfer.jl`](https://github.com/ReactiveBayes/RxInfer.jl) | Fallback engine if pymdp's discrete POMDP proves too rigid for continuous/mixed observation signals | Variational Constrained Bethe Free Energy via reactive message passing; supports hybrid discrete/continuous models. Actively maintained (JOSS paper published). |
| [`ngc-learn`](https://github.com/NACLab/ngc-learn) (NACLab) | Angle B only — predictive-coding / generative memory | JAX-based, requires Python ≥3.10, JAX ≥0.4.28. Actively maintained; companion `ngc-museum` has reference models. |

### 3.2 Orchestration / agent infrastructure (reuse existing stack)

- **LangGraph** — `interrupt()` + checkpointer (e.g. `MemorySaver` or a
  persistent backend) for pausing on `escalate_to_human`; this is the
  first-class, currently-maintained, industry-standard HITL primitive
  (confirmed against current LangChain docs) — wire the EFE control node's
  `escalate` policy directly to it. **LangRun is dropped** from this plan —
  it's not an industry-standard tool and hasn't been fully tested; use
  LangGraph's own checkpointer/interrupt mechanism as the single HITL
  pause point instead of layering a second, less-proven review system on
  top.
- **OPA (Open Policy Agent)** policy gate verdicts — feed into the
  decision-POMDP as an observation modality.
- **OpenTelemetry GenAI semantic conventions** — `gen_ai.agent` /
  `invoke_agent` spans for logging epistemic-vs-pragmatic value per
  decision. Note: as of mid-2026 `gen_ai.*` conventions moved to a
  dedicated `semantic-conventions-genai` repo and agent spans are still
  **experimental** (not stable) — pin the conventions version used and
  expect attribute names to still move.

### 3.3 LLM backbones (need at least one strong tool-calling model)

- One frontier tool-calling model for the agent under test — pick whichever
  the user already has API access to (Claude, GPT-4.x/5, etc.); τ²-bench and
  GAIA are model-agnostic harnesses, so this is a config choice, not a
  research dependency.
- Keep the model **fixed across all conditions** (EFE agent, heuristic
  baseline, router baseline, ReAct baseline) — the independent variable is
  the control loop, not the LLM.

### 3.4 Benchmarks / datasets (all open, verified reachable)

| Benchmark | Use | Verified state (Sep 2026) |
|---|---|---|
| [τ²-bench](https://github.com/sierra-research/tau2-bench) (also mirrored at `LLM360/tau2-bench`; verified variant at `amazon-agi/tau2-bench-verified`) | **Primary.** Tool-agent-user interaction with domain policies; measures reliability via pass^k | Domains now include `mock, airline, retail, telecom, banking_knowledge` (telecom/banking added since the original τ-bench paper). Actively maintained; has a `dev/tau3` branch in progress — pin a commit/tag before running, and record the grader version (τ²-bench has publicly noted non-comparable scores across grading updates). |
| [HiL-Bench](https://arxiv.org/abs/2604.09408) | **Primary.** Selective escalation via Ask-F1 (SWE + text-to-SQL domains, human-validated blockers) | Published 2026-04-10 (arXiv, v2 exists). Confirmed finding: frontier models get 75–89% pass@3 with full info but only 4–24% when they must judge whether to ask — large headroom for a calibrated escalation method. |
| [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) | Secondary — generalist agent success rate | 466 human-verified tasks; public dev/validation set, private test set behind the HF leaderboard (test set never published in plaintext — good contamination defense). Top 2026 scores ~74–75% (Claude Sonnet family leads the Princeton HAL leaderboard as of Apr 2026) vs. ~92% human. |
| WebArena / SWE-bench | Optional generalization check | Standard, open; use only if Angle A finishes early. |

### 3.5 Compute

- CPU is sufficient for pymdp's discrete POMDP inference over the frozen
  small decision space (this is the point of §2 — no GPU needed for the EFE
  engine itself).
- GPU only needed if Angle B (ngc-learn predictive-coding memory) is
  pursued, or if pymdp's JAX backend is used for batched simulation sweeps.
- Budget for LLM API calls: τ²-bench + HiL-Bench + GAIA runs, ×4 conditions
  (EFE, heuristic, router, ReAct), ×multiple trials for pass^k — this is the
  dominant real cost, not compute.

### 3.6 Baselines to implement (must be tuned as hard as the EFE agent)

1. **Heuristic threshold** — `escalate ⟺ p_success < τ_conf ∨ action ∈
   irreversible ∨ cost > budget` (current production-style pattern).
2. **Learned router / contextual bandit** — trained on the same
   observation features the EFE engine consumes. **Must have belief-state
   parity with EFE/VOI** — i.e. access to a running summary of the full
   observation history, not just the latest reading — or any gap in the
   comparison is arguably a feature-engineering artifact, not a finding
   about control mechanisms. (Confirmed necessary by the Stage 0.5
   kill-test — see §4a.)
3. **Decision-theoretic / VOI controller** — an explicit Bayesian
   belief-update + expected-utility computation, the "stronger baseline"
   added after external review (§4a). At kill-test scale this was
   hand-derivable exactly; at Stage 3 scale (real LLM agent, unknown true
   generative model) this becomes an LLM-estimated P(success | context) +
   hand-coded cost/utility function — include it if feasible, since it's
   the sharpest test of what EFE's formalism specifically buys over
   explicit decision theory.
4. **Plain ReAct** — no explicit control-loop reasoning, floor baseline.

## 4. Stage-by-stage plan

### Stage 0 — Setup & literature lock (weeks 1–2)
- Pin exact versions/commits: `pymdp` (≥1.0.1), `tau2-bench` (tag or commit
  SHA, record grader version), `ngc-learn` if Angle B is in scope.
- Stand up LangGraph HITL scaffold with `interrupt()` + checkpointer as the
  escalation sink.
- Freeze the decision-POMDP definition from §2 in a single config file —
  changing it later requires an explicit ADR-style note in this repo.
- Deliverable: `docs/decision-pomdp.md` (frozen schema) + working pymdp
  T-maze/epistemic-chaining demo run locally as an engine sanity check.

### Stage 0.5 — Kill-test (added after external review; completed)

**Why this stage exists:** an external review of this plan (paraphrased):
*"the biggest issue isn't whether pymdp works, it's — what does EFE give
you that a sufficiently good learned router doesn't? Don't start the
20-week project; start a 1-2 week kill-test. If EFE is basically the same
as a good router with more machinery, that's extremely valuable to
discover early."* Agreed, and acted on — this stage was inserted before
any LangGraph/benchmark work, using only what Stage 0 had already built
(a working pymdp engine, the frozen decision-POMDP shape).

**What was built** (`src/aif_orchestrator/kill_test/`): a tiny synthetic
3-state / 3-action escalation environment with noisy observations shaped
like real confidence signals, and five controllers evaluated on 3000
held-out episodes each: random, heuristic, a Q-learning router, a
hand-derived Bayesian VOI controller (the "stronger baseline" the review
specifically asked for), and an EFE controller (pymdp) — VOI and EFE given
the identical true generative model, so the test is about the *mechanism*,
not about who has more information.

**Result (full detail: [`docs/stage0.5-kill-test-results.md`](docs/stage0.5-kill-test-results.md)):**

| Controller | avg reward (95% CI) |
|---|---|
| heuristic | 0.374 [0.345, 0.403] |
| learned_router | 0.432 [0.407, 0.456] |
| voi_decision_theoretic | 0.487 [0.463, 0.510] |
| efe_active_inference | 0.482 [0.458, 0.505] |

- EFE ≈ VOI (CIs heavily overlap) — EFE reproduces near-Bayes-optimal
  value-of-information behavior **without hand-deriving the utility
  math**, which is exactly the sharpened claim in §1.
- EFE/VOI both beat the router with non-overlapping CIs — a real gap, not
  noise, concentrated in unnecessary-escalation rate (0.019 vs. 0.042).
- **Caveat carried into Stage 2:** the router here is memory-limited
  (features = latest observation + step count, not a running belief) —
  some of its gap is plausibly a feature-parity issue. Fix before this
  finding is reported at benchmark scale (§3.6 baseline #2).

**Decision:** proceed to Stage 1.

### Stage 1 — EFE control node (weeks 3–6) — partially complete, blocked

- [x] EFE control node implementing the real decision-POMDP (4 states, 4
  observation modalities, 6 policies) — `src/aif_orchestrator/efe_controller.py`.
  A/B/C/E/D values are hand-specified placeholders matching
  `docs/decision-pomdp.md`'s semantics, not yet calibrated against real
  data (Stage 3 work).
- [x] LangGraph wiring — `src/aif_orchestrator/graph.py`. `escalate_to_human`
  genuinely pauses via `interrupt()` + `MemorySaver` checkpointer and
  resumes correctly with `Command(resume=...)`; multi-turn belief tracking
  proven with a mock (non-LLM) agent step.
- [x] Per-decision epistemic/pragmatic value logging — currently plain
  JSONL (`results/stage1_decision_log.jsonl`), not yet real OTel GenAI
  spans (needs an actual collector/backend — infra decision, not just code).
- [ ] **Blocked:** swap the mock agent step for a real tool-calling LLM
  agent — needs LLM API credentials not present in this environment.
  See `context/TODOS.md` for the exact next steps.
- [ ] Not started: real OPA instance for the `policy_gate` observation
  (mock currently hardcodes `allow`); real confidence derivation
  (self-consistency sampling or a verifier prompt).
- Original deliverable (EFE node on real τ²-bench tasks with real
  epistemic/pragmatic logs) still pending the above.

### Stage 2 — Baselines (weeks 5–7, parallel with Stage 1 tail)
- Implement heuristic-threshold, learned-router, and ReAct baselines against
  the identical LangGraph scaffold and identical observation features.
- Deliverable: all four conditions runnable via one CLI flag on a small task
  subset.

### Stage 3 — Primary evaluation (weeks 8–12)
- Run all four conditions on τ²-bench (retail, airline, telecom) and
  HiL-Bench (SWE + text-to-SQL).
- Metrics: task success / pass^k, Ask-F1, over-asking vs. silent-failure
  rate, human-escalation precision/recall, token/latency/$ cost.
- Ablations: remove epistemic term (pure goal-seeking), vary preference
  prior C, vary planning horizon.
- **Decision checkpoint:** if EFE is within noise of the heuristic baseline
  on both success and Ask-F1, pivot framing to "matches with superior
  interpretability/calibration" (§6) rather than "outperforms."
- Deliverable: results table + ablation plots.

### Stage 4 — Secondary evaluation / generalization (weeks 12–14, optional)
- GAIA validation subset run for all four conditions, if Stage 3 finished on
  schedule.
- Angle B (predictive-coding memory vs. RAG) as an independent side study —
  only if Stage 3 finished early; treat as a second paper, not a dependency.

### Stage 5 — Interpretability analysis (weeks 13–15)
- Decompose logged decisions into epistemic vs. pragmatic value
  contributions; correlate with HiL-Bench's human-validated blocker labels.
- Deliverable: interpretability figures/case studies for the write-up.

### Stage 6 — Writing (weeks 15–20)
- Literature review (structure already scoped in the original plan: FEP/AIF
  foundations → computational instantiations → LLM agents/orchestration →
  convergence frontier → gap statement).
- Methods, results, ablations, limitations, threats to validity.
- Open-source the reference implementation on top of LangGraph.

## 5. Risks / pivot triggers (carried over, still current)

- **Null result** on Stage 3 → reframe as calibration/interpretability win,
  not outcome win (still publishable given the literature gap in §6).
- **EFE latency dominates task latency** → switch from exact enumeration to
  an amortized/variational EFE approximation.
- **pymdp's discrete POMDP too rigid** → switch engine to `RxInfer.jl`.
- **Decision space creep** → explicitly forbidden; any expansion needs a
  written justification and a re-run of the Stage 0 tractability check.
- **Benchmark grader drift** — τ²-bench has already had non-comparable
  re-grades; pin and record grader version for every reported number.
- **Baseline sandbagging** — router/heuristic baselines must get equal
  tuning effort, tracked explicitly, or the comparison is discounted.
- **Router belief-state parity** — confirmed a real issue by the Stage 0.5
  kill-test: a reactive router (features = latest observation only) is not
  a fair comparison against EFE/VOI's running belief. Give the router
  baseline an equivalent state summary before Stage 3 results are reported.

## 6. Why this is still the gap (confirms original framing)

Verified during this planning pass: HiL-Bench (Apr 2026) and τ²-bench's
telecom/banking expansion are both recent enough that no active-inference-
for-LLM paper has used them — the literature gap (toy environments, no
standardized benchmark, no matched baselines) identified in the original
plan still holds as of September 2026.

## 7. Next action

Stage 0 (decision-POMDP schema + pymdp engine sanity check) and Stage 0.5
(kill-test, positive result) are both complete. Next: Stage 1 — build the
EFE control node as an actual LangGraph node, wired to real observation
signals (tool output, confidence, OPA verdict) instead of the kill-test's
synthetic ones, with `escalate_to_human` routed to `interrupt()`.
