"""Stage 5: the epistemic-vs-pragmatic interpretability analysis
(RESEARCH_PLAN.md Stage 5).

RESEARCH_PLAN.md lists "an interpretability method decomposing agent
decisions into epistemic vs pragmatic value" as one of the project's
expected contributions. The raw decomposition has been logged since
Stage 1 (`Decision.epistemic_value` / `.pragmatic_value`, one entry per
candidate policy per decision); this module is what reads it back and
answers the questions that decomposition exists to answer:

  1. When EFE chose a policy, was that choice driven by information gain
     or by goal-seeking? (`driver` per decision)
  2. Does the epistemic term actually do work, or is the pragmatic term
     dominating everything and the active-inference framing decorative?
     (`epistemic_share`, `decisions_driven_by_epistemic`)
  3. Does epistemic value fall as belief concentrates -- i.e. does the
     agent stop asking once it knows? (`epistemic_by_confidence_band`)

Question 2 is the one that can embarrass the thesis, which is exactly
why it's computed here rather than left to a hand-picked case study: if
the epistemic term never changes a decision, "EFE beats X" would be a
claim about a goal-seeking controller with extra machinery.

Baselines have no epistemic term by construction, so an analysis over
their logs is vacuous -- `analyze_decision_log` reports that explicitly
(`has_epistemic_decomposition=False`) rather than returning zeros that
look like a finding.
"""
from dataclasses import dataclass, field

from .decision_log import DecisionRecord, group_by_task

# A decision counts as epistemically driven when the chosen policy's
# information-gain term is what made it win: it must beat the runner-up
# on epistemic value, and it must NOT already win on pragmatic value
# alone (otherwise goal-seeking would have picked it anyway).
CONFIDENCE_BANDS = ((0.0, 0.5, "uncertain"), (0.5, 0.8, "moderate"), (0.8, 1.01, "confident"))


@dataclass
class ControllerInterpretability:
    controller: str
    n_decisions: int
    has_epistemic_decomposition: bool
    policy_counts: dict = field(default_factory=dict)
    mean_epistemic_of_chosen: float = 0.0
    mean_pragmatic_of_chosen: float = 0.0
    epistemic_share: float = 0.0
    decisions_driven_by_epistemic: int = 0
    epistemic_by_confidence_band: dict = field(default_factory=dict)
    mean_turns_per_task: float = 0.0
    escalation_rate: float = 0.0

    @property
    def epistemic_driven_rate(self) -> float:
        return self.decisions_driven_by_epistemic / self.n_decisions if self.n_decisions else 0.0


def _band_for(confidence: float) -> str:
    for low, high, label in CONFIDENCE_BANDS:
        if low <= confidence < high:
            return label
    return CONFIDENCE_BANDS[-1][2]


def _is_epistemically_driven(record: DecisionRecord) -> bool:
    """Would a purely pragmatic (goal-seeking) controller have made the
    same choice? If yes, the epistemic term didn't change anything."""
    if not record.pragmatic_value or not record.epistemic_value:
        return False
    chosen = record.chosen_policy
    if chosen not in record.pragmatic_value or chosen not in record.epistemic_value:
        return False

    pragmatic_winner = max(record.pragmatic_value, key=record.pragmatic_value.get)
    if pragmatic_winner == chosen:
        return False  # goal-seeking alone would have picked this

    others = [p for p in record.epistemic_value if p != chosen]
    if not others:
        return False
    best_other_epistemic = max(record.epistemic_value[p] for p in others)
    return record.epistemic_value[chosen] > best_other_epistemic


def analyze_decision_log(records: list[DecisionRecord]) -> ControllerInterpretability:
    """Analyze one controller's decisions. Pass a filtered list -- mixing
    controllers would average two different mechanisms into one
    meaningless number."""
    if not records:
        raise ValueError("no decision records to analyze")

    controllers = {r.controller for r in records}
    if len(controllers) > 1:
        raise ValueError(
            f"records span multiple controllers ({sorted(controllers)}) -- "
            "filter with load_decision_log(controller=...) first"
        )

    controller = records[0].controller
    has_epistemic = any(r.has_epistemic_decomposition for r in records)

    policy_counts: dict[str, int] = {}
    for record in records:
        policy_counts[record.chosen_policy] = policy_counts.get(record.chosen_policy, 0) + 1

    tasks = group_by_task(records)
    escalations = sum(
        1 for trace in tasks.values()
        if any(r.chosen_policy == "escalate_to_human" for r in trace)
    )

    result = ControllerInterpretability(
        controller=controller,
        n_decisions=len(records),
        has_epistemic_decomposition=has_epistemic,
        policy_counts=dict(sorted(policy_counts.items(), key=lambda kv: -kv[1])),
        mean_turns_per_task=len(records) / len(tasks) if tasks else 0.0,
        escalation_rate=escalations / len(tasks) if tasks else 0.0,
    )
    if not has_epistemic:
        # Baselines zero these fields by construction -- reporting a 0.0
        # epistemic share for them would read like a measurement.
        return result

    epistemic_of_chosen, pragmatic_of_chosen = [], []
    bands: dict[str, list[float]] = {}
    for record in records:
        chosen = record.chosen_policy
        e = record.epistemic_value.get(chosen, 0.0)
        p = record.pragmatic_value.get(chosen, 0.0)
        epistemic_of_chosen.append(e)
        pragmatic_of_chosen.append(p)
        bands.setdefault(_band_for(record.belief_confidence), []).append(e)
        if _is_epistemically_driven(record):
            result.decisions_driven_by_epistemic += 1

    result.mean_epistemic_of_chosen = sum(epistemic_of_chosen) / len(epistemic_of_chosen)
    result.mean_pragmatic_of_chosen = sum(pragmatic_of_chosen) / len(pragmatic_of_chosen)

    # Share of the decision signal attributable to information gain.
    # Magnitudes: pragmatic value is a log-preference (typically
    # negative), epistemic value an information gain (non-negative), so
    # comparing raw sums would be meaningless -- absolute values put
    # them on a comparable footing.
    total = abs(result.mean_epistemic_of_chosen) + abs(result.mean_pragmatic_of_chosen)
    result.epistemic_share = abs(result.mean_epistemic_of_chosen) / total if total else 0.0

    result.epistemic_by_confidence_band = {
        band: sum(values) / len(values) for band, values in sorted(bands.items())
    }
    return result


def format_interpretability_report(analyses: list[ControllerInterpretability]) -> str:
    """Human-readable report -- what actually goes in front of a reader,
    including the honest 'this controller has no epistemic term' case."""
    lines = ["# Interpretability report (Stage 5)", ""]
    for a in analyses:
        lines.append(f"## {a.controller}")
        lines.append(f"- decisions: {a.n_decisions} across "
                     f"{a.n_decisions / a.mean_turns_per_task:.0f} task(s)"
                     if a.mean_turns_per_task else f"- decisions: {a.n_decisions}")
        lines.append(f"- mean turns/task: {a.mean_turns_per_task:.2f}")
        lines.append(f"- escalation rate: {a.escalation_rate:.3f}")
        lines.append(f"- policy mix: {a.policy_counts}")

        if not a.has_epistemic_decomposition:
            lines.append("- **no epistemic decomposition** — this controller has no "
                         "information-gain term by construction, so the "
                         "epistemic/pragmatic split does not apply to it.")
            lines.append("")
            continue

        lines.append(f"- mean epistemic value of chosen policy: {a.mean_epistemic_of_chosen:.4f}")
        lines.append(f"- mean pragmatic value of chosen policy: {a.mean_pragmatic_of_chosen:.4f}")
        lines.append(f"- epistemic share of decision signal: {a.epistemic_share:.3f}")
        lines.append(f"- decisions the epistemic term actually changed: "
                     f"{a.decisions_driven_by_epistemic}/{a.n_decisions} "
                     f"({a.epistemic_driven_rate:.1%})")
        if a.epistemic_by_confidence_band:
            lines.append(f"- mean epistemic value by belief confidence: "
                         f"{ {k: round(v, 4) for k, v in a.epistemic_by_confidence_band.items()} }")
        if a.decisions_driven_by_epistemic == 0:
            lines.append("  > NOTE: the epistemic term never changed a decision in this log. "
                         "A purely goal-seeking controller would have chosen identically — "
                         "worth investigating before citing the active-inference framing "
                         "as load-bearing.")
        lines.append("")
    return "\n".join(lines)
