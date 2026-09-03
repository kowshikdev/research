"""Stage 3 reporting: turns a tau2-bench sweep into the per-controller
comparison table RESEARCH_PLAN.md's Stage 3 actually reports.

Two input sources, deliberately separated by how much they can be
trusted:

  * `results/stage3_tau2/summary.json` -- written by our OWN code
    (tau2_integration/run_stage3_eval.py), so its exact schema is known:
    {"<domain>/<agent>": {n_simulations, avg_reward, pass_1, escalations,
    elapsed_sec}}. This is the primary path and is fully supported.

  * `results/stage3_tau2/<domain>__<agent>.json` -- tau2's own serialized
    `SimulationResults`. The attribute paths used here
    (`simulations[].reward_info.reward`, `simulations[].messages[].tool_calls[].name`)
    are the ones run_stage3_eval.py already reads successfully from the
    live objects, so the serialized key names are expected to match --
    but **this path has never run against a real dump**, because no
    sweep has been executed yet (context/TODOS.md). It is written
    defensively and reports what it could not parse rather than
    guessing. Verify it against the first real sweep before trusting
    per-task numbers from it.

Run: .venv/bin/python -m aif_orchestrator.analysis.stage3_report
"""
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results" / "stage3_tau2"
TRANSFER_TOOL_NAME = "transfer_to_human_agents"


@dataclass
class AgentDomainResult:
    domain: str
    agent: str
    n_simulations: int
    avg_reward: float | None
    pass_1: float | None
    escalations: int
    elapsed_sec: float | None = None
    # Only available from raw dumps (see module docstring).
    escalation_precision: float | None = None
    escalation_recall: float | None = None
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def escalation_rate(self) -> float | None:
        if not self.n_simulations:
            return None
        return self.escalations / self.n_simulations


def load_summary(path=None) -> list[AgentDomainResult]:
    """Load the sweep summary our own runner writes. Fully supported."""
    path = Path(path) if path else RESULTS_DIR / "summary.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no sweep summary at {path} — the Stage 3 sweep has not run yet "
            "(see context/TODOS.md; run tau2_integration/run_stage3_eval.py first)"
        )

    payload = json.loads(path.read_text())
    results = []
    for key, metrics in payload.items():
        domain, _, agent = key.partition("/")
        results.append(AgentDomainResult(
            domain=domain,
            agent=agent,
            n_simulations=metrics.get("n_simulations", 0),
            avg_reward=metrics.get("avg_reward"),
            pass_1=metrics.get("pass_1"),
            escalations=metrics.get("escalations", 0),
            elapsed_sec=metrics.get("elapsed_sec"),
        ))
    return sorted(results, key=lambda r: (r.domain, r.agent))


def enrich_from_raw_dump(result: AgentDomainResult, results_dir=None) -> AgentDomainResult:
    """Best-effort per-task escalation precision/recall from tau2's own
    dump. UNVERIFIED against real data -- see module docstring. Records
    what it couldn't parse in `parse_warnings` instead of returning
    numbers it can't stand behind.

    Ground truth for "should this task have been escalated" is taken as
    'the simulation scored reward 0 without escalating' being a miss and
    'escalated on a task that was otherwise solvable' being a false
    alarm -- a proxy, not tau2's own notion of a correct transfer. Treat
    these two fields as directional until validated against tau2's
    reward_info breakdown on a real sweep.
    """
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    dump = results_dir / f"{result.domain}__{result.agent}.json"
    if not dump.exists():
        result.parse_warnings.append(f"no raw dump at {dump}")
        return result

    try:
        payload = json.loads(dump.read_text())
    except json.JSONDecodeError as exc:
        result.parse_warnings.append(f"raw dump is not valid JSON: {exc}")
        return result

    simulations = payload.get("simulations")
    if not isinstance(simulations, list):
        result.parse_warnings.append(
            "raw dump has no 'simulations' list — tau2's serialized schema may "
            "differ from what this reader expects (never validated against a real dump)"
        )
        return result

    escalated_and_solved = escalated_total = missed = solved_total = 0
    for sim in simulations:
        reward = (sim.get("reward_info") or {}).get("reward")
        escalated = any(
            tc.get("name") == TRANSFER_TOOL_NAME
            for message in (sim.get("messages") or [])
            for tc in (message.get("tool_calls") or [])
        )
        succeeded = reward == 1.0
        if escalated:
            escalated_total += 1
            if succeeded:
                escalated_and_solved += 1
        elif not succeeded:
            missed += 1
        if succeeded:
            solved_total += 1

    if escalated_total:
        result.escalation_precision = escalated_and_solved / escalated_total
    if (escalated_total + missed):
        result.escalation_recall = escalated_total / (escalated_total + missed)
    return result


def format_report(results: list[AgentDomainResult]) -> str:
    lines = ["# Stage 3 tau2-bench sweep report", ""]
    if not results:
        return "\n".join(lines + ["No results."])

    def num(value, fmt=".3f"):
        return "n/a" if value is None else format(value, fmt)

    has_pr = any(r.escalation_precision is not None or r.escalation_recall is not None
                 for r in results)

    for domain in sorted({r.domain for r in results}):
        rows = sorted(
            (r for r in results if r.domain == domain),
            key=lambda r: (r.avg_reward is None, -(r.avg_reward or 0)),
        )
        header = "| agent | sims | avg reward | pass^1 | escalations | escalation rate |"
        divider = "|---|---:|---:|---:|---:|---:|"
        if has_pr:
            header += " esc. precision | esc. recall |"
            divider += "---:|---:|"

        lines += [f"## {domain}", "", header, divider]
        for r in rows:
            row = (f"| {r.agent} | {r.n_simulations} | {num(r.avg_reward)} "
                   f"| {num(r.pass_1)} | {r.escalations} | {num(r.escalation_rate)} |")
            if has_pr:
                row += f" {num(r.escalation_precision)} | {num(r.escalation_recall)} |"
            lines.append(row)
        lines.append("")

    warnings = [(r, w) for r in results for w in r.parse_warnings]
    if warnings:
        lines += ["## Parse warnings", ""]
        lines += [f"- `{r.domain}/{r.agent}`: {w}" for r, w in warnings]
        lines.append("")

    lines += [
        "> Escalation precision/recall (when present) come from the raw-dump "
        "reader, which has never been validated against a real tau2 dump — "
        "see `analysis/stage3_report.py`'s module docstring before citing them.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", default=None, help="path to summary.json")
    parser.add_argument("--raw", action="store_true",
                        help="also attempt per-task enrichment from tau2's raw dumps (unvalidated)")
    parser.add_argument("--out", default=None, help="write the report here instead of stdout")
    args = parser.parse_args()

    results = load_summary(args.summary)
    if args.raw:
        results = [enrich_from_raw_dump(r) for r in results]

    report = format_report(results)
    if args.out:
        Path(args.out).write_text(report)
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
