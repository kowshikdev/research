"""Reader for the JSONL decision logs written by graph.py's
`make_control_step` (and, in the same schema, by tau2_integration's
decision traces).

The logs are append-only by design (docs/langgraph-integration.md), so a
file can contain records from many runs and many controllers
interleaved. Everything here treats that as normal rather than
something to guard against -- filtering is the caller's job, and
`load_decision_log` never silently drops records it doesn't understand.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class DecisionRecord:
    """One control-node decision. Mirrors the dict written by
    graph.make_control_step -- see docs/langgraph-integration.md."""

    task_id: str
    turn: int
    controller: str
    observation: dict
    chosen_policy: str
    belief: dict
    action_marginals: dict = field(default_factory=dict)
    epistemic_value: dict = field(default_factory=dict)
    pragmatic_value: dict = field(default_factory=dict)

    @property
    def has_epistemic_decomposition(self) -> bool:
        """True only for EFE -- the baselines zero these fields by
        construction (baselines/belief.uniform_decision), so an
        interpretability analysis over a baseline's log is vacuous
        rather than wrong, and callers should know which they have."""
        return any(v != 0.0 for v in self.epistemic_value.values())

    @property
    def most_likely_state(self) -> str:
        return max(self.belief, key=self.belief.get)

    @property
    def belief_confidence(self) -> float:
        return max(self.belief.values()) if self.belief else 0.0


REQUIRED_FIELDS = {"task_id", "turn", "observation", "chosen_policy", "belief"}

# `controller` was added to the record schema when graph.py's hardcoded
# efe_control_step was generalized into make_control_step (Stage 2). The
# earliest Stage 1 logs -- including ones committed to this repo -- predate
# it. Those records are all EFE by construction (it was the only
# controller that existed), but labelling them "efe_active_inference"
# here would be inferring data that isn't in the file, so they get an
# explicit legacy marker instead: visible in any report, and impossible
# to mistake for a real controller's results.
LEGACY_CONTROLLER = "unknown (pre-Stage-2 log, no controller field)"


def load_decision_log(path, controller=None, task_id=None) -> list[DecisionRecord]:
    """Load a decision log, optionally filtered by controller/task.

    Malformed lines raise rather than being skipped: a decision log with
    silently-dropped records would produce analysis numbers that look
    fine and are quietly computed over a subset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no decision log at {path}")

    records = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON") from exc

        missing = REQUIRED_FIELDS - payload.keys()
        if missing:
            raise ValueError(f"{path}:{lineno} missing required field(s): {sorted(missing)}")

        record = DecisionRecord(
            task_id=payload["task_id"],
            turn=payload["turn"],
            controller=payload.get("controller", LEGACY_CONTROLLER),
            observation=payload["observation"],
            chosen_policy=payload["chosen_policy"],
            belief=payload["belief"],
            action_marginals=payload.get("action_marginals", {}),
            epistemic_value=payload.get("epistemic_value", {}),
            pragmatic_value=payload.get("pragmatic_value", {}),
        )
        if controller is not None and record.controller != controller:
            continue
        if task_id is not None and record.task_id != task_id:
            continue
        records.append(record)

    return records


def controllers_in(records: list[DecisionRecord]) -> list[str]:
    return sorted({r.controller for r in records})


def group_by_task(records: list[DecisionRecord]) -> dict[str, list[DecisionRecord]]:
    grouped: dict[str, list[DecisionRecord]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record)
    for trace in grouped.values():
        trace.sort(key=lambda r: r.turn)
    return grouped
