"""Tests for the analysis layer (src/aif_orchestrator/analysis/).

This layer is the one that produces the numbers the thesis reports, so
its failure mode is the worst kind: a plausible-looking number computed
over the wrong subset. Hence the emphasis here on the loader refusing to
silently drop records, and on the interpretability analysis refusing to
mix controllers or to report a 0.0 epistemic share for a baseline that
has no epistemic term at all.
"""
import json

import pytest

from aif_orchestrator.analysis.decision_log import (
    DecisionRecord,
    group_by_task,
    load_decision_log,
)
from aif_orchestrator.analysis.interpretability import (
    analyze_decision_log,
    format_interpretability_report,
)
from aif_orchestrator.analysis import stage3_report

POLICIES = ["continue", "retry", "call_tool", "gather_info", "escalate_to_human", "hand_off_to_agent"]


def _record(task_id="t1", turn=0, controller="efe_active_inference", policy="gather_info",
            belief=None, epistemic=None, pragmatic=None):
    return {
        "task_id": task_id,
        "turn": turn,
        "controller": controller,
        "observation": {"tool_result": "partial", "confidence": "medium",
                        "policy_gate": "allow", "retrieval_quality": "adequate"},
        "chosen_policy": policy,
        "belief": belief or {"task_solvable_now": 0.3, "needs_more_info": 0.4,
                             "needs_human": 0.2, "likely_to_fail": 0.1},
        "action_marginals": {p: 1.0 if p == policy else 0.0 for p in POLICIES},
        "epistemic_value": epistemic or {p: 0.0 for p in POLICIES},
        "pragmatic_value": pragmatic or {p: 0.0 for p in POLICIES},
    }


def _write_log(tmp_path, records, name="log.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


# --- loader -----------------------------------------------------------------

def test_loads_records_and_preserves_order(tmp_path):
    path = _write_log(tmp_path, [_record(turn=0), _record(turn=1), _record(turn=2)])
    records = load_decision_log(path)
    assert [r.turn for r in records] == [0, 1, 2]
    assert all(isinstance(r, DecisionRecord) for r in records)


def test_filters_by_controller_and_task(tmp_path):
    path = _write_log(tmp_path, [
        _record(task_id="a", controller="efe_active_inference"),
        _record(task_id="b", controller="heuristic"),
        _record(task_id="a", controller="heuristic"),
    ])
    assert len(load_decision_log(path, controller="heuristic")) == 2
    assert len(load_decision_log(path, task_id="a")) == 2
    assert len(load_decision_log(path, controller="heuristic", task_id="a")) == 1


def test_malformed_lines_raise_rather_than_being_skipped(tmp_path):
    """A silently-skipped record produces analysis numbers computed over
    an unannounced subset -- worse than a crash."""
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(_record()) + "\nnot json at all\n")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_decision_log(path)


def test_missing_required_field_raises(tmp_path):
    incomplete = _record()
    del incomplete["belief"]
    path = tmp_path / "incomplete.jsonl"
    path.write_text(json.dumps(incomplete) + "\n")
    with pytest.raises(ValueError, match="missing required field"):
        load_decision_log(path)


def test_legacy_records_without_a_controller_field_are_labelled_not_guessed(tmp_path):
    """Stage 1 logs predate the `controller` field (added when
    make_control_step was generalized). Those records are EFE by
    construction, but inferring that here would be writing data that
    isn't in the file -- so they get an explicit legacy marker that can't
    be mistaken for a real controller's results."""
    from aif_orchestrator.analysis.decision_log import LEGACY_CONTROLLER

    legacy = _record()
    del legacy["controller"]
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(legacy) + "\n")

    records = load_decision_log(path)
    assert records[0].controller == LEGACY_CONTROLLER
    assert "unknown" in LEGACY_CONTROLLER


def test_the_repos_own_committed_logs_are_loadable():
    """Guards against the analysis layer being unable to read the very
    data this project has been accumulating -- which is exactly what
    happened on first contact with the committed Stage 1 log."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    for name in ("stage1_decision_log.jsonl", "stage2_decision_log.jsonl"):
        log = repo_root / "results" / name
        if not log.exists():
            continue
        records = load_decision_log(log)
        assert records, f"{name} loaded but produced no records"


def test_missing_file_raises_a_useful_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_decision_log(tmp_path / "nope.jsonl")


def test_blank_lines_are_tolerated(tmp_path):
    path = tmp_path / "gappy.jsonl"
    path.write_text(json.dumps(_record()) + "\n\n" + json.dumps(_record(turn=1)) + "\n")
    assert len(load_decision_log(path)) == 2


def test_group_by_task_sorts_turns():
    records = [DecisionRecord(**{**_record(task_id="t", turn=t), "observation": {}})
               for t in (2, 0, 1)]
    grouped = group_by_task(records)
    assert [r.turn for r in grouped["t"]] == [0, 1, 2]


# --- interpretability -------------------------------------------------------

def test_refuses_to_mix_controllers(tmp_path):
    """Averaging two different decision mechanisms into one number would
    be meaningless, so it's an error rather than a silent average."""
    path = _write_log(tmp_path, [
        _record(controller="efe_active_inference"),
        _record(controller="heuristic"),
    ])
    with pytest.raises(ValueError, match="multiple controllers"):
        analyze_decision_log(load_decision_log(path))


def test_baseline_without_epistemic_term_is_reported_as_such(tmp_path):
    """A baseline's zeroed epistemic fields must not be reported as a
    measured epistemic share of 0.0 -- that reads like a finding."""
    path = _write_log(tmp_path, [_record(controller="heuristic", policy="continue")])
    analysis = analyze_decision_log(load_decision_log(path))
    assert analysis.has_epistemic_decomposition is False
    assert analysis.epistemic_share == 0.0
    assert "no epistemic decomposition" in format_interpretability_report([analysis])


def test_detects_when_the_epistemic_term_actually_changed_a_decision(tmp_path):
    """gather_info loses on pragmatic value but wins on information gain
    -- exactly the case the decomposition exists to identify."""
    path = _write_log(tmp_path, [_record(
        policy="gather_info",
        epistemic={**{p: 0.1 for p in POLICIES}, "gather_info": 0.9},
        pragmatic={**{p: -1.0 for p in POLICIES}, "continue": -0.2},
    )])
    analysis = analyze_decision_log(load_decision_log(path))
    assert analysis.has_epistemic_decomposition is True
    assert analysis.decisions_driven_by_epistemic == 1
    assert analysis.epistemic_driven_rate == 1.0


def test_does_not_credit_the_epistemic_term_when_goal_seeking_agrees(tmp_path):
    """If the chosen policy already wins on pragmatic value, the
    epistemic term changed nothing and must not be credited."""
    path = _write_log(tmp_path, [_record(
        policy="continue",
        epistemic={**{p: 0.1 for p in POLICIES}, "continue": 0.9},
        pragmatic={**{p: -1.0 for p in POLICIES}, "continue": -0.1},
    )])
    analysis = analyze_decision_log(load_decision_log(path))
    assert analysis.decisions_driven_by_epistemic == 0


def test_report_flags_a_log_where_epistemic_value_never_mattered(tmp_path):
    """The result that would embarrass the thesis gets called out
    explicitly rather than left for a reader to notice."""
    path = _write_log(tmp_path, [_record(
        policy="continue",
        epistemic={p: 0.5 for p in POLICIES},
        pragmatic={**{p: -1.0 for p in POLICIES}, "continue": -0.1},
    )])
    report = format_interpretability_report([analyze_decision_log(load_decision_log(path))])
    assert "never changed a decision" in report


def test_aggregates_policy_mix_turns_and_escalation_rate(tmp_path):
    path = _write_log(tmp_path, [
        _record(task_id="a", turn=0, policy="gather_info"),
        _record(task_id="a", turn=1, policy="escalate_to_human"),
        _record(task_id="b", turn=0, policy="continue"),
    ])
    analysis = analyze_decision_log(load_decision_log(path))
    assert analysis.n_decisions == 3
    assert analysis.mean_turns_per_task == pytest.approx(1.5)
    assert analysis.escalation_rate == pytest.approx(0.5)  # 1 of 2 tasks escalated
    assert analysis.policy_counts["gather_info"] == 1


def test_epistemic_value_is_bucketed_by_belief_confidence(tmp_path):
    confident = {"task_solvable_now": 0.95, "needs_more_info": 0.03,
                 "needs_human": 0.01, "likely_to_fail": 0.01}
    uncertain = {"task_solvable_now": 0.3, "needs_more_info": 0.3,
                 "needs_human": 0.2, "likely_to_fail": 0.2}
    epistemic = {**{p: 0.1 for p in POLICIES}, "gather_info": 0.8}
    path = _write_log(tmp_path, [
        _record(task_id="a", belief=confident, epistemic=epistemic),
        _record(task_id="b", belief=uncertain, epistemic=epistemic),
    ])
    analysis = analyze_decision_log(load_decision_log(path))
    assert set(analysis.epistemic_by_confidence_band) == {"confident", "uncertain"}


def test_analyze_rejects_empty_input():
    with pytest.raises(ValueError):
        analyze_decision_log([])


# --- stage 3 report ---------------------------------------------------------

def test_stage3_summary_loads_and_reports(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "retail/efe_agent": {"n_simulations": 10, "avg_reward": 0.4, "pass_1": 0.3,
                             "escalations": 2, "elapsed_sec": 12.0},
        "retail/react_agent": {"n_simulations": 10, "avg_reward": 0.2, "pass_1": 0.1,
                               "escalations": 0, "elapsed_sec": 9.0},
    }))
    results = stage3_report.load_summary(summary)
    assert [r.agent for r in results] == ["efe_agent", "react_agent"]
    assert results[0].escalation_rate == pytest.approx(0.2)

    report = stage3_report.format_report(results)
    assert "efe_agent" in report and "react_agent" in report
    # best avg_reward first
    assert report.index("efe_agent") < report.index("react_agent")


def test_stage3_summary_missing_file_says_the_sweep_has_not_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="has not run yet"):
        stage3_report.load_summary(tmp_path / "summary.json")


def test_stage3_raw_enrichment_records_a_warning_when_the_dump_is_absent(tmp_path):
    result = stage3_report.AgentDomainResult(
        domain="retail", agent="efe_agent", n_simulations=1,
        avg_reward=1.0, pass_1=1.0, escalations=0,
    )
    enriched = stage3_report.enrich_from_raw_dump(result, results_dir=tmp_path)
    assert enriched.parse_warnings
    assert enriched.escalation_precision is None


def test_stage3_raw_enrichment_computes_precision_and_recall(tmp_path):
    """Exercises the raw-dump reader against the schema it expects. This
    is the path flagged as never having run against a real tau2 dump --
    the test pins the expected shape so a mismatch shows up as a test
    failure rather than a wrong number in a report."""
    (tmp_path / "retail__efe_agent.json").write_text(json.dumps({
        "simulations": [
            # escalated and solved -> a correct transfer
            {"reward_info": {"reward": 1.0},
             "messages": [{"tool_calls": [{"name": "transfer_to_human_agents"}]}]},
            # escalated, not solved
            {"reward_info": {"reward": 0.0},
             "messages": [{"tool_calls": [{"name": "transfer_to_human_agents"}]}]},
            # not escalated, not solved -> a miss
            {"reward_info": {"reward": 0.0}, "messages": [{"tool_calls": []}]},
            # not escalated, solved -> fine
            {"reward_info": {"reward": 1.0}, "messages": []},
        ]
    }))
    result = stage3_report.AgentDomainResult(
        domain="retail", agent="efe_agent", n_simulations=4,
        avg_reward=0.5, pass_1=0.5, escalations=2,
    )
    enriched = stage3_report.enrich_from_raw_dump(result, results_dir=tmp_path)
    assert enriched.escalation_precision == pytest.approx(0.5)   # 1 of 2 escalations solved
    assert enriched.escalation_recall == pytest.approx(2 / 3)     # 2 escalated of 3 that needed it
    assert not enriched.parse_warnings


def test_stage3_raw_enrichment_warns_on_unexpected_schema(tmp_path):
    (tmp_path / "retail__efe_agent.json").write_text(json.dumps({"unexpected": []}))
    result = stage3_report.AgentDomainResult(
        domain="retail", agent="efe_agent", n_simulations=1,
        avg_reward=1.0, pass_1=1.0, escalations=0,
    )
    enriched = stage3_report.enrich_from_raw_dump(result, results_dir=tmp_path)
    assert any("simulations" in w for w in enriched.parse_warnings)
