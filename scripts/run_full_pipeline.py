"""End-to-end pipeline runner: every stage that can run without network
access or API credentials, in order, with a consolidated report at the
end.

This is the "does the whole thing still work" command. Individual stages
each have their own entry point (and are documented in
context/TODOS.md's verification section); this runs them as one
pipeline and fails loudly on the first broken stage rather than leaving
someone to discover mid-thesis that Stage 2 stopped reproducing.

Deliberately excluded, because they need things this can't assume:
  * Stage 1c real-LLM demo (`graph.py --llm`)      -- API key + network
  * Stage 3 tau2 smoke test / sweep                 -- tau2-bench + network + money

Run: .venv/bin/python scripts/run_full_pipeline.py
     .venv/bin/python scripts/run_full_pipeline.py --quick   (skip the slow evals)
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
PYTHON = sys.executable


@dataclass
class Stage:
    key: str
    label: str
    command: list[str]
    slow: bool = False
    produces: list[str] = field(default_factory=list)


STAGES = [
    Stage("stage0", "Stage 0 — pymdp engine sanity check",
          [PYTHON, "scripts/tmaze_sanity_check.py"],
          produces=["stage0_tmaze_sanity_check.json"]),
    Stage("tests", "Test suite",
          [PYTHON, "-m", "pytest", "-q"]),
    Stage("stage0.5", "Stage 0.5 — kill-test (5 controllers x 3000 episodes)",
          [PYTHON, "-m", "aif_orchestrator.kill_test.run_kill_test"],
          slow=True, produces=["stage0_5_kill_test_results.json"]),
    Stage("stage1", "Stage 1 — LangGraph demo (mock agent, incl. interrupt/resume)",
          [PYTHON, "-m", "aif_orchestrator.graph"]),
    Stage("stage2-plumbing", "Stage 2 — all 5 controllers through the real graph",
          [PYTHON, "-m", "aif_orchestrator.graph", "--stage2"]),
    Stage("stage2-part2", "Stage 2 Part 2 — model-matched comparison (the citable result)",
          [PYTHON, "-m", "aif_orchestrator.baselines.run_model_matched_eval"],
          slow=True, produces=["model_matched_baselines_results.json"]),
    Stage("stage3-cost", "Stage 3 — sweep cost estimate (no network needed)",
          [PYTHON, "scripts/estimate_stage3_cost.py", "--domain", "all"]),
]


def run_stage(stage: Stage, verbose: bool) -> dict:
    print(f"\n=== {stage.label} ===", flush=True)
    started = time.time()
    completed = subprocess.run(
        stage.command, cwd=REPO_ROOT, capture_output=not verbose, text=True,
    )
    elapsed = time.time() - started
    ok = completed.returncode == 0

    if ok:
        print(f"  OK ({elapsed:.1f}s)")
    else:
        print(f"  FAILED (exit {completed.returncode}, {elapsed:.1f}s)")
        if not verbose and completed.stdout:
            print("  --- stdout tail ---")
            print("\n".join(completed.stdout.splitlines()[-20:]))
        if not verbose and completed.stderr:
            print("  --- stderr tail ---")
            print("\n".join(completed.stderr.splitlines()[-20:]))

    return {"stage": stage.key, "label": stage.label, "ok": ok,
            "returncode": completed.returncode, "elapsed_sec": elapsed}


def collect_headline_numbers() -> dict:
    """Pull the numbers worth seeing in one place from whatever result
    files the run produced. Missing files are reported as missing rather
    than silently omitted."""
    headlines = {}

    kill_test = RESULTS_DIR / "stage0_5_kill_test_results.json"
    if kill_test.exists():
        payload = json.loads(kill_test.read_text())
        headlines["stage0.5_kill_test"] = {
            name: {"avg_reward": round(m["avg_reward"], 4),
                   "correct_escalation_rate": m.get("correct_escalation_rate")}
            for name, m in payload.get("results", {}).items()
        }
    else:
        headlines["stage0.5_kill_test"] = "not produced in this run"

    matched = RESULTS_DIR / "model_matched_baselines_results.json"
    if matched.exists():
        payload = json.loads(matched.read_text())
        headlines["stage2_model_matched"] = {
            name: {"avg_reward": round(m["avg_reward"], 4),
                   "correct_escalation_rate": m.get("correct_escalation_rate"),
                   "unnecessary_escalation_rate": m.get("unnecessary_escalation_rate")}
            for name, m in payload.get("results", {}).items()
        }
    else:
        headlines["stage2_model_matched"] = "not produced in this run"

    return headlines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slow evaluation stages")
    parser.add_argument("--verbose", action="store_true", help="stream each stage's output")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=[s.key for s in STAGES], help="run only these stages")
    args = parser.parse_args()

    stages = [s for s in STAGES
              if (args.only is None or s.key in args.only) and not (args.quick and s.slow)]
    if args.quick:
        print("Running in --quick mode: skipping the slow evaluation stages.")

    results = [run_stage(stage, args.verbose) for stage in stages]
    failed = [r for r in results if not r["ok"]]

    print("\n" + "=" * 62)
    print("PIPELINE SUMMARY")
    print("=" * 62)
    for r in results:
        print(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['label']}  ({r['elapsed_sec']:.1f}s)")

    headlines = collect_headline_numbers()
    print("\nHeadline numbers:")
    print(json.dumps(headlines, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    report_path = RESULTS_DIR / "pipeline_run.json"
    report_path.write_text(json.dumps(
        {"stages": results, "headlines": headlines,
         "total_sec": sum(r["elapsed_sec"] for r in results)},
        indent=2,
    ))
    print(f"\nRun report written to {report_path}")

    if failed:
        print(f"\n{len(failed)} stage(s) FAILED: {[r['stage'] for r in failed]}")
        return 1
    print(f"\nAll {len(results)} stage(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
