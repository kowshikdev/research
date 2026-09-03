"""Analysis layer: turns the artifacts the pipeline produces (decision
logs, tau2 sweep results) into the numbers the thesis actually reports.

Everything upstream of this produces raw data; nothing before this read
it back. `interpretability.py` is Stage 5's deliverable
(RESEARCH_PLAN.md) -- the epistemic/pragmatic decomposition has been
logged since Stage 1 precisely so that stage wouldn't be a retrofit.
"""
from .decision_log import DecisionRecord, load_decision_log
from .interpretability import (
    ControllerInterpretability,
    analyze_decision_log,
    format_interpretability_report,
)

__all__ = [
    "DecisionRecord",
    "load_decision_log",
    "ControllerInterpretability",
    "analyze_decision_log",
    "format_interpretability_report",
]
