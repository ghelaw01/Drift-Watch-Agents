"""DriftWatch: catch fairness drift in deployed models, find the cause, and recommend the fix."""
from .agents import AuditResult, agent_audit, rule_audit
from .evaluate import score
from .llm import make_model
from .simulate import Drift, Scenario, benchmark_suite, expected_findings, generate
from .tools import Toolbox

__version__ = "0.1.0"
__all__ = ["AuditResult", "Drift", "Scenario", "Toolbox", "agent_audit", "benchmark_suite",
           "expected_findings", "generate", "make_model", "rule_audit", "score"]
