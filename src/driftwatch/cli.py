"""Command line: dw simulate | dw audit | dw serve"""
from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")


def main(argv=None) -> None:
    from .agents import agent_audit, rule_audit
    from .simulate import Drift, Scenario, generate
    from .tools import Toolbox

    parser = argparse.ArgumentParser(prog="dw", description="DriftWatch: multi-agent fairness drift auditor.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write example prediction logs with a planted problem")
    s.add_argument("--out", default="examples/logs.csv")
    s.add_argument("--group", default="Medicaid")
    s.add_argument("--kind", default="undercoding", choices=["undercoding", "population_shift", "followup_loss", "none"])
    s.add_argument("--start", type=int, default=6)
    s.add_argument("--seed", type=int, default=7)

    a = sub.add_parser("audit", help="audit prediction logs (CSV) for fairness drift")
    a.add_argument("logs")
    a.add_argument("--provider", default="openai")
    a.add_argument("--model", default=None)
    a.add_argument("--rules-only", action="store_true", help="no LLM: fixed rules only")
    a.add_argument("--group-col", default="payer")
    a.add_argument("--baseline-months", type=int, default=3)
    a.add_argument("--report", default="audit_report.md")

    v = sub.add_parser("serve", help="serve the audit tools over MCP (stdio)")
    v.add_argument("logs")
    v.add_argument("--group-col", default="payer")
    v.add_argument("--baseline-months", type=int, default=3)

    args = parser.parse_args(argv)
    if args.cmd == "simulate":
        drifts = [] if args.kind == "none" else [Drift(args.group, args.start, args.kind)]
        df = generate(Scenario("cli", drifts=drifts, seed=args.seed))
        df.to_csv(args.out, index=False)
        print(f"wrote {len(df)} rows to {args.out}")
    elif args.cmd == "audit":
        from .server import load_logs
        box = Toolbox(load_logs(args.logs), args.baseline_months, args.group_col)
        if args.rules_only:
            result = rule_audit(box)
        else:
            from .llm import make_model
            result = agent_audit(box, make_model(args.provider, args.model))
        open(args.report, "w").write(result.report)
        print(json.dumps(result.findings, indent=2))
        print(f"\nreport written to {args.report} ({len(result.steps)} tool calls, {result.seconds:.1f}s)")
    elif args.cmd == "serve":
        from .server import build_server, load_logs
        build_server(load_logs(args.logs), args.baseline_months, args.group_col).run()


if __name__ == "__main__":
    main()
