"""Benchmark: does the audit find the right incidents, causes, and actions?

Rules-only runs for free. Agent runs need an API key:
    python bench/run_bench.py --replicas 3 --models gpt-4o-mini gpt-5.4-mini
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")

from driftwatch import Toolbox, agent_audit, benchmark_suite, expected_findings, generate, make_model, rule_audit, score

OUT = Path(__file__).parent / "results"


def run_one(scenario, logs, system, model):
    box = Toolbox(logs)
    for attempt in range(3):
        try:
            result = rule_audit(box) if system == "rules" else agent_audit(box, model)
            break
        except Exception as e:      # rate limits: wait and retry
            if attempt == 2:
                return {"scenario": scenario.name, "system": system, "error": str(e)}
            time.sleep(10 * (attempt + 1))
    s = score(result.findings, expected_findings(scenario))
    return {"scenario": scenario.name, "system": system, "kinds": [d.kind for d in scenario.drifts],
            "score": s, "findings": result.findings, "report": result.report,
            "tool_calls": len(result.steps), "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens, "seconds": round(result.seconds, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicas", type=int, default=3)
    ap.add_argument("--provider", default="openai")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    suite = benchmark_suite(args.replicas)
    print(f"generating {len(suite)} scenarios ...")
    logs = {s.name: generate(s) for s in suite}

    systems = [("rules", None)] + [(m, make_model(args.provider, m)) for m in args.models]
    for name, model in systems:
        t0 = time.time()
        with ThreadPoolExecutor(args.concurrency if model else 1) as pool:
            rows = list(pool.map(lambda s: run_one(s, logs[s.name], name, model), suite))
        path = OUT / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        ok = sum(r.get("score", {}).get("fully_correct", False) for r in rows)
        errors = sum("error" in r for r in rows)
        print(f"{name:14s} fully correct {ok}/{len(rows)}  errors {errors}  ({time.time() - t0:.0f}s) -> {path}")


if __name__ == "__main__":
    main()
