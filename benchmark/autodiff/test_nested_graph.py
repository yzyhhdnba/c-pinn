#!/usr/bin/env python3
"""Detailed verification for the minimal nested-graph autodiff example.

Checks:
- 1st derivative matches analytic formula on grid + random points
- graph-mode 1st derivative matches detached 1st derivative
- 2nd derivative matches analytic formula
- create_graph result keeps requires_grad=true
- first-derivative graph is larger than forward graph

Outputs:
- benchmark/autodiff/results/nested_graph_test_results.json
- benchmark/autodiff/results/nested_graph_test_report.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"


@dataclass
class CaseResult:
    x: float
    abs_err_d1: float
    abs_err_d2: float
    abs_gap_graph_vs_detached: float
    g0_nodes: int
    g1_nodes: int
    g0_edges: int
    g1_edges: int
    requires_grad: bool


def run_case(binary: Path, x: float) -> dict:
    cmd = [str(binary), f"{x:.17g}", "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"No output for x={x}")
    return json.loads(out)


def evaluate_cases(binary: Path, xs: List[float]) -> List[CaseResult]:
    rows: List[CaseResult] = []
    for x in xs:
        raw = run_case(binary, x)
        rows.append(
            CaseResult(
                x=float(raw["x"]),
                abs_err_d1=float(raw["abs_err_d1"]),
                abs_err_d2=float(raw["abs_err_d2"]),
                abs_gap_graph_vs_detached=abs(float(raw["dy_dx_graph"]) - float(raw["dy_dx_detached"])),
                g0_nodes=int(raw["g0_nodes"]),
                g1_nodes=int(raw["g1_nodes"]),
                g0_edges=int(raw["g0_edges"]),
                g1_edges=int(raw["g1_edges"]),
                requires_grad=bool(raw["dy_dx_graph_requires_grad"]),
            )
        )
    return rows


def summarize(rows: List[CaseResult]) -> dict:
    max_err_d1 = max(r.abs_err_d1 for r in rows)
    max_err_d2 = max(r.abs_err_d2 for r in rows)
    max_gap = max(r.abs_gap_graph_vs_detached for r in rows)
    avg_err_d1 = statistics.fmean(r.abs_err_d1 for r in rows)
    avg_err_d2 = statistics.fmean(r.abs_err_d2 for r in rows)

    any_bad_graph_growth = any((r.g1_nodes <= r.g0_nodes) or (r.g1_edges <= r.g0_edges) for r in rows)
    any_bad_requires_grad = any(not r.requires_grad for r in rows)

    return {
        "num_cases": len(rows),
        "max_abs_err_d1": max_err_d1,
        "max_abs_err_d2": max_err_d2,
        "avg_abs_err_d1": avg_err_d1,
        "avg_abs_err_d2": avg_err_d2,
        "max_gap_graph_vs_detached": max_gap,
        "all_requires_grad_true": not any_bad_requires_grad,
        "all_graphs_grow": not any_bad_graph_growth,
    }


def build_report(grid_summary: dict, random_summary: dict, tol_d1: float, tol_d2: float, tol_gap: float) -> str:
    passed = (
        grid_summary["max_abs_err_d1"] <= tol_d1
        and grid_summary["max_abs_err_d2"] <= tol_d2
        and grid_summary["max_gap_graph_vs_detached"] <= tol_gap
        and grid_summary["all_requires_grad_true"]
        and grid_summary["all_graphs_grow"]
        and random_summary["max_abs_err_d1"] <= tol_d1
        and random_summary["max_abs_err_d2"] <= tol_d2
        and random_summary["max_gap_graph_vs_detached"] <= tol_gap
        and random_summary["all_requires_grad_true"]
        and random_summary["all_graphs_grow"]
    )

    verdict = "PASS" if passed else "FAIL"

    def fmt(v: float) -> str:
        return f"{v:.3e}"

    lines = []
    lines.append("# Autodiff Nested Graph Detailed Test Report")
    lines.append("")
    lines.append(f"Overall verdict: **{verdict}**")
    lines.append("")
    lines.append("## Thresholds")
    lines.append("")
    lines.append(f"- abs_err_d1 <= {tol_d1:.1e}")
    lines.append(f"- abs_err_d2 <= {tol_d2:.1e}")
    lines.append(f"- |dy_dx_graph - dy_dx_detached| <= {tol_gap:.1e}")
    lines.append("- dy_dx_graph.requires_grad must be true")
    lines.append("- graph growth must hold: g1_nodes > g0_nodes and g1_edges > g0_edges")
    lines.append("")
    lines.append("## Grid cases")
    lines.append("")
    lines.append(f"- cases: {grid_summary['num_cases']}")
    lines.append(f"- max_abs_err_d1: {fmt(grid_summary['max_abs_err_d1'])}")
    lines.append(f"- max_abs_err_d2: {fmt(grid_summary['max_abs_err_d2'])}")
    lines.append(f"- avg_abs_err_d1: {fmt(grid_summary['avg_abs_err_d1'])}")
    lines.append(f"- avg_abs_err_d2: {fmt(grid_summary['avg_abs_err_d2'])}")
    lines.append(f"- max_gap_graph_vs_detached: {fmt(grid_summary['max_gap_graph_vs_detached'])}")
    lines.append(f"- all_requires_grad_true: {grid_summary['all_requires_grad_true']}")
    lines.append(f"- all_graphs_grow: {grid_summary['all_graphs_grow']}")
    lines.append("")
    lines.append("## Random cases")
    lines.append("")
    lines.append(f"- cases: {random_summary['num_cases']}")
    lines.append(f"- max_abs_err_d1: {fmt(random_summary['max_abs_err_d1'])}")
    lines.append(f"- max_abs_err_d2: {fmt(random_summary['max_abs_err_d2'])}")
    lines.append(f"- avg_abs_err_d1: {fmt(random_summary['avg_abs_err_d1'])}")
    lines.append(f"- avg_abs_err_d2: {fmt(random_summary['avg_abs_err_d2'])}")
    lines.append(f"- max_gap_graph_vs_detached: {fmt(random_summary['max_gap_graph_vs_detached'])}")
    lines.append(f"- all_requires_grad_true: {random_summary['all_requires_grad_true']}")
    lines.append(f"- all_graphs_grow: {random_summary['all_graphs_grow']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO_ROOT / "build/examples/example_autodiff_nested_graph",
        help="Path to example_autodiff_nested_graph binary",
    )
    parser.add_argument("--random-cases", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tol-d1", type=float, default=1e-12)
    parser.add_argument("--tol-d2", type=float, default=1e-12)
    parser.add_argument("--tol-gap", type=float, default=1e-12)
    args = parser.parse_args()

    binary = args.binary
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    grid_xs = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    grid_rows = evaluate_cases(binary, grid_xs)

    rng = random.Random(args.seed)
    random_xs = [rng.uniform(-2.0, 2.0) for _ in range(args.random_cases)]
    random_rows = evaluate_cases(binary, random_xs)

    grid_summary = summarize(grid_rows)
    random_summary = summarize(random_rows)

    results = {
        "binary": str(binary),
        "thresholds": {
            "tol_d1": args.tol_d1,
            "tol_d2": args.tol_d2,
            "tol_gap": args.tol_gap,
        },
        "grid": grid_summary,
        "random": random_summary,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "nested_graph_test_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = build_report(grid_summary, random_summary, args.tol_d1, args.tol_d2, args.tol_gap)
    out_md = RESULTS_DIR / "nested_graph_test_report.md"
    out_md.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")

    passed = (
        grid_summary["max_abs_err_d1"] <= args.tol_d1
        and grid_summary["max_abs_err_d2"] <= args.tol_d2
        and grid_summary["max_gap_graph_vs_detached"] <= args.tol_gap
        and grid_summary["all_requires_grad_true"]
        and grid_summary["all_graphs_grow"]
        and random_summary["max_abs_err_d1"] <= args.tol_d1
        and random_summary["max_abs_err_d2"] <= args.tol_d2
        and random_summary["max_gap_graph_vs_detached"] <= args.tol_gap
        and random_summary["all_requires_grad_true"]
        and random_summary["all_graphs_grow"]
    )

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"
