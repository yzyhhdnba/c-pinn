#!/usr/bin/env python3
"""Validation for matrix-style mini AD prototype and legacy 3-equation solvers.

What it checks:
1) Matrix AD prototype (with matmul) can run 3 equations and reduce residual loss.
2) Legacy pure C solvers (KdV/Sine-Gordon/Allen-Cahn) still converge from iter 0 to iter 900.

Outputs:
- benchmark/autodiff/results/matrix_graph_test_results.json
- benchmark/autodiff/results/matrix_graph_test_report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"


LOSS_RE = re.compile(r"Iter\s+(\d+)\s+Loss:\s+([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)")


@dataclass
class LegacySummary:
    name: str
    iter0_loss: float
    iter_last: int
    iter_last_loss: float
    decreased: bool


@dataclass
class StressEquationSummary:
    equation: str
    runs: int
    pass_count: int
    pass_rate: float
    ratio_mean: float
    ratio_std: float
    worst_ratio: float
    final_loss_mean: float
    final_loss_std: float
    nan_grad_total: int
    all_nan_free: bool
    all_decreased: bool


def run_cmd(cmd: List[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return proc.stdout


def run_matrix_ad(binary: Path, iters: int, samples: int, seed: int, lr: float, grad_clip: float) -> Dict:
    cmd = [
        str(binary),
        "--json",
        "--equation",
        "all",
        "--iters",
        str(iters),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--grad-clip",
        str(grad_clip),
    ]
    out = run_cmd(cmd).strip()
    if not out:
        raise RuntimeError("No output from matrix AD binary")
    return json.loads(out)


def parse_seeds(seeds_text: str) -> List[int]:
    values: List[int] = []
    for s in seeds_text.split(","):
        t = s.strip()
        if not t:
            continue
        values.append(int(t))
    if not values:
        raise ValueError("At least one stress seed is required")
    return values


def aggregate_stress_runs(stress_runs: List[Dict]) -> List[StressEquationSummary]:
    by_eq: Dict[str, List[Dict]] = {}
    for run in stress_runs:
        for item in run["results"]:
            by_eq.setdefault(item["equation"], []).append(item)

    summaries: List[StressEquationSummary] = []
    for eq, items in sorted(by_eq.items()):
        ratios = [float(x["loss_ratio"]) for x in items]
        finals = [float(x["final_loss"]) for x in items]
        pass_count = sum(1 for x in items if bool(x["loss_decreased"]) and int(x["nan_grad_count"]) == 0)
        nan_total = sum(int(x["nan_grad_count"]) for x in items)

        ratio_mean = statistics.fmean(ratios)
        final_mean = statistics.fmean(finals)
        ratio_std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
        final_std = statistics.stdev(finals) if len(finals) > 1 else 0.0

        summaries.append(
            StressEquationSummary(
                equation=eq,
                runs=len(items),
                pass_count=pass_count,
                pass_rate=pass_count / max(1, len(items)),
                ratio_mean=ratio_mean,
                ratio_std=ratio_std,
                worst_ratio=max(ratios),
                final_loss_mean=final_mean,
                final_loss_std=final_std,
                nan_grad_total=nan_total,
                all_nan_free=nan_total == 0,
                all_decreased=all(bool(x["loss_decreased"]) for x in items),
            )
        )

    return summaries


def run_matrix_stress(binary: Path,
                      seeds: List[int],
                      iters: int,
                      samples: int,
                      lr: float,
                      grad_clip: float) -> Dict:
    runs: List[Dict] = []
    for seed in seeds:
        run = run_matrix_ad(
            binary,
            iters=iters,
            samples=samples,
            seed=seed,
            lr=lr,
            grad_clip=grad_clip,
        )
        runs.append({
            "seed": seed,
            "config": run.get("config", {}),
            "results": run.get("results", []),
        })

    summaries = aggregate_stress_runs([{"results": r["results"]} for r in runs])

    return {
        "config": {
            "iters": iters,
            "samples": samples,
            "seeds": seeds,
            "lr": lr,
            "grad_clip": grad_clip,
            "num_seeds": len(seeds),
        },
        "runs": runs,
        "summary": [
            {
                "equation": s.equation,
                "runs": s.runs,
                "pass_count": s.pass_count,
                "pass_rate": s.pass_rate,
                "ratio_mean": s.ratio_mean,
                "ratio_std": s.ratio_std,
                "worst_ratio": s.worst_ratio,
                "final_loss_mean": s.final_loss_mean,
                "final_loss_std": s.final_loss_std,
                "nan_grad_total": s.nan_grad_total,
                "all_nan_free": s.all_nan_free,
                "all_decreased": s.all_decreased,
            }
            for s in summaries
        ],
    }


def parse_legacy_solver_output(name: str, text: str) -> LegacySummary:
    pairs: List[Tuple[int, float]] = []
    for m in LOSS_RE.finditer(text):
        pairs.append((int(m.group(1)), float(m.group(2))))

    if not pairs:
        raise RuntimeError(f"Cannot parse losses for {name}")

    iter0_loss = pairs[0][1]
    iter_last, iter_last_loss = pairs[-1]
    return LegacySummary(
        name=name,
        iter0_loss=iter0_loss,
        iter_last=iter_last,
        iter_last_loss=iter_last_loss,
        decreased=iter_last_loss < iter0_loss,
    )


def run_legacy_solver(binary: Path, name: str) -> LegacySummary:
    out = run_cmd([str(binary)])
    return parse_legacy_solver_output(name, out)


def build_report(matrix_res: Dict,
                 stress_res: Dict,
                 legacy: List[LegacySummary],
                 verdict: str,
                 stress_min_pass_rate: float) -> str:
    lines: List[str] = []
    lines.append("# Matrix AD + Legacy 3-Equation Regression Report")
    lines.append("")
    lines.append(f"Overall verdict: **{verdict}**")
    lines.append("")

    cfg = matrix_res.get("config", {})
    lines.append("## Matrix AD config")
    lines.append("")
    lines.append(f"- iters: {cfg.get('iters')}")
    lines.append(f"- samples: {cfg.get('samples')}")
    lines.append(f"- seed: {cfg.get('seed')}")
    lines.append(f"- lr: {cfg.get('lr')}")
    lines.append(f"- grad_clip: {cfg.get('grad_clip')}")
    lines.append("")

    lines.append("## Matrix AD results (with matmul)")
    lines.append("")
    for item in matrix_res.get("results", []):
        lines.append(
            "- "
            + f"{item['equation']}: init={item['initial_loss']:.6e}, "
            + f"final={item['final_loss']:.6e}, best={item['best_loss']:.6e}, "
            + f"ratio={item['loss_ratio']:.6f}, decreased={item['loss_decreased']}, "
            + f"nan_grad={item['nan_grad_count']}"
        )
    lines.append("")

    lines.append("## Matrix AD stress test (multi-seed)")
    lines.append("")
    scfg = stress_res.get("config", {})
    lines.append(f"- iters: {scfg.get('iters')}")
    lines.append(f"- samples: {scfg.get('samples')}")
    lines.append(f"- seeds: {scfg.get('seeds')}")
    lines.append(f"- lr: {scfg.get('lr')}")
    lines.append(f"- grad_clip: {scfg.get('grad_clip')}")
    lines.append(f"- min_pass_rate threshold: {stress_min_pass_rate:.2f}")
    lines.append("")

    lines.append("### Aggregate summary")
    lines.append("")
    for item in stress_res.get("summary", []):
        lines.append(
            "- "
            + f"{item['equation']}: runs={item['runs']}, pass={item['pass_count']}, "
            + f"pass_rate={item['pass_rate']:.2f}, ratio_mean={item['ratio_mean']:.6f}, "
            + f"ratio_std={item['ratio_std']:.6f}, worst_ratio={item['worst_ratio']:.6f}, "
            + f"final_mean={item['final_loss_mean']:.6e}, final_std={item['final_loss_std']:.6e}, "
            + f"nan_total={item['nan_grad_total']}"
        )
    lines.append("")

    lines.append("### Per-seed detail")
    lines.append("")
    for run in stress_res.get("runs", []):
        lines.append(f"- seed={run['seed']}")
        for item in run.get("results", []):
            lines.append(
                "  "
                + f"- {item['equation']}: init={item['initial_loss']:.6e}, "
                + f"final={item['final_loss']:.6e}, ratio={item['loss_ratio']:.6f}, "
                + f"decreased={item['loss_decreased']}, nan_grad={item['nan_grad_count']}"
            )
    lines.append("")

    lines.append("## Legacy pure C solver regression")
    lines.append("")
    for s in legacy:
        lines.append(
            "- "
            + f"{s.name}: iter0={s.iter0_loss:.6e}, "
            + f"iter{s.iter_last}={s.iter_last_loss:.6e}, "
            + f"decreased={s.decreased}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-binary", type=Path, default=REPO_ROOT / "build/examples/example_autodiff_matrix_graph")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--grad-clip", type=float, default=50.0)
    parser.add_argument("--stress-seeds", type=str, default="7,17,27,37,47")
    parser.add_argument("--stress-iters", type=int, default=80)
    parser.add_argument("--stress-samples", type=int, default=16)
    parser.add_argument("--stress-min-pass-rate", type=float, default=0.8)
    parser.add_argument(
        "--stress-require-worst-ratio-lt-one",
        action="store_true",
        help="Require worst_ratio < 1.0 for every equation in stress summary",
    )
    parser.add_argument("--kdv-binary", type=Path, default=REPO_ROOT / "build/examples/example_pure_c_kdv")
    parser.add_argument("--sg-binary", type=Path, default=REPO_ROOT / "build/examples/example_pure_c_sine_gordon")
    parser.add_argument("--ac-binary", type=Path, default=REPO_ROOT / "build/examples/example_pure_c_allen_cahn")
    args = parser.parse_args()

    if not args.matrix_binary.exists():
        raise FileNotFoundError(f"Matrix AD binary not found: {args.matrix_binary}")

    matrix_res = run_matrix_ad(
        args.matrix_binary,
        iters=args.iters,
        samples=args.samples,
        seed=args.seed,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    stress_seeds = parse_seeds(args.stress_seeds)
    stress_res = run_matrix_stress(
        args.matrix_binary,
        seeds=stress_seeds,
        iters=args.stress_iters,
        samples=args.stress_samples,
        lr=args.lr,
        grad_clip=args.grad_clip,
    )

    matrix_ok = True
    matrix_fail_reasons: List[str] = []
    for item in matrix_res.get("results", []):
        init_loss = float(item["initial_loss"])
        final_loss = float(item["final_loss"])
        if not math.isfinite(init_loss) or not math.isfinite(final_loss):
            matrix_ok = False
            matrix_fail_reasons.append(f"{item['equation']}: non-finite loss")
            continue
        if int(item["nan_grad_count"]) != 0:
            matrix_ok = False
            matrix_fail_reasons.append(f"{item['equation']}: nan gradients")
        if not bool(item["loss_decreased"]):
            matrix_ok = False
            matrix_fail_reasons.append(f"{item['equation']}: final loss not decreased")

    stress_ok = True
    stress_fail_reasons: List[str] = []
    for item in stress_res.get("summary", []):
        pass_rate = float(item["pass_rate"])
        nan_total = int(item["nan_grad_total"])
        worst_ratio = float(item["worst_ratio"])
        eq = item["equation"]
        if pass_rate < args.stress_min_pass_rate:
            stress_ok = False
            stress_fail_reasons.append(
                f"{eq}: pass_rate {pass_rate:.2f} < {args.stress_min_pass_rate:.2f}"
            )
        if nan_total != 0:
            stress_ok = False
            stress_fail_reasons.append(f"{eq}: nan_grad_total={nan_total}")
        if args.stress_require_worst_ratio_lt_one and worst_ratio >= 1.0:
            stress_ok = False
            stress_fail_reasons.append(f"{eq}: worst_ratio={worst_ratio:.6f} >= 1")

    legacy_bins = [
        ("KdV", args.kdv_binary),
        ("Sine-Gordon", args.sg_binary),
        ("Allen-Cahn", args.ac_binary),
    ]

    legacy_results: List[LegacySummary] = []
    legacy_ok = True
    legacy_fail_reasons: List[str] = []
    for name, binary in legacy_bins:
        if not binary.exists():
            raise FileNotFoundError(f"Legacy binary not found: {binary}")
        s = run_legacy_solver(binary, name)
        legacy_results.append(s)
        if not s.decreased:
            legacy_ok = False
            legacy_fail_reasons.append(f"{name}: iter{s.iter_last} >= iter0")

    passed = matrix_ok and stress_ok and legacy_ok
    verdict = "PASS" if passed else "FAIL"

    results = {
        "verdict": verdict,
        "matrix_ad": matrix_res,
        "stress_test": stress_res,
        "legacy": [
            {
                "name": s.name,
                "iter0_loss": s.iter0_loss,
                "iter_last": s.iter_last,
                "iter_last_loss": s.iter_last_loss,
                "decreased": s.decreased,
            }
            for s in legacy_results
        ],
        "checks": {
            "matrix_ok": matrix_ok,
            "stress_ok": stress_ok,
            "legacy_ok": legacy_ok,
            "matrix_fail_reasons": matrix_fail_reasons,
            "stress_fail_reasons": stress_fail_reasons,
            "legacy_fail_reasons": legacy_fail_reasons,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "matrix_graph_test_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = build_report(
        matrix_res,
        stress_res,
        legacy_results,
        verdict,
        stress_min_pass_rate=args.stress_min_pass_rate,
    )
    out_md = RESULTS_DIR / "matrix_graph_test_report.md"
    out_md.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"
