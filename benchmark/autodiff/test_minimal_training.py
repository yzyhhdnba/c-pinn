#!/usr/bin/env python3
"""Detailed checks for the minimal Node-based autograd training utilities.

Checks:
- Node self-checks pass: grad buffer matches direct grad
- backward accumulates into Node.grad
- zero_grad clears leaf gradient buffers
- sgd_step updates parameter values
- mini PINN training still reduces residual loss for all equations

Outputs:
- benchmark/autodiff/results/minimal_training_results.json
- benchmark/autodiff/results/minimal_training_report.md
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"


def run_json(cmd: list[str]) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError(f"No output from command: {' '.join(cmd)}")
    return json.loads(out)


def main() -> int:
    binary = REPO_ROOT / "build/examples/example_autodiff_matrix_graph"
    if not binary.exists():
        raise FileNotFoundError(f"Binary not found: {binary}")

    self_check = run_json([str(binary), "--json", "--self-check-only"])
    train_check = run_json([str(binary), "--json", "--equation", "all", "--iters", "25", "--samples", "6"])

    sc = self_check["self_check"]
    train_rows = train_check["results"]

    conditions = {
        "grad_buffer_matches_direct": bool(sc["grad_buffer_matches_direct"]),
        "zero_grad_clears_buffers": bool(sc["zero_grad_clears_buffers"]),
        "backward_accumulates_grad": bool(sc["backward_accumulates_grad"]),
        "sgd_step_updates_parameter": bool(sc["sgd_step_updates_parameter"]),
        "all_losses_decreased": all(bool(row["loss_decreased"]) for row in train_rows),
        "all_nan_grad_zero": all(int(row["nan_grad_count"]) == 0 for row in train_rows),
    }

    passed = all(conditions.values())

    results = {
        "binary": str(binary),
        "self_check": sc,
        "training_results": train_rows,
        "conditions": conditions,
        "passed": passed,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "minimal_training_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# 最小 Node Autodiff 训练工具测试报告")
    lines.append("")
    lines.append(f"总体结论：**{'PASS' if passed else 'FAIL'}**")
    lines.append("")
    lines.append("## 自检结果")
    lines.append("")
    lines.append(f"- grad buffer 与 direct grad 一致：{conditions['grad_buffer_matches_direct']}")
    lines.append(f"- zero_grad 清空缓冲：{conditions['zero_grad_clears_buffers']}")
    lines.append(f"- backward 支持累积：{conditions['backward_accumulates_grad']}")
    lines.append(f"- sgd_step 会更新参数：{conditions['sgd_step_updates_parameter']}")
    lines.append(f"- direct_grad：{sc['direct_grad']:.12f}")
    lines.append(f"- buffer_grad：{sc['buffer_grad']:.12f}")
    lines.append(f"- accumulated_grad：{sc['accumulated_grad']:.12f}")
    lines.append(f"- value_before_step：{sc['value_before_step']:.12f}")
    lines.append(f"- value_after_step：{sc['value_after_step']:.12f}")
    lines.append("")
    lines.append("## 训练回归")
    lines.append("")
    for row in train_rows:
        lines.append(
            "- "
            + f"{row['equation']}: init={row['initial_loss']:.6e}, "
            + f"final={row['final_loss']:.6e}, ratio={row['loss_ratio']:.6f}, "
            + f"decreased={row['loss_decreased']}, nan_grad={row['nan_grad_count']}"
        )
    lines.append("")
    lines.append("## 判定项")
    lines.append("")
    for key, value in conditions.items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    out_md = RESULTS_DIR / "minimal_training_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved JSON: {out_json}")
    print(f"Saved Markdown: {out_md}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
