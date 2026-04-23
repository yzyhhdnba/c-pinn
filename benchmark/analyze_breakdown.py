#!/usr/bin/env python3
import json
import re
from pathlib import Path

cpp_lines = Path("benchmark/cpp_breakdown_lines.txt").read_text(encoding="utf-8").strip().splitlines()
cpp = {}
for line in cpp_lines:
    kv = dict(re.findall(r"(\w+)=([^\s]+)", line))
    eq = kv["equation"]
    row = {}
    for k, v in kv.items():
        if k == "equation":
            continue
        row[k] = float(v)
    cpp[eq] = row

torch = json.loads(Path("benchmark/torch_breakdown_results.json").read_text(encoding="utf-8"))

for eq in ["kdv", "sine_gordon", "allen_cahn"]:
    c = cpp[eq]
    t = torch[eq]

    iter_ms = c["profile_total_ms"] / c["iterations"]
    bwd_share = (c["backward_total_ms"] / c["profile_total_ms"]) * 100.0
    fwd_share = (c["forward_stencil_ms"] / c["profile_total_ms"]) * 100.0
    mse_share = (c["mse_ms"] / c["profile_total_ms"]) * 100.0
    single_bw_ms = c["backward_only_ms"] / (c["iterations"] * c["stencil_points"])

    print(eq)
    print(
        "  cpp_iter_ms=%.3f cpp_single_fwd_ms=%.4f cpp_single_refwd_ms=%.4f cpp_single_bw_ms=%.4f"
        % (
            iter_ms,
            c["single_forward_stencil_ms"],
            c["single_reforward_ms"],
            single_bw_ms,
        )
    )
    print(
        "  cpp_share_forward=%.1f%% cpp_share_backward_total=%.1f%% cpp_share_mse=%.3f%%"
        % (fwd_share, bwd_share, mse_share)
    )
    print(
        "  torch_iter_ms=%.3f torch_fwd_ms=%.4f torch_deriv_ms=%.4f torch_bw_ms=%.4f torch_mse_ms=%.4f"
        % (
            t["avg_iter_total_ms"],
            t["avg_model_forward_ms"],
            t["avg_derivative_graph_ms"],
            t["avg_backward_ms"],
            t["avg_mse_ms"],
        )
    )
    print("  ratio_cpp_over_torch=%.2fx" % (iter_ms / t["avg_iter_total_ms"]))
