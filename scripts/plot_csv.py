#!/usr/bin/env python3
"""快速查看 PINN 导出的 CSV 可视化结果。

Usage
-----
python scripts/plot_csv.py path/to/csv --output figure.png
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _detect_dimensions(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    input_cols = [col for col in frame.columns if col.startswith("x")]
    pred_cols = [col for col in frame.columns if col.startswith("pred")]
    if not input_cols or not pred_cols:
        raise ValueError("CSV 文件缺少以 x*/pred* 开头的列")
    return input_cols, pred_cols


def _plot_1d(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    input_cols, pred_cols = _detect_dimensions(frame)
    if len(input_cols) != 1:
        raise ValueError("该 CSV 非 1D 数据")
    frame_sorted = frame.sort_values(input_cols[0])
    x = frame_sorted[input_cols[0]].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in pred_cols:
        ax.plot(x, frame_sorted[col].to_numpy(), label=col)
    target_cols = [col for col in frame.columns if col.startswith("target")]
    for col in target_cols:
        ax.plot(x, frame_sorted[col].to_numpy(), "--", label=col)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.set_title(pathlib.Path(args.csv).name)
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def _plot_2d(frame: pd.DataFrame, args: argparse.Namespace) -> None:
    input_cols, pred_cols = _detect_dimensions(frame)
    if len(input_cols) != 2:
        raise ValueError("该 CSV 非 2D 数据")
    frame_sorted = frame.sort_values(input_cols).reset_index(drop=True)
    unique_x = frame_sorted[input_cols[0]].drop_duplicates().to_numpy()
    unique_y = frame_sorted[input_cols[1]].drop_duplicates().to_numpy()
    nx, ny = unique_x.size, unique_y.size
    if nx * ny != len(frame_sorted):
        raise ValueError("二维 CSV 需要规则网格采样，当前数据无法重塑")

    X = frame_sorted[input_cols[0]].to_numpy().reshape(nx, ny)
    Y = frame_sorted[input_cols[1]].to_numpy().reshape(nx, ny)

    fig, axes = plt.subplots(1, len(pred_cols), figsize=(5 * len(pred_cols), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, col in zip(axes, pred_cols):
        values = frame_sorted[col].to_numpy().reshape(nx, ny)
        pcm = ax.pcolormesh(X, Y, values, shading="auto")
        ax.set_xlabel(input_cols[0])
        ax.set_ylabel(input_cols[1])
        ax.set_title(col)
        fig.colorbar(pcm, ax=ax)
    fig.suptitle(pathlib.Path(args.csv).name)
    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 PINN CSV 输出 (1D/2D)")
    parser.add_argument("csv", type=str, help="导出的 CSV 文件路径")
    parser.add_argument("--dim", type=int, choices=[1, 2], help="数据维度 (1 或 2)，缺省时自动检测")
    parser.add_argument("--output", type=str, help="若提供则保存到该文件，否则弹出窗口")
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV 文件不存在: {csv_path}")

    frame = pd.read_csv(csv_path)
    dim = args.dim
    if dim is None:
        input_cols = [col for col in frame.columns if col.startswith("x")]
        dim = len(input_cols)
    if dim == 1:
        _plot_1d(frame, args)
    elif dim == 2:
        _plot_2d(frame, args)
    else:
        raise SystemExit("暂仅支持 1D/2D 数据可视化")


if __name__ == "__main__":
    main()
