#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare inference_core.py per-tick CSV metrics logs across policy_class runs.

Each CSV is produced by inference_core.py when metrics_log_enable:=true (see
inference_gradcam_single_cam.launch.py / inference_clean_single_cam.launch.py /
inference_visualization_single_cam.launch.py). Point this at one CSV per run
(e.g. one FLOW run and one BSPLINE run on the same physical task) to get a
side-by-side summary and, optionally, overlaid time-series plots.

Usage:
    python3 scripts/compare_policy_runs.py runA.csv runB.csv [...]
    python3 scripts/compare_policy_runs.py \
        logs/inference_metrics/flow_..csv logs/inference_metrics/bspline_..csv \
        --plot --out_dir /tmp/policy_compare

This is an offline analysis tool over logged CSVs; it never touches the
robot or ROS.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _load_run(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.attrs["path"] = path
    return df


def _label_for(df: pd.DataFrame) -> str:
    policy = str(df["policy_class"].mode().iat[0]) if "policy_class" in df.columns else "?"
    base = os.path.splitext(os.path.basename(df.attrs.get("path", "run")))[0]
    return f"{policy}:{base}"


def _summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    duration = float(df["t_elapsed_sec"].max() - df["t_elapsed_sec"].min()) if n else 0.0
    fz_err = (df["cmd_fz_N"] - df["meas_fz_N"]).abs()

    # Smoothness proxy: RMS of the second derivative (jerk-like) of the
    # commanded xyz path. Lower = smoother commanded trajectory.
    xyz = df[["cmd_x_mm", "cmd_y_mm", "cmd_z_mm"]].to_numpy(dtype=np.float64)
    if n >= 3:
        jerk = np.diff(xyz, n=2, axis=0)
        smoothness_rms_mm = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))
    else:
        smoothness_rms_mm = float("nan")

    return {
        "n_ticks": n,
        "duration_sec": duration,
        "contact_ratio": float(df["contact"].mean()) if n else float("nan"),
        "cmd_safety_blocked_ratio": float(df["cmd_safety_blocked"].mean()) if n else float("nan"),
        "meas_fz_mean_N": float(df["meas_fz_N"].mean()) if n else float("nan"),
        "meas_fz_std_N": float(df["meas_fz_N"].std()) if n else float("nan"),
        "fz_tracking_mae_N": float(fz_err.mean()) if n else float("nan"),
        "fz_tracking_max_N": float(fz_err.max()) if n else float("nan"),
        "cmd_fxy_mean_abs_N": float(df[["cmd_fx_N", "cmd_fy_N"]].abs().to_numpy().mean()) if n else float("nan"),
        "cmd_path_smoothness_rms": smoothness_rms_mm,
    }


def _print_summary_table(labels, summaries):
    keys = list(summaries[0].keys())
    col_w = max(28, max(len(l) for l in labels) + 2)
    header = "metric".ljust(24) + "".join(l.ljust(col_w) for l in labels)
    print(header)
    print("-" * len(header))
    for k in keys:
        row = k.ljust(24)
        for s in summaries:
            v = s[k]
            row += (f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(col_w)
        print(row)


def _plot(dfs, labels, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)

    for df, label in zip(dfs, labels):
        t = df["t_elapsed_sec"]
        axes[0].plot(t, df["meas_fz_N"], label=f"{label} meas", alpha=0.8)
        axes[0].plot(t, df["cmd_fz_N"], label=f"{label} cmd", linestyle="--", alpha=0.8)
        axes[1].plot(t, df["cmd_z_mm"], label=label, alpha=0.8)
        axes[2].plot(t, df["contact"], label=label, alpha=0.8)

    axes[0].set_ylabel("Fz [N]")
    axes[0].set_title("Measured vs commanded Fz")
    axes[0].legend(fontsize=8)

    axes[1].set_ylabel("cmd Z [mm]")
    axes[1].set_title("Commanded Z trajectory")
    axes[1].legend(fontsize=8)

    axes[2].set_ylabel("contact")
    axes[2].set_xlabel("elapsed sec")
    axes[2].set_title("Contact state")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "policy_run_comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"[PLOT] saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_paths", nargs="+", help="Two or more inference_core metrics CSV files")
    ap.add_argument("--plot", action="store_true", help="Save overlaid time-series comparison plots")
    ap.add_argument("--out_dir", default="/tmp/policy_compare", help="Directory for --plot output")
    args = ap.parse_args()

    dfs = [_load_run(p) for p in args.csv_paths]
    labels = [_label_for(df) for df in dfs]
    summaries = [_summarize(df) for df in dfs]

    _print_summary_table(labels, summaries)

    if args.plot:
        _plot(dfs, labels, args.out_dir)


if __name__ == "__main__":
    main()
