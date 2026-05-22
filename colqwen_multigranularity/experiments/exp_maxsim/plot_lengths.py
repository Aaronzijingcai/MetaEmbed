from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_rows(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for group_name, group_results in data.items():
        for ds_name, stats in group_results.items():
            q = stats["query_summary"]
            d = stats["target_summary"]
            rows.append(
                {
                    "group": group_name,
                    "dataset": ds_name,
                    "query_count": q["count"],
                    "query_mean": q["mean"],
                    "query_p10": q["p10"],
                    "query_p50": q["p50"],
                    "query_p90": q["p90"],
                    "query_max": q["max"],
                    "target_count": d["count"],
                    "target_mean": d["mean"],
                    "target_p10": d["p10"],
                    "target_p50": d["p50"],
                    "target_p90": d["p90"],
                    "target_max": d["max"],
                    "ratio_p50": d["p50"] / max(q["p50"], 1),
                    "ratio_mean": d["mean"] / max(q["mean"], 1e-9),
                }
            )
    return pd.DataFrame(rows)


def plot_dumbbell(df: pd.DataFrame, out_path: Path) -> None:
    groups = list(df["group"].unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, max(6, 0.45 * len(df) + 2 * len(groups))), squeeze=False)
    query_color = "#1f77b4"
    target_color = "#d62728"

    for ax, group in zip(axes.flatten(), groups):
        sub = df[df["group"] == group].sort_values("ratio_p50", ascending=True).reset_index(drop=True)
        y = range(len(sub))
        ax.hlines(y=y, xmin=sub["query_p50"], xmax=sub["target_p50"], color="#bdbdbd", linewidth=1.8, zorder=1)
        ax.scatter(sub["query_p50"], y, color=query_color, s=45, label="Query p50", zorder=3)
        ax.scatter(sub["target_p50"], y, color=target_color, s=45, label="Target p50", zorder=3)
        for yi, ratio in zip(y, sub["ratio_p50"]):
            ax.text(sub.iloc[yi]["target_p50"] * 1.03, yi, f"x{ratio:.1f}", va="center", fontsize=8, color="#444444")
        ax.set_yticks(list(y))
        ax.set_yticklabels(sub["dataset"].tolist(), fontsize=9)
        ax.set_xscale("log")
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.set_title(group)
        ax.set_xlabel("Sequence length after processor / before MRL loss (log scale)")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2], loc="lower right")

    fig.suptitle("Query/Target Sequence Length Asymmetry", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.sort_values(["group", "ratio_p50"], ascending=[True, False]).copy()
    plot_df["label"] = plot_df["group"] + " | " + plot_df["dataset"]
    fig_h = max(6, 0.35 * len(plot_df))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    colors = plot_df["group"].map({
        "vidore_v1": "#4c78a8",
        "vidore_v2": "#f58518",
        "mmeb": "#54a24b",
    }).fillna("#777777")
    ax.barh(plot_df["label"], plot_df["ratio_p50"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Target p50 / Query p50")
    ax.set_title("Length Asymmetry Ratio by Dataset")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for y, v in enumerate(plot_df["ratio_p50"]):
        ax.text(v + max(plot_df["ratio_p50"]) * 0.01, y, f"x{v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-json", default="colqwen_multigranularity/experiments/exp_maxsim/results/all_lengths.json")
    ap.add_argument("--out-dir", default="colqwen_multigranularity/experiments/exp_maxsim/plots")
    args = ap.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(input_json)
    plot_dumbbell(df, out_dir / "length_asymmetry_dumbbell.png")
    plot_ratio(df, out_dir / "length_asymmetry_ratio.png")
    (out_dir / "length_asymmetry_table.csv").write_text(df.to_csv(index=False))
    print(f"wrote plots to {out_dir}")


if __name__ == '__main__':
    main()
