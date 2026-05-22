from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_rows(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text())
    rows = []
    for subset, stats in data.items():
        q = stats['query_summary']
        d = stats['positive_summary']
        rows.append(
            {
                'subset': subset,
                'query_count': q['count'],
                'query_mean': q['mean'],
                'query_p10': q['p10'],
                'query_p50': q['p50'],
                'query_p90': q['p90'],
                'query_max': q['max'],
                'positive_count': d['count'],
                'positive_mean': d['mean'],
                'positive_p10': d['p10'],
                'positive_p50': d['p50'],
                'positive_p90': d['p90'],
                'positive_max': d['max'],
                'ratio_p50': d['p50'] / max(q['p50'], 1),
                'ratio_mean': d['mean'] / max(q['mean'], 1e-9),
            }
        )
    return pd.DataFrame(rows)


def plot_dumbbell(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.sort_values('ratio_p50', ascending=True).reset_index(drop=True)
    fig_h = max(8, 0.45 * len(sub))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    query_color = '#1f77b4'
    pos_color = '#d62728'
    y = range(len(sub))
    ax.hlines(y=y, xmin=sub['query_p50'], xmax=sub['positive_p50'], color='#bdbdbd', linewidth=1.8, zorder=1)
    ax.scatter(sub['query_p50'], y, color=query_color, s=45, label='Query p50', zorder=3)
    ax.scatter(sub['positive_p50'], y, color=pos_color, s=45, label='Positive p50', zorder=3)
    for yi, ratio in zip(y, sub['ratio_p50']):
        ax.text(sub.iloc[yi]['positive_p50'] * 1.03, yi, f'x{ratio:.4f}', va='center', fontsize=8, color='#444444')
    ax.set_yticks(list(y))
    ax.set_yticklabels(sub['subset'].tolist(), fontsize=9)
    ax.set_xscale('log')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.set_title('Train Query/Positive Sequence Length Asymmetry')
    ax.set_xlabel('Sequence length after processor / before MRL loss (log scale)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_ratio(df: pd.DataFrame, out_path: Path) -> None:
    sub = df.sort_values('ratio_p50', ascending=False).reset_index(drop=True)
    fig_h = max(8, 0.35 * len(sub))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.barh(sub['subset'], sub['ratio_p50'], color='#4c78a8')
    ax.invert_yaxis()
    ax.set_xlabel('Positive p50 / Query p50')
    ax.set_title('Train Length Asymmetry Ratio by Subset')
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    for y, v in enumerate(sub['ratio_p50']):
        ax.text(v + max(sub['ratio_p50']) * 0.01, y, f'x{v:.4f}', va='center', fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-json', default='colqwen_multigranularity/experiments/exp_maxsim/results/train_lengths.json')
    ap.add_argument('--out-dir', default='colqwen_multigranularity/experiments/exp_maxsim/plots')
    args = ap.parse_args()

    df = load_rows(Path(args.input_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dumbbell(df, out_dir / 'train_length_asymmetry_dumbbell.png')
    plot_ratio(df, out_dir / 'train_length_asymmetry_ratio.png')
    (out_dir / 'train_length_asymmetry_table.csv').write_text(df.to_csv(index=False))
    print(f'wrote plots to {out_dir}')


if __name__ == '__main__':
    main()
