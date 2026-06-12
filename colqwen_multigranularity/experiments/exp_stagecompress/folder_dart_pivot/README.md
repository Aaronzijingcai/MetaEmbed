# FolderDartPivot

`folder_dart_pivot/` implements **DART-Pivot Residual HomoFolder**.

It keeps the same Qwen2.5VL / ColQwen2.5 training recipe and the same MRL prefix interface as `folder_homo/`, but changes how cross-granularity novelty is estimated. Instead of comparing every local token with all coarse anchors, it first selects a small set of coarse visual pivots and computes residual novelty against those pivots.

## Method Shape

Target output:

```text
Level 1: G1
Level 2: G1 + R2
Level 3: G1 + R2 + R3
```

Default visual budgets:

```text
G1 = 160
R2 = 160
R3 = 160
Total = 480 visual tokens
```

Core flow:

```text
1. Compress g1 with FOLDER-style merge -> G1.
2. Select top-k visual pivots from G1.
3. Score g2 novelty by distance from those pivots.
4. Compress g2 with protect = saliency + novelty -> R2.
5. Select top-k visual pivots from G1 + R2.
6. Score g3 novelty by distance from those pivots.
7. Compress g3 with FOLDER-style merge -> R3.
```

The pivots are real compressed visual tokens, not synthetic learnable tokens.

## Important Config

| Option | Default | Meaning |
|---|---:|---|
| `BUDGETS` | `160 160 160` | MRL visual token budgets for `G1/R2/R3`. |
| `PIVOT_COUNT` | `32` | Number of coarse visual pivots used for novelty. |
| `PIVOT_SCORE` | `saliency` | Pivot selection score: `saliency`, `norm`, or `uniform`. |
| `NOVELTY_WEIGHT` | `1.0` | Weight of pivot novelty inside token protect score. |
| `GATE_STRENGTH` | `0.25` | Continuous token value scaling strength. |

## Training Rule

Use native Qwen2.5/ColQwen2.5 base and train all of these together:

- LLM LoRA (`--use-peft`)
- `custom_text_proj`
- `folder_dart_pivot`

Default command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/folder_dart_pivot/run_train.sh
```
