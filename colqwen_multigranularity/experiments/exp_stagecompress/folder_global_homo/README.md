# FolderGlobalHomo

`folder_global_homo/` is an isolated implementation of **Global-Guided Residual HomoFolder**.

This is the first implemented follow-up to `folder_homo/`. It keeps the same Qwen2.5VL / ColQwen2.5 training recipe and MRL prefix interface, but adds crop-level global guidance for g2/g3 compression.

## Method Shape

The target output is still fixed-length residual MRL:

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

Compared with `folder_homo/`, this method adds a learned global crop commander:

```text
1. Compress g1 with FOLDER-style merge -> G1.
2. Split g2 into 2 crop groups and g3 into 4 crop groups.
3. Score each local crop with GlobalCropCommander(G1 or G1+R2, crop).
4. Allocate the fixed stage budget across crop groups using deterministic score-ranked remainder allocation.
5. Run FOLDER-style merge inside each crop with its allocated budget.
6. Concatenate crop residuals in deterministic crop order.
```

The crop score is also passed into the token protect/value scaling path, so the commander receives gradient through the MRL loss. It is not only a hard, non-differentiable budget allocator.

## Important Config

| Option | Default | Meaning |
|---|---:|---|
| `BUDGETS` | `160 160 160` | MRL visual token budgets for `G1/R2/R3`. |
| `GLOBAL_GUIDANCE_WEIGHT` | `0.5` | Weight of crop-level global importance inside token protect score. |
| `GLOBAL_MIN_BUDGET_RATIO` | `0.6` | Fraction of each stage budget reserved as per-crop minimum before score-ranked extras. |
| `NOVELTY_WEIGHT` | `1.0` | Weight of novelty relative to coarse anchors. |
| `GATE_STRENGTH` | `0.25` | Continuous token value scaling strength. |

## Training Rule

Use native Qwen2.5/ColQwen2.5 base and train all of these together:

- LLM LoRA (`--use-peft`)
- `custom_text_proj`
- `folder_global_homo`

Do not report MRL-main initialized or compressor-only runs as the mainline result. Those are diagnostics only.

Default command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/folder_global_homo/run_train.sh
```

Run this only after the current `folder_homo` residual-160 training/evaluation is finished or GPUs are intentionally freed.
