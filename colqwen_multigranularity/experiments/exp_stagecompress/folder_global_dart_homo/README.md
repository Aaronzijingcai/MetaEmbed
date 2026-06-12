# FolderGlobalDartHomo

`folder_global_dart_homo/` implements **GlobalCom-DART Fusion** for cross-granularity homogeneity compression.

It combines the crop-level global commander from `folder_global_homo/` with the token-level pivot novelty from `folder_dart_pivot/`.

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
2. Split g2/g3 into crop groups.
3. Use GlobalCropCommander(coarse, crop) to allocate fixed residual budget across crops.
4. Select top-k visual pivots from G1 or G1+R2.
5. Score token novelty inside each crop against the pivots.
6. Run FOLDER-style merge inside each crop with protect = saliency + crop importance + pivot novelty.
7. Concatenate crop residuals in deterministic crop order.
```

This method answers two questions separately: global guidance decides which crop deserves budget, and DART-style pivots decide which tokens in that crop are non-duplicated.

## Important Config

| Option | Default | Meaning |
|---|---:|---|
| `BUDGETS` | `160 160 160` | MRL visual token budgets for `G1/R2/R3`. |
| `PIVOT_COUNT` | `32` | Number of coarse visual pivots used for token novelty. |
| `PIVOT_SCORE` | `saliency` | Pivot selection score: `saliency`, `norm`, or `uniform`. |
| `GLOBAL_GUIDANCE_WEIGHT` | `0.5` | Weight of crop-level global importance inside token protect score. |
| `GLOBAL_MIN_BUDGET_RATIO` | `0.6` | Fraction of each stage budget reserved as per-crop minimum. |
| `NOVELTY_WEIGHT` | `1.0` | Weight of pivot novelty inside token protect score. |
| `GATE_STRENGTH` | `0.25` | Continuous token value scaling strength. |

## Training Rule

Use native Qwen2.5/ColQwen2.5 base and train all of these together:

- LLM LoRA (`--use-peft`)
- `custom_text_proj`
- `folder_global_dart_homo`

Default command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/folder_global_dart_homo/run_train.sh
```
