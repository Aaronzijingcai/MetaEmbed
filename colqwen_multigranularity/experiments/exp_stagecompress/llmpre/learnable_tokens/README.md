# LLM-Pre Global MRL Tokens

This is the active compression direction for StageCompress. It adapts the MetaEmbed global-token idea to multi-granularity ColQwen, while preserving the document-side multi-sampling path as multi-image input.

Core implementation: `modeling_global_mrl_tokens.py`. The model appends learnable Global MRL tokens before the LLM, selects query/doc token groups from LLM hidden states, and then applies `custom_text_proj`. Default groups are `1,1;2,4;4,8;8,16;16,64`.

Default iterative training config: `TRAIN_BSZ=4` and `INTERLEAVED_BSZ=4`. Keep both values equal across model iterations unless the run is explicitly marked as a batch-size probe.

## Smoke

Use two GPUs for the short train + eval path:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CUDA_DEVICE_LIST=0,1 NUM_GPUS=2 MAX_STEPS=8 SAVE_STEPS=4 TRAIN_BSZ=4 INTERLEAVED_BSZ=4 \
  bash experiments/exp_stagecompress/llmpre/smoke_2gpu_train_eval.sh
```

The smoke launcher enables `GLOBAL_MRL_DEBUG=1` by default. Debug logs report input shape, attention lengths, image placeholder counts, `image_grid_thw`, appended prompt-token counts, selected query/doc token widths, output shape, finite checks, and norm range. Set `GLOBAL_MRL_DEBUG=0` to disable this for longer runs.

Full evaluation should use `eval_3sets.sh` with `EVAL_MODE=full`; smoke evaluation uses `EVAL_MODE=smoke` and limits each representative dataset.

## Stage-Inserted Learnable Token Budgets

The current learnable-token line focuses on tokens inserted inside the visual
sequence, at the g1/g2/g3 stage boundaries. Tail-only Global MRL Tokens are kept
as the MetaEmbed-style reference, but they are not the main novelty path.

The inserted-token interpretation is:

```text
g1 + L1 + g2 + L2 + g3 + L3 + text
```

MRL supervision uses cumulative prefixes of the inserted learnable tokens:

```text
L1
L1 + L2
L1 + L2 + L3
```

### Target/Page Budget Sweep

Fix query inserted tokens to `(2,4,8)`, so the query-side MRL prefixes are
`2,6,14`. Sweep only the page/target inserted-token budget.

| Target Sweep | Query Inserted Tokens | Target Inserted Tokens | MRL Groups | Purpose |
|---:|---|---|---|---|
| T1 | `2,4,8` | `4,8,16` | `2,4;6,12;14,28` | small target capacity |
| T2 | `2,4,8` | `8,16,32` | `2,8;6,24;14,56` | main target-capacity setting, query:target approx 1:4 |
| T3 | `2,4,8` | `16,32,64` | `2,16;6,48;14,112` | larger page capacity |
| T4 | `2,4,8` | `32,64,128` | `2,32;6,96;14,224` | high-capacity upper bound |

### Query Budget Sweep

Fix target/page inserted tokens to `(8,16,32)`, so the target-side MRL prefixes
are `8,24,56`. Sweep only the query inserted-token budget.

| Query Sweep | Query Inserted Tokens | Target Inserted Tokens | MRL Groups | Purpose |
|---:|---|---|---|---|
| Q1 | `1,2,4` | `8,16,32` | `1,8;3,24;7,56` | very short query side |
| Q2 | `2,4,8` | `8,16,32` | `2,8;6,24;14,56` | default balanced setting |
| Q3 | `4,8,16` | `8,16,32` | `4,8;12,24;28,56` | larger query capacity |
| Q4 | `8,16,32` | `8,16,32` | `8,8;24,24;56,56` | symmetric upper query setting |

### Experiment Priority

The sweep should be run in a controlled order rather than all at once. The main
rule is: first establish the page/target budget under a fixed query budget, then
study the query budget under the best or default page budget.

| Priority | Run | Query Inserted Tokens | Target Inserted Tokens | Why |
|---:|---|---|---|---|
| P0 | Single-GPU smoke | `2,4,8` | `8,16,32` | Code-path validation only; confirms train/eval shapes and MRL groups before spending formal compute. |
| P1 | T2 main | `2,4,8` | `8,16,32` | Default main setting; query prefix `2,6,14`, target prefix `8,24,56`, roughly keeps MetaEmbed-style query:target asymmetry. |
| P2 | T3 capacity-up | `2,4,8` | `16,32,64` | Tests whether page-side capacity is the main bottleneck. |
| P3 | T1 capacity-down | `2,4,8` | `4,8,16` | Tests how much performance drops when target capacity is small. |
| P4 | T4 upper bound | `2,4,8` | `32,64,128` | Expensive high-capacity upper bound; run after T2/T3 show promise. |
| P5 | Q1 query-down | `1,2,4` | `8,16,32` | Tests whether query can be shorter while target remains rich. |
| P6 | Q3 query-up | `4,8,16` | `8,16,32` | Tests whether adding query capacity helps once target is fixed. |
| P7 | Q4 symmetric | `8,16,32` | `8,16,32` | Symmetric upper query setting; mainly for analysis, not the expected main setting. |
| P8 | Tail-placement control | `2,4,8` | `8,16,32` | Same Q2/T2 budget but place all learnable tokens at the sequence tail; isolates whether stage insertion is better than MetaEmbed-style tail placement. |

Orthogonality regularization should not be swept across all budgets initially.
Use Q2/T2 for the first orthogonality ablation with `ORTH_LAMBDA=0,0.01,0.05`,
then reuse the best value for the target/query budget sweep.

Default stage-inserted training uses Q2/T2:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
env QUERY_STAGE_MRL_TOKENS=2,4,8 DOC_STAGE_MRL_TOKENS=8,16,32 bash experiments/exp_stagecompress/llmpre/learnable_tokens/run_stage_interleaved_budgetmatch_train.sh
```

Single-GPU smoke train + tiny eval:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/llmpre/learnable_tokens/smoke_stage_interleaved_single_gpu_train_eval.sh
```

Default stage-inserted evaluation:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
env QUERY_STAGE_MRL_TOKENS=2,4,8 DOC_STAGE_MRL_TOKENS=8,16,32 bash experiments/exp_stagecompress/llmpre/learnable_tokens/eval_stage_interleaved_budgetmatch_3sets.sh
```

Use `ORTH_LAMBDA=0.01/0.05/0.1` to enable the ReMatch-style orthogonality
regularizer for learnable-token diversity.
