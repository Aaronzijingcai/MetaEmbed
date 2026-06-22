# Badcase MaxSim Scoring Ablation

This folder contains eval-only scoring ablations motivated by the observed badcases:

- prompt/special/query-template tokens contribute high MaxSim to both positive and negative documents;
- repeated query augmentation tokens can add non-semantic score mass;
- many query tokens sometimes hit a small number of document tokens, producing concentrated interactions.

The changes are disabled by default. Original evaluation is unchanged unless the following options are passed through `eval.py` / `eval_3sets.sh`:

| Option | Default | Effect |
|---|---:|---|
| `QUERY_AUGMENTATION_REPEATS` | `10` | Controls repeated `<|endoftext|>` query augmentation in the processor. |
| `MAXSIM_QUERY_DROP_PREFIX` | `0` | Drops this many leading query embedding positions before scoring. Diagnostic only. |
| `MAXSIM_QUERY_DROP_SUFFIX` | `0` | Drops this many trailing query embedding positions before scoring. Diagnostic only. |
| `MAXSIM_QUERY_AGG` | `sum` | One of `sum`, `mean`, `topk_mean`. |
| `MAXSIM_QUERY_TOPK` | `0` | Number of query-token MaxSim scores used by `topk_mean`; default inside code is `min(8, query_len)`. |
| `MAXSIM_LENGTH_NORM_ALPHA` | `0.0` | Divides summed MaxSim by `query_len ** alpha`. |
| `MAXSIM_HIT_PENALTY_WEIGHT` | `0.0` | Penalizes query-document pairs where too many query tokens hit the same doc token. |
| `MAXSIM_HIT_PENALTY_THRESHOLD` | `0.35` | Hit-concentration fraction above which the penalty starts. |

Recommended first run on the current strongest checkpoint:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
CHECKPOINT=experiments/exp_stagecompress/runs/folder_homo_residual160_native_qwen25_lora_linear_folder_bsz4_gc_20260611_163512/checkpoint-2500 \
NUM_GPUS=8 CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
BATCH_QUERY=32 BATCH_PASSAGE=32 BATCH_SCORE=128 NUM_WORKERS=0 \
bash experiments/exp_stagecompress/badcase_maxsim/run_folder_homo_scoring_ablation.sh
```

Interpretation priority:

1. If `qaug0` or `qaug2` beats `baseline_qaug10`, query augmentation noise is real and should be reported.
2. If `qaug0_trim_suffix8` helps, implement precise token-id masking next instead of relying on positional trimming.
3. If `qaug0_topk8_mean` helps, MaxSim sum is over-counting template/common query tokens.
4. If `qaug0_hitpenalty` helps, hit concentration is a real scoring-side defect.
