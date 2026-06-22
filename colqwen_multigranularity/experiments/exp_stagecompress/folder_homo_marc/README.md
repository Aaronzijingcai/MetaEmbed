# MARC-v1: MaxSim-Aware Residual Compression

MARC keeps late-interaction retrieval unchanged at inference time:

```text
query -> query tokens
document -> FolderHomo/MARC compressed document tokens
score(q, d) = sum_i max_j sim(q_i, d_j)
```

The difference is only in training. When `MARC_ENABLED=1`, the FolderHomo compressor records each active stage's pre-compression tokens and scorer logits during the document forward pass. The loss then uses the positive query-document MaxSim interaction to build a soft utility target over the source document tokens:

```text
A = Q D_stage^T
u_j = sum_i softmax_beta(A_i)_j
L_marc = KL(softmax(saliency_logits), normalize(u))
```

This trains the document-side scorer to preserve tokens that are actually useful under MaxSim, while still allowing offline document indexing at inference time.

## Recommended First Run

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
RUN_NAME=folder_homo_marc_v1_b160_160_160_3k \
MARC_ENABLED=1 MARC_WEIGHT=0.1 MARC_BETA=20 \
BUDGETS="160 160 160" MAX_STEPS=3000 SAVE_STEPS=500 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 GRAD_ACCUM_STEPS=1 \
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
bash experiments/exp_stagecompress/folder_homo/run_train.sh
```

## Ablation knobs

- `MARC_WEIGHT=0.05/0.1/0.2`: strength of MaxSim utility distillation.
- `MARC_BETA=10/20/40`: sharpness of the interaction-derived utility target.
- Keep `BUDGETS="160 160 160"` for direct comparison with residual160.

## Evaluation

Use the existing FolderHomo evaluation path on `checkpoint-3000`:

```bash
CHECKPOINT=experiments/exp_stagecompress/runs/folder_homo_marc_v1_b160_160_160_3k/checkpoint-3000 \
OUT_DIR=experiments/exp_stagecompress/runs/folder_homo_marc_v1_b160_160_160_3k/eval/full_3sets_checkpoint3000_budget160 \
BUDGETS="160 160 160" BATCH_QUERY=16 BATCH_PASSAGE=24 BATCH_SCORE=64 NUM_WORKERS=0 \
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
bash experiments/exp_stagecompress/folder_homo/eval_3sets.sh
```

## Latest Formal Result

See `../analysis/marc_v1_result_and_margin_plan.md` for the MARC-v1 3k result, failure analysis, and the narrowed MARC-v2 margin-aware plan.

## MARC-v2: Margin-Aware Mode

MARC-v2 keeps the same inference path as MARC-v1. Query and document are still encoded separately, and retrieval still uses late interaction:

```text
score(q, d) = sum_i max_j sim(q_i, d_j)
```

The training target changes from positive-only token utility to margin-aware token utility. For each query token, the loss compares the positive document's MaxSim score against the hardest negative score from in-batch documents and the paired hard negative:

```text
s_i+ = max_j q_i^T d_j+
s_i- = max_{d-} max_k q_i^T d_k-
v_i = softplus((s_i- + margin - s_i+) / tau)
u_j = sum_i v_i * softmax_beta(q_i^T x_j+)
```

This means a source token receives stronger supervision when it supports query tokens whose positive evidence is threatened by negatives. The method is still end-to-end and does not use a teacher model.

### Fast 50-step sanity run

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity

RUN_NAME=folder_homo_marc_v2_margin_b160_160_160_sanity50 \
MARC_ENABLED=1 MARC_MODE=margin MARC_WEIGHT=0.02 MARC_BETA=20 MARC_MARGIN=0.02 MARC_TAU=0.05 \
BUDGETS="160 160 160" MAX_STEPS=50 SAVE_STEPS=50 \
TRAIN_BSZ=4 INTERLEAVED_BSZ=4 GRAD_ACCUM_STEPS=1 \
CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 NUM_GPUS=8 \
bash experiments/exp_stagecompress/folder_homo/run_train.sh
```

Expected sanity signals in the training log:

- `marc2_stage_count > 0`
- `marc2_margin_violation` is present and finite
- `marc2_weighted / loss` is not near MARC-v1's `0.04%` level

### 300/500-step quick gate

```bash
CHECKPOINT=experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_sanity50/checkpoint-50 \
TRAIN_LOG=experiments/exp_stagecompress/runs/folder_homo_marc_v2_margin_b160_160_160_sanity50/logs/<train_log>.log \
NUM_GPUS=8 CUDA_DEVICE_LIST=0,1,2,3,4,5,6,7 \
BUDGETS="160 160 160" \
EVAL_MAX_QUERIES=16 EVAL_MAX_CORPUS=96 \
BATCH_QUERY=8 BATCH_PASSAGE=8 BATCH_SCORE=32 NUM_WORKERS=0 \
bash experiments/exp_stagecompress/analysis/run_quick_gate.sh
```

For a real early checkpoint, prefer `MAX_STEPS=300 SAVE_STEPS=300` first, then `MAX_STEPS=500 SAVE_STEPS=500` only if the 300-step signals are healthy.

Initial hyperparameters:

- `MARC_MODE=margin`
- `MARC_WEIGHT=0.02`
- `MARC_BETA=20`
- `MARC_MARGIN=0.02`
- `MARC_TAU=0.05`

Use the quick gate to decide whether `MARC_WEIGHT` should be adjusted. If the auxiliary/main ratio is still too weak, increase `MARC_WEIGHT`; if it is above the gate range or destabilizes MRL losses, reduce it.
