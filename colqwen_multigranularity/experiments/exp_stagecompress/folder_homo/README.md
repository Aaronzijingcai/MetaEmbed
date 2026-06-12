# FolderHomo

`folder_homo/` is the active homogeneity mainline for StageCompress.

The goal is to build an isolated trainable compressor inspired by the successful MLP-after FOLDER result, then study whether g1/g2/g3 and multi-image/multi-crop tokens can be compressed through explicit homogeneity/redundancy modeling.

## Relationship To FOLDER

| Item | Role |
|---|---|
| `mlppost/strategies/strategy5_folder.py` | Mature MLP-after FOLDER reference. Do not modify it for this line. |
| `folder_homo/` | New isolated homogeneity model. Use this for current experiments. |
| `mainlines/homogeneity/` | Clean entry point with links back to this implementation and the FOLDER reference. |

## Method Shape

The method keeps the MRL budgets `g1/g2/g3 = 160/320/640`, but compresses stages hierarchically:

- `g1`: compressed with a FOLDER-style redundancy merge.
- `g2`: compressed with coarse anchors from compressed `g1`; tokens redundant with coarse anchors are easier to merge.
- `g3`: compressed with coarse anchors from compressed `g1+g2`.

This branch intentionally does not modify `mlppost/strategy5_folder.py`.

## Valid Mainline Training

Use native Qwen2.5/ColQwen2.5 base and train all of these together:

- LLM LoRA (`--use-peft`)
- `custom_text_proj`
- `folder_homo`

Do not report MRL-main initialized or compressor-only runs as the mainline result.
Those are diagnostics only.

Current corrected run:

```text
experiments/exp_stagecompress/runs/folder_homo_native_qwen25_lora_linear_folder_bsz4_20260610_102541
```

Default command:

```bash
cd /MURE-V2/code/MetaEmbed/colqwen_multigranularity
bash experiments/exp_stagecompress/folder_homo/run_train.sh
```

Formal command templates are in `../FORMAL_8GPU_COMMANDS.md`.
