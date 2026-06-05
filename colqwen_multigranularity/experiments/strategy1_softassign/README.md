# Soft Assignment Experiment

This experiment implements pre-LLM visual token compression.

Pipeline:

```text
pixel_values -> visual encoder -> stage-wise soft assignment
             -> compact input_ids / attention_mask / image_grid_thw
             -> get_rope_index(compact sequence)
             -> language_model
```

Compact row layout:

```text
text_without_vision_markers
+ <vision_start> C1 <vision_end>
+ <vision_start> C2 <vision_end>
+ <vision_start> C3 <vision_end>
```

Four-GPU training:

```bash
bash colqwen_multigranularity/experiments/strategy1_softassign/train_4gpu.sh
```

B2/B3/B4 per-stage serial training:

```bash
./colqwen_multigranularity/experiments/strategy1_softassign/train_b2_b3_b4_serial.sh
```

The four-GPU script follows the current `mrl_main` defaults: LoRA is enabled by
default in the shared training parser, and the script passes `--use-peft` unless
`USE_PEFT=0` is set. The model also implements explicit Soft Assignment export:

```text
strategy1_softassign.bin
strategy1_softassign_config.json
```

Text-query focused evaluation:

```bash
CHECKPOINT=/path/to/checkpoint \
bash colqwen_multigranularity/experiments/strategy1_softassign/eval_textquery_focus.sh
```
