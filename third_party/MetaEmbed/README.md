# MetaEmbed

MetaEmbed is a framework for multimodal retrieval that rethinks how multimodal embeddings are constructed and interacted with at scale. During training, a fixed number of learnable **Meta Tokens** are appended to the input sequence. At test-time, their last-layer contextualized representations serve as compact yet expressive **multi-vector embeddings**. Through the proposed **Matryoshka Multi-Vector Retrieval (MMR)** training, MetaEmbed learns to organize information by granularity across multiple vectors, enabling test-time scaling where users can balance retrieval quality against efficiency by selecting the number of tokens used for indexing and retrieval.

- Paper: https://arxiv.org/abs/2509.18095

## Matryoshka Multi-Vector Retrieval (MMR)

Standard multi-vector retrieval uses a fixed number of embedding vectors per query/document, offering no flexibility to trade off quality vs. efficiency at test time. MetaEmbed addresses this with **Matryoshka Multi-Vector Retrieval (MMR)**, a training strategy inspired by Matryoshka Representation Learning but applied at the *token level* across multiple vectors.

During training, MMR jointly optimizes late-interaction retrieval losses at **multiple granularities** of the multi-vector representation. Specifically, the loss is computed over several `(num_query_tokens, num_doc_tokens)` groups simultaneously, e.g., `(1,1), (2,4), (4,8), (8,16), (16,64)`. This teaches the model to organize the most critical information into the first few Meta Tokens, while progressively encoding finer-grained details into subsequent tokens.

The MMR groups are configured in the loss function:

```yaml
loss_func:
  (): colpali_engine.loss.late_interaction_losses.ColbertInBatchNegativeLoss
  normalize_scores: false
  temperature: 0.03
  mrl_groups: [ [1, 1, 1.0], [2, 4, 1.0], [4, 8, 1.0], [8, 16, 1.0], [16, 64, 1.0] ]
```

Each entry `[num_query_tokens, num_doc_tokens, weight]` defines a granularity level and its loss weight.

## Installation

```bash
pip install torch transformers==4.55.0 peft accelerate
pip install -r requirements.txt
```

## Quick Start

### Model Loading

```python
from colpali_engine.models import LastQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5 import ColQwen2_5_Processor

model = LastQwen2_5.from_pretrained(
    "your-model-path",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_liger_kernel=True,
    dim=128,
    num_query_prompt_tokens=16,
    num_doc_prompt_tokens=64,
)
model = model.to("cuda")
model.eval()

processor = ColQwen2_5_Processor.from_pretrained("your-model-path")
```

### Encoding Queries and Documents

```python
# Encode a text query
query = "What is the capital of France?"
query_inputs = processor(text=query, return_tensors="pt").to("cuda")
query_emb = model(**query_inputs, is_query=True)

# Encode a document image
from PIL import Image
image = Image.open("document.png")
doc_inputs = processor(images=image, return_tensors="pt").to("cuda")
doc_emb = model(**doc_inputs, is_query=False)

# Compute similarity (late interaction / MaxSim)
sim = (query_emb @ doc_emb.T).max(dim=-1).values.sum()
```

## Training

### Basic Training

```bash
python scripts/train/train_colbert.py --config-file scripts/7b_mrl.yaml
```

### Training Configuration

Training is configured via YAML files. See `scripts/7b_mrl.yaml` for a full example. Key options include:

```yaml
config:
  (): colpali_engine.trainer.colmodel_training.ColModelTrainingConfig
  output_dir: !path ./output/lastqwen2.5-7b-mrl

  model:
    (): colpali_engine.utils.transformers_wrappers.AllPurposeWrapper
    class_to_instantiate: !ext colpali_engine.models.LastQwen2_5
    pretrained_model_name_or_path: "./models/colqwen2.5-7B-base"
    torch_dtype: !ext torch.bfloat16
    use_liger_kernel: true
    dim: -1
    num_query_prompt_tokens: 16   # number of Meta Tokens for queries
    num_doc_prompt_tokens: 64     # number of Meta Tokens for documents

  loss_func:
    (): colpali_engine.loss.late_interaction_losses.ColbertInBatchNegativeLoss
    temperature: 0.03
    mrl_groups: [ [1, 1, 1.0], [2, 4, 1.0], [4, 8, 1.0], [8, 16, 1.0], [16, 64, 1.0] ]

  peft_config:
    (): peft.LoraConfig
    r: 32
    lora_alpha: 32
    target_modules: '(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)'
    modules_to_save: ['prompt_embed_tokens', 'custom_text_proj']
```

### Distributed Training with FSDP

```bash
accelerate launch --config_file fsdp.yaml \
    scripts/train/train_colbert.py \
    --config-file scripts/7b_mrl.yaml
```

## Evaluation

### In-Training Evaluation

Enable periodic evaluation during training by specifying eval datasets in the config:

```yaml
config:
  vidore_eval_frequency: 400
  eval_dataset_loader: !import ./data/test_data_vidore_beir.yaml
  eval_dataset_loader_mmeb: !import ./data/test_data_mast_mmeb_v3.yaml
```

### Evaluation Configs

Evaluation dataset configs are available in `scripts/data/`:

- `test_data_vidore_beir.yaml` — ViDoRe benchmark
- `test_data_mast_mmeb_v3.yaml` — MMEB benchmark
- `test_data_mast_v2.yaml` — Additional evaluation suite

## Project Structure

```
MetaEmbed/
├── colpali_engine/
│   ├── models/           # Model implementations
│   │   ├── qwen2_5/      # Qwen2.5-VL models (Last, Col, Bi)
│   │   ├── qwen3/        # Qwen3-VL models
│   │   ├── llama3vision/ # Llama 3.2 Vision models
│   │   ├── paligemma/    # PaliGemma models
│   │   └── idefics3/     # Idefics3 models
│   ├── loss/             # Loss functions (incl. MMR losses)
│   ├── trainer/          # Training utilities
│   ├── collators/        # Data collators
│   ├── compression/      # Token pooling / compression
│   ├── interpretability/ # Attention maps and similarity maps
│   └── utils/            # Utility functions
├── vidore_benchmark/     # Evaluation framework
│   ├── evaluation/       # Evaluators (ViDoRe, MMEB, BEIR, etc.)
│   ├── retrievers/       # Retriever implementations
│   └── utils/            # Evaluation utilities
└── scripts/
    ├── train/            # Training scripts
    ├── data/             # Evaluation dataset configs
    ├── 7b_mrl.yaml       # Main training config (7B with MMR)
    └── moca_data_ratios_v3_nommE5.yaml  # Data mixing ratios
```

## Citation

```bibtex
@inproceedings{
    xiao2026metaembed,
    title={MetaEmbed: Scaling Multimodal Retrieval at Test-Time with Flexible Late Interaction},
    author={Zilin Xiao and Qi Ma and Mengting Gu and Chun-cheng Jason Chen and Xintao Chen and Vicente Ordonez and Vijai Mohan},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=yKDqg9HwZX}
}
```

## License

The majority of MetaEmbed is licensed under CC-BY-NC, however portions of the project are available under separate license terms: colpali and vidore-benchmark are licensed under the MIT license.

## Acknowledgments

We thank the following open-source projects:

- [ColPali](https://github.com/illuin-tech/colpali)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [vidore-benchmark](https://github.com/illuin-tech/vidore-benchmark)
