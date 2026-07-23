import json
from pathlib import Path

from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5_VLConfig

from colpali_engine.models.qwen2_5.lastqwen2_5.modeling_lastqwen2_5_new import (
    LastQwen2_5,
)


MODEL_PATH = Path(
    "/MURE-V2/code/MetaEmbed/colqwen_multigranularity/models/colqwen2.5-base"
)
OUTPUT_PATH = Path("/tmp/metaembed_official_lora_audit.json")


def classify(name: str) -> str:
    if ".visual." in name:
        return "visual_lora"
    if ".language_model." in name:
        return "language_lora"
    if "prompt_embed_tokens" in name:
        return "prompt_embed_tokens"
    if "custom_text_proj" in name:
        return "custom_text_proj"
    return "other"


def audit_case(config, dim: int):
    with init_empty_weights():
        model = LastQwen2_5(
            config,
            dim=dim,
            num_query_prompt_tokens=16,
            num_doc_prompt_tokens=64,
        )
    model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)
    groups = {}
    module_names = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = classify(name)
        stats = groups.setdefault(group, {"tensors": 0, "numel": 0})
        stats["tensors"] += 1
        stats["numel"] += parameter.numel()
        if ".lora_" in name:
            module_name = name.split(".lora_", 1)[0]
            module_names.setdefault(group, set()).add(module_name)
    return {
        "dim": dim,
        "groups": groups,
        "lora_module_counts": {
            group: len(names) for group, names in module_names.items()
        },
        "lora_module_examples": {
            group: sorted(names)[:12] for group, names in module_names.items()
        },
    }


config = Qwen2_5_VLConfig.from_pretrained(MODEL_PATH)
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    init_lora_weights="gaussian",
    bias="none",
    task_type="FEATURE_EXTRACTION",
    target_modules="(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$)",
    modules_to_save=["prompt_embed_tokens", "custom_text_proj"],
)
result = {
    "model_config": str(MODEL_PATH),
    "peft_target_modules": peft_config.target_modules,
    "modules_to_save": sorted(peft_config.modules_to_save),
    "cases": {
        "official_yaml_dim_minus_1": audit_case(config, -1),
        "mure_custom_projection_dim_128": audit_case(config, 128),
    },
}
OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
