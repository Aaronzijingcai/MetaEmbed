import os
import random
from typing import Any, Dict


# from https://github.com/Code-kunkun/LamRA/blob/72863991e62f364d734231b6e3327a985c229984/dataset/datasets_mbeir.py#L394
def _load_query_instructions(instructions_path):
    """Validate and load instructions for MBEIR dataset."""
    # Validate the path and file extension
    assert os.path.exists(
        instructions_path
    ), f"Instructions Path {instructions_path} does not exist"
    assert instructions_path.endswith(
        ".tsv"
    ), f"Instructions Path {instructions_path} is not a tsv file"
    prompts_dict = {}
    with open(instructions_path, "r") as f:
        next(f)  # Skip the header line
        for line in f.readlines():
            parts = line.strip().split("\t")
            # Construct the key to be dataset_id, query_modality, cand_modality
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]  # Filters out any empty prompts
            prompts_dict[key] = prompts
    query_instructions = prompts_dict
    return query_instructions


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


# from LamRA preprocessing
def get_rand_cand(cand_list):
    return random.choice(cand_list)


def format_string(s):
    """Strip the string, remove carriage returns, and capitalize the first character."""
    s = (
        (s or "").replace("\r", "").strip().strip('"')
    )
    if s:  # If the string is not empty
        s = s[0].upper() + s[1:]  # Capitalize the first character
        s = (
            s + "." if s[-1] not in [".", "?", "!"] else s
        )  # Add a period at the end of the string
    return s


def _get_random_query_prompt(
    dataset_id, query_modality, cand_modality, query_instructions
):
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    prompts = query_instructions.get(key, [])
    assert prompts, f"Cannot find prompts for {key}"
    prompt = format_string(random.choice(prompts))  # randomly select a prompt
    assert prompt, f"Prompt is empty for {key}"
    return prompt


# MOVED from mm_collator logic
# usage: insert task-specific prompt to the query (with randomness)
def process_mbeir_example(
    example, query_instructions, selected_pos_doc=None, selected_pos_doc_modality=None
):
    query_modality = example["query_modality"]
    qid = example["qid"]
    query_dataset_id = qid.split(":")[0]
    if selected_pos_doc_modality is None:
        doc_modality = selected_pos_doc["modality"]
    else:
        doc_modality = selected_pos_doc_modality

    query_instruction = _get_random_query_prompt(
        query_dataset_id,
        query_modality,
        doc_modality,
        query_instructions,
    )
    query_text = format_string(f"{query_instruction} {example['query_txt']}")
    if selected_pos_doc is None:  # at eval time, no documents were selected
        return {
            "query_text": query_text,
            "query_img": example["query_img"],
        }

    return {
        "query_text": query_text,
        "query_img": example["query_img"],
        "doc_text": selected_pos_doc["txt"],
        "doc_img": selected_pos_doc["img"],
    }
