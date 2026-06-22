from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_REPO_ROOT = _PROJECT_DIR.parent
_VENDOR_DIR = _PROJECT_DIR / "vendor"
if _VENDOR_DIR.exists():
    _VENDOR_PATH = str(_VENDOR_DIR)
    if _VENDOR_PATH in sys.path:
        sys.path.remove(_VENDOR_PATH)
    sys.path.insert(0, _VENDOR_PATH)
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

os.environ.setdefault("MURE_CACHE_ROOT", str(_PROJECT_DIR / ".cache"))
os.environ.setdefault("HF_HOME", str(Path(os.environ["MURE_CACHE_ROOT"]) / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("TMPDIR", str(Path(os.environ["MURE_CACHE_ROOT"]) / "tmp"))
os.environ.setdefault("DATA_DIR", str(_PROJECT_DIR / "data_dir") + "/")
os.environ.setdefault("CACHED_DATA_DIR", str(_PROJECT_DIR / "cached_data_dir"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import configue
import torch
import torch.nn.functional as F

from colqwen_multigranularity import eval as base_eval
from colqwen_multigranularity.core import normalize_granularities
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.config import FolderHomoConfig
from colqwen_multigranularity.experiments.exp_stagecompress.folder_homo.modeling_folder_homo import build_folder_homo_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe whether near-duplicate document tokens carry useful MaxSim evidence.")
    p.add_argument("--model-name-or-path", default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    p.add_argument("--processor-name-or-path", default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-config", default=str(_PROJECT_DIR / "configs/eval/test_data_vidore_v2_partial_3sets.yaml"))
    p.add_argument("--datasets", nargs="*", default=["esg_reports_human_labeled_v2", "economics_reports_v2"])
    p.add_argument("--max-queries", type=int, default=16)
    p.add_argument("--candidate-negatives", type=int, default=96)
    p.add_argument("--hard-negatives", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--similarity-threshold", type=float, default=0.88)
    p.add_argument("--top-hit-fraction", type=float, default=0.30)
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument("--folder-homo-budgets", type=int, nargs=3, default=[160, 160, 160])
    p.add_argument("--folder-homo-compress-stages", default="all")
    p.add_argument("--folder-homo-novelty-weight", type=float, default=1.0)
    p.add_argument("--folder-homo-gate-strength", type=float, default=0.25)
    p.add_argument("--folder-homo-folder-alpha", type=float, default=1.0)
    p.add_argument("--folder-homo-eval-prefix-level", type=int, default=3)
    p.add_argument("--granularities", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--query-augmentation-repeats", type=int, default=10)
    p.add_argument("--document-augmentation-repeats", type=int, default=0)
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include-text-tokens", action="store_true", help="Include leading document text/special tokens in duplicate clustering. Default probes visual tokens only when possible.")
    p.add_argument("--dry-dataset", action="store_true", help="Only print dataset schema and exit.")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> FolderHomoConfig:
    return FolderHomoConfig(
        enabled=True,
        budgets=tuple(args.folder_homo_budgets),
        compress_stages=args.folder_homo_compress_stages,
        novelty_weight=float(args.folder_homo_novelty_weight),
        gate_strength=float(args.folder_homo_gate_strength),
        folder_alpha=float(args.folder_homo_folder_alpha),
        eval_prefix_level=int(args.folder_homo_eval_prefix_level),
    )


def build_model(args: argparse.Namespace):
    checkpoint = Path(args.checkpoint)
    model = build_folder_homo_model(
        args.model_name_or_path,
        granularities=normalize_granularities(args.granularities),
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        adapter_path=str(checkpoint),
        eval_mode=True,
        folder_homo_config=build_config(args),
    )
    extra_state = checkpoint / "folder_homo.pt"
    if extra_state.exists():
        for name, submodule in model.named_modules():
            if name.endswith("folder_homo"):
                submodule.load_state_dict(torch.load(extra_state, map_location="cpu"), strict=False)
                break
    elif (checkpoint / "pytorch_model.bin").exists():
        state_dict = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
    return model


def build_processor(args: argparse.Namespace):
    class Obj:
        pass
    o = Obj()
    o.processor_name_or_path = args.processor_name_or_path
    o.granularities = args.granularities
    o.truncation_len = 16384
    o.processor_max_length = None
    o.use_simple_prompt = True
    o.resize_crops_to_page = True
    o.crop_resize_mode = None
    o.query_augmentation_repeats = args.query_augmentation_repeats
    o.document_augmentation_repeats = args.document_augmentation_repeats
    o.drop_query_text_if_image = False
    o.drop_doc_text_if_image = False
    return base_eval.build_processor(o)


def dataset_id(row: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        if key in row and row[key] is not None:
            return str(row[key])
    raise KeyError(f"Cannot find id in row keys: {list(row.keys())}")


def query_text(row: dict[str, Any]) -> str:
    for key in ("query", "text", "question"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(row)


def corpus_image(row: dict[str, Any]):
    if "image" in row:
        return row["image"]
    for key in ("img", "image_pil"):
        if key in row:
            return row[key]
    raise KeyError(f"Cannot find image in corpus row keys: {list(row.keys())}")


def qrels_for_query(qrels: Any, qid: str) -> dict[str, float]:
    if isinstance(qrels, dict):
        rels = qrels.get(qid) or qrels.get(str(qid)) or {}
        if isinstance(rels, dict):
            return {str(k): float(v) for k, v in rels.items() if float(v) > 0}
        return {str(k): 1.0 for k in rels}
    out = {}
    names = set(getattr(qrels, "column_names", []))
    qkey = "query_id" if "query_id" in names else ("query-id" if "query-id" in names else ("qid" if "qid" in names else None))
    dkey = "corpus_id" if "corpus_id" in names else ("corpus-id" if "corpus-id" in names else ("doc_id" if "doc_id" in names else ("did" if "did" in names else None)))
    skey = "score" if "score" in names else None
    if qkey is None or dkey is None:
        return out
    for row in qrels:
        if str(row[qkey]) != qid:
            continue
        score = float(row[skey]) if skey is not None else 1.0
        if score > 0:
            out[str(row[dkey])] = score
    return out


def batch_encode_queries(model, processor, texts: list[str], device: torch.device) -> list[torch.Tensor]:
    batches = processor.process_queries(texts).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb = model(**batches, is_query=True)
        else:
            emb = model(**batches, is_query=True)
    return [x.detach().float().cpu() for x in emb]


def batch_encode_docs(model, processor, images: list[Any], device: torch.device) -> list[torch.Tensor]:
    batches = processor.process_images(images).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                emb = model(**batches, is_query=False)
        else:
            emb = model(**batches, is_query=False)
    return [x.detach().float().cpu() for x in emb]


def maxsim_details(q: torch.Tensor, d: torch.Tensor) -> dict[str, torch.Tensor | float]:
    qn = F.normalize(q.float(), dim=-1, eps=1e-12)
    dn = F.normalize(d.float(), dim=-1, eps=1e-12)
    sim = qn @ dn.t()
    vals, idx = sim.max(dim=-1)
    return {
        "score": float(vals.sum().item()),
        "vals": vals,
        "idx": idx,
        "sim": sim,
    }


def duplicate_mask(d: torch.Tensor, *, threshold: float, min_cluster_size: int, start_index: int = 0) -> tuple[torch.Tensor, dict[str, float]]:
    n = int(d.shape[0])
    mask = torch.zeros(n, dtype=torch.bool)
    if n <= 1 or start_index >= n:
        return mask, {"duplicate_token_fraction": 0.0, "mean_max_neighbor": 0.0, "candidate_tokens": max(n - start_index, 0)}
    sub = F.normalize(d[start_index:].float(), dim=-1, eps=1e-12)
    sim = sub @ sub.t()
    sim.fill_diagonal_(-1.0)
    max_neighbor = sim.max(dim=-1).values
    local_dup = max_neighbor >= float(threshold)
    if min_cluster_size > 2:
        counts = (sim >= float(threshold)).sum(dim=-1) + 1
        local_dup = local_dup & (counts >= int(min_cluster_size))
    mask[start_index:] = local_dup
    return mask, {
        "duplicate_token_fraction": float(local_dup.float().mean().item()) if local_dup.numel() else 0.0,
        "mean_max_neighbor": float(max_neighbor.mean().item()) if max_neighbor.numel() else 0.0,
        "candidate_tokens": int(sub.shape[0]),
    }


def contribution_stats(details: dict[str, Any], dup: torch.Tensor, top_hit_fraction: float) -> dict[str, float]:
    vals: torch.Tensor = details["vals"]
    idx: torch.Tensor = details["idx"]
    hit_dup = dup[idx]
    score = vals.sum().clamp_min(1e-12)
    order = torch.argsort(vals, descending=True)
    k = max(1, int(math.ceil(float(top_hit_fraction) * vals.numel())))
    top = order[:k]
    non_dup_vals = vals[~hit_dup]
    dup_vals = vals[hit_dup]
    return {
        "score": float(vals.sum().item()),
        "hit_duplicate_fraction": float(hit_dup.float().mean().item()) if hit_dup.numel() else 0.0,
        "score_duplicate_fraction": float(vals[hit_dup].sum().item() / score.item()) if hit_dup.any() else 0.0,
        "top_hit_duplicate_fraction": float(hit_dup[top].float().mean().item()) if top.numel() else 0.0,
        "mean_duplicate_hit_score": float(dup_vals.mean().item()) if dup_vals.numel() else 0.0,
        "mean_nonduplicate_hit_score": float(non_dup_vals.mean().item()) if non_dup_vals.numel() else 0.0,
        "unique_doc_hits": float(idx.unique().numel()),
        "query_tokens": float(vals.numel()),
    }


def doc_start_index(d: torch.Tensor, args: argparse.Namespace) -> int:
    if args.include_text_tokens:
        return 0
    # FolderHomo output is text tokens followed by compressed visual tokens.
    # Query-independent document text/special tokens are usually short; the exact
    # boundary is not exposed by the eval model, so we conservatively skip the
    # first few tokens to focus the duplicate probe on compressed visual evidence.
    return min(8, max(0, int(d.shape[0]) - sum(args.folder_homo_budgets)))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "n": 0.0}
    xs = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(xs.mean().item()),
        "median": float(xs.median().item()),
        "n": float(xs.numel()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_loader = configue.load(Path(args.eval_config))
    selected = {name: fac for name, fac in eval_loader.items() if name in set(args.datasets)}
    if not selected:
        raise ValueError(f"No requested datasets found. requested={args.datasets}, available={list(eval_loader.keys())}")

    if args.dry_dataset:
        schemas = {}
        for name, fac in selected.items():
            ds = fac()
            schemas[name] = {
                "queries_columns": list(getattr(ds["queries"], "column_names", [])),
                "corpus_columns": list(getattr(ds["corpus"], "column_names", [])),
                "qrels_type": type(ds.get("qrels")).__name__,
                "num_queries": len(ds["queries"]),
                "num_corpus": len(ds["corpus"]),
            }
        print(json.dumps(schemas, indent=2, ensure_ascii=False))
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device).eval()
    processor = build_processor(args)

    per_example = []
    aggregate = defaultdict(list)

    for ds_name, fac in selected.items():
        dataset = fac()
        queries = dataset["queries"]
        corpus = dataset["corpus"]
        qrels = dataset.get("qrels")
        corpus_ids = [dataset_id(corpus[i], ("corpus_id", "doc_id", "corpus-id", "did", "id")) for i in range(len(corpus))]
        corpus_index = {did: i for i, did in enumerate(corpus_ids)}

        doc_cache: dict[int, torch.Tensor] = {}

        def get_doc_emb(doc_idx: int) -> torch.Tensor:
            cached = doc_cache.get(int(doc_idx))
            if cached is not None:
                return cached
            emb = batch_encode_docs(model, processor, [corpus_image(corpus[int(doc_idx)])], device)[0]
            doc_cache[int(doc_idx)] = emb
            return emb

        def get_doc_embs(doc_indices: list[int]) -> list[torch.Tensor]:
            missing = [int(i) for i in doc_indices if int(i) not in doc_cache]
            for start in range(0, len(missing), args.batch_size):
                batch_indices = missing[start:start + args.batch_size]
                if not batch_indices:
                    continue
                batch_embs = batch_encode_docs(model, processor, [corpus_image(corpus[i]) for i in batch_indices], device)
                for idx, emb in zip(batch_indices, batch_embs):
                    doc_cache[int(idx)] = emb
            return [doc_cache[int(i)] for i in doc_indices]

        query_limit = min(args.max_queries, len(queries)) if args.max_queries > 0 else len(queries)
        for qi in range(query_limit):
            qrow = queries[qi]
            qid = dataset_id(qrow, ("query_id", "query-id", "qid", "id"))
            rels = qrels_for_query(qrels, qid)
            pos_ids = [did for did in rels if did in corpus_index]
            if not pos_ids:
                continue
            pos_id = pos_ids[0]
            pos_idx = corpus_index[pos_id]

            candidate_indices = []
            for ci in range(len(corpus)):
                if ci == pos_idx:
                    continue
                if corpus_ids[ci] in rels:
                    continue
                candidate_indices.append(ci)
                if len(candidate_indices) >= args.candidate_negatives:
                    break
            if not candidate_indices:
                continue

            q_emb = batch_encode_queries(model, processor, [query_text(qrow)], device)[0]
            pos_emb = get_doc_emb(pos_idx)
            cand_embs = get_doc_embs(candidate_indices)

            neg_scores = []
            neg_details = []
            for local_i, emb in enumerate(cand_embs):
                det = maxsim_details(q_emb, emb)
                neg_scores.append(float(det["score"]))
                neg_details.append((candidate_indices[local_i], emb, det))
            hard = sorted(zip(neg_scores, neg_details), key=lambda x: x[0], reverse=True)[: max(1, args.hard_negatives)]

            pos_det = maxsim_details(q_emb, pos_emb)
            pos_start = doc_start_index(pos_emb, args)
            pos_dup, pos_dup_meta = duplicate_mask(pos_emb, threshold=args.similarity_threshold, min_cluster_size=args.min_cluster_size, start_index=pos_start)
            pos_stats = contribution_stats(pos_det, pos_dup, args.top_hit_fraction)

            hard_rows = []
            for neg_score, (neg_idx, neg_emb, neg_det) in hard:
                neg_start = doc_start_index(neg_emb, args)
                neg_dup, neg_dup_meta = duplicate_mask(neg_emb, threshold=args.similarity_threshold, min_cluster_size=args.min_cluster_size, start_index=neg_start)
                neg_stats = contribution_stats(neg_det, neg_dup, args.top_hit_fraction)
                margin = float(pos_stats["score"] - neg_stats["score"])
                dup_margin = float(pos_stats["score"] * pos_stats["score_duplicate_fraction"] - neg_stats["score"] * neg_stats["score_duplicate_fraction"])
                nondup_margin = float(pos_stats["score"] * (1.0 - pos_stats["score_duplicate_fraction"]) - neg_stats["score"] * (1.0 - neg_stats["score_duplicate_fraction"]))
                hard_rows.append({
                    "neg_doc_id": corpus_ids[neg_idx],
                    "neg_score": neg_stats["score"],
                    "margin": margin,
                    "duplicate_margin_component": dup_margin,
                    "nonduplicate_margin_component": nondup_margin,
                    "neg_duplicate_token_fraction": neg_dup_meta["duplicate_token_fraction"],
                    "neg_hit_duplicate_fraction": neg_stats["hit_duplicate_fraction"],
                    "neg_score_duplicate_fraction": neg_stats["score_duplicate_fraction"],
                    "neg_top_hit_duplicate_fraction": neg_stats["top_hit_duplicate_fraction"],
                })
                aggregate[f"{ds_name}/margin"].append(margin)
                aggregate[f"{ds_name}/duplicate_margin_component"].append(dup_margin)
                aggregate[f"{ds_name}/nonduplicate_margin_component"].append(nondup_margin)
                aggregate[f"{ds_name}/neg_hit_duplicate_fraction"].append(neg_stats["hit_duplicate_fraction"])
                aggregate[f"{ds_name}/neg_score_duplicate_fraction"].append(neg_stats["score_duplicate_fraction"])

            row = {
                "dataset": ds_name,
                "query_id": qid,
                "query": query_text(qrow),
                "positive_doc_id": pos_id,
                "positive_score": pos_stats["score"],
                "positive_duplicate_token_fraction": pos_dup_meta["duplicate_token_fraction"],
                "positive_mean_max_neighbor": pos_dup_meta["mean_max_neighbor"],
                "positive_hit_duplicate_fraction": pos_stats["hit_duplicate_fraction"],
                "positive_score_duplicate_fraction": pos_stats["score_duplicate_fraction"],
                "positive_top_hit_duplicate_fraction": pos_stats["top_hit_duplicate_fraction"],
                "positive_mean_duplicate_hit_score": pos_stats["mean_duplicate_hit_score"],
                "positive_mean_nonduplicate_hit_score": pos_stats["mean_nonduplicate_hit_score"],
                "hard_negatives": hard_rows,
            }
            per_example.append(row)
            aggregate[f"{ds_name}/positive_duplicate_token_fraction"].append(pos_dup_meta["duplicate_token_fraction"])
            aggregate[f"{ds_name}/positive_hit_duplicate_fraction"].append(pos_stats["hit_duplicate_fraction"])
            aggregate[f"{ds_name}/positive_score_duplicate_fraction"].append(pos_stats["score_duplicate_fraction"])
            aggregate[f"{ds_name}/positive_top_hit_duplicate_fraction"].append(pos_stats["top_hit_duplicate_fraction"])

            print(json.dumps({
                "event": "probe_example",
                "dataset": ds_name,
                "query_id": qid,
                "pos_score_dup_frac": round(pos_stats["score_duplicate_fraction"], 4),
                "pos_hit_dup_frac": round(pos_stats["hit_duplicate_fraction"], 4),
                "best_margin": round(hard_rows[0]["margin"], 4) if hard_rows else None,
                "best_dup_margin_component": round(hard_rows[0]["duplicate_margin_component"], 4) if hard_rows else None,
            }, ensure_ascii=False), flush=True)

    summary = {key: summarize(vals) for key, vals in aggregate.items()}
    all_pos_score_dup = [row["positive_score_duplicate_fraction"] for row in per_example]
    all_pos_hit_dup = [row["positive_hit_duplicate_fraction"] for row in per_example]
    all_dup_margin = [hn["duplicate_margin_component"] for row in per_example for hn in row["hard_negatives"]]
    all_nondup_margin = [hn["nonduplicate_margin_component"] for row in per_example for hn in row["hard_negatives"]]
    summary["overall/positive_score_duplicate_fraction"] = summarize(all_pos_score_dup)
    summary["overall/positive_hit_duplicate_fraction"] = summarize(all_pos_hit_dup)
    summary["overall/duplicate_margin_component"] = summarize(all_dup_margin)
    summary["overall/nonduplicate_margin_component"] = summarize(all_nondup_margin)
    summary["config"] = {
        "checkpoint": args.checkpoint,
        "datasets": args.datasets,
        "max_queries": args.max_queries,
        "candidate_negatives": args.candidate_negatives,
        "hard_negatives": args.hard_negatives,
        "similarity_threshold": args.similarity_threshold,
        "min_cluster_size": args.min_cluster_size,
        "include_text_tokens": args.include_text_tokens,
    }

    (out_dir / "per_example.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in per_example) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# MaxSim Duplicate Token Probe",
        "",
        "This diagnostic tests whether near-duplicate compressed document tokens are actually used by MaxSim and whether they contribute to positive-vs-negative margin.",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Datasets: `{', '.join(args.datasets)}`",
        f"Duplicate threshold: `{args.similarity_threshold}`",
        "",
        "## Overall",
        "",
        "| Signal | Mean | Median | N |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "overall/positive_hit_duplicate_fraction",
        "overall/positive_score_duplicate_fraction",
        "overall/duplicate_margin_component",
        "overall/nonduplicate_margin_component",
    ]:
        val = summary.get(key, {})
        lines.append(f"| {key} | {val.get("mean", 0.0):.4f} | {val.get("median", 0.0):.4f} | {val.get("n", 0.0):.0f} |")
    lines.extend(["", "## Dataset Breakdown", "", "| Signal | Mean | Median | N |", "|---|---:|---:|---:|"])
    for key in sorted(k for k in summary if "/" in k and not k.startswith("overall/")):
        val = summary[key]
        if not isinstance(val, dict) or "mean" not in val:
            continue
        lines.append(f"| {key} | {val.get("mean", 0.0):.4f} | {val.get("median", 0.0):.4f} | {val.get("n", 0.0):.0f} |")
    lines.extend([
        "",
        "## Reading Guide",
        "",
        "- `positive_hit_duplicate_fraction`: fraction of query tokens whose MaxSim winner is a near-duplicate document token.",
        "- `positive_score_duplicate_fraction`: fraction of the positive MaxSim score contributed by near-duplicate document tokens.",
        "- `duplicate_margin_component`: positive duplicate-token score minus hard-negative duplicate-token score. Positive values support the hypothesis that repeated-looking tokens help ranking.",
        "- `nonduplicate_margin_component`: the corresponding margin contribution from non-duplicate token hits.",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "probe_done", "output_dir": str(out_dir), "examples": len(per_example)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
