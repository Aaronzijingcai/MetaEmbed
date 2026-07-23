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
    p = argparse.ArgumentParser(description="Eval-only sweep over duplicate-token cluster quotas.")
    p.add_argument("--model-name-or-path", default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    p.add_argument("--processor-name-or-path", default=str(_PROJECT_DIR / "models/colqwen2.5-base"))
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval-config", default=str(_PROJECT_DIR / "configs/eval/test_data_vidore_v2_partial_3sets.yaml"))
    p.add_argument("--datasets", nargs="*", default=["esg_reports_human_labeled_v2", "economics_reports_v2"])
    p.add_argument("--max-queries", type=int, default=16)
    p.add_argument("--max-corpus", type=int, default=256, help="0 means all corpus pages.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--similarity-threshold", type=float, default=0.88)
    p.add_argument("--quotas", nargs="*", default=["1", "2", "4", "all"])
    p.add_argument("--anchor-quotas", nargs="*", default=[], help="Oracle-style query-anchor quota labels, e.g. anchor2 anchor4.")
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
    p.add_argument("--include-text-tokens", action="store_true")
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


def doc_start_index(d: torch.Tensor, args: argparse.Namespace) -> int:
    if args.include_text_tokens:
        return 0
    return min(8, max(0, int(d.shape[0]) - sum(args.folder_homo_budgets)))


def duplicate_components(d: torch.Tensor, *, threshold: float, start_index: int) -> tuple[list[torch.Tensor], int]:
    n = int(d.shape[0])
    if n <= 1 or start_index >= n:
        return [], max(n - start_index, 0)
    sub = F.normalize(d[start_index:].float(), dim=-1, eps=1e-12)
    sim = sub @ sub.t()
    sim.fill_diagonal_(-1.0)
    m = int(sub.shape[0])
    visited = torch.zeros(m, dtype=torch.bool)
    components = []
    for i in range(m):
        if visited[i]:
            continue
        stack = [i]
        members = []
        visited[i] = True
        while stack:
            u = stack.pop()
            members.append(u)
            neigh = torch.nonzero(sim[u] >= threshold, as_tuple=False).flatten().tolist()
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
        components.append(torch.tensor(members, dtype=torch.long) + int(start_index))
    return components, m


def quota_keep_mask(d: torch.Tensor, *, threshold: float, quota: int | None, start_index: int, token_strength: torch.Tensor | None = None) -> torch.Tensor:
    n = int(d.shape[0])
    keep = torch.ones(n, dtype=torch.bool)
    if quota is None or quota <= 0 or n <= 1 or start_index >= n:
        return keep
    components, _ = duplicate_components(d, threshold=threshold, start_index=start_index)
    local_keep = torch.zeros(n, dtype=torch.bool)
    if start_index > 0:
        local_keep[:start_index] = True
    if token_strength is None:
        token_strength = d.float().norm(dim=-1)
    for members in components:
        if members.numel() <= quota:
            local_keep[members] = True
        else:
            order = torch.argsort(token_strength[members], descending=True)[:quota]
            local_keep[members[order]] = True
    keep = local_keep
    return keep


def anchor_token_strength(queries: list[torch.Tensor], d: torch.Tensor) -> torch.Tensor:
    dn = F.normalize(d.float(), dim=-1, eps=1e-12)
    strength = torch.zeros(d.shape[0], dtype=torch.float32)
    for q in queries:
        qn = F.normalize(q.float(), dim=-1, eps=1e-12)
        vals, idx = (qn @ dn.t()).max(dim=-1)
        strength.scatter_add_(0, idx, vals.float().clamp_min(0.0))
    return strength


def score_matrix(queries: list[torch.Tensor], docs: list[torch.Tensor]) -> torch.Tensor:
    scores = torch.empty(len(queries), len(docs), dtype=torch.float32)
    doc_norm = [F.normalize(d.float(), dim=-1, eps=1e-12) for d in docs]
    for i, q in enumerate(queries):
        qn = F.normalize(q.float(), dim=-1, eps=1e-12)
        for j, dn in enumerate(doc_norm):
            scores[i, j] = (qn @ dn.t()).max(dim=-1).values.sum()
    return scores


def ndcg_at_k(ranked: list[str], rels: dict[str, float], k: int = 5) -> float:
    dcg = 0.0
    for rank, did in enumerate(ranked[:k], start=1):
        rel = float(rels.get(str(did), 0.0))
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1.0)
    ideal = sorted([float(v) for v in rels.values() if float(v) > 0], reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1.0) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, float], k: int = 5) -> float:
    positives = {str(k) for k, v in rels.items() if float(v) > 0}
    if not positives:
        return 0.0
    return len(set(str(x) for x in ranked[:k]) & positives) / len(positives)


def mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_specs: list[tuple[str, int | None, str]] = []
    for q in args.quotas:
        label = str(q).lower()
        sweep_specs.append((label, None if label in {"all", "none", "inf"} else int(q), "norm"))
    for q in args.anchor_quotas:
        label = str(q).lower()
        if not label.startswith("anchor"):
            raise ValueError(f"anchor quota must look like anchor2/anchor4, got {q}")
        sweep_specs.append((label, int(label.replace("anchor", "")), "anchor"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device).eval()
    processor = build_processor(args)
    eval_loader = configue.load(Path(args.eval_config))
    selected = {name: fac for name, fac in eval_loader.items() if name in set(args.datasets)}
    if not selected:
        raise ValueError(f"No requested datasets found. requested={args.datasets}, available={list(eval_loader.keys())}")

    summary: dict[str, Any] = {"config": vars(args), "datasets": {}}
    all_rows = []
    for ds_name, fac in selected.items():
        dataset = fac()
        queries_ds = dataset["queries"]
        corpus_ds = dataset["corpus"]
        qrels = dataset.get("qrels")
        corpus_ids_all = [dataset_id(corpus_ds[i], ("corpus_id", "doc_id", "corpus-id", "did", "id")) for i in range(len(corpus_ds))]
        corpus_index = {did: i for i, did in enumerate(corpus_ids_all)}
        query_limit = min(args.max_queries, len(queries_ds)) if args.max_queries > 0 else len(queries_ds)

        query_rows = []
        required_positive_indices = set()
        for qi in range(query_limit):
            qrow = queries_ds[qi]
            qid = dataset_id(qrow, ("query_id", "query-id", "qid", "id"))
            rels = qrels_for_query(qrels, qid)
            pos_ids = [did for did in rels if did in corpus_index]
            if not pos_ids:
                continue
            for did in pos_ids:
                required_positive_indices.add(corpus_index[did])
            query_rows.append((qid, query_text(qrow), rels))

        max_corpus = len(corpus_ds) if args.max_corpus <= 0 else min(args.max_corpus, len(corpus_ds))
        selected_indices = set(range(max_corpus)) | required_positive_indices
        selected_indices = sorted(selected_indices)
        selected_ids = [corpus_ids_all[i] for i in selected_indices]

        print(json.dumps({"event": "encode_start", "dataset": ds_name, "queries": len(query_rows), "docs": len(selected_indices)}, ensure_ascii=False), flush=True)
        q_embs = []
        for start in range(0, len(query_rows), args.batch_size):
            q_embs.extend(batch_encode_queries(model, processor, [row[1] for row in query_rows[start:start + args.batch_size]], device))
        d_embs = []
        for start in range(0, len(selected_indices), args.batch_size):
            batch_indices = selected_indices[start:start + args.batch_size]
            d_embs.extend(batch_encode_docs(model, processor, [corpus_image(corpus_ds[i]) for i in batch_indices], device))
            print(json.dumps({"event": "encoded_docs", "dataset": ds_name, "done": min(start + args.batch_size, len(selected_indices)), "total": len(selected_indices)}, ensure_ascii=False), flush=True)

        ds_result = {}
        doc_anchor_strengths: list[torch.Tensor] | None = None
        for label, quota, mode in sweep_specs:
            pruned_docs = []
            token_counts = []
            if mode == "anchor" and doc_anchor_strengths is None:
                doc_anchor_strengths = [anchor_token_strength(q_embs, d) for d in d_embs]
            for di, d in enumerate(d_embs):
                strength = doc_anchor_strengths[di] if mode == "anchor" and doc_anchor_strengths is not None else None
                mask = quota_keep_mask(d, threshold=args.similarity_threshold, quota=quota, start_index=doc_start_index(d, args), token_strength=strength)
                pruned_docs.append(d[mask])
                token_counts.append(float(mask.sum().item()))
            scores = score_matrix(q_embs, pruned_docs)
            ndcgs, r1s, r5s = [], [], []
            rows = []
            for qi, (qid, _qtext, rels) in enumerate(query_rows):
                order = torch.argsort(scores[qi], descending=True).tolist()
                ranked_ids = [selected_ids[j] for j in order]
                ndcg = ndcg_at_k(ranked_ids, rels, 5)
                r1 = recall_at_k(ranked_ids, rels, 1)
                r5 = recall_at_k(ranked_ids, rels, 5)
                ndcgs.append(ndcg)
                r1s.append(r1)
                r5s.append(r5)
                rows.append({"dataset": ds_name, "quota": label, "query_id": qid, "ndcg_at_5": ndcg, "recall_at_1": r1, "recall_at_5": r5, "top5": ranked_ids[:5]})
            result = {
                "ndcg_at_5": mean(ndcgs),
                "recall_at_1": mean(r1s),
                "recall_at_5": mean(r5s),
                "mean_doc_tokens": mean(token_counts),
                "docs": len(pruned_docs),
                "queries": len(query_rows),
            }
            ds_result[label] = result
            all_rows.extend(rows)
            print(json.dumps({"event": "quota_result", "dataset": ds_name, "quota": label, **result}, ensure_ascii=False), flush=True)
        summary["datasets"][ds_name] = ds_result

    # Macro average across datasets for each quota.
    macro = {}
    labels = [label for label, _quota, _mode in sweep_specs]
    for label in labels:
        vals = [summary["datasets"][ds][label] for ds in summary["datasets"] if label in summary["datasets"][ds]]
        macro[label] = {
            "ndcg_at_5": mean([v["ndcg_at_5"] for v in vals]),
            "recall_at_1": mean([v["recall_at_1"] for v in vals]),
            "recall_at_5": mean([v["recall_at_5"] for v in vals]),
            "mean_doc_tokens": mean([v["mean_doc_tokens"] for v in vals]),
        }
    summary["macro"] = macro

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "per_query.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in all_rows) + "\n", encoding="utf-8")

    lines = [
        "# Duplicate Quota Sweep",
        "",
        f"Checkpoint: `{args.checkpoint}`",
        f"Datasets: `{', '.join(args.datasets)}`",
        f"Duplicate threshold: `{args.similarity_threshold}`",
        "",
        "## Macro Average",
        "",
        "| Quota | nDCG@5 | Recall@1 | Recall@5 | Mean Doc Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, res in macro.items():
        lines.append(f"| {label} | {res['ndcg_at_5']:.4f} | {res['recall_at_1']:.4f} | {res['recall_at_5']:.4f} | {res['mean_doc_tokens']:.1f} |")
    lines.extend(["", "## Dataset Breakdown", "", "| Dataset | Quota | nDCG@5 | Recall@1 | Recall@5 | Mean Doc Tokens |", "|---|---:|---:|---:|---:|---:|"])
    for ds_name, ds_result in summary["datasets"].items():
        for label, res in ds_result.items():
            lines.append(f"| {ds_name} | {label} | {res['ndcg_at_5']:.4f} | {res['recall_at_1']:.4f} | {res['recall_at_5']:.4f} | {res['mean_doc_tokens']:.1f} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"event": "sweep_done", "output_dir": str(out_dir)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
