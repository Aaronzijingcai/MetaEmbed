
from pathlib import Path
import json, re
import pandas as pd

BASE = Path('/MURE-V2/code/MetaEmbed/data_dir/MMEB-eval-beir-v3')
NAMES = [
 'MMEB-eval-FashionIQ-beir-v3','MMEB-eval-Country211-beir-v3','MMEB-eval-CIRR-beir-v3',
 'MMEB-eval-InfographicsVQA-beir-v3','MMEB-eval-Visual7W-beir-v3','MMEB-eval-GQA-beir-v3',
 'MMEB-eval-ChartQA-beir-v3','MMEB-eval-A-OKVQA-beir-v3','MMEB-eval-ScienceQA-beir-v3','MMEB-eval-OK-VQA-beir-v3'
]
def read_one(d, part):
    files = sorted((d/part).glob('*.parquet'))
    if not files:
        return pd.DataFrame()
    cols = {'queries': ['query-id', 'query_txt', 'local-did'], 'corpus': ['corpus-id', 'txt'], 'qrels': ['query-id', 'corpus-id', 'score']}[part]
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f, columns=cols))
        except Exception:
            dfs.append(pd.read_parquet(f))
    return pd.concat(dfs, ignore_index=True)

def norm_text(s):
    return re.sub(r'\s+', ' ', str(s or '')).strip().lower()

def short(s, n=80):
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    return s[:n] + ('...' if len(s) > n else '')

rows=[]
for name in NAMES:
    d=BASE/name
    if not d.exists():
        continue
    q=read_one(d,'queries')
    c=read_one(d,'corpus')
    c_by_id=dict(zip(c['corpus-id'].astype(str).tolist(), c['txt'].astype(str).tolist()))
    cand_counts=[]; pos_lens=[]; neg_lens=[]; local_unique_ratios=[]; local_same_text=[]; local_empty=[]
    examples=[]
    for qi, qr in q.iterrows():
        local=[str(x) for x in list(qr['local-did'])]
        if not local: continue
        gt=local[0]
        texts=[]
        for did in local:
            texts.append(c_by_id.get(did, ''))
        nt=[norm_text(t) for t in texts]
        cand_counts.append(len(local))
        pos_lens.append(len(norm_text(texts[0])))
        if len(texts)>1:
            neg_lens.extend([len(norm_text(x)) for x in texts[1:]])
        local_unique_ratios.append(len(set(nt))/len(nt) if nt else 0)
        local_same_text.append(1.0 if len(set(nt)) <= 1 else 0.0)
        local_empty.append(sum(1 for x in nt if not x)/len(nt) if nt else 0)
        if len(examples)<3:
            examples.append({
                'query': short(qr.get('query_txt','')),
                'candidate_count': len(local),
                'gt_text': short(texts[0] if texts else ''),
                'neg_texts': [short(x) for x in texts[1:6]],
                'unique_texts_in_local': len(set(nt)),
            })
    def mean(xs): return sum(xs)/len(xs) if xs else None
    def med(xs):
        if not xs: return None
        xs=sorted(xs); return xs[len(xs)//2]
    item={
        'dataset': name.replace('MMEB-eval-','').replace('-beir-v3',''),
        'queries': int(len(q)), 'corpus': int(len(c)),
        'local_candidates_mean': mean(cand_counts),
        'local_candidates_median': med(cand_counts),
        'local_candidates_min': min(cand_counts) if cand_counts else None,
        'local_candidates_max': max(cand_counts) if cand_counts else None,
        'pos_text_len_median': med(pos_lens),
        'neg_text_len_median': med(neg_lens),
        'local_unique_text_ratio_mean': mean(local_unique_ratios),
        'local_all_same_text_ratio': mean(local_same_text),
        'local_empty_text_frac_mean': mean(local_empty),
        'examples': examples,
    }
    rows.append(item)

outdir=Path('MMEB任务课程学习/data_audit')
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'local_candidate_audit.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
lines=['# MMEB Local Candidate Audit','']
lines.append('| Dataset | Q | Corpus | Cand med | Cand max | Pos len med | Neg len med | Local uniq text ratio | All same text | Empty text frac |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
for r in rows:
    lines.append(f"| {r['dataset']} | {r['queries']} | {r['corpus']} | {r['local_candidates_median']} | {r['local_candidates_max']} | {r['pos_text_len_median']} | {r['neg_text_len_median']} | {r['local_unique_text_ratio_mean']:.3f} | {r['local_all_same_text_ratio']:.3f} | {r['local_empty_text_frac_mean']:.3f} |")
lines.append('\n## Examples\n')
for r in rows:
    lines.append(f"### {r['dataset']}")
    for ex in r['examples']:
        lines.append(f"- cand={ex['candidate_count']} uniq={ex['unique_texts_in_local']} q=`{ex['query']}` gt=`{ex['gt_text']}` neg={ex['neg_texts']}")
    lines.append('')
(outdir/'local_candidate_audit.md').write_text('\n'.join(lines), encoding='utf-8')
print(outdir/'local_candidate_audit.md')
