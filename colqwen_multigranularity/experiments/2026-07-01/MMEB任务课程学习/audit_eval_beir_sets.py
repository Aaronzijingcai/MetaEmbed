
from pathlib import Path
import json, re
import pandas as pd

BASE = Path('/MURE-V2/code/MetaEmbed/data_dir/MMEB-eval-beir-v3')
NAMES = [
 'MMEB-eval-FashionIQ-beir-v3','MMEB-eval-Country211-beir-v3','MMEB-eval-CIRR-beir-v3',
 'MMEB-eval-InfographicsVQA-beir-v3','MMEB-eval-Visual7W-beir-v3','MMEB-eval-GQA-beir-v3',
 'MMEB-eval-ChartQA-beir-v3','MMEB-eval-A-OKVQA-beir-v3','MMEB-eval-ScienceQA-beir-v3','MMEB-eval-OK-VQA-beir-v3',
 'MMEB-eval-DocVQA-beir-v3','MMEB-eval-TextVQA-beir-v3','MMEB-eval-VizWiz-beir-v3','MMEB-eval-ImageNet-1K-beir-v3',
 'MMEB-eval-SUN397-beir-v3','MMEB-eval-VOC2007-beir-v3','MMEB-eval-ImageNet-A-beir-v3','MMEB-eval-ImageNet-R-beir-v3','MMEB-eval-ObjectNet-beir-v3'
]

def read_one(d, part):
    files = sorted((d/part).glob('*.parquet'))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def text_col(df):
    for c in ['query_txt','txt','text','title','contents','query','qry','content']:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == 'object':
            return c
    return None

def image_cols(df):
    return [c for c in df.columns if 'image' in c.lower() or 'img' in c.lower() or c.lower() in ('pixel_values',)]

def has_nonempty(v):
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    s = str(v)
    return bool(s and s.lower() not in ('none','nan'))

def trunc(s, n=120):
    s = re.sub(r'\s+', ' ', str(s or '')).strip()
    return s[:n] + ('...' if len(s) > n else '')

out=[]
for name in NAMES:
    d=BASE/name
    if not d.exists():
        out.append({'dataset': name, 'exists': False})
        continue
    q=read_one(d,'queries')
    c=read_one(d,'corpus')
    r=read_one(d,'qrels')
    qt=text_col(q); ct=text_col(c)
    qi=image_cols(q); ci=image_cols(c)
    q_texts=q[qt].astype(str) if qt else pd.Series([], dtype=str)
    c_texts=c[ct].astype(str) if ct else pd.Series([], dtype=str)
    q_has_img=sum(any(has_nonempty(row.get(ic)) for ic in qi) for _, row in q.head(200).iterrows()) if qi else 0
    c_has_img=sum(any(has_nonempty(row.get(ic)) for ic in ci) for _, row in c.head(200).iterrows()) if ci else 0
    # qrels cols vary; count positive docs per query if possible
    qid_col = next((x for x in ['query-id','query_id','qid'] if x in r.columns), None)
    doc_col = next((x for x in ['corpus-id','doc_id','corpus_id','pid'] if x in r.columns), None)
    pos_per_q = None
    if qid_col:
        pos_per_q = r.groupby(qid_col).size().describe().to_dict()
    item={
      'dataset': name.replace('MMEB-eval-','').replace('-beir-v3',''),
      'queries': int(len(q)), 'corpus': int(len(c)), 'qrels': int(len(r)),
      'query_columns': list(q.columns), 'corpus_columns': list(c.columns), 'qrels_columns': list(r.columns),
      'query_text_col': qt, 'corpus_text_col': ct,
      'query_image_cols': qi, 'corpus_image_cols': ci,
      'query_has_image_first200': int(q_has_img), 'corpus_has_image_first200': int(c_has_img),
      'query_text_len_mean': float(q_texts.str.len().mean()) if len(q_texts) else None,
      'query_text_len_median': float(q_texts.str.len().median()) if len(q_texts) else None,
      'corpus_text_len_mean': float(c_texts.str.len().mean()) if len(c_texts) else None,
      'corpus_text_len_median': float(c_texts.str.len().median()) if len(c_texts) else None,
      'corpus_unique_text_ratio': float(c_texts.nunique()/len(c_texts)) if len(c_texts) else None,
      'corpus_top_texts': c_texts.value_counts().head(5).to_dict() if len(c_texts) else {},
      'sample_queries': [trunc(x) for x in q_texts.head(3).tolist()],
      'sample_corpus': [trunc(x) for x in c_texts.head(5).tolist()],
      'pos_per_query': pos_per_q,
    }
    out.append(item)

outdir=Path('MMEB任务课程学习/data_audit')
outdir.mkdir(parents=True, exist_ok=True)
(outdir/'eval_beir_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
lines=['# MMEB Eval BEIR Audit','']
lines.append('| Dataset | Q | Corpus | Q img/200 | C img/200 | Q len med | C len med | Corpus uniq | Notes |')
lines.append('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
for x in out:
    if not x.get('exists', True):
        lines.append(f"| {x['dataset']} | missing | | | | | | | |")
        continue
    notes=[]
    if x['corpus_text_len_median'] is not None and x['corpus_text_len_median'] <= 20: notes.append('short corpus text')
    if x['corpus_unique_text_ratio'] is not None and x['corpus_unique_text_ratio'] < .5: notes.append('many duplicate labels')
    if x['query_has_image_first200']>0 and x['corpus_has_image_first200']==0: notes.append('image/query -> text corpus')
    if x['query_has_image_first200']>0 and x['corpus_has_image_first200']>0: notes.append('image-query -> image corpus')
    if x['query_has_image_first200']==0 and x['corpus_has_image_first200']>0: notes.append('text-query -> image corpus')
    lines.append(f"| {x['dataset']} | {x['queries']} | {x['corpus']} | {x['query_has_image_first200']} | {x['corpus_has_image_first200']} | {x['query_text_len_median']:.1f} | {x['corpus_text_len_median']:.1f} | {x['corpus_unique_text_ratio']:.3f} | {'; '.join(notes)} |")
lines.append('\n## Samples\n')
for x in out:
    if not x.get('exists', True): continue
    lines.append(f"### {x['dataset']}")
    lines.append(f"- query samples: `{x['sample_queries']}`")
    lines.append(f"- corpus samples: `{x['sample_corpus']}`")
    top = list(x['corpus_top_texts'].items())[:5]
    lines.append(f"- top corpus texts: `{top}`")
    lines.append('')
(outdir/'eval_beir_audit.md').write_text('\n'.join(lines), encoding='utf-8')
print(outdir/'eval_beir_audit.md')
