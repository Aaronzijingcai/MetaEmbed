from pathlib import Path
import json, os, yaml
from datasets import load_dataset

ROOT = Path('/MURE-V2/code/MetaEmbed/colqwen_multigranularity/experiments/2026-07-01')
CONFIGS = [
    ROOT/'MMEB任务课程学习/configs/train_vqa_hard.yaml',
    ROOT/'MMEB任务课程学习/configs/train_vqa_hard_replay20.yaml',
    ROOT/'MMEB任务课程学习/configs/train_compositional_hard.yaml',
]
OUT_DIR = ROOT/'MMEB任务课程学习/data_audit'
OUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_PATH = os.environ.get('DATA_DIR', '/MURE-V2/code/MetaEmbed/data_dir/')
DATASET = BASE_PATH.rstrip('/') + '/MoCa_train_with_image'
SAMPLE_INDICES = [0, 1, 2, 10, 99]
TEXT_KEYS = ['qry','pos_text','neg_text']
IMAGE_KEYS = ['qry_image','pos_image']

def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)

def text_preview(x, n=240):
    if x is None:
        return None
    if isinstance(x, list):
        return [text_preview(y, n=120) for y in x[:3]]
    s=str(x).replace('\n','\\n')
    return s[:n] + ('...' if len(s)>n else '')

def image_info(x):
    if x is None:
        return {'present': False}
    info={'present': True, 'type': type(x).__name__}
    try:
        info['size'] = list(x.size)
        info['mode'] = x.mode
    except Exception as e:
        info['image_error'] = repr(e)
    return info

def audit_subset(subset, meta):
    n = int(meta.get('num_samples', 0) or 0)
    split = f'original[:{n}]' if n else 'original'
    ds = load_dataset(DATASET, subset, split=split, num_proc=1)
    length=len(ds)
    idxs=[i for i in SAMPLE_INDICES if i < length]
    stats={
        'subset': subset,
        'configured_num_samples': n or None,
        'loaded_len': length,
        'features': list(ds.features.keys()),
        'samples': [],
        'aggregate': {
            'checked': 0,
            'query_has_image': 0,
            'pos_has_image': 0,
            'neg_image_slots_present': {},
            'empty_qry': 0,
            'empty_pos_text': 0,
            'empty_neg_text': 0,
            'pos_equals_query_text': 0,
            'neg_text_contains_pos_text': 0,
        }
    }
    for idx in idxs:
        ex=ds[idx]
        agg=stats['aggregate']
        agg['checked'] += 1
        qry=ex.get('qry')
        pos=ex.get('pos_text')
        neg=ex.get('neg_text')
        qimg=ex.get('qry_image')
        pimg=ex.get('pos_image')
        if qimg is not None: agg['query_has_image'] += 1
        if pimg is not None: agg['pos_has_image'] += 1
        if not str(qry or '').strip(): agg['empty_qry'] += 1
        if not str(pos or '').strip(): agg['empty_pos_text'] += 1
        if not neg: agg['empty_neg_text'] += 1
        if str(qry or '').strip() == str(pos or '').strip(): agg['pos_equals_query_text'] += 1
        if isinstance(neg, list) and str(pos or '').strip() and any(str(pos).strip()==str(x).strip() for x in neg):
            agg['neg_text_contains_pos_text'] += 1
        for k in ex.keys():
            if k.startswith('neg_image_'):
                agg['neg_image_slots_present'][k] = agg['neg_image_slots_present'].get(k,0) + (1 if ex.get(k) is not None else 0)
        sample={
            'idx': idx,
            'keys': list(ex.keys()),
            'qry': text_preview(qry),
            'pos_text': text_preview(pos),
            'neg_text': text_preview(neg),
            'qry_image': image_info(qimg),
            'pos_image': image_info(pimg),
            'neg_images': {k:image_info(ex.get(k)) for k in ex.keys() if k.startswith('neg_image_')},
        }
        stats['samples'].append(sample)
    return stats

all_stats=[]
seen=[]
for cfg in CONFIGS:
    data=load_yaml(cfg)
    for subset,meta in data.items():
        if subset not in seen:
            seen.append(subset)
            print('AUDIT', subset, meta, flush=True)
            all_stats.append(audit_subset(subset, meta or {}))

json_path=OUT_DIR/'raw_sample_audit.json'
json_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2))
md=[]
md.append('# MMEB Raw Sample Audit\n')
md.append(f'Dataset path: `{DATASET}`\n')
for st in all_stats:
    a=st['aggregate']
    md.append(f"\n## {st['subset']}\n")
    md.append(f"- loaded_len: {st['loaded_len']}\n")
    md.append(f"- features: `{', '.join(st['features'])}`\n")
    md.append(f"- checked: {a['checked']}, query_has_image: {a['query_has_image']}, pos_has_image: {a['pos_has_image']}\n")
    md.append(f"- empty_qry: {a['empty_qry']}, empty_pos_text: {a['empty_pos_text']}, empty_neg_text: {a['empty_neg_text']}\n")
    md.append(f"- pos_equals_query_text: {a['pos_equals_query_text']}, neg_text_contains_pos_text: {a['neg_text_contains_pos_text']}\n")
    md.append(f"- neg_image_slots_present: `{a['neg_image_slots_present']}`\n")
    md.append('\n| idx | q_img | p_img | qry | pos_text | neg_text preview |\n')
    md.append('| ---: | ---: | ---: | --- | --- | --- |\n')
    for sm in st['samples'][:5]:
        neg=sm['neg_text']
        if isinstance(neg, list):
            neg=' ; '.join(str(x) for x in neg[:2])
        md.append(f"| {sm['idx']} | {sm['qry_image']['present']} | {sm['pos_image']['present']} | {sm['qry']} | {sm['pos_text']} | {neg} |\n")
md_path=OUT_DIR/'raw_sample_audit.md'
md_path.write_text(''.join(md))
print('WROTE', json_path)
print('WROTE', md_path)
