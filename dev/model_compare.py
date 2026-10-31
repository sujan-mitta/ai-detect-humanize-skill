import json,sys,random
from pathlib import Path
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'../humanize/scripts')
import external_eval as ee, ai_score, legacy_features as lf
cache=ee.load_cache(); ext=ee.load_texts()
gem=[r for r in json.load(open('runs/gemini_rows.json',encoding='utf-8')) if r['split']=='test']
rows=[json.loads(l) for l in (ee.EXT/'coling_train.jsonl').read_text(encoding='utf-8').splitlines()]
used={ee.key(r['text']) for r in lf.subset()}|{ee.key(r['text']) for r in json.load(open('runs/legacy_subset_extra.json',encoding='utf-8'))}|{ee.key(t['text']) for t in ext}|{ee.key(r['text']) for r in json.load(open('runs/gemini_rows.json',encoding='utf-8'))}
fam={'gpt4':('gpt4o','gpt4'),'gpt3.5':('gpt-3.5-turbo','gpt-35'),'llama3':('llama3-70b','llama3-8b'),'gemma':('gemma2-9b-it','gemma-7b-it'),'mixtral+cohere':('mixtral-8x7b','cohere')}
fa={k:[r for r in rows if r['label']==1 and r['generator'] in g and ee.key(r['text']) in cache and 'neural' in cache[ee.key(r['text'])] and ee.key(r['text']) not in used] for k,g in fam.items()}
hu=[r for r in rows if r['label']==0 and ee.key(r['text']) in cache and 'neural' in cache[ee.key(r['text'])] and ee.key(r['text']) not in used]
def sc(rs): return np.array([ee.score(t['text'],cache[ee.key(t['text'])],'max') or 0 for t in rs])
for name,path in (('before (v4c)','runs/model_v4c_pre_v7.json'),('after (v7)','../humanize/scripts/model.json')):
    ai_score.MODEL_PATH=Path(path); ai_score._MODELS=None
    thr=ai_score.load_models()['max']['thresholds']['ai']/100
    print(f'== {name}: verdict threshold {thr:.2f}')
    h=sc([t for t in ext if t['label']==0]); a=sc([t for t in ext if t['label']==1]); g=sc(gem)
    print(f'  external: humans >=.6 {np.mean(h>=.6):.1%} >=verdict {np.mean(h>=thr):.1%} | AI >=.6 {np.mean(a>=.6):.1%} >=verdict {np.mean(a>=thr):.1%}')
    print(f'  gemini heldout: >=.6 {np.mean(g>=.6):.1%} >=verdict {np.mean(g>=thr):.1%}')
    ho=sc(hu); print(f'  fresh humans n={len(ho)} >=.6 {np.mean(ho>=.6):.1%} >=verdict {np.mean(ho>=thr):.1%}')
    print('  families (n, >=.6, >=verdict): '+' | '.join(f'{k} {len(v)} {np.mean(sc(v)>=.6):.0%} {np.mean(sc(v)>=thr):.0%}' for k,v in fa.items() if v))
