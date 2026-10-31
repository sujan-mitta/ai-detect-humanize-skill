import json,sys,random,collections
import numpy as np
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'../humanize/scripts')
import external_eval as ee, ai_score
random.seed(99)
# same human held-out set as family_eval + external eval humans
rows=[json.loads(l) for l in (ee.EXT/'coling_train.jsonl').read_text(encoding='utf-8').splitlines()]
cache=ee.load_cache()
hum=[r for r in rows if r['label']==0 and ee.key(r['text']) in cache and 'neural' in cache[ee.key(r['text'])]]
ext=[t for t in ee.load_texts() if t['label']==0]
print(len(hum),'cached coling-train humans (incl. legacy training ones),',len(ext),'eval humans')
for name in ['model_v4_pre_extra.json','model_v4b_pre_gemini.json','../../humanize/scripts/model.json']:
    ai_score.MODEL_PATH=Path('runs')/name if '..' not in name else Path('../humanize/scripts/model.json')
    ai_score._MODELS=None
    out=[]
    for t in ext:
        out.append(ee.score(t['text'],cache[ee.key(t['text'])],'max'))
    out=np.array([x for x in out if x is not None])
    thr=ai_score.load_models()['max']['thresholds']['ai']/100
    print(f'{name[:28]:28s} external humans n={len(out)} >=0.60: {np.mean(out>=.6):.1%}  >=verdict({thr:.2f}): {np.mean(out>=thr):.1%}  mean {out.mean():.2f}')
