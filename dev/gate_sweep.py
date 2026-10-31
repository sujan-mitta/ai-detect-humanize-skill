import json,sys,random,collections
import numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'../humanize/scripts')
import external_eval as ee, ai_score
cache=ee.load_cache()
ext=ee.load_texts()
gem=[r for r in json.load(open('runs/gemini_rows.json',encoding='utf-8')) if r['split']=='test']
fam=[json.loads(l) for l in (ee.EXT/'coling_train.jsonl').read_text(encoding='utf-8').splitlines()]
fam=[r for r in fam if r['label']==1 and r['generator'] in ('llama3-70b','llama3-8b','gemma2-9b-it','gemma-7b-it','gpt-3.5-turbo','gpt-35','cohere','mixtral-8x7b') and ee.key(r['text']) in cache and 'neural' in cache[ee.key(r['text'])]]
print(len(fam),'cached other-family AI (some were legacy training rows)')
m=ai_score.load_models(); thr=m['max']['thresholds']['ai']/100
def run(rows,labelkey=None):
    return np.array([ee.score(t['text'],cache[ee.key(t['text'])],'max') or 0 for t in rows])
hum=[t for t in ext if t['label']==0]; aiext=[t for t in ext if t['label']==1]
for gate in (0,60,70,80,88,92):
    m['legacy']['gate']=gate
    h=run(hum); a=run(aiext); g=run(gem)
    print(f'gate {gate:3d}: ext humans >=.60 {np.mean(h>=.6):5.1%} >=verdict {np.mean(h>=thr):5.1%} | ext AI >=.60 {np.mean(a>=.6):5.1%} >=verdict {np.mean(a>=thr):5.1%} | gemini heldout >=.60 {np.mean(g>=.6):5.1%} >=verdict {np.mean(g>=thr):5.1%}')
