"""Resumable paired research benchmark. All public measurements exclude keys."""
import argparse, csv, hashlib, json, math, os, re, secrets, time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
VARIANTS=[
 {'id':'baseline','name':'Previous-token, strength 2','scheme':'previous','delta':2.0,'gamma':.25},
 {'id':'stronger','name':'Previous-token, strength 3','scheme':'previous','delta':3.0,'gamma':.25},
 {'id':'fixed','name':'Fixed vocabulary, strength 3','scheme':'fixed','delta':3.0,'gamma':.25},
]

def write_text_atomic(path,value):
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.{secrets.token_hex(4)}.tmp')
    temp.write_text(value,encoding='utf-8')
    try:
        for attempt in range(40):
            try:
                os.replace(temp,path)
                break
            except PermissionError:
                if attempt==39:raise
                time.sleep(.05)
    finally:
        temp.unlink(missing_ok=True)

def write_json(path,value):
    write_text_atomic(path,json.dumps(value,ensure_ascii=False,indent=2))

def read_json(path,default=None):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

def read_rows(path):
    rows=[]
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            try:rows.append(json.loads(line))
            except ValueError:pass # A writer may be appending the final line.
    except OSError:pass
    return rows

def wilson(k,n):
    if not n:return None
    z=1.95996398454;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-r),min(1,c+r)]

def plan(limit=20,seeds=3,with_length_sweep=True):
    cells=[]
    for p in range(limit):
        for seed in range(seeds):
            for v in VARIANTS:
                cells.append(dict(id=f'p{p+1:02}-s{seed}-n256-{v["id"]}',prompt_id=p+1,seed_index=seed,max_new_tokens=256,variant=v['id'],stage='main'))
    if with_length_sweep:
        for p in range(min(limit,5)):
            for length in [128,512]:
                for v in VARIANTS:
                    cells.append(dict(id=f'p{p+1:02}-s0-n{length}-{v["id"]}',prompt_id=p+1,seed_index=0,max_new_tokens=length,variant=v['id'],stage='length_sweep'))
    return cells

def create_run(limit=20,seeds=3,with_length_sweep=True):
    folder=ROOT/'results'/('research-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    (folder/'samples').mkdir(parents=True)
    (folder/'watermark.key').write_bytes(secrets.token_bytes(32))
    topics=json.loads((ROOT/'prompts.json').read_text(encoding='utf-8'))[:limit]
    manifest=dict(kind='research',model='Qwen/Qwen2.5-0.5B-Instruct',revision='main',created_at=datetime.now(timezone.utc).isoformat(),variants=VARIANTS,topics=topics,seeds=seeds,base_seed=1234,cutoff=4,temperature=.8,top_p=.95,plan=plan(limit,seeds,with_length_sweep),
      sources=[{'title':'Provable Robust Watermarking for AI-Generated Text','url':'https://proceedings.iclr.cc/paper_files/paper/2024/hash/beae9ed5316bcc48e616754c06c11875-Abstract-Conference.html'},{'title':'A Semantic Invariant Robust Watermark for Large Language Models','url':'https://arxiv.org/abs/2310.06356'}],
      scope={'fixed':'Experimental fixed-vocabulary variant inspired by Unigram-Watermark. Uses unique-token scoring; not an exact paper reproduction.', 'main':'Same topics, seeds, length and cutoff across methods. Seed 0 development, seed 1 exploratory calibration, seed 2 held out.', 'quality':'Human meaning, fluency and factual-fidelity ratings must be collected. Lexical overlap is descriptive only.', 'false_positives':'Controls are unwatermarked AI text, not a human corpus. Human false-positive rate remains unmeasured.', 'translation':'Not run: the two translation models are not cached. No synthetic substitution replaces translation.', 'semantic':'Pending: dedicated semantic encoder and faithful reference implementation are not installed. Current fixed method is NOT semantic watermarking.'})
    write_json(folder/'manifest.json',manifest)
    write_json(folder/'status.json',dict(status='queued',completed=0,total=len(manifest['plan']),scored_texts=0,current='Waiting for model',started_at=None,elapsed_seconds=0))
    return folder

def fixed_score(ids,scheme,threshold):
    # A fixed vocabulary partition has no previous-token context. Count distinct IDs
    # so repeatedly generating one favored token does not multiply the evidence.
    tokens=set(ids)&scheme.eligible;green=len(tokens&scheme.green_set(0));n=len(tokens)
    z=(green-scheme.gamma*n)/math.sqrt(n*scheme.gamma*(1-scheme.gamma)) if n else None
    return dict(scored_tokens=n,green_tokens=green,green_ratio=green/n if n else None,z_score=z,detected=z is not None and z>=threshold,threshold=threshold,scoring_unit='distinct token IDs',p_value_approx=None)

def similarity(a,b):
    left=set(re.findall(r'\w+',a.lower()));right=set(re.findall(r'\w+',b.lower()))
    return len(left&right)/len(left|right) if left|right else None

def aggregate(folder):
    manifest=read_json(folder/'manifest.json',{});rows=read_rows(folder/'measurements.jsonl');reviews=read_json(folder/'reviews.json',{})
    out=[]
    for variant in manifest.get('variants',[]):
        group=[r for r in rows if r['variant']==variant['id'] and r['stage']=='main']
        # Calibration is selected exclusively from seed-index 1 controls.
        calibration=[r['z_score'] for r in group if r['condition']=='control' and r['seed_index']==1 and r['z_score'] is not None]
        cutoff=max(4,max(calibration)+1e-6) if calibration else None
        conditions={}
        for condition in ['control','original','paraphrase_1','paraphrase_2','paraphrase_3']:
            selected=[r for r in group if r['condition']==condition];n=len(selected);k=sum(r['detected'] for r in selected)
            held=[r for r in selected if r['seed_index']==2];hn=len(held);hk=sum(r['z_score'] is not None and r['z_score']>=cutoff for r in held) if cutoff is not None else None
            conditions[condition]=dict(n=n,detected=k,rate=k/n if n else None,interval=wilson(k,n),mean_z=sum(r['z_score'] for r in selected if r['z_score'] is not None)/sum(r['z_score'] is not None for r in selected) if any(r['z_score'] is not None for r in selected) else None,heldout_n=hn,heldout_detected=hk,heldout_rate=hk/hn if hn and hk is not None else None,limit_reached=sum(r['hit_token_limit'] for r in selected))
        reviewed=[r for r in group if r['sample_id'] in reviews]
        out.append(dict(**variant,conditions=conditions,calibration_n=len(calibration),calibrated_cutoff=cutoff,reviewed=len(reviewed),human_fpr=None))
    return dict(variants=out,rows=rows,reviews=reviews,limitations=manifest.get('scope',{}))

def run(folder):
    os.environ.setdefault('HF_HOME',str(ROOT/'.runtime/huggingface'))
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
    import torch, numpy as np, transformers
    from generate import Generator
    from watermark import WatermarkScheme,make_processor,vocabulary_ids,tokenizer_fingerprint
    from detect import detect_ids
    from attacks import paraphrase_prompt
    torch.set_num_threads(min(4,torch.get_num_threads()))
    manifest=read_json(folder/'manifest.json');prior=read_json(folder/'status.json',{});started=time.monotonic();elapsed=prior.get('elapsed_seconds',0); start_label=manifest['created_at']
    rows=read_rows(folder/'measurements.jsonl');done=set(read_json(folder/'completed.json',[]));plans=manifest['plan'];key=(folder/'watermark.key').read_bytes()
    def status(state,current):
        write_json(folder/'status.json',dict(status=state,completed=len(done),total=len(plans),scored_texts=len(rows),current=current,started_at=start_label,elapsed_seconds=round(elapsed+time.monotonic()-started),updated_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid()))
    def stop():return (folder/'stop.request').exists()
    try:
        status('loading','Loading cached Qwen model');generator=Generator(manifest['model'],manifest['revision']);tokenizer=generator.tokenizer
        manifest.update(resolved_model_commit=getattr(generator.model.config,'_commit_hash',None),tokenizer_sha256=tokenizer_fingerprint(tokenizer),torch_version=torch.__version__,transformers_version=transformers.__version__,numpy_version=np.__version__,device=generator.device)
        write_json(folder/'manifest.json',manifest)
        class FixedScheme(WatermarkScheme):
            def green_ids(self,previous_token):return super().green_ids(0)
            def green_set(self,previous_token):return super().green_set(0)
        schemes={v['id']:(FixedScheme if v['scheme']=='fixed' else WatermarkScheme)(vocabulary_ids(tokenizer),key,v['gamma']) for v in VARIANTS}
        processors={v['id']:make_processor(schemes[v['id']],v['delta']) for v in VARIANTS};settings=SimpleNamespace(temperature=manifest['temperature'],top_p=manifest['top_p'],max_new_tokens=256)
        eos=generator.model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        def generate(prompt,seed,processor=None):
            t=time.monotonic();text,ids=generator.generate(prompt,seed,settings,processor)
            return dict(text=text,ids=ids,generation_seconds=time.monotonic()-t,hit_token_limit=len(ids)>=settings.max_new_tokens and (not ids or ids[-1] not in eos))
        def record(cell,condition,sample,original):
            sid=cell['id']+'-'+condition;scheme=schemes[cell['variant']];ids=tokenizer.encode(sample['text'],add_special_tokens=False)
            stats=fixed_score(ids,scheme,manifest['cutoff']) if cell['variant']=='fixed' else {**detect_ids(ids,scheme,manifest['cutoff']), 'scoring_unit':'distinct token pairs'}
            row={**cell,'sample_id':sid,'condition':condition,**stats,'token_count':len(ids),'raw_token_count':len(sample['ids']),'characters':len(sample['text']),'generation_seconds':round(sample['generation_seconds'],3),'hit_token_limit':sample['hit_token_limit'],'length_ratio':len(ids)/max(1,len(tokenizer.encode(original,add_special_tokens=False))),'lexical_overlap':similarity(original,sample['text']) if condition.startswith('paraphrase') else None}
            write_json(folder/'samples'/(sid+'.json'),dict(**row,text=sample['text'],raw_ids=sample['ids']))
            # Upsert on resume; never duplicate a partially completed cell's rows.
            existing=next((i for i,r in enumerate(rows) if r['sample_id']==sid),None)
            if existing is not None:rows[existing]=row
            else:rows.append(row)
            write_text_atomic(folder/'measurements.jsonl',''.join(json.dumps(r)+'\n' for r in rows))
            with (folder/'results.csv').open('w',newline='',encoding='utf-8') as h:
                w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            status('running',f'Topic {cell["prompt_id"]}/{len(manifest["topics"])}; seed {cell["seed_index"]}; {cell["variant"]}; {condition}; {settings.max_new_tokens} tokens')
        for cell in plans:
            if cell['id'] in done:continue
            if stop():status('stopped','Stopped at a checkpoint. Resume retains measurements.');return
            settings.max_new_tokens=cell['max_new_tokens'];seed=manifest['base_seed']+100*cell['prompt_id']+cell['seed_index']
            label=f'Topic {cell["prompt_id"]}; seed {cell["seed_index"]}; {cell["variant"]}; {settings.max_new_tokens} tokens'
            status('running',label+'; generating')
            controlfile=folder/'samples'/f'control-p{cell["prompt_id"]}-s{cell["seed_index"]}-n{settings.max_new_tokens}.json'
            topic=manifest['topics'][cell['prompt_id']-1]
            prompt=topic+' Write two complete prose paragraphs, about 120 words total. Return only the passage and finish the final sentence.'
            control=read_json(controlfile)
            if control is None:control=generate(prompt,seed);write_json(controlfile,control)
            record(cell,'control',control,control['text'])
            original=read_json(folder/'samples'/(cell['id']+'-original.json'))
            if original is None:original=generate(prompt,seed,processors[cell['variant']])
            else:original=dict(text=original['text'],ids=original['raw_ids'],generation_seconds=original['generation_seconds'],hit_token_limit=original['hit_token_limit'])
            record(cell,'original',original,original['text']);current=original['text']
            for index in range(1,4):
                if stop():status('stopped','Stopped. Resume continues this cell without discarding saved passages.');return
                saved=read_json(folder/'samples'/(cell['id']+f'-paraphrase_{index}.json'))
                if saved is None:sample=generate(paraphrase_prompt(current),seed+1000*index)
                else:sample=dict(text=saved['text'],ids=saved['raw_ids'],generation_seconds=saved['generation_seconds'],hit_token_limit=saved['hit_token_limit'])
                record(cell,f'paraphrase_{index}',sample,original['text']);current=sample['text']
            done.add(cell['id']);write_json(folder/'completed.json',sorted(done));status('running',label+' complete')
        status('complete','All local-model comparisons finished. Human quality review and additional-model stages remain.');write_json(folder/'summary.json',aggregate(folder)['variants'])
    except Exception as e:
        status('failed',str(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--limit',type=int,default=20);p.add_argument('--seeds',type=int,default=3);p.add_argument('--no-length-sweep',action='store_true');args=p.parse_args()
    folder=args.run or create_run(args.limit,args.seeds,not args.no_length_sweep)
    print(folder.resolve(),flush=True);run(folder)
