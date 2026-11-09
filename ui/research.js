const $ = id => document.getElementById(id);
let token='', runName='', snapshot=null, timer=null, selectedSample=null, inFlight=false, passageSequence=0;
const sampleCache=new Map();
const names={baseline:'Previous-token · strength 2',stronger:'Previous-token · strength 3',fixed:'Fixed vocabulary · strength 3'};
const stages=['original','paraphrase_1','paraphrase_2','paraphrase_3'];
const colors={baseline:'#65716d',stronger:'#087f74',fixed:'#287ca3'};
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function showError(e){$('error').textContent=e.message;$('error').hidden=false;}
function clearError(){$('error').hidden=true;}
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json','X-Dashboard-Token':token},body:JSON.stringify(data)}:{});const result=await r.json();if(!r.ok)throw Error(result.error||'Request failed');return result;}
function query(route,extra={}){return '/api/research/'+route+'?'+new URLSearchParams({name:runName,...extra});}
function number(value,d=2){return value==null?'Not measured':Number(value).toFixed(d);}
function duration(seconds){if(seconds==null||!Number.isFinite(seconds))return 'Not estimated';return seconds>=3600?(seconds/3600).toFixed(1)+' h':seconds>=60?Math.round(seconds/60)+' min':Math.round(seconds)+' s';}
function percent(k,n){return n&&k!=null?(100*k/n).toFixed(1)+'% ('+k+'/'+n+')':'Not measured';}
function optionList(el,values){const signature=JSON.stringify(values);if(el.dataset.options===signature)return false;const old=el.value;el.replaceChildren();for(const [value,label]of values){const o=node('option',label);o.value=value;el.append(o);}if(values.some(([v])=>v===old))el.value=old;el.dataset.options=signature;return true;}
async function loadRuns(){const items=await api('/api/research/runs');optionList($('run-select'),items.map(r=>[r.name,r.name+' · '+r.status]));if(!runName&&items.length)runName=items[0].name;if(runName)$('run-select').value=runName;}
async function refresh(){
 if(inFlight)return;inFlight=true;
 try{
  if(!runName){await loadRuns();if(!runName){$('run-state').textContent='No experiments yet';$('current').textContent='Start a run below to generate the first measured comparison.';return;}}
  const requested=runName;const data=await api(query('run'));if(requested!==runName)return;snapshot=data;
  $('connection').textContent='Live · '+new Date().toLocaleTimeString();renderStatus(data);renderComparison(data);renderLengths(data);renderRoadmap(data);
  const changed=renderFilters(data);if(changed||!selectedSample)await loadPassages();
 }catch(e){$('connection').textContent='Connection interrupted';showError(e);}finally{inFlight=false;}
}
function renderStatus(data){
 const status=data.status,active=['queued','loading','running'].includes(status.status);
 $('run-state').textContent=({queued:'Queued',loading:'Loading model',running:'Experiment running',stopped:'Stopped at a checkpoint',complete:'Local comparisons complete',failed:'Run needs attention'})[status.status]||status.status;
 $('run-count').textContent=(status.completed||0)+' / '+status.total+' cells complete';$('run-progress').max=status.total||1;$('run-progress').value=status.completed||0;
 $('current').textContent=status.current||'Waiting for a measurement';for(const option of $('run-select').options)if(option.value===runName)option.textContent=runName+' · '+status.status;$('elapsed').textContent='Elapsed: '+duration(status.elapsed_seconds);
 const remaining=status.completed?status.elapsed_seconds/status.completed*(status.total-status.completed):null;
 $('eta').textContent=active?'Estimated remaining: '+duration(remaining):'Saved results remain available';$('scored').textContent=(status.scored_texts||0)+' scored passages';
 $('stop').disabled=!active;$('resume').hidden=active||status.status==='complete';$('new-run').disabled=active;
 $('csv').href=query('csv');
 const first=data.variants.map(v=>({name:v.name,...v.conditions.paraphrase_1})).filter(v=>v.n);
 const measured=first.reduce((sum,v)=>sum+v.n,0),detected=first.reduce((sum,v)=>sum+v.detected,0);
 $('finding').textContent=measured?detected+' of '+measured+' measured first rewrites have crossed the cutoff across the three methods so far. The method table shows each denominator. No validated winner yet.':'The comparison is running. A larger original score alone is not the target. We are measuring survival after rewriting.';
}
function renderComparison(data){
 const held=$('view').value==='heldout';$('comparison').replaceChildren();
 $('comparison-note').textContent=held?'Cutoffs use only seed 1 controls. This table evaluates seed 2 passages. Calibration remains exploratory and incomplete until every calibration control finishes.':'Every measured main-stage sample at the unchanged cutoff of 4. Development and held-out seeds are pooled in this exploratory view. Length-sweep samples are separate.';
 for(const variant of data.variants){
  const tr=node('tr');tr.append(node('td',variant.name));
  for(const condition of ['original','paraphrase_1','paraphrase_3','control']){
   const s=variant.conditions[condition],td=node('td',held?percent(s.heldout_detected,s.heldout_n):percent(s.detected,s.n));
   if(!held&&s.interval)td.append(node('small','95% interval '+(s.interval[0]*100).toFixed(1)+'–'+(s.interval[1]*100).toFixed(1)+'%'));
   tr.append(td);
  }
  tr.append(node('td',held?number(variant.calibrated_cutoff,3):'4'),node('td',String(variant.reviewed)+' reviewed'));$('comparison').append(tr);
 }
 renderChart(data,held);
}
function svg(tag,attrs={},text){const el=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))el.setAttribute(k,String(v));if(text!==undefined)el.textContent=text;return el;}
function renderChart(data,held){
 const root=$('survival-chart');root.replaceChildren();const image=svg('svg',{viewBox:'0 0 1000 300',role:'img','aria-label':'Detection rate from the original passage through three rewrites'});
 image.append(svg('title',{},'Measured detection rate by rewriting round'));
 const x=i=>70+i*270,y=v=>238-v*190;
 for(const v of [0,.5,1]){image.append(svg('line',{x1:70,y1:y(v),x2:880,y2:y(v),stroke:'#e5e9e8'}));image.append(svg('text',{x:54,y:y(v)+5,'text-anchor':'end',fill:'#52625e','font-size':14},Math.round(v*100)+'%'));}
 ['Original','Rewrite 1','Rewrite 2','Rewrite 3'].forEach((label,i)=>image.append(svg('text',{x:x(i),y:274,'text-anchor':'middle',fill:'#52625e','font-size':15},label)));
 const legend=node('div',undefined,'chart-legend');let count=0;
 for(const v of data.variants){
  const points=stages.map((s,i)=>{const c=v.conditions[s];return {i,value:held?c.heldout_rate:c.rate,n:held?c.heldout_n:c.n};});
  let previous=null;
  for(const p of points){if(p.value==null||!p.n){previous=null;continue;}count++;
   if(previous)image.append(svg('line',{x1:x(previous.i),y1:y(previous.value),x2:x(p.i),y2:y(p.value),stroke:colors[v.id],'stroke-width':2}));
   const dot=svg('circle',{cx:x(p.i),cy:y(p.value),r:4,fill:colors[v.id]});dot.append(svg('title',{},v.name+': '+(p.value*100).toFixed(1)+'% of '+p.n+' samples'));image.append(dot);previous=p;
  }
  const item=node('span');const swatch=node('span',undefined,'legend-swatch');swatch.style.backgroundColor=colors[v.id];item.append(swatch,document.createTextNode(v.name));legend.append(item);
 }
 if(count)root.append(image,legend);else root.append(node('p','No measured samples in this evaluation view yet.','help'));
}
function caseKey(row){return [row.prompt_id,row.seed_index,row.max_new_tokens].join(':');}
function renderFilters(data){
 const originals=data.rows.filter(r=>r.condition==='original'),cases=new Map();for(const r of originals)cases.set(caseKey(r),'Topic '+r.prompt_id+' · seed '+r.seed_index+' · '+r.max_new_tokens+' token cap');
 const previousCase=$('case-select').value,previousMethod=$('method-select').value;optionList($('case-select'),[...cases]);
 const methods=data.variants.filter(v=>originals.some(r=>caseKey(r)===$('case-select').value&&r.variant===v.id));
 optionList($('method-select'),methods.map(v=>[v.id,v.name]));return previousCase!==$('case-select').value||previousMethod!==$('method-select').value;
}
async function getSample(id){const key=runName+'/'+id;if(!sampleCache.has(key))sampleCache.set(key,await api(query('sample',{sample:id})));return sampleCache.get(key);}
function sampleDetails(sample){return 'Score '+number(sample.z_score)+' · '+sample.token_count+' tokens · '+number(sample.generation_seconds,1)+' s to generate'+(sample.hit_token_limit?' · OUTPUT LIMIT REACHED':'')+'. Evidence unit: '+sample.scoring_unit+'.';}
async function loadPassages(){
 const seq=++passageSequence;if(!snapshot)return;selectedSample=null;$('save-review').disabled=true;
 const rows=snapshot.rows.filter(r=>caseKey(r)===$('case-select').value&&r.variant===$('method-select').value),original=rows.find(r=>r.condition==='original'),rewritten=rows.find(r=>r.condition===$('condition-select').value);
 const topic=original?snapshot.manifest.topics[original.prompt_id-1]:'';$('topic').textContent=topic;
 $('rewritten-heading').textContent=({control:'Ordinary control',paraphrase_1:'First rewrite',paraphrase_2:'Second rewrite',paraphrase_3:'Third rewrite'})[$('condition-select').value];
 try{
  const pair=await Promise.all([original?getSample(original.sample_id):null,rewritten?getSample(rewritten.sample_id):null]);if(seq!==passageSequence)return;
  for(const [prefix,sample]of [['original',pair[0]],['rewritten',pair[1]]]){
   $(prefix+'-text').textContent=sample?sample.text:'This passage has not finished yet. The page updates as generations complete.';
   $(prefix+'-details').textContent=sample?sampleDetails(sample):'';$(prefix+'-details').classList.toggle('limit-warning',Boolean(sample?.hit_token_limit));
   $(prefix+'-verdict').textContent=sample?(sample.detected?'Pattern detected':'Below cutoff'):'';
  }
  selectedSample=pair[1]?.sample_id||null;$('save-review').disabled=!selectedSample;
  const review=snapshot.reviews[selectedSample];for(const k of ['meaning','fluency','facts'])$(k).value=review?.[k]||'';$('review-note').value=review?.note||'';$('review-status').textContent=review?'Saved review loaded.':'';
 }catch(e){showError(e);}
}
function renderLengths(data){
 $('lengths').replaceChildren();
 for(const v of data.variants)for(const cap of [128,256,512]){
  const rows=data.rows.filter(r=>r.variant===v.id&&r.max_new_tokens===cap);if(!rows.length)continue;
  const rewritten=rows.filter(r=>r.condition==='paraphrase_1'),originals=rows.filter(r=>r.condition==='original');const tr=node('tr');
  for(const value of [v.name,String(cap),percent(rewritten.filter(r=>r.detected).length,rewritten.length),percent(originals.filter(r=>r.hit_token_limit).length,originals.length),String(rows.length)])tr.append(node('td',value));$('lengths').append(tr);
 }
}
function renderRoadmap(data){
 const root=$('roadmap');root.replaceChildren();
 const entries=[['Local comparison',data.status.status==='complete'?'Measured runs complete. Review the denominators and saved passages above.':'Compares baseline, stronger bias and fixed vocabulary on the same topics and seeds. Current state: '+data.status.status+'.'],['Length sweep','Exploratory caps of 128 and 512 tokens on five topics, separate from the main comparison.'],['Human quality review',Object.keys(data.reviews).length+' samples reviewed. Meaning and fluency remain unverified for unrated samples.'],['Human false-positive rate',data.limitations.false_positives],['Translation benchmark',data.limitations.translation],['Semantic-watermark candidate',data.limitations.semantic]];
 for(const [title,body]of entries){const article=node('article');article.append(node('h3',title),node('p',body));root.append(article);}
 const references=node('article');references.append(node('h3','Research references'));const list=node('div');for(const source of data.manifest.sources){const p=node('p'),a=node('a',source.title);a.href=source.url;a.target='_blank';a.rel='noopener';p.append(a);list.append(p);}references.append(list);root.append(references);
}
$('run-select').onchange=async()=>{runName=$('run-select').value;selectedSample=null;snapshot=null;clearError();await refresh();};
$('view').onchange=()=>{if(snapshot)renderComparison(snapshot);};
$('case-select').onchange=()=>{if(snapshot)renderFilters(snapshot);loadPassages();};$('method-select').onchange=loadPassages;$('condition-select').onchange=loadPassages;
$('stop').onclick=async()=>{try{await api('/api/research/stop',{name:runName});$('stop').disabled=true;$('current').textContent='Stop requested. The current generation will finish, then save a checkpoint.';}catch(e){showError(e);}};
$('resume').onclick=async()=>{try{clearError();await api('/api/research/resume',{name:runName});await refresh();}catch(e){showError(e);}};
$('save-review').onclick=async()=>{try{if(!selectedSample)return;for(const id of ['meaning','fluency','facts'])if(!$(id).value)throw Error('Choose all three review ratings before saving.');const data={name:runName,sample_id:selectedSample,note:$('review-note').value};for(const id of ['meaning','fluency','facts'])data[id]=Number($(id).value);await api('/api/research/review',data);$('review-status').textContent='Review saved for this sample.';await refresh();}catch(e){showError(e);}};
$('export').onclick=async()=>{try{const report=await api(query('export'));const a=node('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));a.download=runName+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}catch(e){showError(e);}};
$('new-run').onclick=async()=>{try{clearError();const result=await api('/api/research/start',{limit:Number($('new-limit').value),seeds:Number($('new-seeds').value),length_sweep:$('new-sweep').checked});runName=result.name;selectedSample=null;await loadRuns();await refresh();}catch(e){showError(e);}};
api('/api/research/init').then(async data=>{token=data.token;await loadRuns();await refresh();timer=setInterval(refresh,3000);}).catch(showError);
