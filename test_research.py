"""Regression checks for paired plans, calibration separation and checkpoint resume."""
import json,tempfile,threading,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from http.server import ThreadingHTTPServer
from urllib.request import Request,urlopen
from urllib.error import HTTPError
import research,research_server
class StatisticsTests(unittest.TestCase):
 def test_atomic_save_retries_windows_reader_lock(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'state.json';replace=research.os.replace;calls=[]
   def transient(source,target):
    calls.append(1)
    if len(calls)==1:raise PermissionError('reader lock')
    return replace(source,target)
   with patch('research.os.replace',side_effect=transient):research.write_json(p,{'saved':True})
   self.assertEqual(research.read_json(p),{'saved':True});self.assertEqual(len(calls),2)
   self.assertEqual(list(Path(d).glob('*.tmp')),[])
 def test_exited_worker_is_visible_as_resumable(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);research.write_json(p/'status.json',{'status':'running','pid':123})
   with patch('research_server.process_alive',return_value=False):self.assertEqual(research_server.run_status(p)['status'],'stopped')
 def test_plan_has_matched_methods_and_lengths(self):
  cells=research.plan();self.assertEqual(len(cells),210);self.assertEqual(len({c['id'] for c in cells}),210)
  main=[c for c in cells if c['stage']=='main'];self.assertEqual(len(main),180)
  for p in range(1,21):
   for seed in range(3):self.assertEqual({c['variant'] for c in main if c['prompt_id']==p and c['seed_index']==seed},{'baseline','stronger','fixed'})
 def test_fixed_score_does_not_multiply_repeated_tokens(self):
  s=SimpleNamespace(eligible=set(range(8)),gamma=.25,green_set=lambda _:frozenset([1,2]))
  self.assertEqual(research.fixed_score([1,2,3],s,4),research.fixed_score([1,1,1,2,2,3,3],s,4))
  self.assertIsNone(research.fixed_score([],s,4)['z_score'])
 def test_intervals_and_holdout_do_not_calibrate_themselves(self):
  self.assertIsNone(research.wilson(0,0));self.assertGreater(research.wilson(0,20)[1],0)
  with tempfile.TemporaryDirectory() as d:
   f=Path(d);research.write_json(f/'manifest.json',{'variants':[research.VARIANTS[0]]})
   rows=[dict(variant='baseline',stage='main',condition='control',seed_index=1,z_score=4.5,detected=True,hit_token_limit=False,sample_id='cal'),dict(variant='baseline',stage='main',condition='control',seed_index=2,z_score=30,detected=True,hit_token_limit=False,sample_id='test')]
   (f/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   v=research.aggregate(f)['variants'][0];self.assertAlmostEqual(v['calibrated_cutoff'],4.500001);self.assertEqual(v['conditions']['control']['heldout_detected'],1)
 def test_reader_keeps_complete_rows_during_append(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'rows';p.write_text('{"value": 1}\n{"value":');self.assertEqual(research.read_rows(p),[{'value':1}])
class CheckpointTests(unittest.TestCase):
 def test_resume_preserves_rows_and_does_not_regenerate_saved_samples(self):
  class Tokenizer:
   all_special_ids=[127]
   def get_vocab(self):return {str(i):i for i in range(128)}
   def encode(self,*args,**kwargs):return [1,2,3,4,5,6]
  class Generator:
   count=0
   def __init__(self,*args):self.tokenizer=Tokenizer();self.device='cpu';self.model=SimpleNamespace(config=SimpleNamespace(_commit_hash='fixture'),generation_config=SimpleNamespace(eos_token_id=127))
   def generate(self,*args):Generator.count+=1;return 'A complete fixture passage.',[1,2,3,4,5,6]
  with tempfile.TemporaryDirectory() as d,patch.object(research,'ROOT',Path(d)),patch('generate.Generator',Generator):
   Path(d,'prompts.json').write_text('["A test topic"]');f=research.create_run(1,1,False);research.run(f)
   self.assertEqual(len(research.read_rows(f/'measurements.jsonl')),15);self.assertEqual(Generator.count,13)
   research.write_json(f/'completed.json',[]);research.run(f)
   rows=research.read_rows(f/'measurements.jsonl');self.assertEqual(len(rows),15);self.assertEqual(len({r['sample_id'] for r in rows}),15);self.assertEqual(Generator.count,13)
class ApiTests(unittest.TestCase):
 def test_boundaries_exports_and_review_validation(self):
  with tempfile.TemporaryDirectory() as d,patch.object(research_server,'ROOT',Path(d)):
   f=Path(d)/'results'/'research-fixture';(f/'samples').mkdir(parents=True)
   research.write_json(f/'manifest.json',{'variants':[]});research.write_json(f/'status.json',{'status':'complete'});(f/'watermark.key').write_text('PRIVATE-DO-NOT-EXPORT')
   row=dict(sample_id='sample',condition='original',variant='baseline');(f/'measurements.jsonl').write_text(json.dumps(row)+'\n');research.write_json(f/'samples'/'sample.json',{'text':'Fixture passage.'})
   server=ThreadingHTTPServer(('127.0.0.1',0),research_server.Handler);threading.Thread(target=server.serve_forever,daemon=True).start();base=f'http://127.0.0.1:{server.server_port}'
   try:
    with self.assertRaises(HTTPError):urlopen(base+'/api/research/run?name=../outside')
    with urlopen(base+'/api/research/export?name=research-fixture') as r:self.assertNotIn('PRIVATE-DO-NOT-EXPORT',r.read().decode())
    with self.assertRaises(HTTPError):urlopen(Request(base+'/api/research/review',data=b'{}',method='POST'))
    body={'name':'research-fixture','sample_id':'sample','meaning':5,'fluency':4,'facts':4,'note':'Fixture review'}
    req=Request(base+'/api/research/review',data=json.dumps(body).encode(),headers={'X-Dashboard-Token':research_server.TOKEN,'Content-Type':'application/json'},method='POST')
    with urlopen(req) as r:self.assertEqual(json.load(r)['status'],'saved')
    body['meaning']=9;req=Request(base+'/api/research/review',data=json.dumps(body).encode(),headers={'X-Dashboard-Token':research_server.TOKEN,'Content-Type':'application/json'},method='POST')
    with self.assertRaises(HTTPError):urlopen(req)
   finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
