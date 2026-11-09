"""Local research dashboard: live measured results, checkpoints and human review."""
import argparse,json,os,secrets,subprocess,sys,time,threading
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs
from research import ROOT,aggregate,create_run,read_json,read_rows,write_json
TOKEN=secrets.token_hex(24);PROCESSES={};MUTATION_LOCK=threading.RLock()
def folder_for(name):
    folder=(ROOT/'results'/name).resolve()
    if folder.parent!=(ROOT/'results').resolve() or not folder.name.startswith('research-') or not (folder/'manifest.json').is_file():raise ValueError('Unknown research run')
    return folder

def process_alive(pid):
    if not pid:return False
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes=[wintypes.HANDLE]
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except OSError:return False

def run_status(folder):
    value=read_json(folder/'status.json',{})
    if value.get('status') in ('loading','running') and not process_alive(value.get('pid')):
        value.update(status='stopped',current='Worker exited. Resume from the saved checkpoint.')
    return value

def runs():
    return [dict(name=p.name,**run_status(p)) for p in sorted((ROOT/'results').glob('research-*'),reverse=True) if (p/'manifest.json').is_file()]

def active():
    for r in runs():
        if r.get('status') in ('queued','loading','running'):
            path=folder_for(r['name'])/'status.json'
            if time.time()-path.stat().st_mtime<180:return r
    return None

def spawn(folder):
    if active():raise ValueError('A research run is already active. Stop it before starting another.')
    (folder/'stop.request').unlink(missing_ok=True)
    env=dict(os.environ,HF_HOME=str(ROOT/'.runtime/huggingface'),PYTHONIOENCODING='utf-8')
    handle=(folder/'worker.log').open('a',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-u','research.py','--run',str(folder)],cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    PROCESSES[folder.name]=(process,handle)
    status=read_json(folder/'status.json',{});status.update(status='loading',current='Starting or resuming local model',pid=process.pid);write_json(folder/'status.json',status)

class Handler(BaseHTTPRequestHandler):
    def send(self,data,status=200,mime='application/json',attachment=None):
        body=json.dumps(data,ensure_ascii=False).encode() if mime=='application/json' else data if isinstance(data,bytes) else data.encode()
        self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
        if attachment:self.send_header('Content-Disposition','attachment; filename="'+attachment+'"')
        self.end_headers();self.wfile.write(body)
    def do_GET(self):
        url=urlparse(self.path);q=parse_qs(url.query)
        try:
            if url.path=='/api/research/init':return self.send(dict(token=TOKEN,runs=runs()))
            if url.path=='/api/research/runs':return self.send(runs())
            if url.path.startswith('/api/research/'):
                folder=folder_for(q.get('name',[''])[0])
                if url.path=='/api/research/run':
                    return self.send(dict(name=folder.name,status=run_status(folder),manifest=read_json(folder/'manifest.json'),**aggregate(folder)))
                if url.path=='/api/research/sample':
                    sid=q.get('sample',[''])[0];valid={r['sample_id'] for r in read_rows(folder/'measurements.jsonl')}
                    if sid not in valid:raise ValueError('Unknown saved passage')
                    return self.send(read_json(folder/'samples'/(sid+'.json')))
                if url.path=='/api/research/csv':
                    rows=read_rows(folder/'measurements.jsonl')
                    import csv,io
                    buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rows[0]) if rows else ['sample_id']);w.writeheader();w.writerows(rows)
                    return self.send(buf.getvalue(),mime='text/csv',attachment=folder.name+'.csv')
                if url.path=='/api/research/export':
                    report=aggregate(folder)
                    report['samples']=[read_json(folder/'samples'/(row['sample_id']+'.json')) for row in report['rows']]
                    return self.send(dict(manifest=read_json(folder/'manifest.json'),status=read_json(folder/'status.json'),**report))
                raise ValueError('Unknown research route')
            files={'/':'research.html','/research.js':'research.js','/research.css':'research.css','/style.css':'style.css'}
            if url.path not in files:return self.send({'error':'Not found'},404)
            path=ROOT/'ui'/files[url.path];mime='text/html' if path.suffix=='.html' else 'text/javascript' if path.suffix=='.js' else 'text/css'
            return self.send(path.read_bytes(),mime=mime)
        except (ValueError,KeyError,OSError,TypeError) as e:return self.send({'error':str(e)},400)
    def do_POST(self):
        with MUTATION_LOCK:
            self.post_action()
    def post_action(self):
        try:
            if self.headers.get('X-Dashboard-Token')!=TOKEN:return self.send({'error':'Reload the research page before saving changes.'},403)
            size=int(self.headers.get('Content-Length',0))
            if not 0<size<=100000:raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(size));action=urlparse(self.path).path
            if action=='/api/research/start':
                if active():raise ValueError('A research run is already active.')
                limit=int(data.get('limit',20));seeds=int(data.get('seeds',3))
                if not 1<=limit<=20 or not 1<=seeds<=3:raise ValueError('Choose 1-20 prompts and 1-3 seeds.')
                folder=create_run(limit,seeds,bool(data.get('length_sweep',True)))
                # A freshly queued run must not lock out its own spawn.
                status=read_json(folder/'status.json');status['status']='ready';write_json(folder/'status.json',status)
                spawn(folder);return self.send({'name':folder.name},202)
            folder=folder_for(data.get('name',''))
            if action=='/api/research/stop':
                (folder/'stop.request').touch();return self.send({'status':'stop_requested'})
            if action=='/api/research/resume':spawn(folder);return self.send({'name':folder.name},202)
            if action=='/api/research/review':
                sid=data.get('sample_id');valid={r['sample_id'] for r in read_rows(folder/'measurements.jsonl')}
                if sid not in valid:raise ValueError('Choose a measured sample')
                review={k:int(data[k]) for k in ['meaning','fluency','facts']}
                if any(not 1<=v<=5 for v in review.values()):raise ValueError('Ratings must be 1-5')
                note=data.get('note','')
                if not isinstance(note,str) or len(note)>4000:raise ValueError('Review note is too long')
                review.update(note=note,saved_at=time.time());reviews=read_json(folder/'reviews.json',{});reviews[sid]=review;write_json(folder/'reviews.json',reviews)
                return self.send({'status':'saved'})
            raise ValueError('Unknown action')
        except (ValueError,KeyError,OSError,TypeError) as e:self.send({'error':str(e)},400)
    def log_message(self,*args):pass
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8766);args=p.parse_args()
    print(f'Research dashboard: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
