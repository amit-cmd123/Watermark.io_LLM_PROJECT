"""Loopback-only dashboard with isolated, cancellable research jobs. No extra dependencies."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
JOBS = {}
LOCK = threading.Lock()
TOKEN = secrets.token_hex(24)


def validate(data):
    data.setdefault('scheme', 'v1')
    if data['scheme'] not in ('v1', 'v2-h1', 'v2-h4', 'v2-min4'):
        raise ValueError('Unknown scheme')
    if data.get('action') not in ('generate', 'detect', 'attack', 'smoke', 'experiment', 'benchmark'):
        raise ValueError('Unknown action')
    if data.get('mode') not in ('offline', 'real'):
        raise ValueError('Choose offline or real mode')
    for name, low, high in [('gamma', .01, .99), ('delta', 0, 10), ('threshold', 0, 20),
                             ('temperature', .1, 2), ('top_p', .01, 1),
                             ('max_new_tokens', 16, 512), ('seed', 0, 2147483647), ('limit', 1, 20)]:
        value = float(data[name])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} must be between {low} and {high}')
        if name in ('max_new_tokens', 'seed', 'limit'):
            if value != int(value):
                raise ValueError(f'{name} must be a whole number')
            data[name] = int(value)
        else:
            data[name] = value
    for name in ('model', 'revision', 'key', 'prompt', 'text', 'attack'):
        if not isinstance(data.get(name), str) or len(data[name]) > 50000:
            raise ValueError(f'Invalid {name}')
    if len(data['key'].encode()) < 16:
        raise ValueError('Use a key of at least 16 bytes')
    if data['action'] == 'generate' and not data['prompt'].strip():
        raise ValueError('Enter a prompt')
    if data['action'] in ('detect', 'attack') and not data['text'].strip():
        raise ValueError('Enter text to test')
    if data['action'] == 'attack' and data['attack'] not in ('paraphrase', 'translation'):
        raise ValueError('Unknown attack')
    if data['action'] in ('attack', 'experiment', 'benchmark') and data['mode'] == 'offline':
        raise ValueError('Select real model mode for language attacks or the full experiment')
    return data


def run_path(name):
    base = (ROOT / 'results').resolve()
    path = (base / name).resolve()
    if path.parent != base or not path.is_dir():
        raise ValueError('Unknown run')
    return path


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def run_info(path, detail=False):
    info = {'name': path.name, 'offline': path.name.startswith('synthetic-smoke-'),
            'status': read_json(path / 'status.json', {'status': 'saved'}),
            'summary': read_json(path / 'summary.json', {})}
    if detail:
        try:
            with (path / 'results.csv').open(encoding='utf-8', newline='') as handle:
                info['rows'] = list(csv.DictReader(handle))
        except OSError:
            info['rows'] = []
        info['texts'] = {p.name: p.read_text(encoding='utf-8') for p in path.glob('*.txt')}
    return info


def start_job(data):
    with LOCK:
        if any(j['status'] in ('running', 'cancelling') for j in JOBS.values()):
            raise ValueError('A job is already running. Cancel it or wait for completion.')
        job_id = secrets.token_hex(8)
        directory = ROOT / '.runtime' / 'ui-jobs' / job_id
        directory.mkdir(parents=True)
        request = directory / 'request.json'
        request.write_text(json.dumps(data), encoding='utf-8')
        output = directory / 'output.json'
        action = data['action']
        if action == 'smoke':
            command = [sys.executable, '-u', 'smoke.py']
        elif action == 'benchmark':
            command = [sys.executable, '-u', 'bench_eval.py', '--model', data['model'],
                       '--limit', str(data['limit']), '--seeds', '2', '--tokens', str(min(data['max_new_tokens'], 300)),
                       '--delta', str(data['delta'])]
        elif action == 'experiment':
            command = [sys.executable, '-u', 'evaluate.py', '--model', data['model'], '--revision', data['revision']]
            key_path = directory / 'watermark.key'
            key_path.write_bytes(data['key'].encode())
            command += ['--key-file', str(key_path)]
            for name in ('gamma', 'delta', 'threshold', 'temperature', 'top_p', 'max_new_tokens', 'seed', 'limit'):
                command += ['--' + name.replace('_', '-'), str(data[name])]
        else:
            command = [sys.executable, '-u', 'ui_worker.py', str(request), str(output)]
        log = directory / 'log.txt'
        env = dict(os.environ, HF_HOME=str(ROOT / '.runtime' / 'huggingface'), HF_HUB_OFFLINE='1',
                   MPLCONFIGDIR=str(ROOT / '.runtime' / 'matplotlib'), PYTHONIOENCODING='utf-8')
        handle = log.open('w', encoding='utf-8')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        job = {'id': job_id, 'status': 'running', 'process': process, 'log_path': log, 'output': output}
        JOBS[job_id] = job

        def monitor():
            code = process.wait()
            handle.close()
            with LOCK:
                job['status'] = 'cancelled' if job['status'] == 'cancelling' else 'complete' if code == 0 else 'failed'
                job['result'] = read_json(output)
            # Requests contain a private key; retain only the separate experiment key when needed.
            request.unlink(missing_ok=True)
        threading.Thread(target=monitor, daemon=True).start()
        return {'id': job_id}


class Handler(BaseHTTPRequestHandler):
    def respond(self, value, status=200):
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path == '/api/init':
                return self.respond({'token': TOKEN, 'key': secrets.token_hex(32),
                                     'prompts': read_json(ROOT / 'prompts.json', [])})
            if url.path == '/api/runs':
                return self.respond([run_info(p) for p in sorted((ROOT / 'results').glob('*'), reverse=True) if p.is_dir()])
            if url.path == '/api/run':
                return self.respond(run_info(run_path(parse_qs(url.query)['name'][0]), True))
            if url.path.startswith('/api/jobs/'):
                job = JOBS[url.path.split('/')[-1]]
                return self.respond({'id': job['id'], 'status': job['status'],
                                     'log': job['log_path'].read_text(encoding='utf-8')[-16000:],
                                     'result': job.get('result')})
            files = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                     '/style.css': ('style.css', 'text/css')}
            if url.path not in files:
                return self.respond({'error': 'Not found'}, 404)
            name, content_type = files[url.path]
            body = (ROOT / 'ui' / name).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', content_type + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ValueError, KeyError) as error:
            self.respond({'error': str(error)}, 400)

    def do_POST(self):
        try:
            if self.headers.get('X-Dashboard-Token') != TOKEN:
                return self.respond({'error': 'Reload the dashboard to authorize this request'}, 403)
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 200000:
                raise ValueError('Invalid request size')
            data = json.loads(self.rfile.read(size))
            if self.path == '/api/jobs':
                return self.respond(start_job(validate(data)), 202)
            if self.path == '/api/cancel':
                with LOCK:
                    job = JOBS[data['id']]
                    if job['status'] == 'running':
                        job['status'] = 'cancelling'
                        job['process'].terminate()
                return self.respond({'status': job['status']})
            self.respond({'error': 'Not found'}, 404)
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.respond({'error': str(error)}, 400)

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    print(f'Watermark Lab: http://127.0.0.1:{args.port}', flush=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job in JOBS.values():
            if job['process'].poll() is None:
                job['process'].terminate()
        server.server_close()
