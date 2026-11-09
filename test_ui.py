"""Exercise local API validation, file boundaries and the offline research workflow."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import ui_server


def fixture():
    return dict(action='generate', mode='offline', model='Qwen/Qwen2.5-0.5B-Instruct',
                revision='main', key='test-watermark-key-at-least-16', prompt='Test prompt',
                text='', attack='paraphrase', gamma=.25, delta=2, threshold=4,
                temperature=.8, top_p=.95, max_new_tokens=32, seed=1234, limit=1)


class DashboardTests(unittest.TestCase):
    def test_validation_and_boundaries(self):
        self.assertEqual(ui_server.validate(fixture())['max_new_tokens'], 32)
        for patch in ({'gamma': float('nan')}, {'max_new_tokens': 20.5},
                      {'action': 'attack'}, {'key': 'short'}, {'action': 'unknown'}):
            with self.assertRaises(ValueError):
                ui_server.validate({**fixture(), **patch})
        with self.assertRaises(ValueError):
            ui_server.run_path('../.runtime')

    def test_http_and_archive(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), ui_server.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base + '/') as response:
                self.assertIn(b'Generate & compare', response.read())
            with urlopen(base + '/api/runs') as response:
                runs = json.load(response)
                self.assertIsInstance(runs, list)
            with self.assertRaises(HTTPError) as raised:
                urlopen(Request(base + '/api/jobs', data=b'{}', method='POST'))
            self.assertEqual(raised.exception.code, 403)
            with self.assertRaises(HTTPError) as raised:
                urlopen(base + '/api/run?name=../.runtime')
            self.assertEqual(raised.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()

    def test_offline_pair_and_detector_agree(self):
        from ui_worker import execute
        result = execute(fixture())
        self.assertTrue(result['offline'])
        self.assertEqual(set(result['samples']), {'Control', 'Watermarked'})
        for sample in result['samples'].values():
            detected = execute({**fixture(), 'action': 'detect', 'text': sample['text']})
            self.assertEqual(sample['stats'], detected['samples']['Detected input']['stats'])
            self.assertEqual(len(sample['tokens']), len(sample['ids']))
        self.assertGreater(result['samples']['Watermarked']['stats']['green_ratio'],
                           result['samples']['Control']['stats']['green_ratio'])


if __name__ == '__main__':
    unittest.main()
