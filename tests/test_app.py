import json
import threading
import time
import unittest
import urllib.error
import urllib.request

from app import JobManager, create_server, safe_path
from predict import ROOT


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_index_and_config_work(self):
        with urllib.request.urlopen(self.url + '/') as response:
            page = response.read().decode()
            self.assertIn('Рукописный', page)
            self.assertIn('Qwen3-VL', page)
            self.assertNotIn('SmolVLM', page)
        with urllib.request.urlopen(self.url + '/api/config') as response:
            config = json.load(response)
        self.assertEqual(len(config['samples']), 50)
        self.assertEqual(config['models'], ['2B', 'Q3'])

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            safe_path(ROOT / 'web', '../predict.py')
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.url + '/data/../predict.py')
        self.assertEqual(caught.exception.code, 404)

    def test_corrupt_upload_has_actionable_error(self):
        request = urllib.request.Request(self.url + '/api/predict',
            data=json.dumps({'image': 'bm90IGFuIGltYWdl'}).encode(),
            headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn('фото', caught.exception.read().decode())

    def test_external_origin_cannot_start_job(self):
        request = urllib.request.Request(self.url + '/api/evaluate', data=b'{}',
            headers={'Content-Type': 'application/json', 'Origin': 'https://example.org'})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        self.assertEqual(caught.exception.code, 403)


class JobTests(unittest.TestCase):
    def test_second_job_rejected_and_first_result_preserved(self):
        manager = JobManager()
        finish = threading.Event()

        def operation(progress):
            finish.wait(2)
            return {'value': 42}

        manager.start(operation)
        with self.assertRaises(RuntimeError):
            manager.start(operation)
        finish.set()
        for _ in range(100):
            if manager.snapshot()['status'] == 'done':
                break
            time.sleep(.01)
        self.assertEqual(manager.snapshot()['result'], {'value': 42})


if __name__ == '__main__':
    unittest.main()
