import hashlib
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from download_models import ensure_file


class DownloadTests(unittest.TestCase):
    def test_corrupt_existing_file_is_repaired_and_valid_file_works_offline(self):
        payload = b'real model bytes for transport test'

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        expected = {'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'model.bin'
                path.write_bytes(b'xx')
                url = f'http://127.0.0.1:{server.server_port}/model'
                ensure_file(url, path, expected)
                self.assertEqual(path.read_bytes(), payload)
                path.write_bytes(b'x' * len(payload))
                ensure_file(url, path, expected)
                self.assertEqual(path.read_bytes(), payload)
                # Рабочий локальный файл не требует доступного HTTP-сервера.
                ensure_file('http://127.0.0.1:1/unavailable', path, expected)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
