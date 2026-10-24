"""Локальный браузерный интерфейс без веб-фреймворка и внешнего сервера."""
import argparse
import base64
import binascii
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from evaluate import load_manifest, run_evaluation
from predict import MAX_IMAGE_BYTES, MODEL_KEYS, ROOT, ModelRunner, prepare_image
from download_models import MODEL_SPECS


def safe_path(base, relative):
    base = base.resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError('Путь за пределами каталога.')
    return path


class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {'status': 'idle', 'done': 0, 'total': 0, 'message': '', 'result': None}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def start(self, operation):
        with self.lock:
            if self.state['status'] == 'running':
                raise RuntimeError('Уже идёт обработка. Дождитесь результата.')
            self.state = {'status': 'running', 'done': 0, 'total': 2,
                          'message': 'Подготовка моделей…', 'result': None}

        def progress(done, total, message):
            with self.lock:
                self.state.update(done=done, total=total, message=message)

        def work():
            try:
                result = operation(progress)
                with self.lock:
                    self.state.update(status='done', result=result, message='Готово')
            except Exception as exc:
                with self.lock:
                    self.state.update(status='error', message=str(exc))

        threading.Thread(target=work, daemon=True).start()


def create_server(port=8765, device='auto'):
    jobs = JobManager()
    runner = ModelRunner(device)
    samples = load_manifest('dev') + load_manifest('test')
    by_id = {sample['id']: sample for sample in samples}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_body(self, body, status=200, content_type='application/json; charset=utf-8'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; img-src 'self' blob: data:; "
                             "style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == '/api/status':
                return self.send_body(jobs.snapshot())
            if path == '/api/config':
                return self.send_body({'models': list(MODEL_KEYS), 'samples': samples,
                                       'device': device, 'max_image_mb': 15})
            if path == '/api/report':
                report_path = ROOT / 'results/latest_test.json'
                if report_path.exists():
                    return self.send_body(json.loads(report_path.read_text(encoding='utf-8')))
                return self.send_body({'metrics': {}, 'predictions': []})
            try:
                if path in ('/', '/index.html'):
                    file = ROOT / 'web/index.html'
                elif path.startswith('/web/'):
                    file = safe_path(ROOT / 'web', path[5:])
                elif path.startswith('/data/images/'):
                    file = safe_path(ROOT / 'data/images', path[13:])
                    if file.suffix.lower() != '.jpg':
                        raise ValueError()
                elif path.startswith('/results/'):
                    file = safe_path(ROOT / 'results', path[9:])
                    if file.suffix not in ('.json', '.csv', '.html'):
                        raise ValueError()
                else:
                    raise ValueError()
                if not file.is_file():
                    raise ValueError()
                content_type = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
                if file.suffix in ('.html', '.css', '.js'):
                    content_type += '; charset=utf-8'
                self.send_body(file.read_bytes(), content_type=content_type)
            except (ValueError, OSError):
                self.send_body({'error': 'Файл не найден.'}, 404)

        def do_POST(self):
            origin = self.headers.get('Origin')
            allowed = {f'http://127.0.0.1:{self.server.server_port}',
                       f'http://localhost:{self.server.server_port}'}
            if origin and origin not in allowed:
                return self.send_body({'error': 'Запрос разрешён только из приложения.'}, 403)
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send_body({'error': 'Требуется JSON.'}, 400)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > MAX_IMAGE_BYTES * 4 // 3 + 1024:
                    raise ValueError('Выберите изображение размером до 15 МБ.')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('Некорректный запрос.')
                path = urlsplit(self.path).path
                if path == '/api/predict':
                    if 'sample_id' in body:
                        sample = by_id.get(body['sample_id'])
                        if sample is None:
                            raise ValueError('Пример не найден.')
                        data = (ROOT / 'data' / sample['file']).read_bytes()
                    else:
                        data = base64.b64decode(body.get('image', ''), validate=True)
                    image = prepare_image(data)

                    def operation(progress):
                        results = []
                        for index, key in enumerate(MODEL_KEYS):
                            progress(index, 2, f'Распознаёт {MODEL_SPECS[key]["name"]}…')
                            results.append(runner.predict(image, key))
                        return {'kind': 'prediction', 'predictions': results}

                    jobs.start(operation)
                elif path == '/api/evaluate':
                    split = body.get('split', 'test')
                    if split not in ('dev', 'test'):
                        raise ValueError('Выберите dev или test.')

                    def operation(progress):
                        return {'kind': 'evaluation', 'report': run_evaluation(split, runner, progress)}

                    jobs.start(operation)
                else:
                    return self.send_body({'error': 'Адрес не найден.'}, 404)
                self.send_body({'status': 'running'}, 202)
            except RuntimeError as exc:
                self.send_body({'error': str(exc)}, 409)
            except (ValueError, TypeError, KeyError, binascii.Error, OSError) as exc:
                self.send_body({'error': str(exc) or 'Некорректное изображение.'}, 400)

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    server.jobs = jobs
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = create_server(args.port, args.device)
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Open {url} in your browser. Close this window to stop.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
