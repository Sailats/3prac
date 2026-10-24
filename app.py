"""Простой локальный сервер: показать страницу, сравнить модели, запустить тест."""
import argparse
import base64
import json
import webbrowser

from flask import Flask, abort, request, send_from_directory
from werkzeug.exceptions import HTTPException

from evaluate import load_manifest, run_evaluation
from predict import MAX_IMAGE_BYTES, MODEL_KEYS, ROOT, ModelRunner, prepare_image

app = Flask(__name__, static_folder='web', static_url_path='/web')
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_BYTES * 4 // 3 + 1024
app.config['TRUSTED_HOSTS'] = ['127.0.0.1', 'localhost']
runner = ModelRunner()
samples = load_manifest('dev') + load_manifest('test')


def read_data():
    """Принять JSON только из локального интерфейса."""
    if request.headers.get('Origin') not in (None, request.host_url.rstrip('/')):
        abort(403, 'Запрос разрешён только из приложения.')
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ValueError('Некорректный JSON-запрос.')
    return data


@app.get('/')
def index():
    return send_from_directory(ROOT / 'web', 'index.html')


@app.get('/api/config')
def config():
    return {'models': list(MODEL_KEYS), 'samples': samples, 'max_image_mb': 15}


@app.get('/api/report')
def report():
    path = ROOT / 'results/latest_test.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'metrics': {}, 'predictions': []}


@app.post('/api/predict')
def predict():
    data = read_data()
    if 'sample_id' in data:
        sample = next((row for row in samples if row['id'] == data['sample_id']), None)
        if sample is None:
            raise ValueError('Пример не найден.')
        image_bytes = (ROOT / 'data' / sample['file']).read_bytes()
    else:
        encoded = data.get('image', '')
        if not isinstance(encoded, str):
            raise ValueError('Передайте изображение в base64.')
        image_bytes = base64.b64decode(encoded, validate=True)
    image = prepare_image(image_bytes)
    return {'kind': 'prediction', 'predictions': [runner.predict(image, key) for key in MODEL_KEYS]}


@app.post('/api/evaluate')
def experiment():
    data = read_data()
    return {'kind': 'evaluation', 'report': run_evaluation(data.get('split', 'test'), runner)}


@app.get('/data/images/<path:name>')
def image_file(name):
    if not name.endswith('.jpg'):
        abort(404)
    return send_from_directory(ROOT / 'data/images', name)


@app.get('/results/<path:name>')
def result_file(name):
    if not name.endswith(('.json', '.csv', '.html')):
        abort(404)
    return send_from_directory(ROOT / 'results', name)


@app.errorhandler(ValueError)
@app.errorhandler(RuntimeError)
@app.errorhandler(HTTPException)
def error_response(error):
    if isinstance(error, HTTPException):
        return {'error': error.description}, error.code
    if isinstance(error, ValueError):
        return {'error': str(error)}, 400
    return {'error': str(error)}, 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    runner.requested_device = args.device
    if not args.no_browser:
        webbrowser.open(f'http://127.0.0.1:{args.port}')
    # Один поток: модели вызываются последовательно, отдельный менеджер задач не нужен.
    app.run(host='127.0.0.1', port=args.port, threaded=False, debug=False, use_reloader=False, load_dotenv=False)


if __name__ == '__main__':
    main()
