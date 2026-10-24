import base64
import io
import unittest
from unittest.mock import patch

from PIL import Image
from app import app


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_index_and_config_work(self):
        with self.client.get('/') as response:
            self.assertEqual(response.status_code, 200)
        config = self.client.get('/api/config').get_json()
        self.assertEqual(len(config['samples']), 50)
        self.assertEqual(config['models'], ['2B', 'Q3'])

    def test_predict_returns_both_results_directly(self):
        buffer = io.BytesIO()
        Image.new('RGB', (80, 40), 'white').save(buffer, 'PNG')
        data = {'image': base64.b64encode(buffer.getvalue()).decode()}
        with patch('app.runner.predict', side_effect=lambda image, key:
                   {'model': key, 'label': 'printed', 'raw': 'printed'}) as predict:
            response = self.client.post('/api/predict', json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['kind'], 'prediction')
        self.assertEqual([r['model'] for r in response.json['predictions']], ['2B', 'Q3'])
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(predict.call_args.args[0].mode, 'RGB')

    def test_experiment_returns_report_directly(self):
        with patch('app.run_evaluation', return_value={'metrics': {'Q3': {'accuracy': 1}}}) as evaluate:
            response = self.client.post('/api/evaluate', json={'split': 'test'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['kind'], 'evaluation')
        self.assertEqual(response.json['report']['metrics']['Q3']['accuracy'], 1)
        self.assertEqual(evaluate.call_args.args[0], 'test')

    def test_corrupt_upload_has_actionable_error(self):
        response = self.client.post('/api/predict', json={'image': 'bm90IGFuIGltYWdl'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('фото', response.json['error'])

    def test_external_origin_cannot_start_inference(self):
        response = self.client.post('/api/evaluate', json={}, headers={'Origin': 'https://example.org'})
        self.assertEqual(response.status_code, 403)

    def test_same_origin_is_allowed_and_foreign_host_is_rejected(self):
        with patch('app.run_evaluation', return_value={'metrics': {}}):
            response = self.client.post('/api/evaluate', json={}, headers={'Origin': 'http://localhost'})
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/api/config', headers={'Host': 'foreign.example'})
        self.assertEqual(response.status_code, 400)

    def test_path_escape_and_private_files_are_rejected(self):
        for url in ('/web/../predict.py', '/data/images/../../predict.py', '/results/../predict.py'):
            self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_malformed_body_and_sample_id_are_rejected(self):
        for body in ([], {'sample_id': []}, {'sample_id': 'absent'}, {'image': 123}):
            response = self.client.post('/api/predict', json=body)
            self.assertEqual(response.status_code, 400, body)


if __name__ == '__main__':
    unittest.main()
