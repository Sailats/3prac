import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from download_models import file_record, local_model_ready, models_directory


class LocalModelTests(unittest.TestCase):
    def test_portable_path_and_existing_local_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            actual = project / 'elsewhere'
            actual.mkdir()
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(models_directory(project), project / 'models')
                (project / '.local-models.txt').write_text(str(actual), encoding='utf-8')
                self.assertEqual(models_directory(project), actual)
                with patch.dict(os.environ, {'TEXT_MODELS_DIR': str(project / 'env')}):
                    self.assertEqual(models_directory(project), project / 'env')

    def test_incomplete_manifest_and_same_size_corruption_are_rejected(self):
        spec = {'repo': 'official/test', 'revision': 'pinned-revision'}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            files = {}
            for name in ('config.json', 'tokenizer.json', 'preprocessor_config.json', 'model.safetensors'):
                (folder / name).write_bytes(b'valid bytes')
                files[name] = file_record(folder / name)
            manifest = {'repo': spec['repo'], 'revision': spec['revision'],
                        'quantization': 'nf4-double-fp16-vision', 'files': files}
            path = folder / 'download-manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertTrue(local_model_ready(folder, spec))
            (folder / 'model.safetensors').write_bytes(b'wrong bytes')
            self.assertFalse(local_model_ready(folder, spec))
            manifest['files'].pop('model.safetensors')
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertFalse(local_model_ready(folder, spec))

    def test_saved_slow_tokenizer_is_accepted_and_must_be_complete(self):
        spec = {'repo': 'official/test', 'revision': 'pinned-revision'}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            files = {}
            for name in ('config.json', 'preprocessor_config.json', 'model.safetensors',
                         'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                (folder / name).write_bytes(b'valid bytes')
                files[name] = file_record(folder / name)
            manifest = {'repo': spec['repo'], 'revision': spec['revision'],
                        'quantization': 'nf4-double-fp16-vision', 'files': files}
            path = folder / 'download-manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertTrue(local_model_ready(folder, spec))
            for name in ('tokenizer_config.json', 'vocab.json', 'merges.txt'):
                incomplete = {**manifest, 'files': {k: v for k, v in files.items() if k != name}}
                path.write_text(json.dumps(incomplete), encoding='utf-8')
                self.assertFalse(local_model_ready(folder, spec), name)
            path.write_text(json.dumps(manifest), encoding='utf-8')
            (folder / 'model.safetensors').write_bytes(b'wrong bytes')
            self.assertFalse(local_model_ready(folder, spec))

    def test_malformed_manifest_structures_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            path = folder / 'download-manifest.json'
            spec = {'repo': 'official/test', 'revision': 'pinned-revision'}
            for value in (None, [], 'text', 42, {'repo': spec['repo'],
                          'revision': spec['revision'], 'quantization': 'nf4-double-fp16-vision',
                          'files': ['config.json']}):
                path.write_text(json.dumps(value), encoding='utf-8')
                with self.subTest(value=value):
                    self.assertFalse(local_model_ready(folder, spec))


if __name__ == '__main__':
    unittest.main()
