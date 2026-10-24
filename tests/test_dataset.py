import collections
import hashlib
import json
import unittest

from evaluate import load_manifest
from predict import ROOT, prepare_image


class DatasetTests(unittest.TestCase):
    def test_balanced_independent_documents_and_valid_images(self):
        dev, test = load_manifest('dev'), load_manifest('test')
        self.assertEqual(collections.Counter(row['label'] for row in dev),
                         {'handwritten': 5, 'printed': 5})
        self.assertEqual(collections.Counter(row['label'] for row in test),
                         {'handwritten': 20, 'printed': 20})
        rows = dev + test
        self.assertEqual(len(set(row['source_file'] for row in rows)), 50)
        self.assertEqual(len(set(row['sha256'] for row in rows)), 50)
        for row in rows:
            data = (ROOT / 'data' / row['file']).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])
            self.assertEqual(prepare_image(data).mode, 'RGB')
        provenance = json.loads((ROOT / 'data/provenance.json').read_text(encoding='utf-8'))
        self.assertEqual(provenance['license'], 'CC BY 4.0')
        self.assertEqual(len(provenance['selection']), 50)


if __name__ == '__main__':
    unittest.main()
