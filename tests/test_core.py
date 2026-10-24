import io
import unittest

from evaluate import calculate_metrics
from predict import parse_answer, prepare_image


class MetricsTests(unittest.TestCase):
    def test_errors_and_abstentions_are_in_denominator(self):
        result = calculate_metrics(
            ['handwritten', 'handwritten', 'printed', 'printed'],
            ['handwritten', 'unknown', 'printed', 'error'],
        )
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['coverage'], .5)
        self.assertEqual(result['classes']['handwritten']['recall'], .5)
        self.assertEqual(result['confusion']['printed']['error'], 1)

    def test_wrong_class_and_invalid_reduce_recall(self):
        result = calculate_metrics(['handwritten', 'printed'], ['printed', 'invalid'])
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['classes']['printed']['precision'], 0)
        self.assertEqual(result['confusion']['handwritten']['printed'], 1)

    def test_empty_metrics_are_defined(self):
        result = calculate_metrics([], [])
        self.assertEqual(result['accuracy'], 0)
        self.assertEqual(result['macro_f1'], 0)

    def test_lengths_must_match(self):
        with self.assertRaises(ValueError):
            calculate_metrics(['printed'], [])


class InputTests(unittest.TestCase):
    def test_answer_must_contain_only_a_label(self):
        self.assertEqual(parse_answer(' Printed. '), 'printed')
        self.assertEqual(parse_answer('"handwritten"'), 'handwritten')
        self.assertEqual(parse_answer('unknown'), 'unknown')
        self.assertEqual(parse_answer('handwritten or printed'), 'invalid')
        self.assertEqual(parse_answer('This is printed'), 'invalid')
        self.assertEqual(parse_answer(''), 'invalid')

    def test_corrupt_image_rejected(self):
        with self.assertRaises(ValueError):
            prepare_image(b'not an image')

    def test_image_is_rgb_and_size_limited(self):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new('RGBA', (2400, 1200), (255, 255, 255, 128)).save(buffer, 'PNG')
        result = prepare_image(buffer.getvalue())
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.size, (1024, 512))


if __name__ == '__main__':
    unittest.main()
