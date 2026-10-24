import unittest

from quantize_models import quantization_config


class QuantizationTests(unittest.TestCase):
    def test_vision_and_output_head_stay_float_while_decoder_is_compressed(self):
        import torch
        import bitsandbytes as bnb
        from transformers.integrations.bitsandbytes import replace_with_bnb_linear

        model = torch.nn.Module()
        model.visual = torch.nn.Sequential(torch.nn.Linear(4, 4))
        model.decoder = torch.nn.Linear(4, 4)
        model.lm_head = torch.nn.Linear(4, 8)
        config = quantization_config()
        converted = replace_with_bnb_linear(model,
            modules_to_not_convert=config.llm_int8_skip_modules, quantization_config=config)
        self.assertIs(type(converted.visual[0]), torch.nn.Linear)
        self.assertIs(type(converted.lm_head), torch.nn.Linear)
        self.assertIsInstance(converted.decoder, bnb.nn.Linear4bit)


if __name__ == '__main__':
    unittest.main()
