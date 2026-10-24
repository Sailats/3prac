"""Сжатие готовых VLM до NF4 без обучения и доступа к тестовым данным."""
import argparse
import gc
import json
from pathlib import Path

from download_models import MODEL_SPECS, QUANTIZATION, file_record, models_directory


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
        # Явный skip-list заменяет штатный список Transformers, поэтому
        # необходимо сохранить также output head с его tied embeddings.
        llm_int8_skip_modules=['visual', 'lm_head'],
    )


def quantize_model(source: Path, target: Path, spec: dict):
    import torch
    import bitsandbytes as bnb
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    target.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Preparing NF4: {spec["name"]} on {device}', flush=True)
    config = quantization_config()
    model = AutoModelForImageTextToText.from_pretrained(
        source, local_files_only=True, trust_remote_code=False,
        dtype=torch.float16, attn_implementation='sdpa',
        device_map={'': device}, quantization_config=config,
    ).eval()
    processor = AutoProcessor.from_pretrained(source, local_files_only=True,
                                               trust_remote_code=False, use_fast=False)
    model.save_pretrained(target, safe_serialization=True, max_shard_size='10GB')
    processor.save_pretrained(target)
    for name in ('README.md', 'LICENSE'):
        if (source / name).is_file():
            (target / name).write_bytes((source / name).read_bytes())
    metadata = {
        'repo': spec['repo'], 'revision': spec['revision'], 'license': 'Apache-2.0',
        'quantization': QUANTIZATION,
        'method': 'Decoder linear layers: NF4 with double quantization. Visual modules: FP16. No training.',
        'versions': {'torch': torch.__version__, 'transformers': transformers.__version__,
                     'bitsandbytes': bnb.__version__},
        'source_weights': spec['weights'],
        'files': {path.name: file_record(path) for path in target.iterdir()
                  if path.is_file() and path.name != 'download-manifest.json'},
    }
    (target / 'download-manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    size = sum(x['size'] for x in metadata['files'].values()) / 1024**3
    print(f'NF4 ready: {spec["name"]}, {size:.2f} GiB', flush=True)
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODEL_SPECS), required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    spec = MODEL_SPECS[args.model]
    quantize_model(args.source, models_directory() / spec['folder'], spec)
