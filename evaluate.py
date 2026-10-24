"""Проверка выборки, честные метрики и сохранение протокола эксперимента."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

from predict import LABELS, MODEL_KEYS, PROMPTS, ROOT, ModelRunner, prepare_image
from support.download_models import MODEL_SPECS

OUTCOMES = (*LABELS, 'unknown', 'invalid', 'error')


def calculate_metrics(truth: list, predictions: list) -> dict:
    if len(truth) != len(predictions):
        raise ValueError('Число меток и ответов должно совпадать.')
    matrix = {actual: {pred: 0 for pred in OUTCOMES} for actual in LABELS}
    for actual, predicted in zip(truth, predictions):
        if actual not in LABELS or predicted not in OUTCOMES:
            raise ValueError('Недопустимый класс.')
        matrix[actual][predicted] += 1
    classes = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in LABELS if other != label)
        fn = sum(matrix[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        classes[label] = dict(precision=precision, recall=recall, f1=f1, support=tp + fn)
    total = len(truth)
    return {
        'n': total,
        'accuracy': sum(matrix[label][label] for label in LABELS) / total if total else 0,
        'coverage': sum(pred in LABELS for pred in predictions) / total if total else 0,
        'macro_f1': sum(c['f1'] for c in classes.values()) / 2,
        'classes': classes, 'confusion': matrix,
        'unknown': predictions.count('unknown'), 'invalid': predictions.count('invalid'),
        'errors': predictions.count('error'),
    }


def load_manifest(split='test') -> list:
    if split not in ('dev', 'test'):
        raise ValueError('Выборка должна быть dev или test.')
    with (ROOT / 'data' / 'manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row['split'] == split]


def run_evaluation(split='test', runner=None) -> dict:
    runner = runner or ModelRunner()
    samples = load_manifest(split)
    rows = []
    # Прогрев на dev не входит во время теста. Инструкция при этом не изменяется.
    warmup = {}
    warm_image = prepare_image((ROOT / 'data' / load_manifest('dev')[0]['file']).read_bytes())
    for key in MODEL_KEYS:
        print(f'Warmup: {MODEL_SPECS[key]["name"]}', flush=True)
        warmup[key] = runner.predict(warm_image, key)
        if warmup[key]['label'] == 'error':
            raise RuntimeError(warmup[key]['error'])
    for index, sample in enumerate(samples):
        image = prepare_image((ROOT / 'data' / sample['file']).read_bytes())
        # Чередование порядка уменьшает влияние постоянного первого места.
        order = MODEL_KEYS if index % 2 == 0 else MODEL_KEYS[::-1]
        for key in order:
            result = runner.predict(image, key)
            rows.append({'id': sample['id'], 'file': sample['file'],
                         'truth': sample['label'], **result})
            print(f'{len(rows)}/{len(samples) * 2}: {sample["id"]} {key}', flush=True)
    summaries = {}
    for key in MODEL_KEYS:
        subset = [row for row in rows if row['model'] == key]
        metrics = calculate_metrics([r['truth'] for r in subset], [r['label'] for r in subset])
        metrics['mean_seconds'] = statistics.mean(r['seconds'] for r in subset) if subset else 0
        metrics['median_seconds'] = statistics.median(r['seconds'] for r in subset) if subset else 0
        summaries[key] = metrics
    import torch
    import transformers
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'split': split,
        'application_version': '3.0-simple',
        'prompts': PROMPTS,
        'prompts_sha256': hashlib.sha256(json.dumps(PROMPTS, sort_keys=True).encode()).hexdigest(),
        'manifest_sha256': hashlib.sha256((ROOT / 'data/manifest.csv').read_bytes()).hexdigest(),
        'settings': {'max_side': 1024, 'max_new_tokens': 12, 'do_sample': False,
                     'warmup': 'one dev image per model', 'order': 'alternating',
                     'quantization': 'decoder NF4 double; vision FP16; compute FP16 on CUDA',
                     'visual_token_limit': 512,
                     'evaluation_note': 'Repeat comparison on the existing 40 images after replacing SmolVLM; '
                                        'the test set has been used before. Per-model prompts were '
                                        'selected on dev; labels and images unchanged. This compares '
                                        'configured pipelines, not models with identical prompt wording.'},
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'transformers': transformers.__version__, 'platform': platform.platform(),
                        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        'models': {key: {'name': spec['name'], 'repo': spec['repo'], 'revision': spec['revision']}
                   for key, spec in MODEL_SPECS.items()},
        'warmup': warmup, 'metrics': summaries, 'predictions': rows,
    }
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    stem = f'{split}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}'
    report['report_file'] = f'{stem}.json'
    (output / f'{stem}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / f'latest_{split}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / f'{stem}.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev', 'test'], default='test')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    report = run_evaluation(args.split, ModelRunner(args.device))
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))
