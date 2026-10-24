"""Воспроизводимая выборка реальных фрагментов rukopys, CC BY 4.0.

Запускать из корня проекта. Не требуется для обычного использования:
готовые изображения и их происхождение уже включены в архив.
"""
import concurrent.futures
import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPO = 'UkrainianCatholicUniversity/rukopys'


def get_bytes(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except Exception:
            if attempt == 2:
                raise


def overlap_ratio(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    area = max(0, right - left) * max(0, bottom - top)
    return area / max(1, (first[2] - first[0]) * (first[3] - first[1]))


def main():
    metadata = get_bytes(f'https://huggingface.co/api/datasets/{REPO}')
    revision = json.loads(metadata)['sha']
    raw = get_bytes(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/train/metadata.jsonl')
    documents = [json.loads(line) for line in raw.decode().splitlines()]
    random.Random(20261024).shuffle(documents)
    excluded_path = ROOT / 'data/selection-exclusions.json'
    excluded = json.loads(excluded_path.read_text(encoding='utf-8')) if excluded_path.exists() else []
    selection, used = [], {entry['source_file'] for entry in excluded}
    for label in ('printed', 'handwritten'):
        for document in documents:
            if document['file_name'] in used:
                continue
            candidates = []
            for region in document['regions']:
                box = region['bbox']
                width, height = box[2] - box[0], box[3] - box[1]
                if (region['type'] != label or len(region.get('text', '')) < 12
                        or width < 150 or height < 30 or width / height > 40):
                    continue
                others = [other for other in document['regions']
                          if other['type'] in ('printed', 'handwritten') and other['type'] != label]
                if any(overlap_ratio(box, other['bbox']) > .02 for other in others):
                    continue
                candidates.append(region)
            if not candidates:
                continue
            # Выбор по геометрии и исходной аннотации, без просмотра ответов моделей.
            region = max(candidates, key=lambda x: min(len(x['text']), 180))
            number = sum(x['label'] == label for x in selection) + 1
            selection.append({'id': f'{label[0]}{number:02d}', 'label': label,
                              'split': 'dev' if number <= 5 else 'test',
                              'source_file': document['file_name'], 'bbox': region['bbox'],
                              'annotation_source': document.get('annotation_source'),
                              'source_text': region.get('text', '')})
            used.add(document['file_name'])
            if number == 25:
                break
    if len(selection) != 50:
        raise RuntimeError(f'Недостаточно независимых документов: {len(selection)}')
    image_dir = ROOT / 'data/images'
    image_dir.mkdir(parents=True, exist_ok=True)

    def download_sample(sample):
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/train/{sample["source_file"]}'
        data = get_bytes(url)
        with Image.open(io.BytesIO(data)) as original:
            crop = original.convert('RGB').crop(sample['bbox'])
            crop.thumbnail((1600, 1200), Image.Resampling.LANCZOS)
            # Небольшое одинаковое белое поле для обоих классов.
            framed = Image.new('RGB', (crop.width + 24, crop.height + 24), 'white')
            framed.paste(crop, (12, 12))
            filename = f'{sample["id"]}.jpg'
            framed.save(image_dir / filename, quality=94)
        sample['file'] = f'images/{filename}'
        sample['source_url'] = url
        sample['sha256'] = hashlib.sha256((image_dir / filename).read_bytes()).hexdigest()
        print(sample['id'], sample['split'], sample['bbox'], flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(download_sample, selection))
    fields = ('id', 'file', 'label', 'split', 'sha256', 'source_file')
    with (ROOT / 'data/manifest.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(selection)
    provenance = {'dataset': REPO, 'revision': revision, 'license': 'CC BY 4.0',
                  'license_url': 'https://creativecommons.org/licenses/by/4.0/',
                  'authors': 'Ukrainian Catholic University, rukopys dataset contributors',
                  'seed': 20261024, 'selection': selection, 'excluded_after_visual_review': excluded,
                  'limitations': 'Small convenience sample; source handwriting writers are not identified; '
                                 'different documents can belong to one writer. Annotation may contain mistakes.'}
    (ROOT / 'data/provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    for label in ('handwritten', 'printed'):
        sheet = Image.new('RGB', (1500, 1250), '#e9edf2')
        draw = ImageDraw.Draw(sheet)
        for index, sample in enumerate(x for x in selection if x['label'] == label):
            x, y = index % 5 * 300, index // 5 * 250
            with Image.open(ROOT / 'data' / sample['file']) as image:
                image.thumbnail((280, 210))
                sheet.paste(image, (x + 10, y + 28))
            draw.text((x + 10, y + 6), f'{sample["id"]} / {sample["split"]}', fill='black')
        sheet.save(ROOT / 'data' / f'contact_{label}.jpg', quality=90)


if __name__ == '__main__':
    main()
