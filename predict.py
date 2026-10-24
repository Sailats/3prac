"""Подготовка изображения и вызов двух настоящих локальных VLM."""
import io
import os
import re
import time
import warnings
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
from download_models import MODEL_SPECS, models_directory

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ('2B', 'Q3')
LABELS = ('handwritten', 'printed')
PROMPTS = {
    '2B': (
        'Look at the letter shapes, not the meaning or language. '
        'Printed text has regular, repeated letter shapes and consistent spacing, '
        'including printed italic fonts and digital timestamps. '
        'Handwritten text has irregular strokes made by a pen or pencil. '
        'Which type is in the image? Reply with exactly printed or handwritten. '
        'If the image has no text, is unreadable, or has both types, reply unknown.'
    ),
    'Q3': (
        'Is this text handwritten or printed? Answer with only handwritten or printed. '
        'If there is no text or both types are present, answer unknown.'
    ),
}
MAX_IMAGE_BYTES = 15 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 30_000_000


def parse_answer(text: str) -> str:
    cleaned = text.strip().lower().strip(' \t\n\r.\"\'`')
    return cleaned if cleaned in (*LABELS, 'unknown') else 'invalid'


def prepare_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError('Выберите изображение размером до 15 МБ.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP', 'BMP'):
                    raise ValueError('Поддерживаются JPG, PNG, WebP и BMP.')
                image = ImageOps.exif_transpose(source)
                if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
                    rgba = image.convert('RGBA')
                    background = Image.new('RGBA', rgba.size, 'white')
                    image = Image.alpha_composite(background, rgba).convert('RGB')
                else:
                    image = image.convert('RGB')
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                return image.copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError('Не удалось открыть фото. Выберите JPG или PNG до 30 мегапикселей.') from exc


class ModelRunner:
    """Ленивая загрузка. В модель попадают только пиксели и инструкция."""

    def __init__(self, device='auto'):
        self.requested_device = device
        self.loaded = {}

    def _load(self, key):
        if key not in MODEL_KEYS:
            raise ValueError('Неизвестная модель.')
        if key in self.loaded:
            return self.loaded[key], 0.0
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        torch.set_num_threads(min(6, os.cpu_count() or 1))
        device = self.requested_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.float16 if device == 'cuda' else torch.float32
        model_root = models_directory()
        spec = MODEL_SPECS[key]
        folder = model_root / spec['folder']
        if not (folder / 'model.safetensors').is_file():
            raise RuntimeError('Модели не установлены. Запустите Start_Windows.bat при подключённом интернете.')
        started = time.perf_counter()
        # local_files_only и trust_remote_code=False исключают сетевой API и чужой код.
        patch = 28 if key == '2B' else 32
        processor_options = {'max_pixels': 512 * patch * patch}
        if key == 'Q3':
            processor_options['min_pixels'] = 65536
        processor = AutoProcessor.from_pretrained(folder, local_files_only=True,
                                                  trust_remote_code=False, use_fast=False,
                                                  **processor_options)
        model = AutoModelForImageTextToText.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation='sdpa', device_map={'': device},
        ).eval()
        self.loaded[key] = (processor, model, device, dtype)
        return self.loaded[key], time.perf_counter() - started

    def predict(self, image: Image.Image, key: str) -> dict:
        started = time.perf_counter()
        try:
            import torch
            (processor, model, device, dtype), load_seconds = self._load(key)
            messages = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': PROMPTS[key]},
            ]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors='pt').to(device)
            for name, tensor in inputs.items():
                if tensor.is_floating_point():
                    inputs[name] = tensor.to(dtype=dtype)
            if device == 'cuda':
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=12, do_sample=False,
                                           use_cache=True)
            if device == 'cuda':
                torch.cuda.synchronize()
            raw = processor.batch_decode(
                generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
            )[0].strip()
            return {
                'model': key, 'label': parse_answer(raw), 'raw': raw,
                'seconds': round(time.perf_counter() - started, 4),
                'generation_seconds': round(time.perf_counter() - inference_start, 4),
                'load_seconds': round(load_seconds, 4), 'device': device,
                'error': None,
            }
        except Exception as exc:
            return {'model': key, 'label': 'error', 'raw': '',
                    'seconds': round(time.perf_counter() - started, 4),
                    'generation_seconds': None, 'load_seconds': None,
                    'device': None, 'error': f'{type(exc).__name__}: {exc}'}
