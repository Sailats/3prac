"""Общедоступные веса: прямое HTTPS-скачивание, без токена и API вывода."""
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ('config.json', 'generation_config.json', 'preprocessor_config.json',
         'processor_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
         'chat_template.json', 'model.safetensors', 'README.md')
MODEL_SPECS = {
    '2B': {
        'repo': 'Qwen/Qwen2-VL-2B-Instruct',
        'folder': 'Qwen2-VL-2B-NF4',
        'source_folder': 'Qwen2-VL-2B-Instruct',
        'name': 'Qwen2-VL 2B',
        'revision': '895c3a49bc3fa70a340399125c650a463535e71c',
        'files': ('LICENSE', 'README.md', 'chat_template.json', 'config.json',
                  'generation_config.json', 'merges.txt', 'model.safetensors.index.json',
                  'preprocessor_config.json', 'tokenizer.json', 'tokenizer_config.json',
                  'vocab.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'),
        'weights': {
            'model-00001-of-00002.safetensors': {'size': 3988609112, 'sha256': '994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9'},
            'model-00002-of-00002.safetensors': {'size': 429441656, 'sha256': '92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833'},
        },
    },
    'Q3': {
        'repo': 'Qwen/Qwen3-VL-2B-Instruct',
        'folder': 'Qwen3-VL-2B-NF4',
        'source_folder': 'Qwen3-VL-2B-Instruct',
        'name': 'Qwen3-VL 2B',
        'revision': '89644892e4d85e24eaac8bacfd4f463576704203',
        'files': ('README.md', 'chat_template.json', 'config.json', 'generation_config.json',
                  'merges.txt', 'model.safetensors', 'preprocessor_config.json',
                  'tokenizer.json', 'tokenizer_config.json', 'video_preprocessor_config.json',
                  'vocab.json'),
        'weights': {'model.safetensors': {'size': 4255140312, 'sha256': '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'}},
    },
}
QUANTIZATION = 'nf4-double-fp16-vision'


def models_directory(project_root=ROOT):
    override = os.environ.get('TEXT_MODELS_DIR')
    if override:
        return Path(override)
    local = project_root / '.local-models.txt'
    if local.is_file():
        path = Path(local.read_text(encoding='utf-8-sig').strip())
        if path.is_dir():
            return path
    return project_root / 'models'


def local_model_ready(folder, spec):
    try:
        manifest = json.loads((folder / 'download-manifest.json').read_text(encoding='utf-8'))
        if (not isinstance(manifest, dict) or manifest.get('repo') != spec['repo'] or manifest.get('revision') != spec['revision']
                or manifest.get('quantization') != QUANTIZATION):
            return False
        files = manifest.get('files', {})
        required = {'config.json', 'preprocessor_config.json', 'model.safetensors'}
        slow_tokenizer = {'tokenizer_config.json', 'vocab.json', 'merges.txt'}
        if (not isinstance(files, dict) or not required <= set(files)
                or not ('tokenizer.json' in files or slow_tokenizer <= set(files))):
            return False
        return all(isinstance(name, str) and Path(name).name == name
                   and isinstance(record, dict) and (folder / name).is_file()
                   and file_record(folder / name) == record
                   for name, record in files.items())
    except (ValueError, OSError, TypeError):
        return False


def file_record(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'size': path.stat().st_size, 'sha256': digest}


def ensure_file(url, destination, expected=None):
    if expected and destination.is_file() and file_record(destination) == expected:
        return
    fetch(url, destination)
    if expected and file_record(destination) != expected:
        raise RuntimeError(f'Checksum mismatch: {destination.name}. Run the installer again.')


def fetch(url, destination):
    part = destination.with_suffix(destination.suffix + '.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'UniversityTextClassifier/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            part.replace(destination)
            return
        except Exception:
            if attempt == 2:
                raise


def download_all():
    root = models_directory()
    root.mkdir(parents=True, exist_ok=True)
    for key, spec in MODEL_SPECS.items():
        repo, pinned = spec['repo'], spec['revision']
        folder = root / spec['folder']
        folder.mkdir(exist_ok=True)
        if local_model_ready(folder, spec):
            print(f'{spec["name"]}: local NF4 files verified', flush=True)
            continue
        source = root / '.source-cache' / spec['source_folder']
        source.mkdir(parents=True, exist_ok=True)
        folder = source
        lock = folder / 'download-manifest.json'
        previous = {}
        if lock.exists():
            try:
                manifest = json.loads(lock.read_text(encoding='utf-8'))
                if (isinstance(manifest, dict) and manifest.get('revision') == pinned
                        and manifest.get('repo') == repo and isinstance(manifest.get('files'), dict)):
                    previous = manifest['files']
            except (ValueError, OSError):
                pass
        if not pinned:
            with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}', timeout=30) as response:
                pinned = json.load(response)['sha']
        print(f'Downloading {repo} @ {pinned}', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs = []
            for name in spec['files']:
                target = folder / name
                expected = spec['weights'].get(name) or previous.get(name)
                jobs.append(pool.submit(ensure_file, f'https://huggingface.co/{repo}/resolve/{pinned}/{name}', target, expected))
            for job in jobs:
                job.result()
        manifest = {'repo': repo, 'revision': pinned, 'license': 'Apache-2.0', 'files': {}}
        for name in spec['files']:
            path = folder / name
            manifest['files'][name] = file_record(path)
        lock.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'{key}: source downloaded and hashed', flush=True)
        from support.quantize_models import quantize_model
        quantize_model(source, root / spec['folder'], spec)


if __name__ == '__main__':
    download_all()
