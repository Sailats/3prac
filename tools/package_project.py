"""Два ZIP: исходники для сдачи и полностью готовая среда Windows."""
import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    for path in ROOT.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in ('.git', '.superpowers', 'models', '.runtime', '.venv', '__pycache__', 'superpowers')
               or part.endswith('.parts') for part in relative.parts):
            continue
        if path.name in ('.local-python.txt', '.local-models.txt', 'progress.md', 'interface-initial.png'):
            continue
        if path.suffix in ('.pyc', '.part', '.zip'):
            continue
        yield path, f'TextType/{relative.as_posix()}'


def write_archive(output, files, runtime_archive=None):
    files = list(files)
    started = time.perf_counter()
    if runtime_archive:
        shutil.copyfile(runtime_archive, output)
    with zipfile.ZipFile(output, 'a' if runtime_archive else 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=1, allowZip64=True) as archive:
        for index, (path, name) in enumerate(files, 1):
            # Веса уже представлены плотными числами: хранение без повторного
            # сжатия сокращает время подготовки полного архива на обычном ПК.
            archive.write(path, name, compress_type=zipfile.ZIP_STORED
                          if path.suffix == '.safetensors' else zipfile.ZIP_DEFLATED)
            if path.stat().st_size > 100_000_000 or index % 2000 == 0:
                print(f'{output.name}: {index}/{len(files)} files ({name})', flush=True)
    print(f'{output.name}: {output.stat().st_size / 1024**3:.3f} GiB, '
          f'{time.perf_counter() - started:.0f} seconds', flush=True)
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f'Archive CRC failed: {bad}')
        names = set(archive.namelist())
        required = {'TextType/Start_Windows.bat', 'TextType/predict.py',
                    'TextType/data/manifest.csv', 'TextType/results/latest_test.json',
                    'TextType/docs/Отчет.html', 'TextType/ИНСТРУКЦИЯ.txt'}
        if not required <= names:
            raise RuntimeError('Required files missing from archive')
    with output.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    output.with_suffix('.zip.sha256.txt').write_text(f'{digest}  {output.name}\n', encoding='ascii')
    return {'file': output.name, 'size': output.stat().st_size, 'sha256': digest,
            'files': len(names), 'crc_verified': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-archive', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = list(source_files())
    records = [write_archive(args.output / 'TextType_Source.zip', sources)]
    files = list(sources)
    from sys import path as search_path
    search_path.insert(0, str(ROOT))
    from support.download_models import MODEL_SPECS, models_directory
    for spec in MODEL_SPECS.values():
        model_root = models_directory() / spec['folder']
        for path in model_root.iterdir():
            if path.is_file() and path.suffix != '.part':
                files.append((path, f'TextType/models/{spec["folder"]}/{path.name}'))
    if not args.runtime_archive:
        for path in args.runtime.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.pyc', '.zip') and path.name != 'python312.zip':
                continue
            files.append((path, f'TextType/.runtime/{path.relative_to(args.runtime).as_posix()}'))
    records.append(write_archive(args.output / 'TextType_Portable_Windows.zip', files, args.runtime_archive))
    (args.output / 'archive_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
