"""Provision the bagel_sft assets (python prepare_bagel_assets.py --assets-dir <dir>).

    data/bagel_example/   upstream's example data (TRAIN.md: bagel_example.zip, sha256-checked): t2i/, editing/, vlm/
    BAGEL-7B-MoT/         the released checkpoint, Hugging Face ByteDance-Seed/BAGEL-7B-MoT at a pinned revision
                          (or copied from a local directory with --model-dir); ema.safetensors must have the formulated
                          sha256

Downloads resume; the data is unpacked in a staging directory and moved into place complete. --offline only checks
what is there.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile

ZIP_URL = 'https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip'
ZIP_SHA256 = '5a2d18a80deacb2b642c38d14d188ca7fb63e1065e16191124271ca2b3eadcad'
MODEL_REPO = 'ByteDance-Seed/BAGEL-7B-MoT'
MODEL_REVISION = '5019f57d168e5816e8f3f701b17cc816bb7cf24b'
EMA_SHA256 = '0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3'
# What the judge's startup checks (environment/bootstrap.py check_assets).
MODEL_FILES = ['ema.safetensors', 'ae.safetensors', 'llm_config.json', 'vit_config.json', 'tokenizer.json',
               'tokenizer_config.json', 'vocab.json', 'merges.txt']
DATA_FILES = ['t2i/chunk_0.parquet', 'editing/parquet_info/seedxedit_multi.json',
              'editing/seedxedit_multi/chunk_0.parquet', 'vlm/llava_ov_si.jsonl']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, destination, want_sha256):
    if destination.is_file() and sha256(destination) == want_sha256:
        print('Verified:', destination, flush=True)
        return
    if destination.exists():
        raise ValueError('Existing file failed checksum; move it aside before retrying: %s' % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': 'bytes=%d-' % offset} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset and response.status == 206
                with partial.open('ab' if resumed else 'wb') as stream:
                    shutil.copyfileobj(response, stream, length=8 << 20)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and offset:          # already complete
                pass
            elif attempt == 3:
                raise
            else:
                time.sleep(2 ** attempt)
                continue
        except (OSError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
            continue
        if sha256(partial) == want_sha256:
            partial.replace(destination)
            print('Downloaded:', destination, flush=True)
            return
        partial.unlink()
    raise ValueError('Download failed checksum: %s' % url)


def check(assets):
    missing = [str(assets / 'BAGEL-7B-MoT' / f) for f in MODEL_FILES if not (assets / 'BAGEL-7B-MoT' / f).is_file()]
    missing += [str(assets / 'data/bagel_example' / f) for f in DATA_FILES
                if not (assets / 'data/bagel_example' / f).is_file()]
    if missing:
        raise ValueError('assets missing: %s' % missing)
    if sha256(assets / 'BAGEL-7B-MoT/ema.safetensors') != EMA_SHA256:
        raise ValueError('BAGEL-7B-MoT/ema.safetensors is not the formulated release (%s)' % MODEL_REVISION)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets-dir', type=Path, default=Path.home() / '.cache/researchboost/bagel_sft')
    parser.add_argument('--offline', action='store_true', help='check existing files without network')
    parser.add_argument('--model-dir', type=Path, help='copy BAGEL-7B-MoT from this directory instead of downloading')
    args = parser.parse_args()
    assets = args.assets_dir.expanduser().resolve()
    if args.offline:
        check(assets)
        print('Offline asset check passed')
        return
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = assets / 'data'
        if not all((data / 'bagel_example' / f).is_file() for f in DATA_FILES):
            if data.exists():
                raise ValueError('Existing %s is incomplete; move it aside before retrying' % data)
            archive = assets / 'downloads/bagel_example.zip'
            fetch(ZIP_URL, archive, ZIP_SHA256)
            staging = assets / '.data-staging'
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(staging)
            staging.rename(data)
            print('Unpacked:', data, flush=True)
        model = assets / 'BAGEL-7B-MoT'
        if not (model / 'ema.safetensors').is_file():
            if args.model_dir:
                shutil.copytree(args.model_dir, model, dirs_exist_ok=True)
            else:
                from huggingface_hub import snapshot_download     # reads HF_ENDPOINT
                snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=str(model),
                                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.md'])
        check(assets)
        for root, dirs, files in os.walk(assets):          # the judge and the agent read them as another user
            for name in dirs + files:
                p = os.path.join(root, name)
                os.chmod(p, os.stat(p).st_mode | (0o555 if name in dirs else 0o444))
    print('Assets ready:', assets)


if __name__ == '__main__':
    main()
