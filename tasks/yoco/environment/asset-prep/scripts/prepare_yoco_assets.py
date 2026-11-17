"""Download and build the fixed YOCO assets (python prepare_yoco_assets.py --assets-dir <dir>).

The corpus is the one the judge was formulated on, rebuilt in the layout of upstream's README from two public files
at pinned revisions (assets/yoco.json):

    downloads/<source file>          the downloaded files (sha256-checked)
    data/Data/json/train.json        [{"name": "fineweb_edu", "weight": 0.8}, {"name": "codeparrot", "weight": 0.2}]
    data/Data/json/valid.json        [] (never read: --disable-validation; the fairseq fork still opens the file)
    data/Data/json/<source>.json     that source's shards, relative to data/Data
    data/Data/shard/<source>/NNNNN.jsonl   10,000 lines of {"text": ...} each, in source-file row order, empty
                                           documents skipped
    tiktoken_cache/<sha1 of the URL> the cl100k_base encoding, where tiktoken looks for it (TIKTOKEN_CACHE_DIR)

Every shard is checked against its frozen sha256 (assets/yoco.json shard_sha256): the rebuilt corpus is byte for byte
the formulated one, or nothing is published. Downloads resume; the corpus is built in a staging directory and moved
into place only when every shard matches. --offline only checks what is there.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


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


def documents(source, path):
    if source['format'] == 'parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=[source['field']]):
            yield from batch.column(0).to_pylist()
    else:
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                yield json.loads(line)[source['field']]


def build(manifest, downloads, staging):
    """data/Data in staging; returns {shard path: sha256}."""
    root = staging / 'Data'
    (root / 'json').mkdir(parents=True)
    got, lines = {}, manifest['lines_per_shard']
    for source in manifest['sources']:
        sdir = root / 'shard' / source['name']
        sdir.mkdir(parents=True)
        shards, buffer, docs = [], [], 0

        def flush():
            rel = 'shard/%s/%05d.jsonl' % (source['name'], len(shards))
            with open(root / rel, 'w', encoding='utf-8') as stream:
                stream.write(''.join(buffer))
            shards.append(rel)
            buffer.clear()

        for text in documents(source, downloads / Path(source['path']).name):
            if not text:
                continue
            buffer.append(json.dumps({'text': text}, ensure_ascii=False) + '\n')
            docs += 1
            if len(buffer) == lines:
                flush()
        if buffer:
            flush()
        if docs != source['docs']:
            raise ValueError('%s: %d documents, expected %d' % (source['name'], docs, source['docs']))
        (root / 'json' / (source['name'] + '.json')).write_text(json.dumps(shards, indent=1) + '\n')
        for rel in shards:
            got[rel] = sha256(root / rel)
        print('%s: %d documents in %d shards' % (source['name'], docs, len(shards)), flush=True)
    (root / 'json' / 'train.json').write_text(json.dumps(
        [{'name': s['name'], 'weight': s['weight']} for s in manifest['sources']], indent=1) + '\n')
    (root / 'json' / 'valid.json').write_text('[]\n')
    return got


def check_corpus(data, manifest):
    want = manifest['shard_sha256']
    root = data / 'Data'
    present = {p.relative_to(root).as_posix() for p in (root / 'shard').rglob('*.jsonl')} if root.is_dir() else set()
    if present != set(want):
        raise ValueError('corpus shards differ from the manifest: %s' % sorted(present ^ set(want))[:5])
    bad = [rel for rel in sorted(want) if sha256(root / rel) != want[rel]]
    if bad:
        raise ValueError('corpus shards fail their frozen sha256: %s' % bad[:5])
    for source in manifest['sources']:
        listed = json.loads((root / 'json' / (source['name'] + '.json')).read_text())
        if sorted(listed) != sorted(r for r in want if r.startswith('shard/%s/' % source['name'])):
            raise ValueError('json/%s.json does not list its shards' % source['name'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets-dir', type=Path, default=Path.home() / '.cache/researchboost/yoco')
    parser.add_argument('--offline', action='store_true', help='check existing files without network')
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    args = parser.parse_args()
    if not args.endpoint.startswith('https://'):
        parser.error('Download endpoint must use HTTPS')
    manifest = json.loads((ROOT / 'assets/yoco.json').read_text())
    assets = args.assets_dir.expanduser().resolve()
    tik = manifest['tiktoken']
    if args.offline:
        check_corpus(assets / 'data', manifest)
        if sha256(assets / 'tiktoken_cache' / tik['cache_name']) != tik['sha256']:
            raise ValueError('tiktoken cache file missing or wrong')
        print('Offline asset check passed')
        return
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fetch(tik['url'], assets / 'tiktoken_cache' / tik['cache_name'], tik['sha256'])
        data = assets / 'data'
        try:
            check_corpus(data, manifest)
            print('Verified: corpus (%d shards)' % len(manifest['shard_sha256']), flush=True)
        except (OSError, ValueError) as exc:
            if data.exists():
                raise ValueError('Existing corpus failed its check (%s); move %s aside before retrying'
                                 % (exc, data)) from exc
            downloads = assets / 'downloads'
            for source in manifest['sources']:
                url = '%s/datasets/%s/resolve/%s/%s' % (args.endpoint.rstrip('/'), source['repo'], source['revision'],
                                                        source['path'])
                fetch(url, downloads / Path(source['path']).name, source['sha256'])
            staging = assets / '.data-staging'
            shutil.rmtree(staging, ignore_errors=True)
            staging.mkdir()
            got = build(manifest, downloads, staging)
            bad = sorted(rel for rel, h in got.items() if manifest['shard_sha256'].get(rel) != h)
            if bad or set(got) != set(manifest['shard_sha256']):
                raise ValueError('rebuilt shards differ from the formulated corpus: %s' % bad[:5])
            staging.rename(data)
            check_corpus(data, manifest)
            print('Built: corpus (%d shards, every sha256 matches)' % len(got), flush=True)
    print('Assets ready:', assets)


if __name__ == '__main__':
    main()
