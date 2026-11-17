"""Download the fixed LLaVA-1.5 assets and rebuild the frozen fixture (standard library only).

    python prepare_assets.py --assets-dir DIR            download missing files, rebuild the fixture and its images
    python prepare_assets.py --assets-dir DIR --offline  check what is there, without network

Layout under DIR (read-only for the task):

    models/vicuna-7b-v1.5/  models/clip-vit-large-patch14-336/  models/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/
    data/llava_v1_5_mix665k.json     instruction data; only used to rebuild the fixture
    data/fixture/<set>.json          the frozen subsets (timing, case_mm, case_text, probe)
    data/images/<llava path>         the 2,319 images those subsets use
    .partial/  .zipcache/            resumable downloads and zip central directories, outside every frozen directory

Weights and the instruction JSON come from pinned Hugging Face revisions and are checked by size and sha256 (git blob
sha1 for small files). Each fixture subset is `json.dumps([source[i] for i in source_indices])` with the validated
run's indices, checked against its sha256. Images are read out of public zips at pinned revisions by HTTP range
requests (central directory, then each member), checked against the member CRC32, and finally the image, CLIP and
projector directories are checked against the validated directory digests (the harness's content_digest).
"""
import argparse
import concurrent.futures as cf
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 64 << 20
WORKERS = int(os.environ.get('RESEARCHBOOST_DOWNLOAD_WORKERS', '8'))


def log(*args):
    print(*args, flush=True)


# ---------------------------------------------------------------- checks

def file_digest(path, algorithm='sha256', size=None):
    digest = hashlib.sha256() if algorithm == 'sha256' else hashlib.sha1()
    if algorithm == 'git_sha1':
        digest.update(('blob %d\0' % size).encode())
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    algorithm = 'sha256' if 'sha256' in entry else 'git_sha1'
    return file_digest(path, algorithm, entry['size']) == entry[algorithm]


def content_digest(path):
    """The harness's content_digest (bench.py): sha256 over sorted relative paths and per-file sha256."""
    path = Path(path)
    digest, count = hashlib.sha256(), 0
    for f in sorted(path.rglob('*')):
        if f.is_file():
            digest.update(str(f.relative_to(path)).encode() + b'\0' + file_digest(f).encode() + b'\n')
            count += 1
    return digest.hexdigest(), count


# ---------------------------------------------------------------- HTTP

def url_of(endpoint, repo, revision, path):
    return '%s/%s/resolve/%s/%s' % (endpoint.rstrip('/'), repo, revision, path)


def get_range(url, start, end, tries=8):
    """Bytes [start, end) of url. Each attempt starts at the hub URL, so expired CDN redirects are re-resolved."""
    error = None
    for attempt in range(tries):
        try:
            request = urllib.request.Request(url, headers={'Range': 'bytes=%d-%d' % (start, end - 1)})
            with urllib.request.urlopen(request, timeout=120) as response:
                # A server that ignores Range answers 200 with the whole file: accept it only when that is the span.
                whole = response.status == 200 and start == 0 and response.headers.get('Content-Length') == str(end)
                if response.status != 206 and not whole:
                    raise ValueError('HTTP %s without the requested range' % response.status)
                data = response.read()
                if len(data) == end - start:
                    return data
                error = ValueError('short read: %d of %d bytes' % (len(data), end - start))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            error = exc
        time.sleep(min(30, 2 ** attempt))
    raise IOError('%s bytes %d-%d failed: %s' % (url, start, end, error))


def remote_size(url):
    request = urllib.request.Request(url, method='HEAD')
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return int(response.headers['Content-Length'])
        except (OSError, urllib.error.URLError, TypeError):
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def download(entry, assets, endpoint):
    """Parallel ranged download into .partial/<destination>, resumable per chunk, published only after the checksum."""
    destination = assets / entry['destination']
    if matches(destination, entry):
        log('Verified:', entry['destination'])
        return
    if destination.exists():
        raise ValueError('Existing file failed checksum; move it aside before retrying: %s' % destination)
    url = url_of(endpoint, entry['repo'], entry['revision'], entry['path'])
    partial = assets / '.partial' / entry['destination']
    done = partial.with_name(partial.name + '.chunks')
    done.mkdir(parents=True, exist_ok=True)
    size = entry['size']
    with open(partial, 'r+b' if partial.exists() else 'w+b') as out:
        out.truncate(size)
        lock = threading.Lock()

        def fetch(k):
            if (done / str(k)).exists():
                return
            start, end = k * CHUNK, min(size, (k + 1) * CHUNK)
            data = get_range(url, start, end)
            with lock:
                os.pwrite(out.fileno(), data, start)
                os.fsync(out.fileno())
            (done / str(k)).touch()
        with cf.ThreadPoolExecutor(WORKERS) as pool:
            list(pool.map(fetch, range(-(-size // CHUNK) if size else 0)))
    if not matches(partial, entry):
        partial.unlink()
        for f in done.iterdir():
            f.unlink()
        raise ValueError('Downloaded file failed checksum (partial removed; rerun): %s' % entry['destination'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(destination)
    for f in done.iterdir():
        f.unlink()
    done.rmdir()
    log('Downloaded:', entry['destination'])


# ---------------------------------------------------------------- fixture

def fixture_rows(source, indices):
    return json.dumps([source[i] for i in indices])


def build_fixture(assets, fixture, source=None):
    """Write each frozen subset; returns the loaded instruction JSON when it had to be read."""
    target = assets / fixture['destination']
    target.mkdir(parents=True, exist_ok=True)
    for name, spec in fixture['sets'].items():
        path = target / (name + '.json')
        if path.is_file() and file_digest(path) == spec['sha256']:
            continue
        if source is None:
            log('Reading', fixture['source'])
            source = json.loads((assets / fixture['source']).read_text())
        text = fixture_rows(source, spec['source_indices'])
        if hashlib.sha256(text.encode()).hexdigest() != spec['sha256']:
            raise ValueError('Fixture subset %s does not reproduce its frozen sha256' % name)
        tmp = assets / '.partial' / ('fixture-%s.json' % name)
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text)
        tmp.replace(path)
        log('Fixture:', path.name, len(spec['source_indices']), 'rows')
    unexpected = sorted(p.name for p in target.iterdir() if p.name not in {n + '.json' for n in fixture['sets']})
    if unexpected:
        raise ValueError('Unexpected files in %s: %s' % (target, unexpected))
    return source


def fixture_images(assets, fixture):
    images = set()
    for name in fixture['sets']:
        for row in json.loads((assets / fixture['destination'] / (name + '.json')).read_text()):
            if 'image' in row:
                images.add(row['image'])
    return sorted(images)


# ---------------------------------------------------------------- images out of remote zips

def parse_central_directory(data):
    """name -> (method, compressed size, uncompressed size, local header offset, crc32), in directory order."""
    entries, p = {}, 0
    while p + 46 <= len(data) and data[p:p + 4] == b'PK\x01\x02':
        method, = struct.unpack('<H', data[p + 10:p + 12])
        crc, csize, usize, nlen, elen, clen = struct.unpack('<IIIHHH', data[p + 16:p + 34])
        offset, = struct.unpack('<I', data[p + 42:p + 46])
        name = data[p + 46:p + 46 + nlen].decode('utf-8', 'replace')
        extra = data[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack('<HH', extra[q:q + 4])
            if hid == 1:              # ZIP64: only the fields that overflowed are present, in this order
                values, r = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize = struct.unpack('<Q', values[r:r + 8])[0]
                    r += 8
                if csize == 0xFFFFFFFF:
                    csize = struct.unpack('<Q', values[r:r + 8])[0]
                    r += 8
                if offset == 0xFFFFFFFF:
                    offset = struct.unpack('<Q', values[r:r + 8])[0]
                    r += 8
            q += 4 + hlen
        entries[name] = (method, csize, usize, offset, crc)
        p += 46 + nlen + elen + clen
    return entries


def central_directory(url, size, cache):
    """Fetch (once, cached) and parse a remote zip's central directory, handling ZIP64."""
    if not cache.exists():
        tail = get_range(url, max(0, size - 131072), size)
        i = tail.rfind(b'PK\x05\x06')
        if i < 0:
            raise ValueError('No end of central directory record in %s' % url)
        _, _, _, _, count, cd_size, cd_offset, _ = struct.unpack('<IHHHHIIH', tail[i:i + 22])
        if 0xFFFFFFFF in (cd_offset, cd_size) or count == 0xFFFF:
            j = tail.rfind(b'PK\x06\x07')
            z64 = struct.unpack('<Q', tail[j + 8:j + 16])[0]
            record = get_range(url, z64, z64 + 56)
            if record[:4] != b'PK\x06\x06':
                raise ValueError('Bad ZIP64 end of central directory in %s' % url)
            count, cd_size, cd_offset = struct.unpack('<QQQ', record[32:56])
        spans = [(a, min(cd_offset + cd_size, a + (8 << 20))) for a in range(cd_offset, cd_offset + cd_size, 8 << 20)]
        with cf.ThreadPoolExecutor(WORKERS) as pool:
            data = b''.join(pool.map(lambda s: get_range(url, *s), spans))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + '.tmp')
        tmp.write_bytes(data)
        tmp.replace(cache)
    return parse_central_directory(cache.read_bytes())


def member_index(entries, member_prefix):
    """Basename -> entry, first in directory order: the validated fixture builder's lookup."""
    index = {}
    for name, entry in entries.items():
        if name.endswith('/'):
            continue
        if member_prefix is None or name.startswith(member_prefix) or ('/' + member_prefix) in name:
            index.setdefault(os.path.basename(name), entry)
    return index


def read_member(url, entry):
    method, csize, usize, offset, crc = entry
    blob = get_range(url, offset, offset + 30 + 1024 + csize)
    if blob[:4] != b'PK\x03\x04':
        raise ValueError('Bad local header at %d in %s' % (offset, url))
    nlen, elen = struct.unpack('<HH', blob[26:30])
    start = 30 + nlen + elen
    if start + csize > len(blob):
        blob += get_range(url, offset + len(blob), offset + start + csize)
    raw = blob[start:start + csize]
    if method == 0:
        data = raw
    elif method == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise ValueError('Unsupported compression method %d' % method)
    if len(data) != usize or zlib.crc32(data) != crc:
        raise ValueError('Member at %d of %s fails its size or CRC32' % (offset, url))
    return data


def fetch_images(assets, manifest, endpoint):
    spec = manifest['images']
    target = assets / spec['destination']
    wanted = fixture_images(assets, manifest['fixture'])
    if len(wanted) != spec['count']:
        raise ValueError('The fixture names %d images, expected %d' % (len(wanted), spec['count']))
    missing = [p for p in wanted if not (target / p).is_file()]
    log('Images: %d needed, %d missing' % (len(wanted), len(missing)))
    by_prefix = {}
    for p in missing:
        prefix = next((k for k in spec['zips'] if p.startswith(k)), None)
        if prefix is None:
            raise ValueError('No source zip for image %s' % p)
        by_prefix.setdefault(prefix, []).append(p)
    for prefix, paths in sorted(by_prefix.items()):
        z = spec['zips'][prefix]
        url = url_of(endpoint, z['repo'], z['revision'], z['path'])
        size = remote_size(url)
        if size != z['size']:
            raise ValueError('%s has %d bytes, expected the pinned %d' % (url, size, z['size']))
        tag = '%s__%s' % (z['repo'].replace('/', '_'), z['path'])
        index = member_index(central_directory(url, size, assets / '.zipcache' / (tag + '.cd')), z['member_prefix'])
        absent = [p for p in paths if os.path.basename(p) not in index]
        if absent:
            raise ValueError('%d images are not in %s, e.g. %s' % (len(absent), url, absent[:3]))
        start = time.time()

        def fetch(p):
            data = read_member(url, index[os.path.basename(p)])
            tmp = assets / '.partial' / 'images' / p
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            (target / p).parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(target / p)
        with cf.ThreadPoolExecutor(2 * WORKERS) as pool:
            list(pool.map(fetch, paths))
        log('  %s: %d images from %s in %.0f s' % (prefix, len(paths), z['repo'], time.time() - start))
    extra = sorted(str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()
                   and str(p.relative_to(target)) not in set(wanted))
    if extra:
        raise ValueError('Unexpected files in %s (move them aside): %s' % (target, extra[:5]))


# ---------------------------------------------------------------- main

def check_validated(assets, manifest):
    for name, spec in manifest['validated_digests'].items():
        path = assets / spec['path']
        if not path.is_dir():
            raise ValueError('Missing %s' % path)
        digest, count = content_digest(path)
        if (digest, count) != (spec['sha256'], spec['files']):
            raise ValueError('%s differs from the validated directory: %s over %d files, expected %s over %d'
                             % (spec['path'], digest, count, spec['sha256'], spec['files']))
        log('Validated digest:', spec['path'], count, 'files')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets-dir', type=Path, default=Path.home() / '.cache/researchboost/llava')
    parser.add_argument('--offline', action='store_true', help='Check existing files without network')
    parser.add_argument('--endpoint', default=os.environ.get('HF_ENDPOINT', 'https://huggingface.co'))
    args = parser.parse_args()
    if not args.endpoint.startswith('https://'):
        parser.error('Download endpoint must use HTTPS')
    manifest = json.loads((ROOT / 'assets/llava.json').read_text())
    assets = args.assets_dir.expanduser().resolve()
    if args.offline:
        for entry in manifest['files']:
            if not matches(assets / entry['destination'], entry):
                raise ValueError('Missing or invalid: %s' % entry['destination'])
        for name, spec in manifest['fixture']['sets'].items():
            path = assets / manifest['fixture']['destination'] / (name + '.json')
            if not path.is_file() or file_digest(path) != spec['sha256']:
                raise ValueError('Missing or invalid fixture subset: %s' % name)
        check_validated(assets, manifest)
        log('Offline asset check passed')
        return
    assets.mkdir(parents=True, exist_ok=True)
    with (assets / '.download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for entry in manifest['files']:
            download(entry, assets, args.endpoint)
        build_fixture(assets, manifest['fixture'])
        fetch_images(assets, manifest, args.endpoint)
        check_validated(assets, manifest)
    log('Assets ready:', assets)


if __name__ == '__main__':
    main()
