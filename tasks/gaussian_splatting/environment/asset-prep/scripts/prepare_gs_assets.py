"""Provision <assets root>/gaussian_splatting on the host (standard library only; idempotent).

    python3 -I prepare_gs_assets.py --assets-dir <assets root>/gaussian_splatting [--offline]

  download/tandt_db.zip           upstream's T&T+DB COLMAP data set (README link), sha256-checked
  tandt/truck/                    the truck scene extracted from it (images/, sparse/0/), plus sparse/0/points3D.ply
                                  written from points3D.bin exactly as upstream writes it on first use (the scene
                                  is mounted read-only, so upstream's own conversion could not write it)
  torch/hub/checkpoints/vgg16-397923af.pth   torchvision's VGG-16 (ImageNet) weights, sha256-checked
  torch/hub/checkpoints/vgg.pth              LPIPS v0.1 linear weights for VGG (lpipsPyTorch's download), sha256-checked
--offline only verifies what is there."""
import argparse
import hashlib
import shutil
import struct
import sys
import urllib.request
import zipfile
from pathlib import Path

FILES = [
    ('https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip', 'download/tandt_db.zip',
     '816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1'),
    ('https://download.pytorch.org/models/vgg16-397923af.pth', 'torch/hub/checkpoints/vgg16-397923af.pth',
     '397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'),
    ('https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth',
     'torch/hub/checkpoints/vgg.pth', 'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868'),
]
SCENE = 'tandt/truck'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, want, offline):
    if dest.exists():
        got = sha256(dest)
        if got != want:
            raise SystemExit(f'{dest}: sha256 {got}, expected {want}; remove it to download again')
        return
    if offline:
        raise SystemExit(f'{dest} is missing (--offline)')
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + '.part')
    print(f'downloading {url}', flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(part, 'wb') as f:
        shutil.copyfileobj(r, f, 1 << 22)
    got = sha256(part)
    if got != want:
        part.unlink()
        raise SystemExit(f'{url}: sha256 {got}, expected {want}')
    part.rename(dest)


PLY_PROPS = [('float', 'x'), ('float', 'y'), ('float', 'z'), ('float', 'nx'), ('float', 'ny'), ('float', 'nz'),
             ('uchar', 'red'), ('uchar', 'green'), ('uchar', 'blue')]


def points_ply(scene):
    """sparse/0/points3D.ply as upstream's readColmapSceneInfo writes it the first time it opens a scene
    (read_points3D_binary, then storePly: x, y, z, nx, ny, nz float32 with zero normals, red, green, blue uint8, through
    plyfile's binary little-endian writer)."""
    ply = scene / 'sparse' / '0' / 'points3D.ply'
    if ply.exists():
        return
    data = (scene / 'sparse' / '0' / 'points3D.bin').read_bytes()
    n, = struct.unpack_from('<Q', data, 0)
    off, rows = 8, []
    for _ in range(n):
        _, x, y, z, r, g, b, _ = struct.unpack_from('<QdddBBBd', data, off)
        track, = struct.unpack_from('<Q', data, off + 43)
        off += 43 + 8 + 8 * track
        rows.append(struct.pack('<ffffffBBB', x, y, z, 0.0, 0.0, 0.0, r, g, b))
    if off != len(data):
        raise SystemExit(f'{scene}: points3D.bin has {len(data) - off} trailing bytes')
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % n
              + ''.join(f'property {t} {k}\n' for t, k in PLY_PROPS) + 'end_header\n')
    part = ply.with_suffix('.ply.part')
    part.write_bytes(header.encode('ascii') + b''.join(rows))
    part.rename(ply)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--assets-dir', required=True)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    root = Path(a.assets_dir)
    for url, rel, want in FILES:
        fetch(url, root / rel, want, a.offline)
    scene = root / SCENE
    if not (scene / 'sparse' / '0').is_dir():
        if a.offline:
            raise SystemExit(f'{scene} is missing (--offline)')
        staging = root / 'tandt.staging'
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(root / 'download/tandt_db.zip') as z:
            members = [m for m in z.namelist() if m.startswith(SCENE + '/')]
            if not members:
                raise SystemExit(f'tandt_db.zip has no {SCENE}/')
            z.extractall(staging, members)
        scene.parent.mkdir(parents=True, exist_ok=True)
        (staging / SCENE).rename(scene)
        shutil.rmtree(staging)
    if a.offline and not (scene / 'sparse' / '0' / 'points3D.ply').exists():
        raise SystemExit(f'{scene}/sparse/0/points3D.ply is missing (--offline)')
    points_ply(scene)
    n = len(list((scene / 'images').iterdir()))
    for d in [root, *root.rglob('*')]:
        d.chmod(d.stat().st_mode | (0o755 if d.is_dir() else 0o644))
    print(f'assets ready in {root}: {SCENE} ({n} images), VGG-16 and LPIPS weights')


if __name__ == '__main__':
    sys.exit(main())
