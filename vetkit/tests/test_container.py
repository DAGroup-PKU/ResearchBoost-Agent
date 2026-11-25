"""Unit tests of the container side (no Docker, no GPU): python3 vetkit/tests/test_container.py"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'container'))
import svckey  # noqa: E402

FIELDS = dict(task='example_task', judge_hash='a' * 64, image_digest='sha256:' + 'b' * 64,
              gpu_model='NVIDIA H200', world_size=1, driver='570.148.08', prep_version='v2')


class KeyTest(unittest.TestCase):
    def test_stable_and_sensitive(self):
        k = svckey.pack_key(**FIELDS)
        self.assertEqual(k, svckey.pack_key(**dict(FIELDS, world_size='1')))
        for f, v in (('driver', '575.0'), ('gpu_model', 'NVIDIA H100 80GB HBM3'), ('image_digest', 'local:sha256:' + 'c' * 64),
                     ('prep_version', 'v3'), ('judge_hash', 'd' * 64), ('world_size', 2), ('task', 'llava')):
            self.assertNotEqual(k, svckey.pack_key(**dict(FIELDS, **{f: v})), f)

    def test_canonical_json(self):
        self.assertEqual(svckey.canonical(svckey.key_fields(**FIELDS)),
                         '{"driver":"570.148.08","gpu_model":"NVIDIA H200","image_digest":"sha256:%s",'
                         '"judge_hash":"%s","prep_version":"v2","schema":"rbb-pack/1","task":"example_task",'
                         '"world_size":1}' % ('b' * 64, 'a' * 64))

    def test_missing_and_unknown(self):
        with self.assertRaises(ValueError):
            svckey.pack_key(**dict(FIELDS, driver=''))
        with self.assertRaises(ValueError):
            svckey.pack_key(**dict(FIELDS, extra='x'))

    def test_image_digest(self):
        self.assertEqual(svckey.image_digest_of(ref='r/x@sha256:' + 'e' * 64), 'sha256:' + 'e' * 64)
        self.assertEqual(svckey.image_digest_of(image_id='sha256:' + 'f' * 64), 'local:sha256:' + 'f' * 64)


def restore(art, allow):
    r = subprocess.run([sys.executable, '-I', str(HERE / 'container/restore_bundle.py'), '--artifacts', str(art),
                        '--allow', *allow], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout)


class RestoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sub = self.tmp / 'submission'
        self.env = self.tmp / 'candidate-env'
        self.outside = self.tmp / 'judge-ws'
        for d in (self.sub / 'tree', self.env / 'bin', self.outside):
            d.mkdir(parents=True)
        (self.sub / 'tree/old.py').write_text('old')
        (self.sub / 'switches.json').write_text('{}')
        os.symlink('/usr/bin/python3', self.env / 'bin/python')          # an image interpreter link
        (self.outside / 'bench.py').write_text('judge')
        self.art = self.tmp / 'artifacts'
        (self.art / 'tree').mkdir(parents=True)
        (self.art / 'tree/new.py').write_text('new')
        (self.art / 'switches.json').write_text('{"environment": {}}')
        (self.art / 'env/lib').mkdir(parents=True)
        (self.art / 'env/lib/pkg.py').write_text('pkg')
        self.allow = [str(self.sub / 'tree'), str(self.sub / 'switches.json'), str(self.env)]

    def manifest(self, extra=()):
        m = [{'source': '/logs/artifacts', 'destination': 'artifacts/logs/artifacts', 'type': 'directory',
              'status': 'empty'},
             {'source': str(self.sub / 'tree'), 'destination': 'artifacts/tree', 'type': 'directory', 'status': 'ok'},
             {'source': str(self.sub / 'switches.json'), 'destination': 'artifacts/switches.json', 'type': 'file',
              'status': 'ok'},
             {'source': str(self.env), 'destination': 'artifacts/env', 'type': 'directory', 'status': 'ok'}]
        (self.art / 'manifest.json').write_text(json.dumps(m + list(extra)))

    def test_restore(self):
        self.manifest()
        code, rep = restore(self.art, self.allow)
        self.assertEqual((code, rep['status']), (0, 'ok'), rep)
        self.assertFalse((self.sub / 'tree/old.py').exists())
        self.assertEqual((self.sub / 'tree/new.py').read_text(), 'new')
        self.assertTrue(os.path.islink(self.env / 'bin/python'), 'the image interpreter link is kept')
        self.assertEqual((self.env / 'lib/pkg.py').read_text(), 'pkg')

    def test_symlink_in_bundle_is_not_followed(self):
        os.symlink(str(self.outside / 'bench.py'), self.art / 'tree/evil')
        self.manifest()
        code, rep = restore(self.art, self.allow)
        self.assertEqual(code, 0)
        self.assertTrue(os.path.islink(self.sub / 'tree/evil'))
        self.assertEqual((self.outside / 'bench.py').read_text(), 'judge')
        self.assertTrue(rep['links_outside'])

    def test_file_over_image_link_replaces_link(self):
        (self.art / 'env/bin').mkdir()
        (self.art / 'env/bin/python').write_text('fake interpreter')
        os.unlink(self.env / 'bin/python')
        os.symlink(str(self.outside / 'bench.py'), self.env / 'bin/python')
        self.manifest()
        code, _ = restore(self.art, self.allow)
        self.assertEqual(code, 0)
        self.assertEqual((self.outside / 'bench.py').read_text(), 'judge', 'never written through a link')
        self.assertFalse(os.path.islink(self.env / 'bin/python'))

    def test_dir_link_leaving_target_is_replaced(self):
        os.symlink(str(self.outside), self.env / 'lib')
        self.manifest()
        code, rep = restore(self.art, self.allow)
        self.assertEqual(code, 0)
        self.assertFalse((self.outside / 'pkg.py').exists())
        self.assertIn(str(self.env / 'lib'), rep['replaced_dir_links'])

    def test_unknown_target_refused(self):
        self.manifest([{'source': str(self.outside), 'destination': 'artifacts/tree', 'type': 'directory',
                        'status': 'ok'}])
        code, rep = restore(self.art, self.allow)
        self.assertEqual((code, rep['status']), (3, 'refused'))
        self.assertEqual((self.outside / 'bench.py').read_text(), 'judge')

    def test_destination_escape_refused(self):
        self.manifest()
        m = json.loads((self.art / 'manifest.json').read_text())
        m[1]['destination'] = 'artifacts/../judge-ws'
        (self.art / 'manifest.json').write_text(json.dumps(m))
        code, rep = restore(self.art, self.allow)
        self.assertEqual((code, rep['status']), (3, 'refused'), rep)

    def test_fifo_refused(self):
        os.mkfifo(self.art / 'tree/pipe')
        self.manifest()
        code, rep = restore(self.art, self.allow)
        self.assertEqual(code, 3, rep)


class ScanTest(unittest.TestCase):
    def test_judge_hash_ignores_targets_and_pycache(self):
        sys.path.insert(0, str(HERE / 'container'))
        import svcpack
        ws = Path(tempfile.mkdtemp())
        (ws / 'bench').mkdir()
        (ws / 'bench/bench.py').write_text('x')
        h = svcpack.judge_hash(ws)
        (ws / 'targets/reference').mkdir(parents=True)
        (ws / 'targets/reference/train.py').write_text('y')
        (ws / 'bench/__pycache__').mkdir()
        (ws / 'bench/__pycache__/bench.pyc').write_text('z')
        self.assertEqual(h, svcpack.judge_hash(ws))
        (ws / 'bench/bench.json').write_text('{}')
        self.assertNotEqual(h, svcpack.judge_hash(ws))


class PackRoundTripTest(unittest.TestCase):
    """pristine -> preparation -> collect -> (fresh pristine) -> restore gives the prepared tree."""

    def test_roundtrip(self):
        import argparse
        import hashlib
        import shutil
        sys.path.insert(0, str(HERE / 'container'))
        import svcpack
        tmp = Path(tempfile.mkdtemp())
        root = tmp / 'ws'

        def pristine():
            shutil.rmtree(root, ignore_errors=True)
            (root / 'bench').mkdir(parents=True)
            (root / 'bench/bench.py').write_text('code')
            (root / 'bench/helper.py').write_text('helper')
            (root / 'prep').mkdir()
            (root / 'prep/make_accept.py').write_text('prep tool')
        pristine()
        before = svcpack.scan([str(root)])
        before.update(roots=[str(root)], keyinfo={})
        (tmp / 'pristine.json').write_text(json.dumps(before))
        # "preparation"
        (root / 'bench/records').mkdir()
        (root / 'bench/records/ref.json').write_text('{"r": 1}')
        (root / 'bench/helper.py').write_text('helper changed')
        shutil.rmtree(root / 'prep')
        os.symlink('records/ref.json', root / 'bench/latest')
        after = svcpack.scan([str(root)])
        svcpack.cmd_collect(argparse.Namespace(task='nanochat', pristine=str(tmp / 'pristine.json'),
                                               out=str(tmp / 'pack')))
        mb = (tmp / 'pack/manifest.json').read_bytes()
        meta = {'content_sha256': hashlib.sha256(mb).hexdigest(), 'key': 'k', 'key_fields': {}}
        (tmp / 'pack/pack.json').write_text(json.dumps(meta))
        _, manifest, collect = svcpack.check_integrity(tmp / 'pack')
        self.assertEqual(set(manifest), {str(root / 'bench/records/ref.json'), str(root / 'bench/helper.py'),
                                         str(root / 'bench/latest')})
        pristine()
        ctx = svcpack.Ctx('nanochat', type('M', (), {})(), pack=tmp / 'pack', meta=meta)
        svcpack.restore(ctx, manifest, collect)
        self.assertEqual(svcpack.scan([str(root)])['files'], after['files'])
        self.assertFalse((root / 'prep').exists())
        # a corrupted pack file is refused
        f = tmp / 'pack/files' / str(root / 'bench/helper.py').lstrip('/')
        f.write_text('tampered')
        with self.assertRaises(svcpack.Mismatch):
            svcpack.check_integrity(tmp / 'pack')


class SafePathTest(unittest.TestCase):
    """Protocol supplement C10: the judge's PATH and programs are those only root can change."""

    def test_root_only(self):
        import safepath
        for p in ('/usr/bin', '/usr/bin/env', '/usr/bin/../bin', '/var'):     # /var: a link on some systems
            self.assertTrue(safepath.root_only(p), p)
        self.assertFalse(safepath.root_only('/usr/bin/no-such-program'))
        with tempfile.TemporaryDirectory() as tmp:
            open_dir = Path(tmp) / 'open'
            (open_dir / 'bin').mkdir(parents=True)
            os.chmod(str(open_dir), 0o777)                    # writable by others, whoever owns it (also as root)
            os.symlink('/usr/bin', str(open_dir / 'link'))   # a link someone else could point elsewhere
            for p in (open_dir, open_dir / 'bin', open_dir / 'link', open_dir / 'link/env', open_dir / 'link/../bin'):
                self.assertFalse(safepath.root_only(str(p)), p)
            value = '%s:/usr/bin:relative::%s:/no/such/dir:/usr/bin' % (open_dir / 'bin', open_dir / 'link')
            self.assertEqual(safepath.safe_path(value), '/usr/bin')
            tool = [sys.executable, str(HERE / 'container/safepath.py')]
            r = subprocess.run(tool + ['path', value], stdout=subprocess.PIPE, universal_newlines=True)
            self.assertEqual((r.returncode, r.stdout.strip()), (0, '/usr/bin'))
            self.assertEqual(subprocess.run(tool + ['check', '/usr/bin/env']).returncode, 0)
            self.assertEqual(subprocess.run(tool + ['check', '/usr/bin/env', str(open_dir / 'link/env')],
                                            stderr=subprocess.DEVNULL).returncode, 1)


@unittest.skipUnless(sys.platform.startswith('linux') and os.geteuid() == 0, 'Linux, as root')
class GuardTest(unittest.TestCase):
    """An escapee (double fork + setsid) of a non-root child is killed within ticks; the child's exit status passes
    through."""

    def test_escapee_killed(self):
        import time
        tmp = Path(tempfile.mkdtemp())
        mark = tmp / 'daemon-alive'
        code = ("import os, time, sys\n"
                "if os.fork() == 0:\n"
                "    os.setsid()\n"
                "    if os.fork() == 0:\n"
                "        for i in range(300):\n"
                "            open(%r, 'w').write(str(i)); time.sleep(0.01)\n"
                "        os._exit(0)\n"
                "    os._exit(0)\n"
                "time.sleep(1.5); sys.exit(7)\n" % str(mark))
        os.chmod(tmp, 0o777)
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / 'container/guard.py'), '--report', str(tmp / 'g.json'), '--',
                            'runuser', '-u', 'nobody', '--', sys.executable, '-c', code])
        self.assertEqual(r.returncode, 7)
        rep = json.loads((tmp / 'g.json').read_text())
        self.assertGreaterEqual(rep['killed'], 1, rep)
        n = int(mark.read_text()) if mark.exists() else 0
        self.assertLess(n, 20, 'the escapee lived %d ticks' % n)
        self.assertLess(time.time() - t0, 5)


if __name__ == '__main__':
    unittest.main()
