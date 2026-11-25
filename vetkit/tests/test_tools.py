"""Unit tests of the host tools (no Docker, no GPU): python3 vetkit/tests/test_tools.py"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

KIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KIT / 'bin'))
import _hostlib  # noqa: E402
import leak_guard  # noqa: E402
import onboard_check  # noqa: E402
import refpack  # noqa: E402
import run_trial  # noqa: E402

EXAMPLE = KIT / 'examples' / 'tasks' / 'nanochat'
try:                                   # reading task.toml: Python >= 3.11, or the toml package (Harbor installs it)
    import tomllib  # noqa: F401
    HAVE_TOML = True
except ImportError:
    try:
        import toml  # noqa: F401
        HAVE_TOML = True
    except ImportError:
        HAVE_TOML = False
needs_toml = unittest.skipUnless(HAVE_TOML, 'needs Python >= 3.11 or the toml package')


def copy_example(tmp):
    dst = Path(tmp) / 'tasks' / 'nanochat'
    shutil.copytree(str(EXAMPLE), str(dst))
    return dst


class CpuTest(unittest.TestCase):
    def test_cpulist_roundtrip(self):
        self.assertEqual(_hostlib.parse_cpulist('0-3,8,10-11'), [(0, 3), (8, 8), (10, 11)])
        self.assertEqual(_hostlib.format_cpulist([3, 1, 2, 0, 8, 10, 11]), '0-3,8,10-11')

    def test_numa_share(self):
        # 8 GPUs, two NUMA nodes of 48 cores + 48 hyperthreads each
        lists = {str(g): ('0-47,96-143' if g < 4 else '48-95,144-191') for g in range(8)}
        self.assertEqual(_hostlib.gpu_cpuset(['0'], lists), '0-11,96-107')
        self.assertEqual(_hostlib.gpu_cpuset(['0', '1'], lists), '0-23,96-119')
        self.assertEqual(_hostlib.gpu_cpuset(['4', '5', '6', '7'], lists), '48-95,144-191')
        self.assertEqual(_hostlib.resolve_cpus('0-3,5', ['0']), '0-3,5')
        self.assertIsNone(_hostlib.resolve_cpus('none', ['0']))


class RunTrialTest(unittest.TestCase):
    def test_parse_trial(self):
        tasks = EXAMPLE.parent
        self.assertEqual(run_trial.parse_trial('nanochat:nop', tasks)['agent'], 'nop')
        t = run_trial.parse_trial('nanochat:claude-code:anthropic/some-model', tasks)
        self.assertEqual((t['agent'], t['model']), ('claude-code', 'anthropic/some-model'))
        t = run_trial.parse_trial('nanochat:agents.my_agent:MyAgent:provider/model', tasks)
        self.assertEqual((t['agent'], t['label'], t['model']), ('agents.my_agent:MyAgent', 'MyAgent', 'provider/model'))
        with self.assertRaises(ValueError):
            run_trial.parse_trial('nope:nop', tasks)

    def test_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = run_trial.gpu_override(Path(tmp) / 'o.yaml', ['2', '3'], '24-47', '/scratch/gpus-2-3/nanochat')
            text = p.read_text()
            self.assertIn('device_ids: ["2", "3"]', text)
            self.assertIn('cpuset: "24-47"', text)
            self.assertIn('/scratch/gpus-2-3/nanochat:/tmp', text)

    def test_override_cpu_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = run_trial.gpu_override(Path(tmp) / 'o.yaml', [], '0-23,96-119').read_text()
            self.assertIn('cpuset: "0-23,96-119"', text)
            self.assertNotIn('device_ids', text)

    @needs_toml
    def test_dry_run_and_secrets(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / 'r.env'
            env.write_text('RESEARCHBOOST_ASSETS_ROOT=/a\nRESEARCHBOOST_SCRATCH_ROOT=/s\n')
            out = subprocess.run([sys.executable, str(KIT / 'bin/run_trial.py'), '--tasks-dir', str(EXAMPLE.parent),
                                  '--env-file', str(env), '--trial', 'nanochat:nop', '--gpu-ids', '0,1',
                                  '--jobs-dir', str(Path(tmp) / 'jobs'), '--agent-timeout-min', '60', '--dry-run'],
                                 capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn('--agent-timeout-multiplier 0.1', out.stdout)      # 60 min of the task's 10 h
            self.assertIn('-i nanochat -a nop', out.stdout)
            env.write_text('RESEARCHBOOST_SMOKE=1\n')
            out = subprocess.run([sys.executable, str(KIT / 'bin/run_trial.py'), '--tasks-dir', str(EXAMPLE.parent),
                                  '--env-file', str(env), '--trial', 'nanochat:nop', '--dry-run'],
                                 capture_output=True, text=True)
            self.assertNotEqual(out.returncode, 0)                           # the runner owns RESEARCHBOOST_SMOKE
            d = Path(tmp) / 'job'
            d.mkdir()
            (d / 'log.txt').write_text('key=not-a-real-secret-0123 ok')
            self.assertEqual(len(run_trial.redact_tree(d, ['not-a-real-secret-0123'])), 1)
            self.assertEqual((d / 'log.txt').read_text(), 'key=<redacted> ok')
            with self.assertRaises(SystemExit):                              # wrong GPU count for the task
                run_trial.parse_args(['--tasks-dir', str(EXAMPLE.parent), '--env-file', str(Path(tmp) / 'r.env'),
                                      '--trial', 'nanochat:nop', '--gpu-ids', '0'])


@needs_toml
class OnboardCheckTest(unittest.TestCase):
    def test_example_passes(self):
        facts, problems, _ = onboard_check.check(EXAMPLE, allow_dirty=True)
        self.assertEqual(problems, [])
        self.assertEqual(facts['gpus'], 2)

    def test_broken_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = copy_example(tmp)
            toml = (td / 'task.toml').read_text()
            toml = toml.replace('"/opt/candidate-env"]', ']').replace('environment_mode = "shared"',
                                                                      'environment_mode = "separate"')
            (td / 'task.toml').write_text(toml)
            with open(td / 'instruction.md', 'a') as f:
                f.write('\nYou have 10 hours.\n')
            (td / 'vetting').mkdir()
            bj = td / 'environment/workspace/bench/bench.json'
            d = json.loads(bj.read_text())
            d['world_size'] = 4
            bj.write_text(json.dumps(d))
            _, problems, _ = onboard_check.check(td, allow_dirty=True)
            text = '\n'.join(problems)
            for want in ('artifacts', 'environment_mode', 'duration', 'vetting inside the task', 'world_size 4'):
                self.assertIn(want, text)


class VerifierPathTest(unittest.TestCase):
    """Protocol supplement C10: the verifier runs nothing by bare name from the inherited PATH."""

    def test_example_clean(self):
        problems, warnings = [], []
        onboard_check.check_verifier(EXAMPLE, problems, warnings)
        self.assertEqual(problems, [])
        self.assertTrue(any('/opt/candidate-env/bin' in w for w in warnings), warnings)

    def test_bare_names_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = copy_example(tmp)
            with open(td / 'tests/verify.py', 'a') as f:
                f.write('\nsubprocess.run(["pkill", "-u", AGENT])\n'
                        'subprocess.run([RUNUSER, "-u", AGENT, "--", "git", "log"])\n'
                        'os.system("nvidia-smi -L")\n'
                        'subprocess.run(git + ["log", "--stat"])\n'          # arguments, not a program
                        '# subprocess.run(["pgrep", "x"]) in a comment\n')
            (td / 'tests/test.sh').write_text('#!/bin/bash\nset -euo pipefail\nexec python3 /tests/verify.py\n')
            problems = []
            onboard_check.check_verifier(td, problems, [])
            names = sorted(p.split("'")[1] for p in problems)
            self.assertEqual(names, ['git', 'nvidia-smi', 'pkill', 'python3'], problems)


class LeakGuardTest(unittest.TestCase):
    def test_example_clean(self):
        self.assertEqual(leak_guard.scan_task(EXAMPLE), [])

    def test_planted_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = copy_example(tmp)
            with open(td / 'instruction.md', 'a') as f:
                f.write('\nA 3.2x speedup was reached on H100 by fusing the norm.\n')
            (td / 'environment/workspace/bench/WORK_LOG.md').write_text('notes')
            deny = Path(tmp) / 'deny.txt'
            deny.write_text('# the ladder switch\nfusing the norm\n')
            found = leak_guard.scan_task(td, leak_guard.load_deny(deny))
            df = td / 'environment/Dockerfile'
            (td / 'environment/extra').mkdir()
            (td / 'environment/extra/a.md').write_text('profiling shows the hotspot')
            df.write_text(df.read_text() + '\nCOPY ["extra/", "/opt/x/"]\nCOPY missing.txt /opt/z\n')
            found = leak_guard.scan_task(td, leak_guard.load_deny(deny))
            self.assertTrue(any(f['file'] == 'environment/extra/a.md' for f in found), 'JSON-form COPY scanned')
            self.assertTrue(any(f['match'] == 'missing.txt' for f in found), 'missing COPY source reported')
            reasons = {f['reason'] for f in found}
            self.assertIn('speedup target or result', reasons)
            self.assertTrue(any(r.startswith('other hardware') for r in reasons))
            self.assertTrue(any(r.startswith('task deny list') for r in reasons))
            self.assertTrue(any('WORK_LOG' in f['match'] for f in found))


@needs_toml
class RefpackTest(unittest.TestCase):
    def args(self, **kw):
        base = dict(task='nanochat', tasks_dir=str(EXAMPLE.parent), adapter=None, python=None, timing_file=None,
                    timing_field=None, band=None, dirs=None, time_tree_wrap=None)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_task_module_defaults(self):
        t = refpack.Task(self.args())
        self.assertEqual((t.adapter, t.python, t.gpus), ('nanochat', '/opt/nanochat-env/bin/python', 2))
        self.assertEqual(t.artifacts, ['/submission/nanochat', '/submission/switches.json', '/opt/candidate-env'])
        self.assertNotIn('RBB_SVC_TASK_CONFIG', t.env())

    @unittest.skipUnless(shutil.which('docker'), 'needs the docker CLI with Compose (no daemon)')
    def test_compose(self):
        """The judge container's Compose file from the task's own service: no network, entry.sh, these GPUs and
        CPUs, a fresh /tmp, the kit read-only, the pack read-only, no prep-pack, the task's CPU and memory limits."""
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / 'r.env'
            env.write_text('RESEARCHBOOST_ASSETS_ROOT=%s/assets\nRESEARCHBOOST_SCRATCH_ROOT=%s/scratch\n' % (tmp, tmp))
            a = argparse.Namespace(env_file=str(env), extra_docker_compose=[], judge_env=['X_FLAG=1'], scratch=None)
            task = refpack.Task(self.args())
            path, project = refpack.make_compose(
                a, task, {'id': 'sha256:' + 'a' * 64}, 'judge', Path(tmp), ['2', '3'], '24-47,120-143',
                [(Path(tmp) / 'pack', '/opt/rbb-svc/pack', True)], {'RBB_SVC_KEY': 'k'})
            main = json.loads(path.read_text())['services']['main']
            self.assertEqual(main['network_mode'], 'none')
            self.assertEqual(main['entrypoint'], ['/bin/bash', '/opt/rbb-svc/entry.sh', 'judge'])
            self.assertEqual(main['deploy']['resources']['reservations']['devices'][0]['device_ids'], ['2', '3'])
            self.assertEqual(main['cpuset'], '24-47,120-143')
            self.assertEqual((main['cpus'], main['mem_limit']), (48.0, '262144m'))
            vols = {v['target']: v for v in main['volumes']}
            self.assertEqual(vols['/tmp']['source'], str(Path(tmp) / 'tmp'))
            self.assertTrue(vols['/opt/rbb-svc']['read_only'] and vols['/opt/rbb-svc/pack']['read_only'])
            self.assertTrue(vols['/opt/researchboost/assets']['read_only'])
            self.assertEqual(main['environment']['RBB_SVC_PYTHON'], '/opt/nanochat-env/bin/python')
            self.assertEqual(main['environment']['X_FLAG'], '1')
            self.assertNotIn('build', main)

    def test_other_adapter_is_configured(self):
        mod = KIT / 'container' / 'tasks' / 'zz_test_module.py'
        mod.write_text("PYTHON = '/opt/zz/bin/python'\n")
        try:
            t = refpack.Task(self.args(adapter='zz_test_module', python='/opt/x/bin/python'))
            self.assertEqual(json.loads(t.env()['RBB_SVC_TASK_CONFIG']), {'adapter': 'zz_test_module'})
            self.assertEqual(t.env()['RBB_SVC_COMPAT'], 'never')
        finally:
            mod.unlink()

    def test_generic_needs_timing(self):
        with self.assertRaises(SystemExit):
            refpack.Task(self.args(adapter='generic', python='/opt/x/bin/python'))
        t = refpack.Task(self.args(adapter='generic', python='/opt/x/bin/python', timing_file='task/timing_ref.json',
                                   timing_field='s_per_update', band=0.08))
        cfg = json.loads(t.env()['RBB_SVC_TASK_CONFIG'])
        self.assertEqual(cfg['generic']['timing_field'], 's_per_update')
        self.assertEqual(cfg['generic']['band'], 0.08)


if __name__ == '__main__':
    unittest.main()
