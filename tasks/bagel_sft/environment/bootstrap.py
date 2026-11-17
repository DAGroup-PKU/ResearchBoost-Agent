"""Prepare machine-specific records before the agent starts; never runs during grading.

Runs as root on the four allocated GPUs, following the judge's acceptance procedure:
  1. the assets: the released BAGEL-7B-MoT EMA weights must have the formulated SHA-256, upstream's bagel_example data
     must be complete;
  2. the reference recorded twice in deterministic mode (the two records must be identical, else this host cannot
     judge bitwise and preparation stops), its record frozen as bench/records/ref.json;
  3. the reference's trajectories for seeds 1-6, seed 1 again (it must reproduce exactly), and the frozen delta of
     every endpoint (bench/traj/); the reference's gate-2b run on the held-out interleaved data configuration, twice
     (the second must reproduce the first exactly), frozen as bench/conform/ref.json;
  4. the frozen input fingerprints, every one readable by the agent user;
  5. the unchanged tree's quick check (PASS, bitwise);
  6. this machine's reference timing: timing.reference_runs runs of the reference, their mean and spread frozen as
     task/timing/reference.json (what `time` and the agent's check read the speedup against, and the final
     evaluation's reference drift);
  7. the unchanged tree's full check (UNAVAILABLE: no benefit) and acceptance controls: AdamW beta2 0.999 must be
     INCORRECT (exact items), a tree that skips the EMA update must be INCORRECT (statistical gate), a tree that
     normalizes the CE loss in another order (divide, then sum) must PASS (bitwise on 4 GPUs: the factor 4 is exact),
     and a tree whose RMSNorm divides by the square root instead of multiplying by its reciprocal (not bitwise, the
     same function) must PASS on the statistical gate; two trees whose attention masks are wrong only on interleaved
     layouts (noise spans visible to later splits; a bidirectional span that also sees the next one), bitwise
     identical to the reference on the recipe's data, must be INCORRECT at gate 2b (nonconforming).
Any failure stops the container.

With a preparation pack (tasks/bagel_sft/prep-pack/ws4 in the repository, or $ASSETS/pack; made from an earlier full preparation on the same GPU model and the same
assets: the reference record, trajectories and gate-2b run), steps 2-3 restore the pack and the reference is recorded
once and run once on the interleaved data: both must be bitwise identical to the pack's, else the full preparation
runs. The quick check and this machine's
reference timing follow; the full check and the controls stand from the pack's preparation.

RESEARCHBOOST_SMOKE=1 (integration tests) keeps the two records and the quick check; trajectories, the timing baseline,
the full check and the controls are skipped, so only quick checks can be graded.
"""
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WS = Path('/opt/researchboost/ws')
ASSETS = Path(os.environ.get('ASSETS', '/opt/researchboost/assets'))
CANDIDATE = '/submission/bagel_sft'
PY = sys.executable
SCRATCH = Path('/tmp/bagel-prep')
MIN_GPU_MEMORY_GIB = 130
DEFAULT_WORLD_SIZE = 4
EMA_SHA256 = '0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3'   # BAGEL-7B-MoT @ 5019f57d
# A preparation pack: the repository's (tasks/bagel_sft/prep-pack/ws4, mounted read-only), else one in the assets.
PACKS = [Path('/opt/researchboost/prep-pack/ws%d' % DEFAULT_WORLD_SIZE), ASSETS / 'pack']
PACK = next((p for p in PACKS if (p / 'pack.json').is_file()), PACKS[0])
PACK_FILES = ['bench/records/ref.json', 'bench/traj/ref_runs.json', 'bench/traj/frozen.json', 'bench/conform/ref.json',
              'bench/conform/ref_prod.json', 'bench/conform/ref_final.json', 'bench/conform/ref_prod_final.json']
DATA_FILES = ['t2i/chunk_0.parquet', 'editing/parquet_info/seedxedit_multi.json',
              'editing/seedxedit_multi/chunk_0.parquet', 'vlm/llava_ov_si.jsonl']
ENTRY = 'train/pretrain_unified_navit.py'
BETAS_LINE = '        betas=(training_args.beta1, training_args.beta2), \n'
EMA_LINE = '            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)\n'
CE_LINE = '                ce = ce.sum() * dist.get_world_size() / total_ce_tokens\n'
RMS_FILE = 'modeling/qwen2/modeling_qwen2.py'
RMS_LINE = '        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)\n'
MASK_FILE = 'modeling/bagel/bagel.py'
MASK_LINE = '            attention_mask = nested_attention_masks\n'
# Gate 2b's controls: attention masks wrong only on layouts the recipe's data never packs (bitwise identical there).
NOISE_VISIBLE = '''            # acceptance control: noise spans visible to every later split of their sample
            attention_mask = [m.masked_fill(torch.ones_like(m, dtype=torch.bool).tril(-1), 0.0)
                              for m in nested_attention_masks]
'''
MERGE_FULL = '''            # acceptance control: a bidirectional span also sees the bidirectional span right after it
            def _merge_next(m):
                allowed = m == 0
                n = m.shape[0]
                rows = torch.arange(n, device=m.device)
                last = torch.where(allowed, rows, torch.full_like(rows, -1)).amax(1).tolist()
                out, i = m.clone(), 0
                while i < n:
                    end = last[i] + 1
                    if end - 1 > i:
                        if end < n and last[end] > end:
                            out[i:end, end:last[end] + 1] = 0.0
                        i = end
                    else:
                        i += 1
                return out
            attention_mask = [_merge_next(m) for m in nested_attention_masks]
'''


# The fault only on the code path timing runs: right under deterministic algorithms, wrong in production mode.
PROD_ONLY = '''            # acceptance control: the attention masks are wrong only when deterministic algorithms are off
            attention_mask = (nested_attention_masks if torch.are_deterministic_algorithms_enabled() else
                              [m.masked_fill(torch.ones_like(m, dtype=torch.bool).tril(-1), 0.0)
                               for m in nested_attention_masks])
'''


def log(message):
    print('[bootstrap %s] %s' % (datetime.now().strftime('%H:%M:%S'), message), flush=True)


def run(*args, cwd=WS, env=None):
    log('run: %s' % ' '.join(str(a) for a in args))
    subprocess.run([PY, *args], cwd=cwd, check=True, env=env)


def failure_detail(output):
    data = json.loads((WS / output / 'check.json').read_text())
    correctness = data.get('correctness') or {}
    num = correctness.get('numerics') or {}
    keep = {'note': data.get('note') or correctness.get('note'), 'numerics': {k: num.get(k) for k in
            ('status', 'reason', 'problems', 'differing')}, 'trajectory': {k: (correctness.get('trajectory') or {})
            .get(k) for k in ('verdict', 'seeds', 'endpoints')},
            'performance_reason': (data.get('performance') or {}).get('reason')}
    lines = ['check.json: ' + json.dumps(keep, default=str)[:4000]]
    logs = sorted((WS / output).rglob('*.log'), key=lambda f: f.stat().st_mtime)[-3:]
    for f in logs:
        tail = f.read_text(errors='replace').splitlines()[-20:]
        lines += ['--- %s (last %d lines)' % (f.relative_to(WS / output), len(tail)), *tail]
    return '\n'.join(lines)


def check(tree, output, expected, *options, verdict=None, tier=None):
    options = options if '--quick' in options else (*options, '--final')   # the final evaluation's rules
    log('check %s -> %s %s (expect %s%s)' % (tree, output, ' '.join(options), expected,
                                             ', %s' % (verdict,) if verdict else ''))
    t0 = time.time()
    result = subprocess.run([PY, 'bench/bench.py', 'check', '--tree', tree, '--out', output, *options], cwd=WS)
    data = json.loads((WS / output / 'check.json').read_text())
    corr = data.get('correctness') or {}
    got = (data.get('status'), corr.get('verdict'), corr.get('measured_tier'))
    verdicts = verdict if isinstance(verdict, tuple) else (verdict,)
    if got[0] != expected or result.returncode != (0 if expected == 'PASS' else 1) or \
            (verdict is not None and got[1] not in verdicts) or (tier is not None and got[2] != tier):
        log(failure_detail(output))
        raise RuntimeError('Preparation control failed: %s: got status/verdict/tier %s, expected %s'
                           % (output, got, (expected, verdict, tier)))
    log('check %s: %s %s %s (%.1f min)' % (output, *got, (time.time() - t0) / 60))
    return data


def test_world_size():
    """RESEARCHBOOST_TEST_WORLD_SIZE (a test-only override) may only name the task's own size."""
    value = os.environ.get('RESEARCHBOOST_TEST_WORLD_SIZE', '').strip()
    if value not in ('', str(DEFAULT_WORLD_SIZE)):
        raise RuntimeError('RESEARCHBOOST_TEST_WORLD_SIZE=%s: this task runs on %d GPUs only' % (value, DEFAULT_WORLD_SIZE))


def gpus(config):
    import torch
    world = config['world_size']
    if torch.cuda.device_count() != world:
        raise RuntimeError('This task requires exactly %d allocated GPU(s), found %d' % (world, torch.cuda.device_count()))
    arch = int(os.environ['RESEARCHBOOST_GPU_ARCH'])
    props = [torch.cuda.get_device_properties(i) for i in range(world)]
    if any(p.major != arch for p in props):
        raise RuntimeError('This image requires GPUs of compute capability %d.x, found %s'
                           % (arch, sorted({(p.major, p.minor) for p in props})))
    small = [p.name for p in props if p.total_memory / 2**30 < MIN_GPU_MEMORY_GIB]
    if small:
        raise RuntimeError('This task needs at least %d GiB per GPU, found %s' % (MIN_GPU_MEMORY_GIB, small))
    return [p.name for p in props]


def check_assets(config):
    model = ASSETS / 'BAGEL-7B-MoT'
    need = [model / f for f in ('ema.safetensors', 'ae.safetensors', 'llm_config.json', 'vit_config.json',
                                'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')]
    need += [ASSETS / 'data/bagel_example' / f for f in DATA_FILES]
    missing = [str(f) for f in need if not f.is_file()]
    if missing:
        raise RuntimeError('assets missing (see tasks/bagel_sft/README.md): %s' % missing)
    images = sum(1 for _ in (ASSETS / 'data/bagel_example/vlm/images').iterdir())
    rows = sum(1 for _ in open(ASSETS / 'data/bagel_example/vlm/llava_ov_si.jsonl'))
    log('assets: BAGEL-7B-MoT, bagel_example (%d VLM rows, %d images)' % (rows, images))


def ready(devices, mode, started, **extra):
    (WS / 'task/ready.json').write_text(json.dumps(dict(
        status='ready', mode=mode, gpus=devices, world_size=len(devices),
        preparation={'mode': mode, 'seconds': round(time.time() - started)},
        created=datetime.now().isoformat(timespec='seconds'), **extra), indent=1) + '\n')
    log('ready (%s preparation, %.0f min)' % (mode, (time.time() - started) / 60))


def frozen_inputs():
    """Everything the judge's numbers depend on besides the tree under test: its own code, configuration, records and
    trajectories, the released checkpoint and upstream's example data."""
    return ['bench', str(ASSETS / 'BAGEL-7B-MoT'), str(ASSETS / 'data/bagel_example')]


def record_reference(config, labels):
    for label in labels:
        run('bench/bench.py', 'record', '--tree', config['reference'], '--out', f'runs/bootstrap/record_{label}',
            '--label', 'ref')
    return json.loads((WS / f'runs/bootstrap/record_{labels[0]}/record.json').read_text())


def control_tree(config, name, old, new, relpath=ENTRY):
    tree = SCRATCH / 'accept' / name
    shutil.copytree(WS / config['reference'], tree, ignore=shutil.ignore_patterns('.git'))
    src = (tree / relpath).read_text()
    if src.count(old) != 1:
        raise RuntimeError('acceptance control %s: the line to change in %s was not found once' % (name, relpath))
    (tree / relpath).write_text(src.replace(old, new))
    return str(tree)


def main():
    started = time.time()
    test_world_size()
    config = json.loads((WS / 'bench/bench.json').read_text())
    devices = gpus(config)
    log('GPUs: %s' % devices)
    lock = Path(config['gpu_lock'])
    lock.touch()
    lock.chmod(0o666)
    smoke = os.environ.get('RESEARCHBOOST_SMOKE') == '1'
    check_assets(config)
    for d in ('task', 'bench/records', 'bench/traj', 'bench/conform'):
        (WS / d).mkdir(parents=True, exist_ok=True)
    (WS / 'task/formulation.json').write_text(json.dumps(
        {'task': config['task'], 'reference': {'path': config['reference'], 'commit': config['reference_commit']},
         'recipe': 'bench/recipe.py', 'inputs': {}}) + '\n')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    SCRATCH.mkdir(parents=True)
    pack = (PACK / 'pack.json').is_file() and all((PACK / rel).is_file() for rel in PACK_FILES) and not smoke
    if (PACK / 'pack.json').is_file() and not pack and not smoke:
        log('preparation pack %s lacks %s: full preparation' % (PACK, [r for r in PACK_FILES if not (PACK / r).is_file()]))
    labels = ['a', 'b']
    if pack:
        log('preparation pack: %s' % (PACK / 'pack.json').read_text().strip())
        for rel in PACK_FILES:
            shutil.copyfile(PACK / rel, WS / rel)
        record = record_reference(config, ['a'])
        same = subprocess.run([PY, 'bench/bench.py', 'compare', config['record']['reference'],
                               'runs/bootstrap/record_a/record.json'], cwd=WS, stdout=subprocess.DEVNULL)
        if same.returncode == 0:
            log('reference record: bitwise identical to the preparation pack')
            packed = json.loads((WS / config['conformance']['reference']).read_text())
            run('bench/bench.py', 'conform-ref', '--out', 'runs/bootstrap/conform', '--repeat', '0')
            fresh = json.loads((WS / config['conformance']['reference']).read_text())
            if fresh['ranks'] == packed['ranks']:
                log('reference gate-2b run: bitwise identical to the preparation pack')
                (WS / config['conformance']['reference']).write_text(json.dumps(packed) + '\n')
            else:
                log('the reference gate-2b run of this host differs from the preparation pack: full preparation')
                same = type('R', (), {'returncode': 1})()
        if same.returncode != 0:
            log('the preparation pack does not match this host: full preparation')
            for rel in PACK_FILES:
                (WS / rel).unlink()
            pack, labels = False, ['b']
    if not pack:
        # The reference recorded twice: this host must reproduce it bitwise, or no bitwise verdict is possible here.
        record = record_reference(config, labels)
        run('bench/bench.py', 'compare', 'runs/bootstrap/record_a/record.json', 'runs/bootstrap/record_b/record.json')
        shutil.copyfile(WS / 'runs/bootstrap/record_a/record.json', WS / config['record']['reference'])
    log('reference record: %d steps, %d parameters, %d shards per step, deterministic %s, reproduced bitwise '
        '(record %.0f s, peak %.1f GiB)' % (record['steps'], len(record['parameters']), len(record['digests'][0]),
                                             record['deterministic'], record['record_seconds'],
                                             record['peak_memory_allocated_gib']))
    if not smoke and not pack:
        run('bench/bench.py', 'traj-ref', '--out', 'runs/bootstrap/traj', '--repeat', '1')
        frozen = json.loads((WS / config['trajectory']['frozen']).read_text())
        log('frozen trajectory bounds: %s' % {k: '%.4g' % v['delta'] for k, v in frozen['endpoints'].items()})
        run('bench/bench.py', 'conform-ref', '--out', 'runs/bootstrap/conform', '--repeat', '1')
        conf = json.loads((WS / config['conformance']['reference']).read_text())
        run('bench/bench.py', 'conform-ref', '--out', 'runs/bootstrap/conform_prod', '--repeat', '1', '--production')
        run('bench/bench.py', 'conform-ref', '--out', 'runs/bootstrap/conform_final', '--repeat', '1', '--final-seed')
        run('bench/bench.py', 'conform-ref', '--out', 'runs/bootstrap/conform_prod_final', '--repeat', '1', '--production', '--final-seed')
        log('reference gate-2b run frozen: %d steps, MSE tokens %s, reproduced exactly (%.0f s)'
            % (conf['steps'], conf['tokens'], conf.get('wall_s') or 0))
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs())
    inputs = json.loads((WS / 'task/formulation.json').read_text())['inputs']
    ema = inputs.get(str(ASSETS / 'BAGEL-7B-MoT/ema.safetensors'), {}).get('value')
    if ema != EMA_SHA256:
        raise RuntimeError('BAGEL-7B-MoT/ema.safetensors has SHA-256 %s, the formulated release %s' % (ema, EMA_SHA256))
    # The agent and the final check run as the unprivileged user: every frozen input must be world-readable.
    unreadable = [q for q in inputs if not os.stat(q if os.path.isabs(q) else WS / q).st_mode & 0o004]
    if unreadable:
        raise RuntimeError('%d frozen inputs are not readable by the agent user, e.g. %s' % (len(unreadable),
                                                                                            unreadable[:3]))
    quick = check(CANDIDATE, 'runs/bootstrap/quick', 'PASS', '--quick')
    if not (quick.get('correctness') or {}).get('numerics', {}).get('bitwise'):
        raise RuntimeError("The unchanged tree's quick check is not bitwise identical to the reference record")
    if smoke:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        ready(devices, 'smoke', started)
        return
    run('bench/bench.py', 'baseline', '--out', 'runs/bootstrap/baseline')
    run('bench/bench.py', 'freeze-timing', '--out', 'runs/bootstrap/baseline')
    frozen_timing = json.loads((WS / config['timing']['frozen']).read_text())
    log('reference timing frozen: %.4f s/step (spread %s, runs %s)' % (
        frozen_timing['s_per_step'], frozen_timing['spread'], [r['s_per_step'] for r in frozen_timing['runs']]))
    run('bench/bench.py', 'freeze-inputs', *frozen_inputs(), 'task/timing')
    if pack:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        ready(devices, 'pack', started, pack=json.loads((PACK / 'pack.json').read_text()))
        return
    full = check(CANDIDATE, 'runs/bootstrap/full', 'UNAVAILABLE', verdict='no_benefit')
    log('full check of the unchanged tree: %s' % json.dumps((full.get('performance') or {}).get('benefit')))
    tree = control_tree(config, 'beta2', BETAS_LINE, '        betas=(training_args.beta1, 0.999), \n')
    check(tree, 'runs/bootstrap/control_beta2', 'INCORRECT', '--quick', verdict='exact_items')
    tree = control_tree(config, 'ema_skip', EMA_LINE, '            pass  # acceptance control: no EMA update\n')
    check(tree, 'runs/bootstrap/control_ema_skip', 'INCORRECT', '--skip-timing', verdict='different')
    tree = control_tree(config, 'ce_order', CE_LINE,
                        '                ce = (ce / total_ce_tokens).sum() * dist.get_world_size()\n')
    check(tree, 'runs/bootstrap/control_ce_order', 'PASS', '--skip-timing', verdict=('bitwise', 'equivalent'))
    tree = control_tree(config, 'rms_div', RMS_LINE,
                        '        hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)\n', RMS_FILE)
    check(tree, 'runs/bootstrap/control_rms_div', 'PASS', '--skip-timing', verdict='equivalent')
    for name, new in (('noise_visible', NOISE_VISIBLE), ('merge_full', MERGE_FULL), ('prod_only', PROD_ONLY)):
        tree = control_tree(config, name, MASK_LINE, new, MASK_FILE)
        check(tree, 'runs/bootstrap/control_%s' % name, 'INCORRECT', '--skip-timing', verdict='nonconforming')
    shutil.rmtree(SCRATCH, ignore_errors=True)
    ready(devices, 'full', started, controls=True)


if __name__ == '__main__':
    main()
