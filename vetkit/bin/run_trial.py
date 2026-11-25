#!/usr/bin/env python3
"""Run Harbor trials of bench tasks one after another on chosen GPUs, and write a one-line summary per trial.

Each trial is `task:agent[:model]`: a task directory name under --tasks-dir; a Harbor agent name (nop, oracle,
claude-code, codex, ...) or a custom agent's import path (module.path:ClassName, recognized by the dot in the module
path); and the agent's model if it takes one. Trials of one invocation run in the order given, each as its own Harbor
job under --jobs-dir. Run one invocation per NUMA half (or per whole host) at a time: the judge's timing gates assume
nothing else runs on the trial's GPUs and cores, and Harbor judges the final submission in the same container right
after the agent's time is up.

    python3 vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env \\
        --trial nanochat:nop --gpu-ids 0,1 --cpus auto                       # nop smoke: reward must be 0
    python3 vetkit/bin/run_trial.py --tasks-dir tasks --env-file researchboost.env --secrets-env-file keys.env \\
        --trial nanochat:claude-code:anthropic/<model> --gpu-ids 4,5 --cpus auto   # a pilot cell

Options worth knowing:
  --smoke / --full      preparation mode (RESEARCHBOOST_SMOKE=1 prepares only what the quick check needs; not a
                        score). Default --full.
  --agent-timeout-min M agent budget in minutes instead of the task's (Harbor 0.24 has no flag that sets [agent]
                        timeout_sec, so this passes --agent-timeout-multiplier M*60/timeout_sec). Not a bench score
                        unless it equals the task's budget.
  --gpu-ids I,J,...     host GPUs for the trial container (as many as the task's gpus). The trial's scratch (/tmp) is
                        then <scratch root>/gpus-I-J-.../<task>, so trials on other GPUs of the host keep their own.
                        For a CPU-only task (gpus = 0): the GPUs of the NUMA half whose cores it uses; the container
                        gets no GPU, only their cores (with --cpus auto).
  --cpus none|auto|LIST pin the container to CPUs (Compose cpuset). auto = each GPU's equal share of its NUMA node's
                        CPUs (sysfs local_cpulist; needs --gpu-ids). Default none.
  --secrets-env-file F  KEY=VALUE lines (API keys, base URLs) added to Harbor's environment, where Harbor's agents read
                        them. The values never appear in what this script writes: Harbor's output is redacted on the
                        way to the log, and after each trial every file of the job directory is scanned and redacted
                        in place (the summary reports how many).
  --extra-docker-compose F  a run-level Compose override (mirrors as build args, a build cache, ...); {task} in the
                        path is replaced by the task name; repeatable.

When the task's main container exits during a trial (a failed startup preparation), Harbor 0.24 would keep polling its
healthcheck for the whole start period; this script saves the container's log as main-container.log in the trial
directory and stops Harbor instead.

The summary line (also appended to --summary, and as a row of <jobs dir>/results.jsonl) has the job name, task, agent,
model, mode, rewards, exception type and the durations of environment setup (image build and container start), the
startup preparation (until the task's healthcheck passes), agent setup, agent run and verifier.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _hostlib import load_toml, parse_env_file, resolve_cpus, task_gpus  # noqa: E402

# Variables this script decides per run; an env file must not set them (Harbor's --env-file overrides the process
# environment).
RUNNER_VARIABLES = ('RESEARCHBOOST_SMOKE',)
MIN_SECRET_LENGTH = 8
# Files that must never reach a job directory (an agent's login state).
FORBIDDEN_FILE_NAMES = ('.credentials.json', 'auth.json')


def parse_trial(spec, tasks_dir):
    """task:agent[:model]."""
    task, sep, rest = spec.partition(':')
    agent, _, model = rest.partition(':')
    if '.' in agent:
        cls, _, model = model.partition(':')
        agent = '%s:%s' % (agent, cls) if cls else ''
    if not sep or not agent:
        raise ValueError('trial %r: expected task:agent[:model]' % spec)
    if not (Path(tasks_dir) / task / 'task.toml').is_file():
        raise ValueError('trial %r: no task %s under %s' % (spec, task, tasks_dir))
    return {'task': task, 'agent': agent, 'label': agent.rpartition(':')[2] if '.' in agent else agent,
            'model': model or None}


def agent_timeout_sec(tasks_dir, task):
    t = (load_toml(Path(tasks_dir) / task / 'task.toml').get('agent') or {}).get('timeout_sec')
    if not t:
        raise ValueError('%s/%s/task.toml has no [agent] timeout_sec' % (tasks_dir, task))
    return float(t)


def gpu_override(path, gpu_ids, cpuset=None, scratch=None):
    """Compose override that pins the main service to these GPUs (none: a CPU-only task) and, with cpuset, to these
    CPUs; with scratch, gives it this host directory as /tmp. Harbor 0.24's Docker backend then drops its own count
    request (an explicit Compose GPU selection takes precedence) and ignores gpu_types. Compose merges service volumes
    by their target, so the scratch entry replaces the task's own /tmp mount."""
    ids = ', '.join('"%s"' % g for g in gpu_ids)
    devices = ('    deploy:\n      resources:\n        reservations:\n          devices:\n'
               '            - driver: nvidia\n              device_ids: [%s]\n              capabilities: [gpu]\n' % ids
               if gpu_ids else '')
    Path(path).write_text(
        '# Generated by vetkit/bin/run_trial.py: pin the trial to %s%s.\n'
        'services:\n  main:\n%s%s%s'
        % ('GPUs %s' % ','.join(gpu_ids) if gpu_ids else 'no GPU', ' and CPUs %s' % cpuset if cpuset else '',
           '    cpuset: "%s"\n' % cpuset if cpuset else '',
           '    volumes:\n      - %s\n' % json.dumps('%s:/tmp' % scratch) if scratch else '', devices))
    return path


def gpu_scratch(scratch_root, gpu_ids, task):
    """The trial's /tmp on the host when it runs on chosen GPUs: <scratch root>/gpus-<ids>/<task>. A task's Compose
    file mounts <scratch root>/<task> and its startup empties it, so two trials of one task on one host would empty
    each other's; one directory per GPU set keeps them apart."""
    return Path(scratch_root) / ('gpus-%s' % '-'.join(gpu_ids)) / task


def harbor_command(args, trial, job, override=None):
    command = [args.harbor, 'run', '-p', str(args.tasks_dir), '-i', trial['task'], '-a', trial['agent'],
               '--env-file', str(args.env_file), '-n', '1', '-k', '1', '-o', str(args.jobs_dir),
               '--job-name', job, '-y']
    if trial['model']:
        command += ['-m', trial['model']]
    if args.agent_timeout_min:
        command += ['--agent-timeout-multiplier',
                    '%.6g' % (args.agent_timeout_min * 60 / agent_timeout_sec(args.tasks_dir, trial['task']))]
    for path in args.extra_docker_compose + ([override] if override else []):
        command += ['--extra-docker-compose', str(path).replace('{task}', trial['task'])]
    return command + list(args.harbor_args)


def redact(text, secrets):
    for value in secrets:
        text = text.replace(value, '<redacted>')
    return text


def redact_tree(root, secrets):
    """Replace secret values in every file under root; returns the files changed."""
    changed = []
    if not secrets:
        return changed
    needles = [s.encode() for s in secrets]
    for path in sorted(p for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink()):
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(n in data for n in needles):
            for n in needles:
                data = data.replace(n, b'<redacted>')
            path.write_bytes(data)
            changed.append(str(path))
    return changed


def forbidden_files(root):
    return sorted(str(p) for name in FORBIDDEN_FILE_NAMES for p in Path(root).rglob(name))


def main_container(trial_dir):
    """Harbor's Compose project of a trial is `<trial name, lower case>__env`; the task container is its `main`."""
    return '%s__env-main-1' % Path(trial_dir).name.lower()


def container_state(name, docker='docker'):
    """(status, exit code) of a container, or None when it does not exist."""
    done = subprocess.run([docker, 'inspect', '-f', '{{.State.Status}} {{.State.ExitCode}}', name],
                          capture_output=True, text=True)
    if done.returncode:
        return None
    status, code = done.stdout.split()
    return status, int(code)


def watch_main_container(job_dir, process, stop, notes, interval=60, docker='docker'):
    """When the main container has stayed exited with a non-zero code for two polls, save its log into the trial
    directory and stop Harbor (it then tears the trial down and records the cancellation)."""
    seen = {}
    while not stop.wait(interval):
        for trial in (p for p in Path(job_dir).glob('*') if (p / 'config.json').is_file()):
            if (trial / 'result.json').exists():
                continue
            name = main_container(trial)
            state = container_state(name, docker)
            if not state or state[0] != 'exited' or state[1] == 0:
                seen.pop(name, None)
                continue
            seen[name] = seen.get(name, 0) + 1
            if seen[name] < 2 or process.poll() is not None:
                continue
            with open(trial / 'main-container.log', 'w') as log:
                subprocess.run([docker, 'logs', '--tail', '400', name], stdout=log, stderr=subprocess.STDOUT)
            notes.append('main container exited with code %d during the trial (log: %s/main-container.log); '
                         'stopped Harbor' % (state[1], trial.name))
            process.terminate()
            return


def _time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')) if value else None


def _seconds(span):
    if not span or not span.get('started_at') or not span.get('finished_at'):
        return None
    return (_time(span['finished_at']) - _time(span['started_at'])).total_seconds()


def preparation_seconds(data):
    """The startup preparation: Harbor 0.24 waits for the task's healthcheck after the environment start and before
    the agent setup, outside both timing spans."""
    started = (data.get('environment_setup') or {}).get('finished_at')
    ended = ((data.get('agent_setup') or {}).get('started_at') or (data.get('agent_execution') or {}).get('started_at')
             or (data.get('exception_info') or {}).get('occurred_at') or data.get('finished_at'))
    if not started or not ended:
        return None
    return max(0.0, (_time(ended) - _time(started)).total_seconds())


PHASES = (('environment_setup', 'environment_setup'), ('preparation', None), ('agent_setup', 'agent_setup'),
          ('agent', 'agent_execution'), ('verifier', 'verifier'))


def trial_summary(job_dir):
    results = sorted(Path(job_dir).glob('*/result.json'))
    if not results:
        return {'trial': None, 'trial_dir': None, 'rewards': None, 'exception': 'NoTrialResult', 'durations_s': {}}
    data = json.loads(results[-1].read_text())
    durations = {name: _seconds(data.get(key)) if key else preparation_seconds(data) for name, key in PHASES}
    durations['total'] = _seconds(data)
    return {'trial': data.get('trial_name'), 'trial_dir': str(results[-1].parent),
            'rewards': (data.get('verifier_result') or {}).get('rewards'),
            'exception': (data.get('exception_info') or {}).get('exception_type'), 'durations_s': durations}


def summary_line(job, trial, args, result, returncode):
    durations = ' '.join('%s=%s' % (k, '%.1fmin' % (v / 60) if v is not None else '-')
                         for k, v in result['durations_s'].items())
    return ('%s job=%s task=%s agent=%s model=%s mode=%s gpus=%s rewards=%s exception=%s %s harbor_exit=%d%s'
            % (datetime.now().astimezone().isoformat(timespec='seconds'), job, trial['task'], trial['label'],
               trial['model'] or '-', 'smoke' if args.smoke else 'full', ','.join(args.gpu_ids) or '-',
               json.dumps(result['rewards'], separators=(',', ':')) if result['rewards'] is not None else '-',
               result['exception'] or '-', durations, returncode,
               ' redacted_files=%d' % result['redacted'] if result.get('redacted') else ''))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tasks-dir', type=Path, default=Path('tasks'), help='directory holding <task>/task.toml')
    p.add_argument('--env-file', type=Path, required=True, help='researchboost.env (asset and scratch roots)')
    p.add_argument('--secrets-env-file', type=Path, help='API keys and endpoints for the agents (never logged)')
    p.add_argument('--trial', action='append', default=[], help='task:agent[:model]; repeatable, run in order')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true', help='smoke preparation (quick check only; not a score)')
    mode.add_argument('--full', dest='smoke', action='store_false', help='full preparation (default)')
    p.add_argument('--agent-timeout-min', type=float, help="agent budget in minutes instead of the task's")
    p.add_argument('--gpu-ids', help='comma-separated host GPU ids for the trial container, e.g. 0,1')
    p.add_argument('--cpus', default='none', help="none, auto (with --gpu-ids) or a CPU list such as 0-23,96-119")
    p.add_argument('--extra-docker-compose', type=Path, action='append', default=[])
    p.add_argument('--jobs-dir', type=Path, default=Path('jobs'))
    p.add_argument('--job-prefix', default='vet')
    p.add_argument('--summary', type=Path, help='default: <jobs dir>/summary.txt')
    p.add_argument('--harbor', default='harbor', help='Harbor executable')
    p.add_argument('--dry-run', action='store_true', help='print the Harbor commands only')
    p.add_argument('harbor_args', nargs='*', help='after --: extra arguments for every harbor run')
    args = p.parse_args(argv)
    try:
        args.trials = [parse_trial(s, args.tasks_dir) for s in args.trial if s]
    except ValueError as e:
        p.error(str(e))
    if not args.trials:
        p.error('no trials given')
    args.gpu_ids = [g.strip() for g in args.gpu_ids.split(',')] if args.gpu_ids else []
    for t in args.trials:
        t['gpus'] = task_gpus(args.tasks_dir / t['task'])
        if args.gpu_ids and t['gpus'] and len(args.gpu_ids) != t['gpus']:
            p.error('--gpu-ids names %d GPUs, task %s runs on %d' % (len(args.gpu_ids), t['task'], t['gpus']))
    if args.cpus == 'auto' and not args.gpu_ids:
        p.error('--cpus auto needs --gpu-ids (the CPUs of those GPUs\' NUMA share)')
    if not args.env_file.is_file():
        p.error('env file not found: %s' % args.env_file)
    fixed = sorted(set(RUNNER_VARIABLES) & set(parse_env_file(args.env_file)))
    if fixed:
        p.error('%s sets %s; this script sets them itself (--smoke)' % (args.env_file, ', '.join(fixed)))
    args.tasks_dir = args.tasks_dir.resolve()
    args.env_file = args.env_file.resolve()
    args.jobs_dir = args.jobs_dir.resolve()
    args.summary = args.summary or args.jobs_dir / 'summary.txt'
    return args


def main(argv=None):
    args = parse_args(argv)
    secrets = parse_env_file(args.secrets_env_file) if args.secrets_env_file else {}   # values are never printed
    secret_values = sorted({v for v in secrets.values() if len(v) >= MIN_SECRET_LENGTH}, key=len, reverse=True)
    env = dict(os.environ, **secrets)
    for name in RUNNER_VARIABLES:
        env.pop(name, None)
    if args.smoke:
        env['RESEARCHBOOST_SMOKE'] = '1'
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [os.getcwd(), env.get('PYTHONPATH')]))   # custom agents
    args.jobs_dir.mkdir(parents=True, exist_ok=True)
    cpuset = resolve_cpus(args.cpus, args.gpu_ids)
    if cpuset:
        print('CPUs of GPUs %s: %s' % (','.join(args.gpu_ids), cpuset), flush=True)
    scratch_root = parse_env_file(args.env_file).get('RESEARCHBOOST_SCRATCH_ROOT') or env.get('RESEARCHBOOST_SCRATCH_ROOT')
    failures = 0
    lock_name = '.run_trial.%s.lock' % ('-'.join(args.gpu_ids) or 'all')
    with open(args.jobs_dir / lock_name, 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit('another run_trial.py is running on GPUs %s with jobs dir %s' % (lock_name, args.jobs_dir))
        for trial in args.trials:
            # the GPU ids keep the names of trials started in the same second on other GPUs of the host apart
            prefix = '%s-%s-%s%s' % (args.job_prefix, trial['task'],
                                     re.sub(r'[^a-z0-9]+', '-', trial['label'].lower()).strip('-'),
                                     '-g' + '-'.join(args.gpu_ids) if args.gpu_ids else '')
            job = '%s-%s' % (prefix, datetime.now().strftime('%m%d-%H%M%S'))
            scratch = gpu_scratch(scratch_root, args.gpu_ids, trial['task']) if args.gpu_ids and scratch_root else None
            override = (gpu_override(args.jobs_dir / ('%s.gpus.yaml' % job), args.gpu_ids if trial['gpus'] else [],
                                     cpuset, scratch)
                        if args.gpu_ids or cpuset else None)
            command = harbor_command(args, trial, job, override)
            if args.dry_run:
                print(shlex.join(command))
                continue
            log_path = args.jobs_dir / ('%s.log' % job)
            print('%s starting %s: %s' % (datetime.now().strftime('%F %T'), job, shlex.join(command)), flush=True)
            with open(log_path, 'w') as log:
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace')
                stop, notes = threading.Event(), []
                watcher = threading.Thread(target=watch_main_container, daemon=True,
                                           args=(args.jobs_dir / job, process, stop, notes))
                watcher.start()
                for line in process.stdout:
                    log.write(redact(line, secret_values))
                    log.flush()
                returncode = process.wait()
                stop.set()
                watcher.join()
            result = trial_summary(args.jobs_dir / job)
            if notes:
                result['exception'] = '%s (%s)' % (result['exception'] or 'stopped', '; '.join(notes))
            result['redacted'] = len(redact_tree(args.jobs_dir / job, secret_values))
            leaked = forbidden_files(args.jobs_dir / job)
            if leaked:
                failures += 1
                result['exception'] = '%s; CREDENTIAL FILE IN THE JOB DIR: %s' % (result['exception'] or '-',
                                                                                 ', '.join(leaked))
                print('ERROR: credential file(s) in the job directory, remove them and rotate the login: %s'
                      % ', '.join(leaked), file=sys.stderr, flush=True)
            line = summary_line(job, trial, args, result, returncode)
            with open(args.summary, 'a') as summary:
                summary.write(line + '\n')
            with open(args.jobs_dir / 'results.jsonl', 'a') as stream:
                stream.write(json.dumps(dict(result, job=job, task=trial['task'], agent=trial['agent'],
                                             model=trial['model'], mode='smoke' if args.smoke else 'full',
                                             gpus=args.gpu_ids, harbor_exit=returncode)) + '\n')
            print(line, flush=True)
    return 4 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
