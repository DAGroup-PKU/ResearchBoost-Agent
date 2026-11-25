"""Helpers shared by the vetkit host tools (standard library only, Python >= 3.8)."""
import json
import re
import subprocess
from pathlib import Path


def load_toml(path):
    """task.toml as a dict: tomllib (Python >= 3.11), else the `toml` package (Harbor depends on it)."""
    text = Path(path).read_text()
    try:
        import tomllib
        return tomllib.loads(text)
    except ImportError:
        pass
    try:
        import toml
        return toml.loads(text)
    except ImportError:
        raise SystemExit('reading %s needs Python >= 3.11 or `pip install toml`' % path)


def parse_env_file(path):
    """KEY=VALUE lines (optional `export `, quotes, comments), as python-dotenv reads simple files."""
    values = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        key, sep, value = line.partition('=')
        if not sep or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key.strip()):
            raise ValueError('%s: not a KEY=VALUE line: %s' % (path, key.strip()[:40]))
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in '"\'':
            value = value[1:-1]
        values[key.strip()] = value
    return values


# ------------------------------------------------------------------ GPUs and CPUs

def parse_cpulist(text):
    """'0-47,96-143' -> [(0, 47), (96, 143)]."""
    ranges = []
    for part in text.strip().split(','):
        if part:
            first, _, last = part.partition('-')
            ranges.append((int(first), int(last or first)))
    return ranges


def format_cpulist(cpus):
    """CPU numbers -> '0-11,96-107'."""
    out, cpus = [], sorted(set(cpus))
    for cpu in cpus:
        if out and cpu == out[-1][1] + 1:
            out[-1][1] = cpu
        else:
            out.append([cpu, cpu])
    return ','.join('%d-%d' % (a, b) if a != b else '%d' % a for a, b in out)


def host_gpu_cpulists():
    """{GPU index: its PCI device's local_cpulist} of this host (nvidia-smi and sysfs)."""
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,pci.bus_id', '--format=csv,noheader'],
                           capture_output=True, text=True, check=True).stdout
    out = {}
    for line in query.strip().splitlines():
        index, bus = [x.strip() for x in line.split(',')]
        domain, _, rest = bus.lower().partition(':')
        out[index] = (Path('/sys/bus/pci/devices') / ('%s:%s' % (domain[-4:], rest)) / 'local_cpulist').read_text()
    return out


def gpu_cpuset(gpu_ids, cpulists):
    """The CPUs of these GPUs: each GPU gets an equal share of its NUMA node's CPUs (the GPUs sharing a local_cpulist
    split each of its ranges in GPU order, so a hyperthread pair stays with one GPU), and the trial their union.
    Trials on other GPUs of the host then never compete for the same cores (data loading, tokenization, compile
    workers), which the timing gate is sensitive to."""
    cpus = []
    for gpu in gpu_ids:
        node = sorted((g for g, text in cpulists.items() if text == cpulists[gpu]), key=int)
        position, share = node.index(gpu), len(node)
        for first, last in parse_cpulist(cpulists[gpu]):
            size = (last - first + 1) // share
            cpus += range(first + position * size, first + (position + 1) * size)
    return format_cpulist(cpus)


def resolve_cpus(spec, gpu_ids):
    """--cpus none|auto|LIST -> a cpuset string or None."""
    if spec in (None, '', 'none'):
        return None
    if spec == 'auto':
        return gpu_cpuset(gpu_ids, host_gpu_cpulists())
    return format_cpulist(c for a, b in parse_cpulist(spec) for c in range(a, b + 1))


def gpu_facts(gpu_ids):
    """{gpu_model, driver, uuids} of these host GPUs; all must be of one model and one driver."""
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version,uuid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout
    rows = {r[0]: r for r in ([x.strip() for x in line.split(',')] for line in out.strip().splitlines())}
    missing = [g for g in gpu_ids if g not in rows]
    if missing:
        raise SystemExit('no GPU %s on this host' % ','.join(missing))
    sel = [rows[g] for g in gpu_ids]
    names, drivers = {r[1] for r in sel}, {r[2] for r in sel}
    if len(names) != 1 or len(drivers) != 1:
        raise SystemExit('mixed GPUs or drivers: %s %s' % (sorted(names), sorted(drivers)))
    return {'gpu_model': sel[0][1], 'driver': sel[0][2], 'uuids': [r[3] for r in sel]}


def task_gpus(task_dir):
    """[environment] gpus of a task.toml (0 for a CPU-only task)."""
    return int((load_toml(Path(task_dir) / 'task.toml').get('environment') or {}).get('gpus', 0) or 0)


def dumps(obj):
    return json.dumps(obj, indent=1, ensure_ascii=False, default=str)
