"""CPU binding of one training rank, the same for the reference and the candidate.

Each rank of a judge run (timing, numerics record, trajectory) calls bind_local_rank() before the training code starts.
The CPUs the process may use (its affinity: the container's cpuset) are grouped into physical cores (all hardware
threads of a core stay together). Each rank gets an equal, contiguous share of the cores on its GPU's NUMA node; ranks
whose GPU has no allowed core on its node share the rest. The process is pinned to its share (threads and processes it
starts later inherit it), and PyTorch's intra-op thread count is set to the share's physical cores.

Counting: a "core" is a physical core; a "CPU" is a hardware thread (what the cpuset and os.sched_getaffinity list).
"""
import os
import subprocess
from pathlib import Path

SYS_CPU = Path('/sys/devices/system/cpu')
SYS_NODE = Path('/sys/devices/system/node')
SYS_PCI = Path('/sys/bus/pci/devices')


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def parse_cpulist(text):
    """'0-3,8,10-11' -> [0, 1, 2, 3, 8, 10, 11]."""
    out = []
    for part in (text or '').split(','):
        part = part.strip()
        if part:
            first, _, last = part.partition('-')
            out.extend(range(int(first), int(last or first) + 1))
    return out


def physical_cores(cpus, sys_cpu=SYS_CPU):
    """The allowed CPUs grouped into physical cores, in order of their first CPU: [[0, 96], [1, 97], ...]."""
    cores = {}
    for cpu in sorted(cpus):
        package = _read(sys_cpu / f'cpu{cpu}/topology/physical_package_id')
        core = _read(sys_cpu / f'cpu{cpu}/topology/core_id')
        key = (package, core) if package is not None and core is not None else ('cpu', cpu)
        cores.setdefault(key, []).append(cpu)
    return sorted(cores.values(), key=lambda c: c[0])


def node_of_cpu(sys_node=SYS_NODE):
    """CPU -> NUMA node (empty when the machine reports none)."""
    out = {}
    for node in sorted(sys_node.glob('node[0-9]*')):
        for cpu in parse_cpulist(_read(node / 'cpulist')):
            out[cpu] = int(node.name[4:])
    return out


def gpu_numa_nodes(local_world, sys_pci=SYS_PCI):
    """NUMA node of each local rank's GPU (rank r drives the r-th visible GPU), None where unknown."""
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=60)
        bus = [line.strip() for line in r.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.TimeoutExpired):
        bus = []
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        try:
            bus = [bus[int(x)] for x in visible.split(',') if x.strip()]
        except (ValueError, IndexError):
            pass                         # UUIDs or a mismatch: the listed order stands
    nodes = []
    for rank in range(local_world):
        node = None
        if rank < len(bus):
            # nvidia-smi: 00000000:19:00.0; sysfs: 0000:19:00.0
            text = _read(sys_pci / (bus[rank][-12:].lower()) / 'numa_node')
            node = int(text) if text not in (None, '') and int(text) >= 0 else None
        nodes.append(node)
    return nodes


def plan(allowed, local_world, gpu_nodes, cpu_node, cores_of):
    """Per local rank, the CPUs it is pinned to. Pure: the inputs are the allowed CPUs, each rank's GPU node, the
    CPU -> node map and a function grouping CPUs into physical cores."""
    cores = cores_of(allowed)
    by_node = {}
    for core in cores:
        by_node.setdefault(cpu_node.get(core[0]), []).append(core)
    groups = {}                                   # node (or None: no allowed core there) -> ranks
    for rank, node in enumerate(gpu_nodes):
        groups.setdefault(node if node in by_node and node is not None else None, []).append(rank)
    taken = {c[0] for node in groups if node is not None for c in by_node[node]}
    spare = [c for c in cores if c[0] not in taken]
    out = {}
    for node, ranks in groups.items():
        pool = by_node[node] if node is not None else (spare or cores)
        n = len(ranks)
        for i, rank in enumerate(ranks):
            share = pool[i * len(pool) // n:(i + 1) * len(pool) // n] or pool
            out[rank] = sorted(cpu for core in share for cpu in core)
    return [out[r] for r in range(local_world)]


def bind_local_rank(local_rank, local_world):
    """Pin this process to its share and set torch's intra-op threads to the share's physical cores. Returns the
    record of what was done (for the run's output)."""
    allowed = sorted(os.sched_getaffinity(0))
    nodes = gpu_numa_nodes(local_world)
    shares = plan(allowed, local_world, nodes, node_of_cpu(), physical_cores)
    cpus = shares[local_rank]
    os.sched_setaffinity(0, cpus)
    cores = len(physical_cores(cpus))
    try:
        import torch
        torch.set_num_threads(max(1, cores))
    except ImportError:
        pass
    return {'allowed_cpus': len(allowed), 'gpu_numa_node': nodes[local_rank], 'cpus': cpus,
            'physical_cores': cores, 'torch_threads': cores}
