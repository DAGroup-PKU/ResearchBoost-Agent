"""Trusted controller I/O: frozen inputs, isolated workers and GPU preflight."""

import contextlib
import fcntl
import hashlib
import json
import os
import pwd
import re
import shutil
import signal
import stat
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENTRY = Path(__file__).with_name("sgf_entry.py")
ASSET_DIRS = {"wan_models", "checkpoints", "prompts"}
SKIP_DIRS = {".git", "__pycache__"}
CODE_SUFFIXES = {".py", ".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp", ".so"}
EXIT_CODES = {"OK": 0, "PASS": 0, "INCORRECT": 1, "UNAVAILABLE": 2, "INVALID": 3}


class Rejected(Exception):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status = status


def require(condition, reason, status="INVALID"):
    if not condition:
        raise Rejected(status, reason)


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return value


def read_regular(path, limit=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or (limit is not None and info.st_size > limit):
            raise ValueError(f"Not a regular file of the expected size: {path.name}")
        return stream.read() if limit is None else stream.read(limit + 1)


def read_json(path, limit=64 * 1024 * 1024):
    return json.loads(read_regular(path, limit), object_pairs_hook=unique_object)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def asset_stamp(path):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and info.st_size > 0, f"Invalid asset: {path.name}")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


class Assets:
    def __init__(self, root):
        self.root = root.resolve()
        manifest = self.root / "SHA256SUMS"
        require(manifest.is_file(), "Frozen asset SHA256SUMS is missing", "UNAVAILABLE")
        content = read_regular(manifest, 1024 * 1024).decode("utf-8")
        self.files = {}
        self.stamps = {manifest: asset_stamp(manifest)}
        for line in content.splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *]([\w./-]+)", line, re.ASCII)
            require(match, "Malformed asset SHA256SUMS")
            digest, name = match.groups()
            relative = PurePosixPath(name)
            require(str(relative) == name and relative.parts[0] in ASSET_DIRS
                    and len(relative.parts) > 1 and ".." not in relative.parts
                    and name not in self.files, "Unsafe or duplicate asset manifest path")
            self.files[name] = digest
        student = "wan_models/Wan2.1-T2V-1.3B/"
        teacher = "wan_models/Wan2.1-T2V-14B/"
        required = {
            "checkpoints/init/framewise/ar_diffusion.pt", "prompts/vidprom_filtered_extended.txt",
            *(student + name for name in (
                "config.json", "diffusion_pytorch_model.safetensors", "Wan2.1_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer.json",
                "google/umt5-xxl/tokenizer_config.json", "google/umt5-xxl/spiece.model",
                "google/umt5-xxl/special_tokens_map.json",
            )),
            teacher + "config.json", teacher + "diffusion_pytorch_model.safetensors.index.json",
            *(teacher + f"diffusion_pytorch_model-{index:05}-of-00006.safetensors" for index in range(1, 7)),
        }
        require(required <= self.files.keys(), "Frozen asset manifest is incomplete", "UNAVAILABLE")
        for name in sorted(self.files):
            path = self.root / name
            require(path.is_file(), f"Frozen asset is missing: {name}", "UNAVAILABLE")
            require(path.resolve() == path, f"Asset path contains a symlink: {name}")
            self.stamps[path] = asset_stamp(path)
        for name, digest in self.files.items():
            with (self.root / name).open("rb") as stream:
                actual = hashlib.sha256()
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    actual.update(chunk)
            require(actual.hexdigest() == digest, f"Frozen asset hash mismatch: {name}")
        self.verify_unchanged()
        self.digest = fingerprint(self.files)

    def verify_unchanged(self):
        require(all(path.resolve() == path and asset_stamp(path) == stamp
                    for path, stamp in self.stamps.items()), "Frozen assets changed during evaluation")

    def overlay(self, tree):
        for name in self.files:
            target = tree / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(self.root / name)


def source_files(tree):
    require(tree.is_dir(), "Training checkout is missing", "UNAVAILABLE")
    for directory, subdirs, files in os.walk(tree, followlinks=False):
        relative = Path(directory).relative_to(tree)
        excluded = SKIP_DIRS | (ASSET_DIRS if relative == Path(".") else set())
        subdirs[:] = sorted(set(subdirs) - excluded)
        for subdir in subdirs:
            require(not (Path(directory) / subdir).is_symlink(), "Source directory symlinks are not supported")
        for name in sorted(files):
            if name not in excluded and not name.endswith((".pyc", ".pyo")):
                yield relative / name


def snapshot(tree, destination=None):
    all_files, code = {}, {}
    for relative in source_files(tree):
        path = tree / relative
        data = read_regular(path, 256 * 1024 * 1024)
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        require(len(data) <= 256 * 1024 * 1024, "Source file is too large")
        digest = hashlib.sha256(data).hexdigest()
        all_files[str(relative)] = [digest, mode]
        if relative.suffix in CODE_SUFFIXES:
            code[str(relative)] = digest
        if destination is not None:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(mode)
    require("trainer/distillation.py" in code, "Missing upstream trainer", "UNAVAILABLE")
    return fingerprint(all_files), fingerprint(code)


def pinned_tree(checkout, commit, destination, env):
    require(checkout.is_dir(), "Pinned reference checkout is missing", "UNAVAILABLE")
    command = ["/usr/bin/git", "-C", str(checkout), "archive", "--format=tar", commit]
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE) as process:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for entry in archive:
                relative = PurePosixPath(entry.name)
                require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe reference archive")
                if relative.parts[0] in ASSET_DIRS:
                    continue
                require(entry.isdir() or entry.isfile(), "Unsupported reference archive entry")
                target = destination / relative
                if entry.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(entry) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(0o755 if entry.mode & 0o111 else 0o644)
        process.stdout.read()
        require(process.wait() == 0, "Pinned reference revision is unavailable", "UNAVAILABLE")


@contextlib.contextmanager
def gpu_lock(path):
    with os.fdopen(os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600), "rb+") as lock:
        info = os.fstat(lock.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and not info.st_mode & 0o022,
                "Untrusted GPU lock", "UNAVAILABLE")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Rejected("UNAVAILABLE", "SGF GPU judge lock is busy") from None
        yield


def worker_env(home, path=SYSTEM_PATH):
    return {
        "PATH": path, "HOME": str(home), "TMPDIR": str(home), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": str(home / "hf"), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "ENABLE_WANDB": "0", "WANDB_MODE": "disabled", "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_IFNAME": "=lo", "GLOO_SOCKET_IFNAME": "lo",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    }


def gpu_devices(count, env):
    binary = shutil.which("nvidia-smi", path=env["PATH"])
    require(binary, "nvidia-smi is unavailable", "UNAVAILABLE")
    binary = Path(binary).resolve()
    if os.geteuid() == 0:
        for component in (binary, *binary.parents):
            info = component.stat()
            require(info.st_uid == 0 and not info.st_mode & 0o022, "Untrusted GPU query executable", "UNAVAILABLE")
    probe = subprocess.run([str(binary), "--query-gpu=uuid,name,memory.used", "--format=csv,noheader,nounits"],
                           env=env, capture_output=True, text=True, timeout=15)
    require(probe.returncode == 0, "Cannot query NVIDIA GPUs", "UNAVAILABLE")
    rows = [tuple(field.strip() for field in line.split(",")) for line in probe.stdout.splitlines()]
    devices = [uuid for uuid, name, used in rows if "H200" in name and used.isdecimal() and int(used) < 1024]
    require(len(devices) == count and len(set(devices)) == count,
            f"Expected exactly {count} idle H200 GPUs; found {len(devices)}", "UNAVAILABLE")
    return ",".join(devices)


def worker_identity(name):
    if name is None:
        return {}, None
    require(os.geteuid() == 0, "Selecting a worker account requires root", "UNAVAILABLE")
    account = pwd.getpwnam(name)
    require(account.pw_uid != 0, "Worker must not run as root")
    return {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": []}, account


def stop_worker(account):
    if account is None:
        return
    deadline = time.monotonic() + 10
    while True:
        active = False
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal():
                continue
            try:
                with contextlib.ExitStack() as stack:
                    handle = os.pidfd_open(int(directory.name))
                    stack.callback(os.close, handle)
                    fields = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines()
                                  if ":" in line)
                    if int(fields["Uid"].split()[0]) == account.pw_uid and fields["State"].split()[0] not in {"Z", "X"}:
                        signal.pidfd_send_signal(handle, signal.SIGKILL)
                        active = True
            except (FileNotFoundError, ProcessLookupError):
                continue
        if not active:
            return
        require(time.monotonic() < deadline, "Worker processes survived cleanup")
        time.sleep(0.1)


def launch(python, tree, run_dir, spec, devices, identity, account, timeout, signatures, path):
    run_dir.mkdir(mode=0o711)
    output = run_dir / "ranks"
    output.mkdir(mode=0o700)
    if account is not None:
        os.chown(output, account.pw_uid, account.pw_gid)
    spec_path = run_dir / "spec.json"
    spec_path.write_text(json.dumps(spec) + "\n")
    spec_path.chmod(0o644)
    env = worker_env(output, path)
    env["CUDA_VISIBLE_DEVICES"] = devices
    command = [str(python), "-I", "-B", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               "--local-addr=127.0.0.1",
               f"--nproc-per-node={spec['world_size']}", "--no-python", str(python), "-I", "-B",
               str(ENTRY), "--worker", str(spec_path), str(output)]
    log_path = run_dir / "train.log"
    try:
        with log_path.open("w") as log, subprocess.Popen(
            command, cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, **identity,
        ) as process:
            try:
                code = process.wait(timeout=timeout)
            finally:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
    finally:
        try:
            stop_worker(account)
        finally:
            run_dir.chmod(0o700)
    patterns = [(re.compile(item["regex"]), item["reason"]) for item in signatures]
    with log_path.open(errors="replace") as log:
        for line in log:
            for pattern, reason in patterns:
                require(not pattern.search(line), f"Invalid training log: {reason}")
    require(code == 0, f"Four-rank training failed ({code}); see {log_path}")
    return [read_json(output / f"rank-{rank}.json") for rank in range(spec["world_size"])]
