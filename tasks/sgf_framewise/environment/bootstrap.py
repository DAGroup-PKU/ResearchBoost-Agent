"""Prepare the offline training runtime without starting GPU work."""

import hashlib
import json
import subprocess
from pathlib import Path

WORKSPACE = Path("/opt/researchboost/ws")
CANDIDATE = Path("/submission/sgf_framewise")
READY = Path("/run/researchboost-ready")
WHEELHOUSE = Path("/opt/researchboost/assets/runtime")
REFERENCE_ENV = Path("/opt/reference-env")
CANDIDATE_ENV = Path("/opt/candidate-env")
REFERENCE_PYTHON = str(REFERENCE_ENV / "bin/python")


def purelib(python):
    return Path(subprocess.check_output(
        [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True
    ).strip())


def install_runtime():
    lock = WHEELHOUSE / "requirements.lock"
    if not lock.is_file():
        return "unprovisioned"
    installed = REFERENCE_ENV / ".runtime-lock"
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    if installed.is_file():
        if installed.read_text().strip() != digest:
            raise RuntimeError("Offline runtime lock changed; rebuild the environment")
        return "provisioned"
    subprocess.run([REFERENCE_PYTHON, "-I", "-m", "pip", "--isolated", "install", "--no-index",
                    "--find-links", str(WHEELHOUSE), "--only-binary=:all:", "--require-hashes",
                    "--no-cache-dir", "-r", str(lock)], check=True)
    packages = purelib(REFERENCE_PYTHON)
    candidate_packages = CANDIDATE_ENV / packages.relative_to(REFERENCE_ENV)
    (candidate_packages / "reference-runtime.pth").write_text(str(packages) + "\n")
    installed.write_text(digest + "\n")
    return "provisioned"


def main():
    READY.unlink(missing_ok=True)
    config = json.loads((WORKSPACE / "bench/bench.json").read_text())
    reference = WORKSPACE / config["reference"]
    commit = subprocess.check_output(
        ["/usr/bin/git", "-C", str(reference), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != config["reference_commit"]:
        raise SystemExit("Unexpected reference revision")
    for tree in (reference, CANDIDATE):
        if not (tree / "scripts/train_self_gradient_forcing_framewise.sh").is_file():
            raise SystemExit(f"Missing SGF launcher in {tree}")
    metadata = WORKSPACE / "task/ready.json"
    metadata.parent.mkdir(exist_ok=True)
    metadata.write_text(json.dumps({"status": "ready", "mode": "gpu", "runtime": install_runtime(),
                                    "reference_commit": commit}) + "\n")
    READY.touch()


if __name__ == "__main__":
    main()
