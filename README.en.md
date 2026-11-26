# rsi-bench-vetkit

**English** · [简体中文](README.md)

Use this kit to turn a training repository into a candidate task for [RSI-Infra Bench](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/bench) and decide whether to include it in the benchmark. It includes a vetting protocol, a Claude Code skill, and tools for running the benchmark judge on your own hardware.

| Directory | What you'll find |
|---|---|
| [`skills/rbb-vet/`](skills/rbb-vet/SKILL.md) | The task-vetting skill. The [protocol](skills/rbb-vet/references/protocol.md) defines five stages: (A) desk review, (B) estimating the speedup ceiling and assessing baseline strength, (C) packaging the task and testing the judge, (D) pilot runs, and (E) final review. It sets the checks, thresholds, and scorecards for each stage, with additional rules in the practice supplement. The [guide](skills/rbb-vet/references/guide.md) walks through the process, and [lessons learned](skills/rbb-vet/references/lessons.md) covers common pitfalls. The [scripts](skills/rbb-vet/scripts/) measure hardware peaks (`peaks.py`), estimate T0 and T1 bounds (`bound.py`), calculate false rejection rates (`frr.py`), and produce a scorecard (`scorecard.py`). |
| [`vetkit/`](vetkit/README.md) | Tools for running the judge on a single machine. Lint a task with `onboard_check.py`, scan for information leaks with `leak_guard.py`, and run a Harbor trial on selected GPUs with NUMA-aware CPU pinning using `run_trial.py`. Use `refpack.py` and `container/` to build and verify reference packs and rejudge completed trials in a clean container. This directory also includes the NanoChat example from the benchmark branch, unit tests, and a Docker-only end-to-end test. |
| [`tools/`](tools/) | Run `scan_sensitive.py` before committing. The script contains the generic rules. Keep organization-specific terms outside the repository as described in the [script](tools/scan_sensitive.py). Record reviewed false positives and the reason for each exception in [`tools/scan_allow.txt`](tools/scan_allow.txt). |

## Requirements

- Linux with root access, Docker with Compose v2, the NVIDIA Container Toolkit, and Hopper GPUs (H100 or H200 SXM). The public leaderboard uses H200-class GPUs.
- Python 3.11 or later, or an earlier version with the `toml` package installed. Install Harbor with `pip install harbor==0.24.0`.
- The vetting scripts in `skills/rbb-vet/scripts/` require NumPy. The peak measurement script, `peaks.py`, also requires PyTorch.

Run the self-tests from the repository root:

```bash
python3 -m pytest skills/rbb-vet/scripts/test_vet.py
python3 vetkit/tests/test_tools.py && python3 vetkit/tests/test_container.py
# Docker only: lint a CPU-only toy task, build and verify its reference pack, then judge submissions.
bash vetkit/tests/e2e/run_e2e.sh
```

## Install the skill

```bash
git clone -b vetkit https://github.com/DAGroup-PKU/ResearchBoost-Agent.git rsi-bench-vetkit
mkdir -p ~/.claude/skills
ln -s "$PWD/rsi-bench-vetkit/skills/rbb-vet" ~/.claude/skills/rbb-vet
```

Use a symlink so the skill can find `../../vetkit` in the checkout. In Claude Code, ask “Vet this repository as a candidate benchmark task,” or enter `/rbb-vet <task-name> <repository-URL> [commit]`. You can also give the [protocol](skills/rbb-vet/references/protocol.md) to another agent as its prompt, or follow the [guide](skills/rbb-vet/references/guide.md) yourself.

## Quick start

Follow the [VetKit walkthrough](vetkit/README.md) with the [NanoChat example](vetkit/examples/tasks/nanochat/). Lint the task, scan the files visible to the agent for leaks, and run a no-op smoke test. The no-op agent must receive a reward of zero. Then build a reference pack, verify it on a different set of GPUs, and rejudge a completed trial from its artifacts.

## Contribution rules

- Keep target speedups, previous results, known hotspots, optimization flag names, optimization ideas, and references to other hardware out of anything the agent can see (protocol rule 1). Store private vetting data in `vetting/<task-name>/`, outside the task and its image. Keep that data out of public releases.
- Use publicly available code, weights, and data. Pin each to a revision and record its checksum (protocol rule 5).
- Before committing, run `python3 tools/scan_sensitive.py` and include the scan result in your pull request description.
