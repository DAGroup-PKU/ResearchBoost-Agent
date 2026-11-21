<div align="center">

<a href="https://github.com/DAGroup-PKU/ResearchBoost-Agent"><img src="docs/assets/readme/banner-en.jpg" alt="RSI-Infra Bench: how much faster can an agent make real training?" width="100%"></a>

<h3>Eight training-speed tasks for coding agents on Harbor,<br>scored by a frozen judge that accepts only the same training, faster</h3>

<p>
<a href="#leaderboard"><img src="docs/assets/readme/button-leaderboard-en.png" alt="Leaderboard" height="48"></a>
<a href="https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/"><img src="docs/assets/readme/button-blog-en.png" alt="Blog post" height="48"></a>
<a href="#how-to-run"><img src="docs/assets/readme/button-run-en.png" alt="How to run" height="48"></a>
<a href="https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/main"><img src="docs/assets/readme/button-agent-en.png" alt="RBA Home" height="48"></a>
</p>

<p>
<a href="https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/leaderboard/index.html"><img alt="Leaderboard" src="https://img.shields.io/badge/leaderboard-live-8C0000?style=flat-square"></a>
<a href="docs/eval-protocol.md"><img alt="Eight tasks" src="https://img.shields.io/badge/tasks-8-0891B2?style=flat-square"></a>
<a href="https://pypi.org/project/harbor/0.24.0/"><img alt="Harbor 0.24" src="https://img.shields.io/badge/Harbor-0.24.0-7C3AED?style=flat-square"></a>
<a href="docs/how-to-run.md#hardware"><img alt="Hopper GPUs" src="https://img.shields.io/badge/GPUs-Hopper%20%E2%89%A5130%20GiB-2A78D6?style=flat-square"></a>
<a href="https://github.com/DAGroup-PKU/ResearchBoost-Agent"><img alt="GitHub stars" src="https://img.shields.io/github/stars/DAGroup-PKU/ResearchBoost-Agent?style=flat-square&logo=github&color=241B1C"></a>
<a href="https://discord.gg/5k7Rr9hut"><img alt="Discord" src="https://img.shields.io/badge/Discord-join-5865F2?style=flat-square&logo=discord&logoColor=white"></a>
</p>

</div>

<br>

## LeaderBoard

### 8 training-optimization tasks · 10 hours per task

#### Overall

![Overall](docs/assets/readme/leaderboard-scores.png)

#### Per-Task

![Per-Task](docs/assets/readme/leaderboard-table.png)

Final judged speedups against each task’s baseline.

<p align="center">
<a href="https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/leaderboard/index.html"><img src="docs/assets/readme/button-bench-board-en.png" alt="Interactive leaderboard and run traces" height="56"></a>
<a href="https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/#analysis"><img src="docs/assets/readme/button-bench-blog-en.png" alt="How the models differ, in the blog post" height="56"></a>
</p>

<sub>\* GPT-6 Astra: suspected nerf; it feels different from our early testing.
GPT-6.2 Astra (18 November) and GPT-6.2 Sol (19 November) joined with the same setup (Codex, reasoning effort max, 10 hours).
Scores come from an ECI fit over all models together, so every rank is dynamic:
each new model refits the scale, and the scores and ranks of the others can move.</sub>

## Eval Protocol

Each agent optimizes a fixed training workload on the same GPUs.
A frozen judge verifies correctness and measures the final speedup.

- **Correctness:** bitwise identity or statistical equivalence,
  depending on the task.
- **Timing:** startup and warmup are excluded; the judge measures
  the task’s fixed workload.
- **Score:** an ECI-style fit of the eight normalized log-speedups,
  anchored at Claude Opus 5.5 = 40 and Kimi k3.1-pro = 10.

[Tasks, numerical checks and scoring details →](docs/eval-protocol.md)

## How To Run

Requires Linux, Docker with NVIDIA GPU support, and the task’s
specified Hopper GPUs.

1. Clone the benchmark and install Harbor:

   ```bash
   git clone --branch bench --single-branch \
     https://github.com/DAGroup-PKU/ResearchBoost-Agent.git RSI-Infra-Bench
   cd RSI-Infra-Bench
   pip install harbor==0.24.0
   cp researchboost.env.example researchboost.env
   ```

2. Configure the environment and prepare the task’s assets
   following the [setup guide](docs/how-to-run.md).

3. Run one task:

   ```bash
   harbor run -p tasks -i llava --env-file researchboost.env \
     -a <agent> -m <model> -n 1
   harbor view ./jobs
   ```

[GPU requirements, assets and build instructions →](docs/how-to-run.md)

## Citation

```bibtex
@misc{researchBoost_Agent_Team_RSI_Infra_Bench_2026,
  author = {{researchBoost-Agent Team}},
  title = {{RSI-Infra Bench Preview}},
  url = {https://ai4ai-for-anything.github.io/blog/rsi-infra-bench/},
  year = {2026}
}
```
