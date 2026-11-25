# vetkit：在自己的机器上验一道 RSI-Infra Bench 题

验题规程（A–E）和流程在 [`skills/rbb-vet`](../skills/rbb-vet/SKILL.md)：规范是 [protocol.md](../skills/rbb-vet/references/protocol.md)，上限、误拒率、打分卡的工具在 `skills/rbb-vet/scripts/`。
这个目录是判卷侧的工具：一个贡献者在一台（或几台）自己的 Hopper 机器上打包、冒烟、跑 pilot、复判一道题需要的最小部分。不依赖任何调度器、队列、registry 或共享存储。

题目本身的格式（Harbor 任务、`task.toml`、镜像、资产、判卷）以 [ResearchBoost-Agent 的 `bench` 分支](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/bench) 为准：先读它的 `docs/how-to-run.md` 和 `docs/eval-protocol.md`。

## 里面有什么

| 路径 | 做什么 | 哪一段 |
|---|---|---|
| `bin/onboard_check.py` | 只读 linter：目录结构、`task.toml` 对规程（artifacts、`[agent]`、`[verifier]`、健康检查）、标准判卷接口（`bench.py record / compare / time / check --final`，`bench.json` 的字段）、题面里不写时长、vetting 数据没进镜像、`tests/` 不按名字调用常见程序（规程补充 C10）、没有未提交的改动。装了 Harbor 时也用 Harbor 自己的模型校验。退出码 2 = 有问题 | C 收尾 |
| `bin/leak_guard.py` | 扫 agent 能看到的所有文件（题面、Dockerfile COPY 进去的、prep-pack），可选再扫构建好的镜像；`--deny` 给这道题自己的禁词（已有成绩、阶梯里的开关名、热点）。命中都要人看，再人读一遍 | C |
| `bin/run_trial.py` | 直接跑 Harbor trial：`--gpu-ids` 指定卡、`--cpus auto` 按 NUMA 等分绑核、每组卡一个 `/tmp`、主容器退出时及时停掉 Harbor、密钥脱敏、每格一行汇总（`results.jsonl`） | C nop、D pilot |
| `bin/refpack.py` | 参考包：`build` 独占整机做一次完整准备并发布；`verify` 在别的卡 / 机器上恢复并验证（参考记录逐位一致、参考计时落在 band 里）；`judge` 从一个跑完的 trial 的 artifacts 在干净容器里重判；`show` 列出包 | C、D 复判、E |
| `container/` | `refpack.py` 只读挂进任务容器的代码（`/opt/rbb-svc`）：`entry.sh`（build / trial / judge 三种角色）、`svcpack.py`（pristine / collect / verify）、`svckey.py`（包的 key）、`restore_bundle.py`（以 agent 用户把提交物放回原位）、`guard.py`（终判时杀掉逃逸的候选进程）、`safepath.py`（哪些目录和程序只有 root 能改）、`tasks/`（每道题的包模块；`_generic.py` 适用于有标准判卷接口的题） | — |
| `examples/tasks/nanochat/` | 一道已发布的题（预训练，2 卡，N0 + N1），原样取自 ResearchBoost-Agent `bench` 分支，当模板和冒烟用。见 [examples/README.md](examples/README.md) | — |
| `tests/` | `test_tools.py`、`test_container.py`：单元测试（不要 GPU，也不要 Docker 守护进程）；`e2e/run_e2e.sh`：在一道纯 CPU 的玩具题上把 lint → 建包 → 验证 → 三份提交的判卷 → 篡改的包被拒整条走一遍，只要 Docker | — |

## 需要的环境

- Linux，root，Docker（Compose v2）、NVIDIA container toolkit，Hopper GPU（H100 / H200 SXM）。公开榜在 H200 级的卡上跑（每卡 ≥ 130 GiB）。
- Python ≥ 3.11（或装了 `toml` 包；装 Harbor 时会带上）。工具只用标准库。
- `pip install harbor==0.24.0`。
- 先跑一遍自测：
  ```bash
  python3 vetkit/tests/test_tools.py && python3 vetkit/tests/test_container.py
  bash vetkit/tests/e2e/run_e2e.sh        # 只要 Docker；E2E_WORK 要在 Docker 能挂载的目录（笔记本上的虚拟机：家目录下）
  ```

## 用法：拿示例题走一遍

```bash
cd vetkit/examples
cp researchboost.env.example researchboost.env      # 设 RESEARCHBOOST_ASSETS_ROOT、RESEARCHBOOST_SCRATCH_ROOT（绝对路径）
# 资产：见 tasks/nanochat/README.md（ClimbMix 分片 + 用任务镜像训练上游的 tokenizer）

K=../bin
python3 $K/onboard_check.py tasks/nanochat                    # 1. linter
python3 $K/leak_guard.py tasks/nanochat                       # 2. 泄漏扫描（再人读一遍）
python3 $K/run_trial.py --tasks-dir tasks --env-file researchboost.env \
    --trial nanochat:nop --gpu-ids 0,1 --cpus auto            # 3. nop 冒烟：完整准备 + 最终评测，reward 必须为 0
```

### 参考包

题要在多组卡、多台机器上跑 D 段，或者有采样类统计门（规程补充 C6）时，先建一个参考包：所有 trial 和复判都对同一份冻结的参考判。

```bash
docker build -t vet/nanochat tasks/nanochat/environment     # 镜像源、代理用 --build-arg（见 bench 分支 docs/how-to-run.md）
P="--task nanochat --tasks-dir tasks --image vet/nanochat --env-file researchboost.env --store /srv/packs"
python3 $K/refpack.py build  $P --gpu-ids 0,1 --cpus auto --spread-max 0.05   # 独占整台机器；约等于一次完整启动准备
python3 $K/refpack.py verify $P --gpu-ids 4,5 --cpus auto                     # 另一组卡 / 另一台同型号机器
python3 $K/refpack.py show --store /srv/packs
```

- `refpack.py` 的 build / verify / judge 以 **root** 运行：包里有只给判卷看的东西（留出数据、冻结的参考），包目录发布时设成 0700；判卷的输出目录也必须归 root，否则和 agent 同 uid 的宿主用户能让候选往里放文件。`--test-non-root` 只给测试用。
- **key** = sha256(任务, 判卷哈希, 镜像 ID, GPU 型号, 卡数, 驱动, 准备版本)。镜像 ID 是镜像 config 的 digest：同一个镜像不管是本地构建、`docker pull` 还是 `docker load` 来的都一样。判卷哈希是镜像里准备之前的判卷工作区（`/opt/researchboost/ws`，不含 `targets/` 和 `__pycache__`）的 sha256，所以判卷一改就是另一个包。换了 GPU 型号或驱动，也要另建一个包。
- 包发布一次（临时名再 rename），之后不改。不再信任的包手动移走；`refpack.py` 从不悄悄替换，也从不在验证失败时偷偷做一次完整准备（退出 81 / 82）。
- `--spread-max` 是建包的质量门：包冻结的参考计时决定 agent 自己 check 的门槛（1 + 2 × spread），建包时测得的参考波动超过它就不发布（退出 75），换个安静的时间重建。
- `verify` 里那一次参考计时和包里冻结值的比较（band）只检查机器快慢，不是候选的门槛（规程补充 C7）。
- 镜像里的 CUDA forward-compat 库默认不加载（`--compat never`，和多数题自己的 entrypoint 一样）；题目自己的 entrypoint 会加载时用 `--compat always`（或 `auto`：只在比宿主机驱动新时）。包模式跑的库要和 trial 跑的一样。
- 题目没有自己的包模块时用通用模块：`--adapter generic --timing-file task/timing_ref.json --timing-field s_per_update [--band B]`，还要 `--python <镜像里训练环境的 python>`。要求标准判卷接口（`onboard_check.py` 会查）。题目有特别的验证步骤（例如每个 rank 都要用某个 kernel、某个文件由机器自己算），就照 `container/tasks/nanochat.py` 写一个 `container/tasks/<题名>.py`。

### pilot 和复判

```bash
# 一格占一个 NUMA 半边：卡和它们的核（--cpus auto）。另一半可以同时跑另一格。
python3 $K/run_trial.py --tasks-dir tasks --env-file researchboost.env --secrets-env-file keys.env \
    --trial nanochat:claude-code:anthropic/<model> --gpu-ids 0,1 --cpus auto
# 复判：从跑完的 trial 的 artifacts，在干净容器里（同一个镜像、包只读、不联网、提交物以 agent 用户放回）
python3 $K/refpack.py judge $P --gpu-ids 0,1 --cpus auto \
    --artifacts jobs/<job>/<trial>/artifacts --out judged/<job>-1
```

- Harbor 在 agent 到点后立即在同一个容器里做最终评测：这时这一半上不能有别的东西（规程补充 D1）。可疑的结果用 `refpack.py judge` 在独占的整台机器上再判一次。
- `judge` 里 `entry.sh` 自己的命令只用系统目录（`/usr/sbin:/usr/bin:/sbin:/bin`）；交给 `test.sh` 的 `PATH` 是镜像的 `PATH` 去掉不是 root-only 的目录（多半是 agent 的候选环境），记在输出的 `svc/judge-path.txt`；以 root 跑 `guard.py` 的解释器也必须是 root-only，否则不判（规程补充 C10）。这只管 `refpack.py judge`：Harbor 的 shared verifier 不经过 vetkit，题目自己的 `verify.py` 要做到同样的事。
- `judge` 的输出：`check.json`、`reward.json`（只有判卷用 reward 认证了的才算）、`judge.json`（包 key、镜像、判卷哈希、GPU、核、守护进程杀掉的逃逸进程）、`verifier/`、`svc/`。E 段的“final 重判 2 次都 PASS”就是跑两次 `judge`。
- `keys.env` 是 agent 用的 API key 和 base URL，`run_trial.py` 不会把它们写进日志；跑完后会扫一遍 job 目录做脱敏。
- 纯 CPU 的题（`gpus = 0`）：`--gpu-ids` 写这个 NUMA 半边的卡，容器拿不到 GPU，只拿到这些卡对应的核（`--cpus auto`），计时期间这半边的卡也不要给别的计时任务。

## 有调度器或队列时

这些工具都是“在这一组卡上跑这一件事”。有 Slurm、k8s 或自己的作业系统时，把每次调用包成一个作业：
- `refpack.py build`、B 段计时：独占整台机器的作业；
- `run_trial.py` 的一格：一个 NUMA 半边（卡 + 核）的作业，`--gpu-ids` 用调度器分给你的卡；
- `refpack.py verify` / `judge`：一个 NUMA 半边；可疑结果的复判：整台机器。
多台机器共用参考包时，把 `--store` 放在共享目录上，各机器在计时前把要用的包复制到本地盘再用（计时期间不读网络盘）。

## 和维护者那边的服务的关系

`container/` 来自 bench 维护者那边判卷服务的容器侧代码，`bin/` 是它的单机版：包的内容格式、key 的组成（维护者那边的镜像字段用 registry digest，所以两边的 key 不相同）和验证规则一样，你建的包和维护者建的包可以逐个文件比对（参考记录、冻结的参考）。交接时把包 key、`judge.json` 和 vetting 结论一起交过去。
