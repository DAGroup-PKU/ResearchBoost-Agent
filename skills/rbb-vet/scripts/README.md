# scripts/：验题工具

配合 [../references/guide.md](../references/guide.md) 用。全部是单文件脚本：
- `frr.py`、`bound.py`、`scorecard.py` 只要 Python 3 和 numpy；
- `peaks.py` 要 torch（GPU 路径要 CUDA；没有 GPU 时只能跑 `shapes` 和 CPU 自测）。

| 脚本 | 哪一段 | 做什么 |
|---|---|---|
| `peaks.py shapes` | B | 从 torch.profiler 的 trace（`record_shapes=True`）抽出每步的 GEMM 形状 |
| `peaks.py measure` | B | 单卡实测：每个 GEMM / 注意力形状的时间、HBM 拷贝带宽、矩阵乘期间的时钟和功耗 |
| `peaks.py comm` | B | `torchrun` 下实测集合通信的 bus bandwidth（nccl-tests 口径） |
| `bound.py` | B | 由 `stages.json` 算 T0（手册峰值）和 T1（实测峰值），写进打分卡 |
| `frr.py` | C | 统计门对合法改写的误拒率：参照表、给定目标的 k、由实测 z 值反推 |
| `scorecard.py` | 每段结尾 | 打分卡逐项对阈值，给出下一步 |

`example/` 里是格式示例（`stages`、`peaks`、`vetting`），数都是编的，只示范格式。真峰值用 `peaks.py` 测。

自测（CPU 上跑；装了 torch 时也测 `peaks.py` 的 CPU 模式）：

```bash
python3 -m pytest test_vet.py   # 要 pytest、numpy
```
