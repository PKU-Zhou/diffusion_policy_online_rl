# GPU 推理动作频率统计

在 GPU 上运行单次推理，统计每次动作生成的耗时，计算动作频率（Hz）。

## 文件

- `timed_infer.py` — 带逐动作计时的推理脚本（派生自 `zczhou/scripts/single_infer.py`）
- `run_timed_infer.sh` — GPU 启动入口
- `analyze.py` — 读原始耗时数据，生成 markdown 统计报告
- `results/` — 运行产物（已 gitignore）

## 快速开始

```bash
# 1. GPU 上跑一个完整 episode，落盘逐步耗时
bash zczhou/infer_freq/run_timed_infer.sh

# 2. 生成统计报告（含动作频率）
python zczhou/infer_freq/analyze.py \
    --input zczhou/infer_freq/results/latencies_<timestamp>.json
```

也可以用 `conda` 环境的 python 直接调 `timed_infer.py`：

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. \
    python zczhou/infer_freq/timed_infer.py \
    --out zczhou/infer_freq/results/latencies_$(date +%Y%m%d_%H%M%S).json
```

## 常用参数

透传给 `timed_infer.py`：

| 参数 | 说明 |
|---|---|
| `--policy policy-XXX.pkl` | 指定 checkpoint（默认取 sample_step 最大者） |
| `--max_steps N` | 只跑前 N 步（默认完整 episode） |
| `--seed S` | 随机种子（默认 0） |
| `--quant` | 走 INT8 量化路径做对照 |

shell 环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPU` | `3` | 使用的显卡序号 |
| `DEVICE` | `gpu` | `gpu` / `cpu` |
| `LOG_DIR` | 内置默认 run | 实验目录 |
| `RESULTS_DIR` | `infer_freq/results` | 结果与日志目录 |

## 输出

`results/latencies_<ts>.json` 原始数据：

- `warmup_act_s`：首步 policy_fn 耗时（含 XLA 编译）
- `act_latencies_s[]`：稳态每步动作生成耗时（含 GPU 同步）
- `step_latencies_s[]`：稳态每步动作生成 + env.step 耗时
- `meta`：环境、checkpoint、seed、JAX 设备、版本等

`results/report_<ts>.md` 统计报告：

- 稳态耗时 mean / std / min / p50 / p90 / p95 / p99 / max
- **动作频率**：纯推理 mean / p50 频率，闭环 step 频率
- 与 MuJoCo 环境控制频率（HalfCheetah-v4 为 20 Hz）的实时性对比

## 计时口径

- JAX 在 GPU 上异步发射 kernel，必须 `jax.block_until_ready(act)` 后再停表，
  否则测到的只是 host 发射开销
- 首步 `policy_fn` 调用触发 XLA 编译，耗时远超稳态，单独记为 warmup，不计入频率
- 计时用 `time.perf_counter()`，host 端开销约微秒级，对毫秒级推理可忽略
