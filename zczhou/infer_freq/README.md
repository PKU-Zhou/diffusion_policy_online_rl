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

## 动作延迟实验

观察观测-动作延迟对任务回报的影响。MuJoCo 是确定性步进仿真，`time.sleep` 期间
环境物理状态冻结，无法模拟"延迟期间环境继续演化"，因此采用**动作延迟队列**：

```mermaid
flowchart LR
    obs_t["观测 obs_t"] --> policy["策略生成 act_t"]
    policy --> queue["延迟队列 FIFO"]
    queue -->|取出 act_(t-N)| env["env.step 推进"]
    env --> obs_next["obs_(t+1)"]
    obs_next --> policy
```

- 策略在 t 时刻生成的动作入队尾，同时从队首取出 N 步前的动作送 `env.step`
- 环境持续演化，动作生效滞后 N 步，等效于真实机器人的观测-动作延迟
- 前 N 步队列未满时用零动作填充（延迟期间无控制输入）
- N=0 时退化为无延迟 baseline

注意口径：环境控制频率恒为 20 Hz 不变，此处的"延迟"指动作信息滞后 N 个控制周期，
而非降低环境仿真频率。

### 用法

```bash
# 无延迟 baseline（N=0）
bash zczhou/infer_freq/run_delay_infer.sh --delay_steps 0

# 延迟 1 个控制周期（HalfCheetah-v4 为 50 ms 仿真时间）
bash zczhou/infer_freq/run_delay_infer.sh --delay_steps 1

# 对比分析，生成衰减报告
python zczhou/infer_freq/analyze_delay.py \
    --inputs zczhou/infer_freq/results/delay_N0_*.json zczhou/infer_freq/results/delay_N1_*.json
```

### 文件

- `delay_infer.py` — 带动作延迟队列的推理脚本，`--delay_steps N` 控制延迟步数
- `run_delay_infer.sh` — GPU 启动入口（`DELAY=N` 环境变量等价 `--delay_steps N`）
- `analyze_delay.py` — 对比多档延迟的回报衰减，给出下一档建议

### 输出

`results/delay_N<N>_<ts>.json`：在计时数据基础上增加 `delay_steps`、`ctrl_dt_s`、
`delay_sim_ms`（延迟对应仿真时间）字段。

`results/report_delay_<ts>.md`：各档延迟的回报、相对 baseline 衰减、下一档建议。

### 实测结果（HalfCheetah-v4，RTX 5090，seed=0，FP32）

| 延迟步数 N | 延迟仿真时间 | ep_ret | 相对 baseline |
|---|---|---|---|
| 0 | 0 ms | 11765.71 | baseline |
| 1 | 50 ms | 618.83 | -94.74% |

结论：该 SDAC 策略对动作延迟极度敏感——仅滞后 1 个控制周期（50 ms 仿真时间），
回报就衰减约 95%，说明策略高度依赖即时的观测-动作闭环，几乎无延迟容忍度。

实现注意：延迟队列用 `collections.deque` 时**不能设 `maxlen=N`**——预填 N 个零动作后
队列已满，`append` 会挤掉队首使延迟失效。正确做法是预填 N 个零动作后每步
`append` + `popleft`，队列长度恒为 N+1，取出恰好是 N 步前的动作。
