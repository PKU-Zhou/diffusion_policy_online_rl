# update_step 内部延迟细分分析

基于独立微基准实测（RTX 5090，JAX GPU后端，与 `configs/profiling_short.json` 相同超参：
HalfCheetah-v4 维度、batch_size=256、diffusion_steps=20、num_particles=32、reverse_mc_num=64）。

- 微基准脚本：`zczhou/profiling/profile_update_step.py`
- 分析脚本：`zczhou/profiling/utils/analyze_update_step.py`
- 原始数据：`profile_update_step_20260902_095236.json`
- 运行日志：`profile_update_step.log`

## 测量方法

`SDAC.stateless_update`（`relax/algorithm/sdac.py:84-249`）被 `jax.jit` 整体编译成单一
XLA 图（`relax/algorithm/base.py:16`），内部无法插入 Python 计时器。因此本实验将其
各子步骤按代码顺序**镜像拆分**，每个子步骤单独 jit 编译，预热后循环 200 次、每次
`jax.block_until_ready` 强制同步计时。同时测量完整融合版 `_update` 作为对照。

注意：独立 jit 会失去跨阶段的算子融合与异步流水，因此"各阶段之和"略大于融合版是
预期行为，差值即融合收益。

## 核心结果

### 主流程细分（每个子步骤独立 jit）

| 子步骤 | 代码位置 (sdac.py) | 平均耗时(ms) | 占比 |
| --- | --- | ---: | ---: |
| `get_action(next_obs)` 目标动作采样 | :113 | 4.3067 | 27.23% |
| 目标Q计算（2次Q前向） | :114-117 | 0.1502 | 0.95% |
| Critic前向+反向 ×2 | :119-125 | 0.3428 | 2.17% |
| `get_action(obs)` 当前动作采样 | :130 | 4.2819 | 27.07% |
| `q_sample`加噪 + 64倍MC扩展 | :131-144 | 0.0906 | 0.57% |
| Policy前向+反向（有效batch=16384） | :146-170 | 5.4780 | 34.63% |
| 优化器+目标网络更新（全更新分支） | :172-216 | 1.1677 | 7.38% |
| **各阶段之和** | | **15.8179** | **100%** |

### 融合版对照与代表性验证

| 项目 | 平均耗时(ms) |
| --- | ---: |
| 融合版 `_update`（step=0 全更新分支） | 14.5496 |
| 融合版 `_update`（step=1 仅critic分支） | 14.6401 |
| trainer实际路径 `algorithm.update`（含 `float()` 主机同步） | 15.9464 |
| 镜像各阶段之和 − 融合版 | +1.2683（融合/流水收益） |
| **full_train 实测 `update_step` 平均** | **18.7143** |

- 微基准 trainer 路径（15.95ms）达到 full_train 实测（18.71ms）的 **85.2%**，
  剩余差异来自实训中的 GPU 与采样进程竞争、buffer 采样数据搬运、日志累加等，
  微基准具有良好代表性。
- 延迟更新机制（`delay_update=2`）**几乎不省时间**：step=0（policy+target全更新）与
  step=1（仅critic）耗时几乎相同（14.55 vs 14.64ms）。因为 `jax.lax.cond` 两个分支都
  编译进图中，且梯度计算（真正贵的部分）无论如何都会执行，跳过的只是廉价的
  optax 参数应用。

## 关键发现

### 发现1：两次 get_action 合计占 54.3%，是最大瓶颈

`stateless_update` 调用了两次 `agent.get_action`（sdac.py:113 计算 `next_action`、
:130 计算 `new_action`），各约 4.3ms，合计约 8.6ms，占各阶段之和的 **54.3%**。

内窥测量显示（`relax/network/diffv2.py:41-63`）：

| get_action 内部组件 | 耗时(ms) | 说明 |
| --- | ---: | --- |
| 32粒子 × 20步去噪（`p_sample` vmap） | 3.9286 | 占 get_action 的 ~91% |
| 单粒子 × 20步去噪 | 0.7770 | 20步序贯 `lax.scan`，每步一次策略网前向 |
| 32粒子Q择优 | 0.4471 | 64次Q前向 + argmax |

单粒子已需 0.78ms（20步序贯去噪的固有延迟，每步约39μs，GPU利用率极低），32粒子
只涨到 3.93ms（并行摊薄），说明瓶颈是**20步序贯依赖链**本身：每步 kernel 太小、
无法并行的时间轴太长。

**注意**：这两次 `get_action` 是**纯前向推理（无梯度）**，用于构造 TD 目标和
policy 损失的锚点动作。

### 发现2：policy 梯度确实只有1步去噪，且反向传播本身并不是主要成本

`reverse_samping_weighted_p_loss`（`relax/utils/diffusion.py:147-154`）中梯度路径只
穿过**一次**去噪网络前向 `model(t, x_t)`；`tilde_at` 在损失函数外生成，梯度不穿过
任何多步去噪链。**"20步去噪的梯度计算"的说法是错误的**（本仓库创新点正是1步梯度）。

policy 更新（5.48ms，34.6%）的内窥：

| 组件 | 耗时(ms) |
| --- | ---: |
| policy损失 前向+反向（完整 value_and_grad） | 5.4780 |
| 仅前向（loss计算，不求梯度） | 1.1440 |
| 其中：宽batch(16384) Q前向 ×2 | 0.7588 |
| 其中：宽batch(16384) 去噪网前向 ×1 | 0.4584 |

即反向传播部分约 4.3ms。它贵不是因为"20步梯度"，而是因为 64 倍 MC 扩展把有效
batch 撑到 256×64=16384，在这个宽 batch 上做一次前向+反向。

### 发现3：Critic 更新非常便宜（2.17%）

Q 网络（3×256 MLP）在 batch=256 上的两次前向+反向只要 0.34ms。之前文档中
"Critic 和 Actor 都需要更新，且扩散 Actor 特别复杂"的表述虽方向正确，但量级上
Critic 完全不构成瓶颈。

### 发现4：update 内部没有环境交互

代码证实 update 阶段完全没有环境交互（环境交互只发生在 sample 阶段，
`relax/trainer/off_policy.py:208-246`）。update 内部由"前向推理（两次
get_action，54%）+ 前向反向（policy/critic 梯度，37%）+ 参数更新（7%）"构成。

## 优化建议（按预期收益排序）

1. **削减 get_action 成本（潜在收益最高，最多可省约54%）**
   - 减少 `num_particles`（32 → 8/4/1）：单粒子仅 0.78ms，若粒子择优对性能增益有限，
     可近乎免费拿回 3ms+/次；
   - 减少去噪步数或改用 DDIM 类少步采样：序贯 20 步是延迟主因；
   - 复用动作:如目标动作 `next_action` 是否可用 target policy 缓存/低频刷新。

2. **降低 MC 扩展倍数（potential ~20%）**
   - `reverse_mc_num=64` 使 policy 梯度的有效 batch 达 16384；若 32/16 倍不损性能，
     policy 更新耗时可近似线性下降。

3. **优化 `float()` 主机同步（约1.4ms/次，~9%）**
   - `Algorithm.update`（base.py:24）对每个 info 标量做 `float(v)` 强制 device→host
     同步。可改为每 N 步同步一次或用 `jax.device_get` 批量取回。

4. **延迟更新（delay_update）当前设计不省时间**
   - 梯度计算无条件执行，`lax.cond` 只跳过廉价的参数应用。若想真正省时，
     需把梯度计算本身放进条件分支（会改变编译图结构，需验证）。

## 与 full_train 结果的换算

full_train 中 `update_step` 总耗时 3742.9s（200,001次 × 18.71ms）。按本细分比例估算：

| 组件 | 估算总耗时 | 占训练主循环 |
| --- | ---: | ---: |
| 两次 get_action（前向推理） | ~2032s | ~49.7% |
| policy 前向+反向 | ~1296s | ~31.7% |
| 优化器+其他 | ~415s | ~10.2% |

即整个 1M 步训练中，约 **34 分钟纯粹花在 update 内部的扩散采样前向推理上**——
这与"梯度计算是瓶颈"的原有结论截然不同，是后续优化的第一优先级。
