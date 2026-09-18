"""分析 timed_infer.py 产出的 latencies_*.json，生成统计报告。

统计稳态动作生成耗时（排除首步 XLA 编译 warmup），给出：
- mean / std / min / p50 / p90 / p95 / p99 / max
- 动作频率（Hz）：纯推理频率（mean / p50）与闭环 step 频率
- 与 MuJoCo 环境控制频率对比，判断是否满足实时性

用法：
    python analyze.py --input results/latencies_<ts>.json
    python analyze.py --input results/latencies_<ts>.json --out results/report_<ts>.md
"""

import argparse
import json
from pathlib import Path

import numpy as np


def query_env_ctrl_freq(env_name: str) -> float | None:
    """查询 MuJoCo 环境的控制频率（frame_skip * timestep 的倒数）。查不到返回 None。"""
    try:
        import gymnasium
        env = gymnasium.make(env_name)
        dt = env.unwrapped.frame_skip * env.unwrapped.model.opt.timestep
        env.close()
        return 1.0 / dt
    except Exception:
        return None


def summarize(name: str, arr: np.ndarray, unit: str = "ms") -> dict:
    scale = 1e3 if unit == "ms" else 1.0
    return {
        "name": name,
        "n": int(arr.size),
        "mean": float(arr.mean() * scale),
        "std": float(arr.std() * scale),
        "min": float(arr.min() * scale),
        "p50": float(np.percentile(arr, 50) * scale),
        "p90": float(np.percentile(arr, 90) * scale),
        "p95": float(np.percentile(arr, 95) * scale),
        "p99": float(np.percentile(arr, 99) * scale),
        "max": float(arr.max() * scale),
        "unit": unit,
    }


def freq_from_mean(arr: np.ndarray) -> float:
    return float(1.0 / arr.mean())


def freq_from_p50(arr: np.ndarray) -> float:
    return float(1.0 / np.percentile(arr, 50))


def render_report(payload: dict, ctrl_freq: float | None) -> str:
    meta = payload["meta"]
    ep = payload["episode"]
    warmup_s = payload["warmup_act_s"]
    act = np.asarray(payload["act_latencies_s"], dtype=np.float64)
    step = np.asarray(payload["step_latencies_s"], dtype=np.float64)

    act_stat = summarize("act", act)
    step_stat = summarize("step", step)

    act_freq_mean = freq_from_mean(act)
    act_freq_p50 = freq_from_p50(act)
    step_freq_mean = freq_from_mean(step)

    lines = []
    lines.append("# GPU 推理动作频率统计报告")
    lines.append("")
    lines.append("## 运行信息")
    lines.append("")
    lines.append(f"- 时间：`{meta['timestamp']}`")
    lines.append(f"- 主机：`{meta['hostname']}`")
    lines.append(f"- 环境：`{meta['env']}` (obs_dim={meta['obs_dim']}, act_dim={meta['act_dim']})")
    lines.append(f"- checkpoint：`{meta['checkpoint']}`")
    lines.append(f"- seed：`{meta['seed']}`")
    lines.append(f"- 量化：`{'INT8' if meta['quant'] else 'off（FP32）'}`"
                 + ("（from_source）" if meta.get("from_source") else ""))
    lines.append(f"- JAX：`{meta['jax_version']}`，backend=`{meta['default_backend']}`，"
                 f"devices={meta['jax_devices']}")
    lines.append("")
    lines.append("## Episode")
    lines.append("")
    lines.append(f"- 步数：{ep['ep_len']}")
    lines.append(f"- 回报：{ep['ep_ret']:.2f}")
    lines.append("")
    lines.append("## XLA 编译预热（首步，不计入稳态统计）")
    lines.append("")
    lines.append(f"- warmup 耗时：**{warmup_s * 1e3:.3f} ms**")
    if act.size > 0:
        ratio = warmup_s / act.mean()
        lines.append(f"- 相对稳态均值倍数：{ratio:.1f}x")
    lines.append("")
    lines.append("## 稳态耗时统计")
    lines.append("")
    lines.append("| 指标 | 动作生成 act (ms) | 整步 step (ms) |")
    lines.append("|---|---|---|")
    for k in ("n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        a = act_stat[k]
        s = step_stat[k]
        if k == "n":
            lines.append(f"| {k} | {a} | {s} |")
        else:
            lines.append(f"| {k} | {a:.3f} | {s:.3f} |")
    lines.append("")
    lines.append("说明：")
    lines.append("- act = `policy_fn(params, obs)` 调用耗时，含 `jax.block_until_ready` GPU 同步")
    lines.append("- step = act + `env.step(act)` 整步耗时")
    lines.append("- 已排除首步 XLA 编译 warmup")
    lines.append("")
    lines.append("## 动作频率")
    lines.append("")
    lines.append("| 口径 | 频率 (Hz) |")
    lines.append("|---|---|")
    lines.append(f"| 纯推理（mean） | **{act_freq_mean:.2f}** |")
    lines.append(f"| 纯推理（p50） | {act_freq_p50:.2f} |")
    lines.append(f"| 闭环 step（mean） | **{step_freq_mean:.2f}** |")
    lines.append("")

    if ctrl_freq is not None:
        lines.append("## 实时性对比")
        lines.append("")
        lines.append(f"- MuJoCo 环境控制频率：**{ctrl_freq:.2f} Hz**"
                     f"（每步 {1e3 / ctrl_freq:.2f} ms 预算）")
        headroom_mean = act_freq_mean / ctrl_freq
        headroom_step = step_freq_mean / ctrl_freq
        lines.append(f"- 纯推理频率 / 控制频率 = **{headroom_mean:.2f}x**"
                     f"（{'满足' if headroom_mean >= 1 else '不满足'}实时性）")
        lines.append(f"- 闭环频率 / 控制频率 = **{headroom_step:.2f}x**"
                     f"（{'满足' if headroom_step >= 1 else '不满足'}实时性）")
        budget_ms = 1e3 / ctrl_freq
        lines.append(f"- 每步预算 {budget_ms:.2f} ms 中，动作生成占 {act_stat['mean']:.3f} ms"
                     f"（{act_stat['mean'] / budget_ms * 100:.1f}%），"
                     f"环境步进占 {step_stat['mean'] - act_stat['mean']:.3f} ms")
        lines.append("")

    lines.append("## 备注")
    lines.append("")
    lines.append("- 计时使用 `time.perf_counter()`，host 端开销约微秒级，对毫秒级推理可忽略")
    lines.append("- `block_until_ready` 确保测到的是 GPU kernel 真实完成时间，而非 host 发射开销")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="latencies_*.json 路径")
    parser.add_argument("--out", type=Path, default=None,
                        help="报告输出路径，默认与 input 同目录的 report_<ts>.md")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        payload = json.load(f)

    ctrl_freq = query_env_ctrl_freq(payload["meta"]["env"])
    report = render_report(payload, ctrl_freq)

    out_path: Path = args.out
    if out_path is None:
        ts = payload["meta"]["timestamp"]
        out_path = args.input.parent / f"report_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"report saved: {out_path}")


if __name__ == "__main__":
    main()
