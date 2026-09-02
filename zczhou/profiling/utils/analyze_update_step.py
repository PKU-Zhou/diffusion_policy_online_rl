"""
update 阶段细分结果分析脚本

读取 profile_update_step.py 产出的 JSON，计算：
  - 各子步骤平均耗时与占比（以镜像各阶段之和为基数）
  - 镜像各阶段之和 vs 融合版 fused_stateless_update 的差值（XLA 融合收益）
  - 与 full_train 实测 update_step 平均耗时的对照

用法：
    python zczhou/profiling/utils/analyze_update_step.py <profile_update_step_*.json> [--markdown] \
        [--full-train <profiling_results_*.json>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

# 镜像 stateless_update 主流程的阶段（顺序即代码顺序），用于求和对照融合版
MAIN_STAGES = [
    ("stage1_next_action_get_action", "get_action(next_obs) 目标动作采样"),
    ("stage2_target_q_backup", "目标Q计算 (2次Q前向)"),
    ("stage3_critic_value_and_grad", "Critic前向+反向 x2"),
    ("stage4_new_action_get_action", "get_action(obs) 当前动作采样"),
    ("stage5_q_sample_and_mc_expand", "q_sample加噪 + 64倍MC扩展"),
    ("stage6_policy_value_and_grad", "Policy前向+反向 (batch=16384)"),
    ("stage7_param_updates_step0_all", "优化器+目标网络更新 (全更新分支)"),
]

# 内窥性质的阶段，仅展示，不参与求和
PROBE_STAGES = [
    ("baseline_dispatch_overhead", "空jit调用 dispatch 开销地板"),
    ("stage1a_p_sample_32particles", "get_action内窥: 32粒子x20步去噪"),
    ("stage1b_p_sample_1particle", "get_action内窥: 单粒子x20步去噪"),
    ("stage1c_particle_q_select", "get_action内窥: 粒子Q择优"),
    ("stage6a_policy_forward_only", "policy loss内窥: 仅前向"),
    ("stage6b_wide_q_forward_only", "policy loss内窥: 仅宽batch Q前向"),
    ("stage6c_wide_denoiser_forward_only", "policy loss内窥: 仅宽batch去噪前向x1"),
    ("stage7_param_updates_step1_criticonly", "优化器更新 (仅critic分支)"),
]

FUSED_STAGES = [
    ("fused_stateless_update_step0", "融合版 _update (step=0 全更新)"),
    ("fused_stateless_update_step1", "融合版 _update (step=1 仅critic)"),
    ("api_algorithm_update_with_host_sync", "trainer实际路径 (含float()主机同步)"),
]


def load_summary(path: Path) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def avg_ms(summary: Dict[str, dict], name: str) -> Optional[float]:
    entry = summary.get(name)
    if entry is None:
        return None
    return entry["average_duration"] * 1e3


def fmt(v: Optional[float]) -> str:
    return f"{v:9.4f}" if v is not None else "      n/a"


def analyze(path: Path, full_train_path: Optional[Path], markdown: bool) -> None:
    summary = load_summary(path)

    main_vals = {name: avg_ms(summary, name) for name, _ in MAIN_STAGES}
    total_main = sum(v for v in main_vals.values() if v is not None)

    fused0 = avg_ms(summary, "fused_stateless_update_step0")
    fused1 = avg_ms(summary, "fused_stateless_update_step1")
    api = avg_ms(summary, "api_algorithm_update_with_host_sync")

    full_train_update_ms = None
    if full_train_path is not None and full_train_path.exists():
        ft_summary = load_summary(full_train_path)
        entry = ft_summary.get("update_step")
        if entry:
            full_train_update_ms = entry["average_duration"] * 1e3

    if markdown:
        print("## update 阶段细分（镜像子步骤，独立 jit 编译）\n")
        print("| 子步骤 | 平均耗时(ms) | 占各阶段之和 |")
        print("| --- | ---: | ---: |")
        for name, label in MAIN_STAGES:
            v = main_vals[name]
            pct = f"{v / total_main * 100:.2f}%" if v is not None and total_main > 0 else "n/a"
            print(f"| {label} | {fmt(v).strip()} | {pct} |")
        print(f"| **各阶段之和** | **{total_main:.4f}** | **100%** |")
        print()
        print("## 内窥基准\n")
        print("| 项目 | 平均耗时(ms) |")
        print("| --- | ---: |")
        for name, label in PROBE_STAGES:
            print(f"| {label} | {fmt(avg_ms(summary, name)).strip()} |")
        print()
        print("## 融合版对照\n")
        print("| 项目 | 平均耗时(ms) |")
        print("| --- | ---: |")
        for name, label in FUSED_STAGES:
            print(f"| {label} | {fmt(avg_ms(summary, name)).strip()} |")
        if fused0 is not None:
            print(f"| 镜像各阶段之和 - 融合版(step0) 差值 | {total_main - fused0:.4f} |")
        if full_train_update_ms is not None:
            print(f"| full_train 实测 update_step 平均 | {full_train_update_ms:.4f} |")
        print()
        return

    print("=" * 78)
    print(f"update 阶段细分分析: {path.name}")
    print("=" * 78)

    print("\n[主流程镜像子步骤]  (独立 jit, 每次 block_until_ready)")
    for name, label in MAIN_STAGES:
        v = main_vals[name]
        pct = f"{v / total_main * 100:6.2f}%" if v is not None and total_main > 0 else "   n/a"
        print(f"  {label:44s} {fmt(v)} ms  {pct}")
    print(f"  {'各阶段之和':44s} {fmt(total_main)} ms  100.00%")

    print("\n[内窥基准]")
    for name, label in PROBE_STAGES:
        print(f"  {label:44s} {fmt(avg_ms(summary, name))} ms")

    print("\n[融合版对照]")
    for name, label in FUSED_STAGES:
        print(f"  {label:44s} {fmt(avg_ms(summary, name))} ms")

    if fused0 is not None:
        gain = total_main - fused0
        print(f"\n  镜像之和 {total_main:.4f} ms vs 融合版(step0) {fused0:.4f} ms"
              f" -> 差值 {gain:+.4f} ms（正值≈跨阶段融合/调度收益）")
    if fused1 is not None and fused0 is not None:
        print(f"  融合版 step0(全更新) - step1(仅critic) = {fused0 - fused1:+.4f} ms")
    if api is not None and fused0 is not None:
        print(f"  trainer实际路径 - 融合版(step0) = {api - fused0:+.4f} ms（float()主机同步开销）")
    if full_train_update_ms is not None:
        print(f"\n  对照 full_train 实测 update_step 平均: {full_train_update_ms:.4f} ms")
        if api is not None:
            print(f"  微基准 api 路径 / full_train 实测 = {api / full_train_update_ms * 100:.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 update 阶段细分 profiling 结果")
    parser.add_argument("result", type=Path, help="profile_update_step_*.json 路径")
    parser.add_argument("--markdown", action="store_true", help="输出Markdown表格")
    parser.add_argument(
        "--full-train", type=Path, default=None,
        help="full_train 的 profiling_results_*.json，用于对照 update_step 实测均值",
    )
    args = parser.parse_args()
    analyze(args.result, args.full_train, args.markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
