"""分析 delay_infer.py 产出的 delay_N*_<ts>.json，对比不同动作延迟步数下的回报衰减。

以 delay_steps=0 的一档为 baseline，计算各档相对回报衰减，给出延迟步数 N、对应仿真
时间、滞后的控制周期数，并根据衰减幅度给出下一档延迟建议。

用法：
    python analyze_delay.py --inputs results/delay_N0_*.json results/delay_N1_*.json
    python analyze_delay.py --inputs results/delay_N*.json --out results/report_delay_<ts>.md
"""

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for p in patterns:
        matched = sorted(glob.glob(p))
        if matched:
            paths.extend(Path(m) for m in matched)
        else:
            candidate = Path(p)
            if candidate.is_file():
                paths.append(candidate)
            else:
                raise FileNotFoundError(f"未匹配到文件: {p}")
    # 去重并保持顺序
    seen = set()
    uniq = []
    for p in paths:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def suggest_next_steps(baseline_ret: float, rows: list[dict]) -> str:
    """根据已测档位的衰减幅度，给出下一档延迟建议。"""
    nonzero = [r for r in rows if r["delay_steps"] > 0]
    if not nonzero:
        return "- 仅 baseline，建议先跑 `--delay_steps 1`"
    smallest = min(nonzero, key=lambda r: r["delay_steps"])
    decay = (baseline_ret - smallest["ep_ret"]) / baseline_ret if baseline_ret > 0 else 0.0
    n = smallest["delay_steps"]
    lines = []
    if abs(decay) < 0.01:
        lines.append(f"- N={n} 衰减 {decay * 100:.2f}%（<1%），任务对该延迟不敏感，"
                     f"建议加大档位：N={n * 2}、N={n * 4}（延迟 {n * 2 * 50}ms / {n * 4 * 50}ms 仿真时间）")
    elif decay > 0:
        lines.append(f"- N={n} 衰减 {decay * 100:.2f}%，已见明显影响")
        candidates = sorted({n + 1, 2 * n})
        cand_str = "、".join(f"N={c}" for c in candidates)
        lines.append(f"- 如需刻画衰减曲线，可加测 {cand_str} 观察趋势")
        lines.append("- 如需更细粒度（<1 个控制周期），需改环境步长或插值，超出当前机制范围")
    else:
        lines.append(f"- N={n} 回报反而上升 {abs(decay) * 100:.2f}%（单 seed 噪声可能），"
                     "建议补 seed 或加档确认")
    return "\n".join(lines)


def render_report(rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: r["delay_steps"])
    baseline = next((r for r in rows if r["delay_steps"] == 0), None)
    baseline_ret = baseline["ep_ret"] if baseline else None

    meta0 = rows[0]["meta"]
    lines = []
    lines.append("# 动作延迟对任务回报影响报告")
    lines.append("")
    lines.append("## 实验设置")
    lines.append("")
    lines.append(f"- 环境：`{meta0['env']}` (obs_dim={meta0['obs_dim']}, act_dim={meta0['act_dim']})")
    lines.append(f"- checkpoint：`{meta0['checkpoint']}`")
    lines.append(f"- seed：`{meta0['seed']}`（单 seed，仅看趋势）")
    lines.append(f"- backend：`{meta0['default_backend']}`，devices={meta0['jax_devices']}")
    ctrl_dt_ms = meta0.get("ctrl_dt_s") * 1e3 if meta0.get("ctrl_dt_s") else None
    if ctrl_dt_ms is not None:
        lines.append(f"- 控制周期（单步仿真时间）：{ctrl_dt_ms:.1f} ms"
                     f"（环境控制频率 {1e3 / ctrl_dt_ms:.1f} Hz）")
    lines.append("")
    lines.append("机制：策略在 t 时刻生成的动作入队，环境每步仍按队列中 N 步前的动作推进，"
                 "环境持续演化而动作生效滞后 N 步；前 N 步用零动作填充。")
    lines.append("注意：环境控制频率恒为 20Hz 不变，此处“延迟”指动作信息滞后 N 个控制周期，"
                 "而非降低环境仿真频率。")
    lines.append("")
    lines.append("## 结果")
    lines.append("")
    header = "| 延迟步数 N | 延迟仿真时间 (ms) | 滞后控制周期 | ep_ret | ep_len | 相对 baseline |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        n = r["delay_steps"]
        delay_ms = r["meta"].get("delay_sim_ms")
        delay_str = f"{delay_ms:.1f}" if delay_ms is not None else "-"
        if baseline_ret is not None and baseline_ret > 0:
            rel = (r["ep_ret"] - baseline_ret) / baseline_ret * 100
            rel_str = f"{rel:+.2f}%" if n > 0 else "baseline"
        else:
            rel_str = "-"
        lines.append(f"| {n} | {delay_str} | {n} | {r['ep_ret']:.2f} | {r['ep_len']} | {rel_str} |")
    lines.append("")

    if baseline_ret is not None:
        lines.append("## 衰减分析")
        lines.append("")
        for r in rows:
            if r["delay_steps"] == 0:
                continue
            decay = (baseline_ret - r["ep_ret"]) / baseline_ret * 100
            lines.append(f"- N={r['delay_steps']}：回报 {r['ep_ret']:.2f}，"
                         f"相对 baseline 衰减 {decay:.2f}%")
        lines.append("")
        lines.append("## 下一档建议")
        lines.append("")
        lines.append(suggest_next_steps(baseline_ret, rows))
        lines.append("")
    else:
        lines.append("## 衰减分析")
        lines.append("")
        lines.append("- 未提供 N=0 baseline 档，无法计算相对衰减")
        lines.append("")

    lines.append("## 备注")
    lines.append("")
    lines.append("- 单 seed 结果仅能看趋势，不具统计显著性；如需稳健结论请补多 seed")
    lines.append("- 原始数据见对应 delay_N*_<ts>.json")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="delay_N*_<ts>.json 路径或 glob（如 results/delay_N*.json）")
    parser.add_argument("--out", type=Path, default=None,
                        help="报告输出路径，默认与首个输入同目录的 report_delay_<ts>.md")
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    rows = []
    for p in paths:
        data = load(p)
        rows.append({
            "path": str(p),
            "delay_steps": data["meta"]["delay_steps"],
            "ep_ret": data["episode"]["ep_ret"],
            "ep_len": data["episode"]["ep_len"],
            "meta": data["meta"],
        })
    if not rows:
        raise ValueError("未加载到任何输入文件")

    report = render_report(rows)

    out_path: Path = args.out
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = paths[0].parent / f"report_delay_{ts}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"report saved: {out_path}")


if __name__ == "__main__":
    main()
