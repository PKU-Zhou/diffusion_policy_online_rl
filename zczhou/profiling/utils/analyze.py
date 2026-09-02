"""
性能数据分析脚本

读取 timer 导出的 JSON，计算各阶段耗时与占比，输出可直接写入文档的表格。

用法:
    python analyze.py <profiling_results.json> [--markdown]
"""

import argparse
import json
from pathlib import Path

# 顶层阶段（TimerRegistry 中 parent 为 None 的计时器）
TOP_PHASES = ['setup_total', 'training_loop', 'finish_total']

# training_loop 的直接子阶段
LOOP_CHILDREN = ['sample_step', 'update_step', 'save_policy', 'evaluator_communication']

# sample_step 的细分
SAMPLE_CHILDREN = ['sample_policy_inference', 'sample_env_step']


def fmt_time(seconds: float) -> str:
    """将秒格式化为易读形式"""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{sec:.0f}s"


def load(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def get(summary: dict, name: str, field: str = 'total_duration', default=0.0):
    return summary.get(name, {}).get(field, default)


def analyze(data: dict) -> dict:
    summary = data['summary']

    setup = get(summary, 'setup_total')
    loop = get(summary, 'training_loop')
    finish = get(summary, 'finish_total')
    warmup = get(summary, 'warmup_total')

    # warmup 在 train() 内部调用，被计入 training_loop，需要剥离出来
    loop_net = loop - warmup
    total = setup + loop + finish

    sample = get(summary, 'sample_step')
    update = get(summary, 'update_step')
    save_policy = get(summary, 'save_policy')
    eval_comm = get(summary, 'evaluator_communication')
    loop_other = loop_net - (sample + update + save_policy + eval_comm)

    result = {
        'total': total,
        'phases': {
            'setup': setup,
            'warmup': warmup,
            'training_loop': loop_net,
            'finish': finish,
        },
        'loop_breakdown': {
            'sample_step': sample,
            'update_step': update,
            'save_policy': save_policy,
            'evaluator_communication': eval_comm,
            'loop_overhead': loop_other,
        },
        'sample_breakdown': {
            'policy_inference': get(summary, 'sample_policy_inference'),
            'env_step': get(summary, 'sample_env_step'),
        },
        'finish_breakdown': {
            'save_state': get(summary, 'finish_save_state'),
            'wait_evaluator': get(summary, 'finish_wait_evaluator'),
        },
        'setup_breakdown': {
            'algorithm_warmup': get(summary, 'algorithm_warmup'),
            'logger_init': get(summary, 'logger_init'),
            'save_network_structure': get(summary, 'save_network_structure'),
            'evaluator_init': get(summary, 'evaluator_init'),
        },
        'counts': {k: get(summary, k, 'count', 0) for k in
                   ['sample_step', 'update_step', 'sample_policy_inference',
                    'sample_env_step', 'save_policy']},
        'averages': {k: get(summary, k, 'average_duration') for k in
                     ['sample_step', 'update_step', 'sample_policy_inference',
                      'sample_env_step']},
    }
    return result


def print_report(r: dict):
    total = r['total']

    def pct(v):
        return f"{v / total * 100:6.2f}%" if total > 0 else "   n/a"

    print("=" * 66)
    print(f"训练总耗时: {fmt_time(total)}  ({total:.2f}s)")
    print("=" * 66)

    print("\n[顶层阶段]")
    for name, label in [('setup', '初始化 (setup)'),
                        ('warmup', 'Warmup'),
                        ('training_loop', '训练主循环'),
                        ('finish', '结束清理 (finish)')]:
        v = r['phases'][name]
        print(f"  {label:24s} {fmt_time(v):>12s}  {pct(v)}")

    loop = r['phases']['training_loop']

    def lpct(v):
        return f"{v / loop * 100:6.2f}%" if loop > 0 else "   n/a"

    print(f"\n[训练主循环内部分解]  (基数 {fmt_time(loop)})")
    for name, label in [('sample_step', 'Sample 阶段'),
                        ('update_step', 'Update 阶段'),
                        ('save_policy', '策略保存'),
                        ('evaluator_communication', '评估器通信'),
                        ('loop_overhead', '循环其他开销')]:
        v = r['loop_breakdown'][name]
        print(f"  {label:24s} {fmt_time(v):>12s}  {lpct(v)}")

    sample = r['loop_breakdown']['sample_step']

    def spct(v):
        return f"{v / sample * 100:6.2f}%" if sample > 0 else "   n/a"

    print(f"\n[Sample 阶段细分]  (基数 {fmt_time(sample)})")
    for name, label in [('policy_inference', '策略推理 (扩散去噪)'),
                        ('env_step', '环境交互 (env.step)')]:
        v = r['sample_breakdown'][name]
        print(f"  {label:24s} {fmt_time(v):>12s}  {spct(v)}")

    print("\n[单步平均耗时]")
    for name, label in [('sample_step', 'sample 单次'),
                        ('sample_policy_inference', '  └ 策略推理'),
                        ('sample_env_step', '  └ 环境交互'),
                        ('update_step', 'update 单次')]:
        avg = r['averages'].get(name, 0)
        cnt = r['counts'].get(name, 0)
        print(f"  {label:24s} {avg * 1000:8.3f}ms  (调用 {cnt} 次)")

    print("\n[初始化细分]")
    for name, label in [('algorithm_warmup', 'JIT 编译预热'),
                        ('save_network_structure', '网络结构保存'),
                        ('logger_init', '日志初始化'),
                        ('evaluator_init', '评估器启动')]:
        print(f"  {label:24s} {fmt_time(r['setup_breakdown'][name]):>12s}")

    print("\n[结束阶段细分]")
    for name, label in [('save_state', '保存最终状态'),
                        ('wait_evaluator', '等待评估器子进程')]:
        v = r['finish_breakdown'][name]
        print(f"  {label:24s} {fmt_time(v):>12s}  {pct(v)}")

    su, up = r['loop_breakdown']['sample_step'], r['loop_breakdown']['update_step']
    if su > 0:
        print(f"\nUpdate / Sample 耗时比: {up / su:.2f} : 1")


def print_markdown(r: dict):
    total = r['total']

    def pct(v):
        return f"{v / total * 100:.2f}%" if total > 0 else "n/a"

    print("\n<!-- 顶层阶段 -->\n")
    print("| 阶段 | 耗时 | 占总时间 |")
    print("| --- | --- | --- |")
    for name, label in [('setup', '初始化 setup'), ('warmup', 'Warmup'),
                        ('training_loop', '训练主循环'), ('finish', '结束清理 finish')]:
        v = r['phases'][name]
        print(f"| {label} | {fmt_time(v)} | {pct(v)} |")
    print(f"| **合计** | **{fmt_time(total)}** | **100%** |")

    loop = r['phases']['training_loop']

    def lpct(v):
        return f"{v / loop * 100:.2f}%" if loop > 0 else "n/a"

    print("\n<!-- 主循环分解 -->\n")
    print("| 主循环内阶段 | 耗时 | 占主循环 | 单次平均 | 调用次数 |")
    print("| --- | --- | --- | --- | --- |")
    for name, label in [('sample_step', 'Sample'), ('update_step', 'Update'),
                        ('save_policy', '策略保存'),
                        ('evaluator_communication', '评估器通信'),
                        ('loop_overhead', '循环其他开销')]:
        v = r['loop_breakdown'][name]
        avg = r['averages'].get(name)
        cnt = r['counts'].get(name, 0)
        avg_s = f"{avg * 1000:.3f}ms" if avg else "-"
        cnt_s = str(cnt) if cnt else "-"
        print(f"| {label} | {fmt_time(v)} | {lpct(v)} | {avg_s} | {cnt_s} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', help='profiling_results.json 路径')
    ap.add_argument('--markdown', action='store_true', help='额外输出 markdown 表格')
    args = ap.parse_args()

    if not Path(args.path).exists():
        raise SystemExit(f"文件不存在: {args.path}")

    r = analyze(load(args.path))
    print_report(r)
    if args.markdown:
        print_markdown(r)


if __name__ == '__main__':
    main()
