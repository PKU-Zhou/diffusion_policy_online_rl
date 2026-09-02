#!/usr/bin/env bash
# 性能分析测试脚本
#
# 用法：
#   bash run_profiling.sh short    # 运行短期测试 (10K步)
#   bash run_profiling.sh full     # 运行完整训练 (1M步)

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZCZHOU_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ZCZHOU_DIR")"

# 设置动态库路径
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:\
$LD_LIBRARY_PATH

GPU=${GPU:-3}
MEM_FRACTION=${MEM_FRACTION:-.9}

# 默认为短期测试
MODE=${1:-short}

if [ "$MODE" = "short" ]; then
    echo "========================================"
    echo "运行短期性能分析测试 (10K steps)"
    echo "========================================"
    CONFIG=$ZCZHOU_DIR/configs/profiling_short.json
    OUTPUT_DIR=$SCRIPT_DIR/results/short_test
    LOG_FILE=$OUTPUT_DIR/profiling_short_test.log
elif [ "$MODE" = "full" ]; then
    echo "========================================"
    echo "运行完整训练性能分析 (1M steps)"
    echo "========================================"
    CONFIG=$ZCZHOU_DIR/configs/default_cfg.json
    OUTPUT_DIR=$SCRIPT_DIR/results/full_train
    LOG_FILE=$OUTPUT_DIR/profiling_full_train.log
else
    echo "错误: 无效的模式 '$MODE'"
    echo "用法: bash run_profiling.sh [short|full]"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置profiling输出路径
PROFILING_OUTPUT="$OUTPUT_DIR/profiling_results_$(date +%Y%m%d_%H%M%S).json"

# 把 JSON 配置展开成命令行参数
mapfile -t TRAIN_ARGS < <(python - "$CONFIG" <<'PY'
import json, sys

with open(sys.argv[1]) as f:
    cfg = json.load(f)

for key, value in cfg.items():
    if isinstance(value, bool):
        if value:
            print(f"--{key}")
    else:
        print(f"--{key}")
        print(value)
PY
)

echo "配置文件: $CONFIG"
echo "日志文件: $LOG_FILE"
echo "Profiling结果: $PROFILING_OUTPUT"
echo ""

# 运行训练
XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
        CUDA_VISIBLE_DEVICES=$GPU \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
        python "$REPO_DIR/scripts/train_mujoco.py" \
        "${TRAIN_ARGS[@]}" \
        --enable_profiling \
        --profiling_output "$PROFILING_OUTPUT" \
        2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ 训练完成"
    echo "========================================"
    echo "日志已保存到: $LOG_FILE"

    # 显示性能摘要（如果结果文件存在）
    if [ -f "$PROFILING_OUTPUT" ]; then
        echo "Profiling结果已保存到: $PROFILING_OUTPUT"
        echo ""
        echo "性能摘要:"
        python -c "
import json
with open('$PROFILING_OUTPUT') as f:
    data = json.load(f)
    print('\n主要阶段耗时:')
    for name, stats in data.get('summary', {}).items():
        print(f'  {name:30s}: {stats[\"total_duration\"]:10.2f}s (avg: {stats[\"average_duration\"]:8.4f}s, count: {stats[\"count\"]})')
" 2>/dev/null || echo "无法解析profiling结果"
    else
        echo ""
        echo "⚠ 警告: 未找到profiling结果文件 ($PROFILING_OUTPUT)"
        echo "  请检查训练日志确认profiling是否正常启用"
    fi
else
    echo ""
    echo "========================================"
    echo "✗ 训练失败 (exit code: $EXIT_CODE)"
    echo "========================================"
    echo "请查看日志文件: $LOG_FILE"
fi

exit $EXIT_CODE
