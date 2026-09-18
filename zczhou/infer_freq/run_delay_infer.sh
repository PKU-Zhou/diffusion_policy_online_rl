#!/usr/bin/env bash
# 动作延迟对任务回报影响实验启动脚本（GPU）
# 通过动作延迟队列模拟观测-动作延迟，跑一个完整 episode
#
# 用法：
#   bash run_delay_infer.sh                          # 默认 N=0 无延迟 baseline，GPU=3
#   bash run_delay_infer.sh --delay_steps 1          # 延迟 1 个控制周期
#   DELAY=2 bash run_delay_infer.sh                  # 等价 --delay_steps 2
#   GPU=1 bash run_delay_infer.sh --delay_steps 1    # 换显卡
#
# 覆盖参数直接追加在命令行末尾即可，会透传给 delay_infer.py。

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZCZHOU_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ZCZHOU_DIR")"

# deterministic.pkl 是 jax 0.9.2 导出的计算图，必须用同版本环境加载
PYTHON=${PYTHON:-/data/home/zch_zhou28/.conda/envs/relax_blackwell/bin/python}
PY_ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"

DEVICE=${DEVICE:-gpu}
GPU=${GPU:-3}
MEM_FRACTION=${MEM_FRACTION:-.9}
DELAY=${DELAY:-0}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR=${RESULTS_DIR:-$SCRIPT_DIR/results}
LOG_FILE=${LOG_FILE:-$RESULTS_DIR/delay_infer_$TIMESTAMP.log}
OUT_JSON=${OUT_JSON:-$RESULTS_DIR/delay_N${DELAY}_$TIMESTAMP.json}

QUANT=${QUANT:-0}

INFER_ARGS=(--out "$OUT_JSON" --delay_steps "$DELAY")
if [ -n "${LOG_DIR:-}" ]; then
    INFER_ARGS+=(--log_dir "$LOG_DIR")
fi
if [ "$QUANT" = "1" ]; then
    INFER_ARGS+=(--quant)
fi
if [ -n "${QUANT_TARGET:-}" ]; then
    INFER_ARGS+=(--quant_target "$QUANT_TARGET")
fi
if [ -n "${ACT_GROUP_SIZE:-}" ]; then
    INFER_ARGS+=(--act_group_size "$ACT_GROUP_SIZE")
fi

# 这一步是为了设置动态库路径
# 如果不设置的话，pip 安装的 nvidia-* 各组件动态库路径未被加载
# 会导致jax退化到CPU
export LD_LIBRARY_PATH=$PY_ENV_ROOT/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$PY_ENV_ROOT/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$PY_ENV_ROOT/lib/python3.11/site-packages/nvidia/cublas/lib:\
$PY_ENV_ROOT/lib/python3.11/site-packages/nvidia/cudnn/lib:\
$LD_LIBRARY_PATH

# 服务器无显示设备，MuJoCo 必须走 EGL 离屏渲染才能出帧
export MUJOCO_GL=${MUJOCO_GL:-egl}

mkdir -p "$RESULTS_DIR"

if [ "$DEVICE" = "cpu" ]; then
    JAX_PLATFORMS=cpu \
            PYTHONPATH=$REPO_DIR:$PYTHONPATH \
            "$PYTHON" "$SCRIPT_DIR/delay_infer.py" \
            "${INFER_ARGS[@]}" "$@" \
            2>&1 | tee "$LOG_FILE"
else
    CUDA_VISIBLE_DEVICES=$GPU \
            XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
            PYTHONPATH=$REPO_DIR:$PYTHONPATH \
            "$PYTHON" "$SCRIPT_DIR/delay_infer.py" \
            "${INFER_ARGS[@]}" "$@" \
            2>&1 | tee "$LOG_FILE"
fi

echo "log    : $LOG_FILE"
echo "result : $OUT_JSON"
