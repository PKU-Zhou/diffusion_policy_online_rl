#!/usr/bin/env bash
# 单次推理脚本
# 用已有权重,进行单次推理
#
# 用法：
#   bash run_single_infer.sh                                   # 默认最新 checkpoint，CPU
#   bash run_single_infer.sh --policy policy-500000-100000.pkl # 指定某个 checkpoint
#   LOG_DIR=/path/to/run bash run_single_infer.sh              # 换实验目录
#   DEVICE=gpu GPU=3 bash run_single_infer.sh                  # 用显卡
#   VIDEO=1 bash run_single_infer.sh                           # 录制视频到 videos/
#   VIDEO=1 bash run_single_infer.sh --max_steps 200           # 只录前 200 步
#   QUANT=1 bash run_single_infer.sh                           # MLP 层 INT8 推理
#   QUANT=1 QUANT_TARGET=policy bash run_single_infer.sh       # 只量化策略网
#   QUANT=1 ACT_GROUP_SIZE=32 bash run_single_infer.sh         # 换激活分组大小
#   QUANT=1 bash run_single_infer.sh --no_skip_first_layer     # 首层也量化（对照）
#
# 覆盖参数直接追加在命令行末尾即可，会透传给 single_infer.py。

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZCZHOU_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ZCZHOU_DIR")"

# deterministic.pkl 是 jax 0.9.2 导出的计算图，必须用同版本环境加载
PYTHON=${PYTHON:-/data/home/zch_zhou28/.conda/envs/relax_blackwell/bin/python}
PY_ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"

DEVICE=${DEVICE:-cpu}
GPU=${GPU:-3}
MEM_FRACTION=${MEM_FRACTION:-.9}
LOG_FILE=${LOG_FILE:-$ZCZHOU_DIR/logs/single_infer.log}

VIDEO=${VIDEO:-0}
QUANT=${QUANT:-0}

INFER_ARGS=()
if [ -n "${LOG_DIR:-}" ]; then
    INFER_ARGS+=(--log_dir "$LOG_DIR")
fi
if [ "$VIDEO" = "1" ]; then
    INFER_ARGS+=(--video)
fi
if [ -n "${VIDEO_DIR:-}" ]; then
    INFER_ARGS+=(--video_dir "$VIDEO_DIR")
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

mkdir -p "$(dirname "$LOG_FILE")"

if [ "$DEVICE" = "cpu" ]; then
    JAX_PLATFORMS=cpu \
            PYTHONPATH=$REPO_DIR:$PYTHONPATH \
            "$PYTHON" "$SCRIPT_DIR/single_infer.py" \
            "${INFER_ARGS[@]}" "$@" \
            2>&1 | tee -a "$LOG_FILE"
else
    CUDA_VISIBLE_DEVICES=$GPU \
            XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
            PYTHONPATH=$REPO_DIR:$PYTHONPATH \
            "$PYTHON" "$SCRIPT_DIR/single_infer.py" \
            "${INFER_ARGS[@]}" "$@" \
            2>&1 | tee -a "$LOG_FILE"
fi
