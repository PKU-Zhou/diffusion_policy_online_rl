#!/usr/bin/env bash
# 单次训练启动脚本
#
# 用法：
#   bash run.sh                                  # 使用默认配置
#   bash run.sh --env Ant-v4 --seed 200          # 覆盖配置中的部分参数
#   CONFIG=../configs/xxx.json bash run.sh       # 换配置文件
#   GPU=1 bash run.sh                            # 指定显卡
#
# 覆盖参数直接追加在命令行末尾即可：argparse 中后出现的同名参数会覆盖先出现的。

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZCZHOU_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$ZCZHOU_DIR")"

CONFIG=${CONFIG:-$ZCZHOU_DIR/configs/default_cfg.json}
GPU=${GPU:-3}
MEM_FRACTION=${MEM_FRACTION:-.9}
LOG_FILE=${LOG_FILE:-$ZCZHOU_DIR/logs/train.log}

# 这一步是为了设置动态库路径
# 如果不设置的话，pip 安装的 nvidia-* 各组件动态库路径未被加载
# 会导致jax退化到CPU
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cusparse/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cublas/lib:\
$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:\
$LD_LIBRARY_PATH

# 把 JSON 配置展开成命令行参数
# 布尔值为 true 时输出开关（如 --debug），为 false 时跳过
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

mkdir -p "$(dirname "$LOG_FILE")"

XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
        CUDA_VISIBLE_DEVICES=$GPU \
        XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION \
        python "$REPO_DIR/scripts/train_mujoco.py" \
        "${TRAIN_ARGS[@]}" "$@" \
        2>&1 | tee -a "$LOG_FILE"
