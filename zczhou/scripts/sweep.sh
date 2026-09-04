#!/usr/bin/env bash
# 批量实验脚本：对指定维度做笛卡尔积，每张卡同时只跑一个实验，跑完自动取下一个
#
# HalfCheetah-v4
# Ant-v4
# Hopper-v4
# Walker2d-v4
# Swimmer-v4
# Reacher-v4
# InvertedPendulum-v4
# InvertedDoublePendulum-v4

# 用法：
#   bash sweep.sh --env HalfCheetah-v4,Ant-v4 --seed 100,200,300
#   bash sweep.sh --alg sdac,dpmd --seed 100,200 --gpus 0,1,3
#   bash sweep.sh --env Ant-v4 --lr 3e-4,1e-4 --dry-run
#
# 说明：
#   - 任意 --key v1,v2,v3 都会被当作扫描维度；只给一个值则该维度固定
#   - --gpus 指定可用显卡列表（默认 3）
#   - --dry-run 只打印将要执行的命令，不实际运行
#   - suffix 由扫描维度自动生成，便于 inspect_results.py 按目录名聚合

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZCZHOU_DIR="$(dirname "$SCRIPT_DIR")"

GPUS=3
DRY_RUN=0
SWEEP_KEYS=()
SWEEP_VALS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --*)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "错误：$1 缺少取值" >&2
                exit 1
            fi
            SWEEP_KEYS+=("${1#--}")
            SWEEP_VALS+=("$2")
            shift 2
            ;;
        *)
            echo "错误：无法识别的参数 $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#SWEEP_KEYS[@]} -eq 0 ]]; then
    echo "错误：至少需要指定一个扫描维度，例如 --seed 100,200" >&2
    exit 1
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

# 生成笛卡尔积：每行是一组完整的参数组合
COMBOS=("")
for i in "${!SWEEP_KEYS[@]}"; do
    key="${SWEEP_KEYS[$i]}"
    IFS=',' read -r -a values <<< "${SWEEP_VALS[$i]}"
    NEXT=()
    for combo in "${COMBOS[@]}"; do
        for value in "${values[@]}"; do
            NEXT+=("${combo}${key}=${value};")
        done
    done
    COMBOS=("${NEXT[@]}")
done

echo "共 ${#COMBOS[@]} 个实验，可用显卡：${GPU_LIST[*]}"
echo

# 每张卡一个后台槽位，槽位空出来才派发下一个实验
declare -A SLOT_PID

launch() {
    local gpu="$1" combo="$2"

    local args=() suffix_parts=()
    local IFS=';'
    for item in $combo; do
        [[ -z "$item" ]] && continue
        local key="${item%%=*}" value="${item#*=}"
        args+=("--$key" "$value")
        suffix_parts+=("${key}_${value}")
    done
    unset IFS

    local suffix
    suffix=$(printf '%s-' "${suffix_parts[@]}")
    suffix="sweep_${suffix%-}"

    local log_file="$ZCZHOU_DIR/logs/sweep/${suffix}.log"

    echo "[GPU $gpu] $suffix"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "         GPU=$gpu bash run.sh ${args[*]} --suffix $suffix"
        return
    fi

    GPU="$gpu" LOG_FILE="$log_file" \
        bash "$SCRIPT_DIR/run.sh" "${args[@]}" --suffix "$suffix" \
        > /dev/null 2>&1 &
    SLOT_PID[$gpu]=$!
}

idx=0
for combo in "${COMBOS[@]}"; do
    # dry-run 不产生真实进程，按轮转展示派发结果
    if [[ $DRY_RUN -eq 1 ]]; then
        launch "${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}" "$combo"
        ((idx++))
        continue
    fi

    target_gpu=""
    while [[ -z "$target_gpu" ]]; do
        for gpu in "${GPU_LIST[@]}"; do
            pid="${SLOT_PID[$gpu]:-}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                target_gpu="$gpu"
                break
            fi
        done
        # 所有卡都忙，等一会儿再检查
        [[ -z "$target_gpu" ]] && sleep 10
    done
    launch "$target_gpu" "$combo"
done

if [[ $DRY_RUN -eq 0 ]]; then
    wait
    echo
    echo "全部实验结束，日志见 $ZCZHOU_DIR/logs/sweep/"
fi
