#!/bin/bash
set -e

# export VLLM_LOGGING_LEVEL=DEBUG
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
VLLM_BIN="$SCRIPT_DIR/.venv/bin/vllm"

if [[ ! -x "$VLLM_BIN" ]]; then
        echo "vllm executable not found: $VLLM_BIN" >&2
        exit 1
fi

# ---- Tunables (override via env) -------------------------------------------
MODEL_PATH="${MODEL_PATH:-/models/DeepSeek-V3.2}"
TP="${TP:-8}"                       # tensor-parallel size == number of GPUs needed
PORT="${PORT:-30000}"
LOG_DIR="${LOG_DIR:-log}"
# A GPU counts as "free" when its used memory is below this many MiB.
GPU_FREE_MEM_MIB="${GPU_FREE_MEM_MIB:-2000}"

# ---- Free-GPU autodetect ----------------------------------------------------
# Honor an explicit CUDA_VISIBLE_DEVICES only when it lists at least $TP GPUs.
# A preset with too few GPUs (e.g. a stale CUDA_VISIBLE_DEVICES=0 in the shell)
# would crash a multi-GPU run, so fall back to probing. Otherwise scan
# nvidia-smi, keep GPUs whose used memory is under the threshold, and pick the
# first $TP of them. Set FORCE_CVD=1 to honor the preset verbatim regardless.
PRESET_CVD="${CUDA_VISIBLE_DEVICES:-}"
NEED_AUTODETECT=1
if [[ -n "$PRESET_CVD" ]]; then
        IFS=',' read -ra PRESET_ARR <<< "$PRESET_CVD"
        if [[ "${FORCE_CVD:-0}" == "1" || ${#PRESET_ARR[@]} -ge $TP ]]; then
                NEED_AUTODETECT=0
                export CUDA_VISIBLE_DEVICES="$PRESET_CVD"
                echo "Using preset CUDA_VISIBLE_DEVICES=$PRESET_CVD"
        else
                echo "Preset CUDA_VISIBLE_DEVICES=$PRESET_CVD has ${#PRESET_ARR[@]} GPU(s) < TP=$TP; ignoring and autodetecting (set FORCE_CVD=1 to override)." >&2
        fi
fi

if [[ "$NEED_AUTODETECT" == "1" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
                echo "nvidia-smi not found; cannot autodetect GPUs. Set CUDA_VISIBLE_DEVICES manually." >&2
                exit 1
        fi

        mapfile -t FREE_GPUS < <(
                nvidia-smi --query-gpu=index,memory.used \
                        --format=csv,noheader,nounits |
                        awk -F', *' -v thr="$GPU_FREE_MEM_MIB" \
                                '$2+0 < thr { print $1 }'
        )

        if (( ${#FREE_GPUS[@]} < TP )); then
                echo "Need $TP free GPU(s) (used mem < ${GPU_FREE_MEM_MIB} MiB) but found only ${#FREE_GPUS[@]}: [${FREE_GPUS[*]}]" >&2
                echo "Current GPU usage:" >&2
                nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                        --format=csv,noheader >&2
                exit 1
        fi

        SELECTED=("${FREE_GPUS[@]:0:$TP}")
        CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${SELECTED[*]}")
        export CUDA_VISIBLE_DEVICES
        echo "Auto-selected $TP free GPU(s): CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export TZ=Asia/Shanghai
# Ephemeral port pool (1024-65535) on this shared netns is exhausted by ~300k
# CLOSE_WAIT sockets in sibling containers, so `bind(("", 0))` fails inside
# EngineCore. The reserved range 20000-32768 is excluded from the ephemeral
# pool, so binding specific ports there always succeeds. VLLM_PORT switches
# vllm's `_get_open_port` from random-bind to explicit-bind starting at this
# port (incrementing on collision).
export VLLM_PORT="${VLLM_PORT:-30100}"

SERVED_NAME="${SERVED_NAME:-$(basename "$MODEL_PATH")}"

mkdir -p "$LOG_DIR"

"$VLLM_BIN" serve "$MODEL_PATH" --port "$PORT" \
        --tensor-parallel-size "$TP" \
        --served-model-name "$SERVED_NAME" \
        --trust-remote-code \
        --max-num-batched-tokens 81920 \
        --no-enable-prefix-caching \
        --gpu-memory-utilization 0.90 \
        --max-num-seqs 512 \
        --block-size 64 \
        --enforce-eager \
        > "$LOG_DIR/vllm.log" 2>&1 &
