#!/bin/bash
# Launch multi-GPU parallel evaluation with model sharding.
#
# Each shard gets GPUS_PER_SHARD GPUs (for large models like Qwen 72B BF16).
# Total GPUs used = NUM_SHARDS * GPUS_PER_SHARD
#
# Usage:
#   bash scripts/launch_eval_multi_gpu.sh [NUM_SHARDS] [GPUS_PER_SHARD] [CONDITIONS] [EXTRA_ARGS...]
#
# CONDITIONS is a comma-separated list (no spaces), e.g. "1,2,3,4,5,6,7,8"
#
# Examples:
#   # Evaluate cond 1-8, 4 shards × 2 GPUs, Qwen 72B BF16
#   bash scripts/launch_eval_multi_gpu.sh 4 2 1,2,3,4,5,6,7,8 --no_quant
#
#   # Evaluate cond 7,8 only, 8 shards × 1 GPU, INT4
#   bash scripts/launch_eval_multi_gpu.sh 8 1 7,8

NUM_SHARDS=${1:-4}
GPUS_PER_SHARD=${2:-2}
CONDITIONS_CSV=${3:-8}
shift 3 2>/dev/null
EXTRA_ARGS="$@"

# Convert comma-separated to space-separated for --conditions
CONDITIONS_ARGS="${CONDITIONS_CSV//,/ }"
CONDITIONS_LABEL="${CONDITIONS_CSV//,/_}"

TOTAL_GPUS=$((NUM_SHARDS * GPUS_PER_SHARD))

LOG_DIR="results/table3/logs/cond${CONDITIONS_LABEL}_eval"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " Multi-GPU Parallel Evaluation"
echo " Shards:         $NUM_SHARDS"
echo " GPUs per shard: $GPUS_PER_SHARD"
echo " Total GPUs:     $TOTAL_GPUS"
echo " Conditions:     $CONDITIONS_ARGS"
echo " Extra args:     $EXTRA_ARGS"
echo "============================================="

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_START=$((SHARD_ID * GPUS_PER_SHARD))
    GPU_END=$((GPU_START + GPUS_PER_SHARD - 1))

    GPU_LIST=""
    for g in $(seq $GPU_START $GPU_END); do
        if [ -n "$GPU_LIST" ]; then
            GPU_LIST="${GPU_LIST},$g"
        else
            GPU_LIST="$g"
        fi
    done

    LOG_FILE="${LOG_DIR}/shard${SHARD_ID}.log"

    echo "[Shard $SHARD_ID] GPUs: $GPU_LIST → $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_LIST python scripts/run_table3.py \
        --conditions $CONDITIONS_ARGS \
        --benchmark t2i_reasonbench \
        --eval_only \
        --shard_id $SHARD_ID \
        --num_shards $NUM_SHARDS \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE" &

    PIDS+=($!)
done

echo ""
echo "All $NUM_SHARDS shards launched. PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/shard*.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/shard*.log"
echo ""
echo "Waiting for all shards to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[Shard $i] FAILED with exit code $EXIT_CODE"
        FAILED=$((FAILED + 1))
    else
        echo "[Shard $i] Completed successfully"
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED shard(s) failed! Skipping merge."
    exit 1
else
    echo ""
    echo "All $NUM_SHARDS shards completed successfully!"
    echo ""
    echo "Merging sharded evaluation results..."
    python scripts/run_table3.py \
        --conditions $CONDITIONS_ARGS \
        --benchmark t2i_reasonbench \
        --merge_eval \
        --merge_num_shards $NUM_SHARDS \
        2>&1 | tee "${LOG_DIR}/merge.log"
    echo ""
    echo "Done! Merged results saved. See ${LOG_DIR}/merge.log"
fi
