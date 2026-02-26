#!/bin/bash
# Launch multi-GPU parallel TTS generation.
# Usage: bash scripts/launch_multi_gpu.sh [NUM_GPUS] [CONDITION] [EXTRA_ARGS...]
# Example: bash scripts/launch_multi_gpu.sh 8 8 --generate_only

NUM_GPUS=${1:-8}
CONDITION=${2:-8}
shift 2 2>/dev/null
EXTRA_ARGS="$@"

LOG_DIR="results/table3/logs/cond${CONDITION}"
mkdir -p "$LOG_DIR"

echo "============================================="
echo " Multi-GPU TTS Generation"
echo " GPUs: $NUM_GPUS"
echo " Condition: $CONDITION"
echo " Extra args: $EXTRA_ARGS"
echo "============================================="

PIDS=()
for SHARD_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${LOG_DIR}/shard${SHARD_ID}.log"

    echo "[GPU $SHARD_ID] Starting shard $SHARD_ID / $NUM_GPUS → $LOG_FILE"

    CUDA_VISIBLE_DEVICES=$SHARD_ID python scripts/run_table3.py \
        --conditions $CONDITION \
        --benchmark t2i_reasonbench \
        --shard_id $SHARD_ID \
        --num_shards $NUM_GPUS \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG_FILE" &

    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS shards launched. PIDs: ${PIDS[*]}"
echo "Logs: ${LOG_DIR}/cond${CONDITION}_shard*.log"
echo ""
echo "Monitor progress:"
echo "  tail -f ${LOG_DIR}/cond${CONDITION}_shard*.log"
echo ""
echo "Waiting for all shards to finish..."

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}"
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[GPU $i] FAILED with exit code $EXIT_CODE"
        FAILED=$((FAILED + 1))
    else
        echo "[GPU $i] Completed successfully"
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "WARNING: $FAILED shard(s) failed!"
    exit 1
else
    echo ""
    echo "All $NUM_GPUS shards completed successfully!"
fi
