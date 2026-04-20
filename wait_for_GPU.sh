#!/bin/bash

REQUIRED_MEM_MB=10240   # 10GB
CHECK_INTERVAL=30       # check mỗi 30s

echo "⏳ Waiting for GPU with at least ${REQUIRED_MEM_MB} MB free..."

while true; do
    # Lấy memory free của tất cả GPU (MB)
    FREE_MEM_LIST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    GPU_ID=0
    for FREE_MEM in $FREE_MEM_LIST; do
        echo "GPU $GPU_ID free memory: ${FREE_MEM} MB"

        if [ "$FREE_MEM" -ge "$REQUIRED_MEM_MB" ]; then
            echo "🚀 Found suitable GPU: $GPU_ID"

            # Set GPU cần dùng
            export CUDA_VISIBLE_DEVICES=$GPU_ID

            # Chạy script của bạn
            bash infer_location.sh

            exit 0
        fi

        GPU_ID=$((GPU_ID + 1))
    done

    echo "❌ No GPU available, retrying in ${CHECK_INTERVAL}s..."
    sleep $CHECK_INTERVAL
done