#!/bin/bash
pwd
REQUIRED_MEM_MB=33240   # 10GB
CHECK_INTERVAL=45
WORKDIR="/mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm"
CONDA_BASE="/mnt/disk1/aiotlab/miniconda"
ENV_NAME="unicombine"

echo "⏳ Waiting for GPU with at least ${REQUIRED_MEM_MB} MB free..."

# 👉 init conda (chuẩn tuyệt đối)
source "$CONDA_BASE/etc/profile.d/conda.sh" || {
    echo "❌ Cannot load conda.sh"
    exit 1
}
pwd
while true; do
    FREE_MEM_LIST=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)

    GPU_ID=0
    for FREE_MEM in $FREE_MEM_LIST; do
        echo "GPU $GPU_ID free memory: ${FREE_MEM} MB"

        if [ "$FREE_MEM" -ge "$REQUIRED_MEM_MB" ]; then
            echo "🚀 Found suitable GPU: $GPU_ID"

            export CUDA_VISIBLE_DEVICES=$GPU_ID

            cd "$WORKDIR" || {
                echo "❌ Cannot cd to $WORKDIR"
                exit 1
            }

            # 👉 activate env
            conda activate "$ENV_NAME" || {
                echo "❌ Cannot activate env $ENV_NAME"
                exit 1
            }

            echo "📂 Current dir: $(pwd)"
            echo "🐍 Python: $(which python)"

            echo "▶️ Running infer_location.sh..."
            bash scripts/infer_location.sh

            exit 0
        fi

        GPU_ID=$((GPU_ID + 1))
    done

    echo "❌ No GPU available, retrying in ${CHECK_INTERVAL}s..."
    sleep $CHECK_INTERVAL
done