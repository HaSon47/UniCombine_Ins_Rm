#!/bin/bash
set -e
export HF_TOKEN=$HF_TOKEN
echo "=============================="
echo " HuggingFace Login"
echo "=============================="
hf auth login --token $HF_TOKEN

echo "=============================="
echo " Creating checkpoint directory"
echo "=============================="
mkdir -p ckpt

echo "=============================="
echo " Downloading FLUX.1-schnell"
echo "=============================="
hf download black-forest-labs/FLUX.1-schnell \
    --local-dir ./ckpt/FLUX.1-schnell

echo "=============================="
echo " Downloading Condition-LoRA"
echo "=============================="
hf download Xuan-World/UniCombine \
    --include "Condition_LoRA/*" \
    --local-dir ./ckpt 

echo "=============================="
echo " Downloading Denoising-LoRA"
echo "=============================="
hf download Xuan-World/UniCombine \
    --include "Denoising_LoRA/*" \
    --local-dir ./ckpt 

echo "=============================="
echo " All downloads completed"
echo "=============================="