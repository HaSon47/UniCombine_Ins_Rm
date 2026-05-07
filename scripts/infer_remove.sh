#!/bin/bash

python inference_remove.py \
--condition_types subject fill \
--denoising_lora_dir /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/output_remove/train_result/26_04_21-16:37/ckpt/checkpoint-25000 \
--denoising_lora_name fill_remove_subject \
--denoising_lora_weight 1.0 \
--test_dir /mnt/disk1/aiotlab/hachi/data/Location_with_anno_test/Location_with_anno_test \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/Location_with_anno_test/Remove/Uni_ft2_remove_25000/Turn_1 \
--turn 1 \
--padding 255 \
--use_mask
