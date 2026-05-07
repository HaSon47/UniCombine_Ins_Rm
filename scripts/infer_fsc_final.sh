#!/bin/bash

echo "--------------run FSC_difficult_no_overlap --------------------"
python inference_fsc_final.py \
--condition_types subject fill \
--denoising_lora_dir /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/output/train_result/26_04_16-16:45/checkpoint-10000 \
--denoising_lora_name subject_fill_finetuned_union \
--denoising_lora_weight 1.0 \
--pretrained_subject_condition_lora_dir ckpt/Condition_LoRA \
--test_dir /mnt/disk1/aiotlab/hachi/data/FSC_final/FSC_difficult_no_overlap \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/FSC_final/Uni_ft1_pad_512/Turn_1/FSC_difficult_no_overlap \
--turn 1 \
--padding

echo "--------------run FSC_difficult_overlap --------------------"
python inference_fsc_final.py \
--condition_types subject fill \
--denoising_lora_dir /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/output/train_result/26_04_16-16:45/checkpoint-10000 \
--denoising_lora_name subject_fill_finetuned_union \
--denoising_lora_weight 1.0 \
--pretrained_subject_condition_lora_dir ckpt/Condition_LoRA \
--test_dir /mnt/disk1/aiotlab/hachi/data/FSC_final/FSC_difficult_overlap \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/FSC_final/Uni_ft1_pad_512/Turn_1/FSC_difficult_overlap \
--turn 1 \
--padding

echo "--------------run FSC_easy_no_overlap --------------------"
python inference_fsc_final.py \
--condition_types subject fill \
--denoising_lora_dir /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/output/train_result/26_04_16-16:45/checkpoint-10000 \
--denoising_lora_name subject_fill_finetuned_union \
--denoising_lora_weight 1.0 \
--pretrained_subject_condition_lora_dir ckpt/Condition_LoRA \
--test_dir /mnt/disk1/aiotlab/hachi/data/FSC_final/FSC_easy_no_overlap \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/FSC_final/Uni_ft1_pad_512/Turn_1/FSC_easy_no_overlap \
--turn 1 \
--padding

echo "--------------run FSC_easy_overlap --------------------"
python inference_fsc_final.py \
--condition_types subject fill \
--denoising_lora_dir /mnt/disk1/aiotlab/hachi/code/UniCombine_Ins_Rm/output/train_result/26_04_16-16:45/checkpoint-10000 \
--denoising_lora_name subject_fill_finetuned_union \
--denoising_lora_weight 1.0 \
--pretrained_subject_condition_lora_dir ckpt/Condition_LoRA \
--test_dir /mnt/disk1/aiotlab/hachi/data/FSC_final/FSC_easy_overlap \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/FSC_final/Uni_ft1_pad_512/Turn_1/FSC_easy_overlap \
--turn 1 \
--padding