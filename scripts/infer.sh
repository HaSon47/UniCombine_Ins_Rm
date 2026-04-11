#!/bin/bash
# python inference.py \
# --condition_types fill subject \
# --denoising_lora_name subject_fill_union \
# --denoising_lora_weight 1.0 \
# --fill examples/window/background.jpg \
# --subject examples/window/subject.jpg \
# --json "examples/window/1634_rank0_A decorative fabric topper for windows..json" \
# --version training-based

python inference_location_square.py \
--condition_types subject fill \
--denoising_lora_name subject_fill_union \
--denoising_lora_weight 1.0 \
--test_dir /mnt/disk1/aiotlab/hachi/data/Location_squared/Turn_1 \
--version training-based \
--exam_size 224 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/Location_squared/Turn_1/Unicombine

python inference_location_square.py \
--condition_types subject fill \
--denoising_lora_name subject_fill_union \
--denoising_lora_weight 1.0 \
--test_dir /mnt/disk1/aiotlab/hachi/data/Location_squared/Turn_1 \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/Location_squared/Turn_1/Unicombine_512