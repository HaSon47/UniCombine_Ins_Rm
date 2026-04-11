#!/bin/bash
# python inference.py \
# --condition_types fill subject \
# --denoising_lora_name subject_fill_union \
# --denoising_lora_weight 1.0 \
# --fill examples/window/background.jpg \
# --subject examples/window/subject.jpg \
# --json "examples/window/1634_rank0_A decorative fabric topper for windows..json" \
# --version training-based

python inference_location.py \
--condition_types subject fill \
--denoising_lora_name subject_fill_union \
--denoising_lora_weight 1.0 \
--test_dir /mnt/disk1/aiotlab/hachi/data/Location_with_anno_test/Location_with_anno_test \
--version training-based \
--exam_size 512 \
--work_dir /mnt/disk1/aiotlab/hachi/Output/Location_with_anno_test/Unicombine_512/Turn_1 \
--turn 1