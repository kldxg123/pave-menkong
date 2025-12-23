#!/bin/bash

# === 1. 显存分配优化 (防止碎片化导致 OOM) ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === 2. 定义路径变量 (确保与训练时使用的本地底座一致) ===
# 原脚本用的是 HF ID，这里强制改为你本地的路径
MODEL_BASE="./models/llava-onevision-qwen2-7b-ov"
MODEL_PATH="./checkpoints/pave_v5_1_3_lora_avsd_7B_imagebind"

# === 3. 执行推理 ===
# 使用单卡 GPU 0 进行推理 (A100 80G 足够)
CUDA_VISIBLE_DEVICES=0 python eval_pave_avsd.py \
    --model-path ${MODEL_PATH} \
    --model-base ${MODEL_BASE} \
    --model-arg-name VideoFeatModelArgumentsV5_1_3_audio_languagebind_7B \
    --annotation-file ./data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json \
    --video-folder ./data/video_instruction_tuning/avsd/Charades_vu17_test_480 \
    --feature-folder ./data/video_instruction_tuning/avsd/Charades_vu17_test_audio_imagebind_feat \
    --pred-save ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avsd_7B_imagebind.json \
    --for_get_frames_num 32 \
    --mm_spatial_pool_stride 2 \
    --mm_spatial_pool_mode bilinear \
    --mm_newline_position grid \
    --overwrite True \
    --num-workers 8 \
    --conv-mode conv_llava_ov_qwen

# === 4. 执行评分 (CIDEr/BLEU 等) ===
# 只有当上一步生成了 json 文件后才运行
if [ -f "./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avsd_7B_imagebind.json" ]; then
    echo "Prediction generated. Running evaluation..."
    python tools/audio/avsd/run_coco_eval.py \
        --gt-file ./data/video_instruction_tuning/avsd/coco_version_test_gt.json \
        --results-file ./data/video_instruction_tuning/prediction/pave_v5_1_3_lora_avsd_7B_imagebind.json
else
    echo "Error: Prediction file not found!"
fi