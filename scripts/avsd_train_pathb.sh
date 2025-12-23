#!/bin/bash

WANDB__SERVICE_WAIT=500 deepspeed --master_port 60000 train_pave_w_feat.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --annotation_path ./data/video_instruction_tuning/avsd/avsd_train_instruct.json  \
    --fast_path_mapping_path ./data/video_instruction_tuning/avsd/avsd_all_feats_mapping.json \
    --slow_path_mapping_path ./data/video_instruction_tuning/avsd/avsd_all_videos_mapping.json \
    --data_root ./data/video_instruction_tuning/avsd/Charades_v1_audio_imagebind_feat \
    --slow_path_data_root ./data/video_instruction_tuning/avsd/Charades_v1_480 \
    \
    --model_name_or_path /home/app-ahr/.cache/modelscope/hub/Qwen/Qwen3-VL-32B-Instruct \
    --version plain \
    --model_class pave_qwen3vl_pathb:PaveQwen3VLMPathB \
    --dataset_class LazySupervisedVideoDatasetPathB \
    \
    --output_dir ./checkpoints/pave_pathb_qwen3vl_32B_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --bf16 True \
    --tf32 True \
    --mm_newline_position grid \
    --mm_spatial_pool_mode bilinear \
    --feat_combine_method add \
    --fast_feat_type audio \
    --use_slow True \
    --use_slow_feat False \
    --use_fast False \
    --use_fast_feat True

