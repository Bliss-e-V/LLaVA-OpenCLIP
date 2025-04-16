#!/bin/bash

# Default usage
VISION_TOWER_ORIGIN=openai
VISION_TOWER=clip-vit-large-patch14-336
#  --vision_tower $VISION_TOWER_ORIGIN/$VISION_TOWER

# New, OpenCLIP usage:
# VISION_TOWER_OPENCLIP_ORIGIN=UCSC-VLAA
# VISION_TOWER_OPENCLIP=ViT-L-14-CLIPA-336-datacomp1B
#  --vision_tower hf-hub:$VISION_TOWER_OPENCLIP_ORIGIN/$VISION_TOWER_OPENCLIP

# Set name from pretraining (step 1, feature alignment) here
ADAPTER_NAME=/llava-checkpoints/llava-v1.5-7b-openai-clip-vit-large-patch14-336-bs16-pretrain
# ADAPTER_NAME=/llava-checkpoints/llava-v1.5-7b-UCSC-VLAA-ViT-L-14-CLIPA-336-datacomp1B-bs16-pretrain

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /llava-datasets/visual_instruction_tuning/llava_v1_5_mix665k.json \
    --image_folder /llava-datasets/visual_instruction_tuning/ \
    --vision_tower $VISION_TOWER_ORIGIN/$VISION_TOWER \
    --pretrain_mm_mlp_adapter $ADAPTER_NAME/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /llava-checkpoints/llava-v1.5-7b-$VISION_TOWER_ORIGIN-$VISION_TOWER-bs16-b200 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
