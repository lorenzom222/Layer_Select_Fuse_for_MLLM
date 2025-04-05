#!/bin/bash
BASE_MODEL_NAME="test"

# Define common variables
FUSING_STRATEGY="I_D" # Options: E_D, E_M, I_D, I_M
USING_STRATEGY="3-18-23" # Options: 18, 3-18, 3-18-23, former, latter, all
MODEL_NAME="siglip_14_665k" # Specific model name, {Vsiual Encoder}_{LLM size}_{data size}

# Define paths
PRETRAIN_DATA_PATH="./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json"
PRETRAIN_IMAGE_FOLDER="./playground/data/LLaVA-Pretrain/images"
MODEL_PATH="mtgv/MobileLLaMA-1.4B-Base" # mtgv/MobileLLaMA-2.7B-Base
VISION_TOWER="google/siglip-so400m-patch14-384" # google/siglip-so400m-patch14-384  openai/clip-vit-large-patch14-336
FINETUNE_DATA_PATH="./playground/data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json" # ./playground/Cambrian-10M/jsons/cleaned_Cambrian737k.json
FINETUNE_IMAGE_FOLDER="./playground/data" # ./playground/Cambrian-10M


# Pretraining
deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version plain \
    --data_path ${PRETRAIN_DATA_PATH} \
    --image_folder ${PRETRAIN_IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --layer_using_strategy ${USING_STRATEGY} \
    --layer_fusing_strategy ${FUSING_STRATEGY} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 1e-3 \
    --weight_decay 5e-2 \
    --warmup_steps 200 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name ${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}
#--max_steps 10 \
# Fine-tuning
deepspeed llava/train/train.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${MODEL_PATH} \
    --version v1 \
    --data_path ${FINETUNE_DATA_PATH} \
    --image_folder ${FINETUNE_IMAGE_FOLDER} \
    --vision_tower ${VISION_TOWER} \
    --pretrain_mm_mlp_adapter ./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-pretrain-${USING_STRATEGY}-${MODEL_NAME}/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --layer_using_strategy ${USING_STRATEGY} \
    --layer_fusing_strategy ${FUSING_STRATEGY} \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoint/${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_name ${BASE_MODEL_NAME}-${FUSING_STRATEGY}-finetune-${USING_STRATEGY}-${MODEL_NAME}
#     --max_steps 10 \