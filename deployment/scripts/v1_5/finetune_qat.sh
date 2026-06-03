#!/bin/bash
# Single-node debug launcher for LLaVA QAT (group-wise LSQ).
# For multi-node SLURM, see finetune_qat.slurm.

set -euo pipefail

MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"liuhaotian/llava-v1.5-7b"}
VISION_TOWER=${VISION_TOWER:-"openai/clip-vit-large-patch14-336"}
CONV_VERSION=${CONV_VERSION:-"v1"}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"pad"}

# Training data: the two ShareGPT4V annotation JSONs (mix665k + cap100k),
# comma-joined; LazySupervisedDataset concatenates them in order.
DATA_ROOT=${DATA_ROOT:-"./playground/data/ShareGPT4V"}
DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json,${DATA_ROOT}/sharegpt4v_instruct_gpt4-vision_cap100k.json"}
IMAGE_FOLDER=${IMAGE_FOLDER:-"${DATA_ROOT}/data"}

QAT_BITS=${QAT_BITS:-4}
QAT_GROUP_SIZE=${QAT_GROUP_SIZE:-128}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
QAT_SCALE_LR=${QAT_SCALE_LR:-2e-6}
QAT_LEARN_SCALES=${QAT_LEARN_SCALES:-True}
QAT_DEBUG_GRAD_CHECK_STEPS=${QAT_DEBUG_GRAD_CHECK_STEPS:-1}

if [[ "${QAT_LEARN_SCALES,,}" == "true" ]]; then
    QAT_SCALE_MODE="learned-scales"
else
    QAT_SCALE_MODE="fixed-scales"
fi
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/llava-qat-w${QAT_BITS}-g${QAT_GROUP_SIZE}-${QAT_SCALE_MODE}"}

deepspeed llava/train/train_qat.py \
    --deepspeed ./scripts/zero1.json \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --version "${CONV_VERSION}" \
    --data_path "${DATA_PATH}" \
    --image_folder "${IMAGE_FOLDER}" \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "${IMAGE_ASPECT_RATIO}" \
    --group_by_modality_length True \
    --bf16 True \
    --tf32 True \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate "${LEARNING_RATE}" \
    --max_grad_norm 1.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --qat_enable True \
    --qat_bits "${QAT_BITS}" \
    --qat_group_size "${QAT_GROUP_SIZE}" \
    --qat_scale_lr "${QAT_SCALE_LR}" \
    --qat_learn_scales "${QAT_LEARN_SCALES}" \
    --qat_debug_grad_check_steps "${QAT_DEBUG_GRAD_CHECK_STEPS}" \
    --qat_save_quantized True
