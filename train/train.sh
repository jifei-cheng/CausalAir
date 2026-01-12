# Qwen3-8B ------------------------------------------------------------s
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/Qwen/Qwen3-8B \
    --train_type lora \
    --dataset /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/COT/sft_data/train.jsonl \
    --val_dataset /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/COT/sft_data/val.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 500 \
    --save_total_limit 3 \
    --logging_steps 20 \
    --max_length 2048 \
    --output_dir output/Qwen3-8B/loar/sft \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --attn_impl flash_attn \
    --model_author aviation \
    --model_name aviation-sft

# Qwen3-8B/loar/dpo
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
    --rlhf_type dpo \
    --model  /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/Qwen/Qwen3-8B \
    --adapters output/Qwen3-8B/loar/sft/v5-20251120-141155/checkpoint-6630 \
    --ref_adapters output/Qwen3-8B/loar/sft/v5-20251120-141155/checkpoint-6630 \
    --train_type lora \
    --dataset /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/COT/dpo_data/train.jsonl \
    --val_dataset /mnt/e/code/pythonProject/project/jifei/paper/AviationAccidentReport/COT/dpo_data/val.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-6 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/Qwen3-8B/loar/sft/dpo    \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --rpo_alpha 1 \
    --attn_impl flash_attn \
    --dataset_num_proc 4
