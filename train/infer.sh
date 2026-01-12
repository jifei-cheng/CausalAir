CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output_dpo/Qwen3-8B/v4-20251121-235442/checkpoint-1442 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

