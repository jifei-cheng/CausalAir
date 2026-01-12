"""
Generate Responses
LoAR Fine-tuned Model
"""
import os
import json
from tqdm import tqdm  # Progress bar

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download,
    get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift

###########################################
#               Configuration Parameters
###########################################
model_path = './meta-llama/Llama-3.1-8B-Instruct'
# Ensure the checkpoint path is correct
lora_checkpoint = safe_snapshot_download(
    '/output/Llama-3.1-8B/loar/dpo/v1-20251128-221822/checkpoint-1458'
)

# Modify input and output paths
input_file = "./evaluation/contrast_eva/contrast_sample.json"  # Ensure the file name is correct
output_file = "./evaluation/contrast_eva/Llama-3.1-8B.json"

BATCH = 4   # Adjust according to GPU memory, 4090D recommends 2 or 4

###########################################
#           Load Model and Engine
###########################################
print("Loading model...")
model, tokenizer = get_model_tokenizer(model_path, model_type='llama3_1')
model = Swift.from_pretrained(model, lora_checkpoint)

template_type = model.model_meta.template
# If you have a custom system prompt, you can add it here, otherwise, keep the default
template = get_template(template_type, tokenizer, default_system=None)

engine = PtEngine.from_model_template(model, template, max_batch_size=BATCH)

# Configure inference parameters
request_config = RequestConfig(
    max_tokens=2048,
    temperature=0.3, 
    # If deterministic responses are needed, you can lower the temperature
)

###########################################
#       Read JSON File
###########################################
print(f"Reading data from: {input_file}")
with open(input_file, "r", encoding="utf-8") as f:
    # Note: Assuming the input is a standard JSON list format [{}, {}]
    records = json.load(f)

print(f"Loaded {len(records)} records\n")

###########################################
#             Start Inference (Batch Processing)
###########################################

# List to store the final results
final_results = []

print("====== Starting Inference ======")

# Use tqdm to show progress
for idx in tqdm(range(0, len(records), BATCH), desc="Model Inference"):
    # 1. Get current batch of data
    batch_items = records[idx: idx + BATCH]
    
    # 2. Construct inference requests
    infer_requests = []
    valid_batch_items = []  # To correspond requests with original data
    
    for item in batch_items:
        # Get input text
        content = item.get("narr_accp", "")
        
        if not content:
            continue  # Skip empty content

        content = content + "\n\n Please analyze the causes that led to this accident."
            
        # Construct messages format
        # Swift/Qwen typically requires [{'role': 'user', 'content': ...}] format
        messages = [{"role": "user", "content": content}]
        
        infer_requests.append(InferRequest(messages=messages))
        valid_batch_items.append(item)

    if not infer_requests:
        continue

    # 3. Perform batch inference
    responses = engine.infer(infer_requests, request_config)

    # 4. Process and save results
    for original_item, resp in zip(valid_batch_items, responses):
        generated_answer = resp.choices[0].message.content
        
        # Construct output object, including the required fields
        result_obj = {
            "ev_id": original_item.get("ev_id"),
            "Aircraft_Key": original_item.get("Aircraft_Key"),
            "narr_accp": original_item.get("narr_accp"),
            "model_output": generated_answer  # Model generated answer
        }
        
        final_results.append(result_obj)

###########################################
#           Save Results as JSON
###########################################
print(f"\nSaving results to: {output_file}")

with open(output_file, "w", encoding="utf-8") as fout:
    # ensure_ascii=False ensures Chinese characters display correctly, indent=4 ensures a neat format
    json.dump(final_results, fout, ensure_ascii=False, indent=4)

print("\n==== Task Complete ====")
