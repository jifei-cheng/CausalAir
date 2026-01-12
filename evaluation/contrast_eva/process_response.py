"""
Separate the chain of thought and the answer from the model's response
"""
import json
import re
import os

# ================= Configuration Area =================

input_file = ""
# Output file path
output_file = ""
# ===========================================

def process_cot_data(input_path, output_path):
    # 1. Check if the file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found")
        return

    print(f"Reading file: {input_path} ...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Invalid JSON file format, please check the file content.")
        return

    processed_data = []
    
    # 2. Compile regular expression
    # r'<think>(.*?)</think>\s*(.*)'
    # (.*?) : Non-greedy match, extract content inside <think>
    # \s* : Ignore any whitespace characters (newlines, spaces, etc.) after the closing tag
    # (.*)  : Extract the remaining content as the final answer
    # re.DOTALL : Allow the dot to match newline characters, which is important for multi-line text
    pattern = re.compile(r'<think>(.*?)</think>\s*(.*)', re.DOTALL)

    print(f"Processing {len(data)} records...")

    for item in data:
        raw_output = item.get("model_output", "")
        
        # Initialize new fields
        chain_of_thought = ""
        final_answer = raw_output  # Default, if no tags are found, the entire content is treated as the answer

        if raw_output:
            match = pattern.search(raw_output)
            if match:
                # Extract content inside <think> tag and trim whitespace
                chain_of_thought = match.group(1).strip()
                # Extract the content after the <think> tag
                final_answer = match.group(2).strip()
            else:
                # If no <think> tag is found in the data, print a simple message
                print(f"Note: ID {item.get('ev_id')} did not contain a <think> tag, skipping separation.")

        # Construct the new object
        new_item = {
            "ev_id": item.get("ev_id"),
            "Aircraft_Key": item.get("Aircraft_Key"),
            "narr_accp": item.get("narr_accp"),
            # Newly added chain of thought field
            "chain_of_thought": chain_of_thought,
            # Extracted clean answer (here named as 'answer', you can rename it if needed)
            "answer": final_answer
        }
        
        processed_data.append(new_item)

    # 3. Save the result
    print(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print("==== Processing complete ====")

# Execute function
if __name__ == "__main__":
    process_cot_data(input_file, output_file)
