"""
Generate Responses
Ollama Model
"""
import os
import json
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. Configuration Area: Define multiple model configurations
# ==========================================

# Assuming that you've started model services on different ports, or the same port supports different model names.
# Modify the base_url according to the actual address of the deployed service.
MODELS_CONFIG = [
    {
        "model_name": "gpt-oss:20b",
        # The base_url is the address after you deploy the model service (e.g., vLLM or Swift deploy)
        "base_url": "http://192.168.2.4:11434/v1", 
        "api_key": "EMPTY",  # Local deployment usually does not require a key, set to EMPTY
        "output_file": "./evaluation/contrast_eva/gpt-oss-20b.json"
    },
]

# Evaluation dataset
INPUT_FILE = "./evaluation/contrast_eva/contrast_sample.json"

# ==========================================
# 2. Define Prompt Template
# ==========================================


prompt_text = """You are a professional aviation accident investigator with analytical capabilities comparable to those of the NTSB (National Transportation Safety Board).

**Task:**
I will provide an *accident narrative*.
Based on the known facts in the narrative, directly provide the *cause of the accident*.

**Output requirements:**
* Output *only* the accident cause itself
* Analysis must be concise, professional, and evidence-based
* Do not add explanations or extra words
* Do not speculate
* Do not include phrases like “the cause is”

**Accident narrative:**
{content}
"""

# Use LangChain's template system
prompt_template = ChatPromptTemplate.from_messages([
    ("user", prompt_text),
])

# ==========================================
# 3. Main Execution Logic
# ==========================================

def run_evaluation():
    # 1. Read input data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found")
        return

    print(f"Reading data: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    # 2. Iterate through the list of models and execute sequentially
    for config in MODELS_CONFIG:
        current_model = config["model_name"]
        current_base_url = config["base_url"]
        output_path = config["output_file"]
        
        print(f"\n" + "="*50)
        print(f"Processing model: {current_model}")
        print(f"API address: {current_base_url}")
        print(f"="*50)

        # Initialize LangChain LLM
        # Set temperature as 0.3, as in your previous code
        llm = ChatOpenAI(
            model=current_model,
            openai_api_base=current_base_url, # Note: in LangChain the parameter name is usually openai_api_base or base_url
            openai_api_key=config["api_key"],
            temperature=0.3,
            max_retries=3  # Simple retry mechanism
        )

        # Construct chain (Chain)
        chain = prompt_template | llm

        final_results = []
        
        # Start inference loop
        # Use tqdm to show the current model's progress
        for item in tqdm(records, desc=f"Running {current_model}"):
            content = item.get("narr_accp", "")
            
            if not content:
                continue

            try:
                # Call LangChain
                response = chain.invoke({"content": content})
                generated_answer = response.content

                result_obj = {
                    "ev_id": item.get("ev_id"),
                    "Aircraft_Key": item.get("Aircraft_Key"),
                    "narr_accp": item.get("narr_accp"),
                    "model_output": generated_answer,
                    "model_name": current_model # Record which model generated the output
                }
                final_results.append(result_obj)

            except Exception as e:
                print(f"\n[Error] Error processing ID {item.get('ev_id')}: {e}")
                # In case of error, choose to either record an empty result or skip
                continue

        # 3. Save the current model's results
        print(f"Saving results to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(final_results, fout, ensure_ascii=False, indent=4)
            
        print(f"Model {current_model} task completed.")

    print("\nAll model tasks completed!")

if __name__ == "__main__":
    run_evaluation()
