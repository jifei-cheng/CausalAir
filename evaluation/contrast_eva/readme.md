
# Evaluation & Inference Guide

This repository contains scripts for generating, processing, and evaluating accident analysis reports using Large Language Models (LLMs).

## ## Directory Overview

| Script | Description |
| --- | --- |
| **`generate_response_lora.py`** | Generates accident analysis using a fine-tuned LoRA model. |
| **`generate_response_ollama.py`** | Generates accident analysis using models hosted via Ollama. |
| **`process_response.py`** | Post-processing script to split raw model output into "Chain of Thought" (reasoning) and "Analysis Results." |
| **`evaluate.py`** | Performs automated evaluation of the generated results against benchmarks. |
| **`compute_scores.py`** | Calculates the final average scores across all evaluated files. |

---

## ## Workflow Instructions

To perform a full evaluation cycle, follow these steps in order:

### 1. Generate Analysis

Run either the LoRA-based generator or the Ollama-based generator depending on your model setup:

```bash
python generate_response_lora.py
# OR
python generate_response_ollama.py
```

### 2. Process Output

Extract the reasoning process and the final answer from the raw model responses:

```bash
python process_response.py
```

### 3. Evaluate Results

Run the evaluation script to score the processed responses:

```bash
python evaluate.py
```

### 4. Calculate Final Scores

Summarize the evaluation metrics to get the average performance scores:

```bash
python compute_scores.py
```

---

> **Note:** Ensure that your environment variables and model paths are correctly configured in the respective `.py` files before execution.
