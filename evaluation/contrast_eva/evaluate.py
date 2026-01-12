"""
Batch Evaluate Chain-of-Thought Scoring Script (Automatically Match Original Data + Output 0-1 Score System)
"""

import json
import asyncio
import aiofiles
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

# =============================
# Initialize LLM
# =============================

llm = ChatOpenAI(
    model="",
    base_url="",
    api_key="",
    temperature=0.3,
    timeout=120,
)



FAITHFULNESS_PROMPT = """
You are an aviation accident investigation expert, and you are now to assess whether a chain of thought is faithful to the accident narrative.

**Accident Narrative**
{narrative}

**Chain of Thought**
{cot}

Please determine whether the chain of thought is strictly based on the narrative, without inventing or adding specific facts that are not present in the narrative.

**Scoring Criteria:**
1 point: A large amount of content does not match the narrative; clear fabrication.
2 points: Considerable fabrication or contradictions.
3 points: Basically based on the narrative, but contains minor expansions or unreasonable inferences.
4 points: Mostly faithful to the narrative, with only slight reasonable inferences.
5 points: Completely faithful to the narrative, with no fabrication whatsoever.

Please output only a number from 1 to 5. Do not output explanations or any other content, and do not repeat the chain of thought.


"""


LOGICALITY_PROMPT = """

You are an aviation accident causal-analysis expert, and you need to evaluate the causal logic of a chain of thought.

**Accident Narrative**
{narrative}

**Chain of Thought**
{cot}

Please determine whether the chain of thought follows a reasonable causal sequence:

* Flight phase → Abnormal event → Pilot actions → Environmental exclusion → Inspection results → Accident chain

**Scoring Criteria:**
1 point: Chaotic reasoning with no causal relationships.
2 points: Multiple leaps in logic; weak causal connections.
3 points: Partial causal chain, but incomplete or not rigorous.
4 points: Logical and complete, with only minor leaps.
5 points: Highly rigorous, coherent, and professional causal chain.

Please output only a number from 1 to 5. Do not output explanations or any other content, and do not repeat the chain of thought.

"""

SUPPORT_PROMPT="""
You are an aviation accident assessment expert, and you must determine whether a chain of thought can reasonably support the official final cause.

**Accident Narrative**
{narrative}

**Chain of Thought**
{cot}

**Official Conclusion**
{cause}

Please evaluate: Based solely on this chain of thought, would you be convinced that the official cause is correct?

**Scoring Criteria:**
1 point: Does not support the conclusion at all, even contradicts it.
2 points: Insufficient support.
3 points: Some steps support the conclusion, but the chain is not complete.
4 points: Basically supports the conclusion, with minor gaps.
5 points: Fully and sufficiently supports the official cause.

Please output only a number from 1 to 5. Do not output explanations or any other content, and do not repeat the chain of thought.

"""

COMPLETENESS_PROMPT= """
You are an aviation accident reasoning-chain reviewer and need to evaluate whether the chain of thought covers the key elements of the narrative.

**Accident Narrative**
{narrative}

**Chain of Thought**
{cot}

Please check whether it includes:

* Flight phase (Phase)
* Abnormal event (Anomaly)
* Pilot action (Pilot action)
* Environmental factors (Weather)
* Mechanical/system exclusion (Mechanical)
* Causal chain description (Causal chain)

**Scoring Rules:**
1 point: Extremely incomplete, only repeats the narrative.
2 points: Missing several important elements.
3 points: Covers some but not all elements.
4 points: Covers most key content.
5 points: Fully covers all key elements.

Please output only a number from 1 to 5. Do not output explanations or any other content, and do not repeat the chain of thought.

"""


NTSB_STYLE_PROMPT= """
You are an investigator familiar with NTSB writing style. Please determine whether the chain of thought conforms to NTSB standards:

**NTSB Style Characteristics**

* Objective and fact-based
* No emotional wording
* No subjective speculation (e.g., “I think,” “it seems”)
* Uses formal terminology (e.g., “the pilot reported...”)
* Clear structure
* No invention of background information not provided

**Chain of Thought**
{cot}

**Scoring Rules:**
1 point: Not like NTSB at all; contains obvious subjectivity and emotional content.
2 points: Partially similar, but with clear inconsistencies.
3 points: Generally consistent, but with minor stylistic deviations.
4 points: Largely consistent, with only slight differences.
5 points: Fully consistent with NTSB style.

Please output only a number from 1 to 5. Do not output explanations or other information.
"""


CAUSAL_ACCURACY_PROMPT = """
You are an aviation safety expert. Evaluate whether the Generated Answer correctly identifies the true causal factors supported by the accident narrative.

**Accident Narrative**
{narrative}

**Generated Answer**
{answer}

**Official Conclusion**
{cause}

Your task is to determine whether the Generated Answer accurately reflects the core causal factors described in the narrative. 
Do NOT compare wording. Focus strictly on whether the causal explanation matches the facts presented in the narrative.

**Scoring Criteria (1–5):**
1 point: Identifies a cause contradicted by the narrative.
2 points: Mentions a factor from the narrative but misses the primary causal element.
3 points: Captures part of the causal relationship but misses key components.
4 points: Mostly accurate; identifies the main narrative-supported cause.
5 points: Fully accurate; clearly reflects the primary cause supported by the narrative.

Output only a number from 1 to 5.
"""

CAUSAL_COMPLETENESS_PROMPT_1 = """
You are an aviation safety expert. Evaluate whether the Generated Answer includes all key causal elements described in the accident narrative.

**Accident Narrative**
{narrative}

**Generated Answer**
{answer}

**Official Conclusion**
{cause}

Do not penalize for concise wording, but deduct points if important causal components are missing.

**Scoring Criteria (1–5):**
1 point: Only unrelated or incorrect causes; misses all key elements.
2 points: Mentions only a minor or partial factor; misses the primary cause.
3 points: Identifies the primary cause but omits major contributing factors.
4 points: Covers the main cause and most contributing factors.
5 points: Fully complete; addresses all key causal components in the narrative.

Output only a number from 1 to 5.
"""

CAUSAL_COMPLETENESS_PROMPT = """
You are an aviation safety expert. Evaluate whether the Generated Answer includes all
**essential causal elements** necessary to convey the core cause of the accident.

The Generated Answer may be brief and does NOT need to include every minor contributing factor.
Do not penalize for concise wording.  
Your task is to check whether the answer covers all **major causal components that are essential for understanding the main causal mechanism** of the accident.

**Accident Narrative**
{narrative}

**Generated Answer**
{answer}

**Official Conclusion**
{cause}

**Scoring Criteria (1–5):**
1 point: Misses all essential causal elements; does not reflect the narrative.
2 points: Mentions only a small part of the essential cause; incomplete.
3 points: Captures the main cause but omits one or more important essential elements.
4 points: Covers the primary cause and almost all essential elements, with minor omissions.
5 points: Fully complete for a brief summary; includes all major essential causal elements needed to understand the core causal chain.

Output only a number from 1 to 5.
"""





CAUSAL_PRECISION_PROMPT = """
You are an aviation safety expert. Evaluate whether the Generated Answer avoids introducing any causal claims not supported by the narrative.

**Accident Narrative**
{narrative}

**Generated Answer**
{answer}

Determine whether the Generated Answer strictly adheres to the narrative facts and does not invent unsupported causes (e.g., mechanical failures, weather issues, pilot actions, or procedural factors not mentioned).

**Scoring Criteria (1–5):**
1 point: Contains major fabricated or contradictory causes.
2 points: Includes multiple unsupported assumptions or invented details.
3 points: Mostly grounded but includes one or two minor unsupported elements.
4 points: Very precise; no major unsupported claims.
5 points: Perfect precision; all causal statements are directly supported by the narrative with zero fabrication.

Output only a number from 1 to 5.
"""


CAUSE_ALIGNMENT_PROMPT = """
You are an aviation safety expert. Evaluate whether the Generated Answer is consistent with the Official Probable Cause while still being supported by the facts presented in the accident narrative.

The Generated Answer does NOT need to match the wording of the official cause. 
It may include additional detail or context from the narrative, as long as it does not contradict the official cause. 
Your task is to judge whether the Generated Answer is aligned with the intent and meaning of the official cause and compatible with the narrative facts.

**Accident Narrative**
{narrative}

**Official Probable Cause**
{cause}

**Generated Answer**
{answer}

**Scoring Criteria (1–5):**
1 point: Contradicts the official cause or narrative facts.
2 points: Mentions a related factor but conflicts with the official cause or misses its key intent.
3 points: Partially aligned; captures part of the official cause but misses important meaning.
4 points: Mostly aligned; consistent with the official cause and narrative, with minor differences.
5 points: Fully aligned; meaningfully consistent with the official cause and fully compatible with the narrative.

Output only a number from 1 to 5.
"""


# =============================
# 1–5 Convert to 0–1
# =============================
def normalize(score):
    return round((score - 1) / 4, 4)

# =============================
# Call Model
# =============================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
async def ask_score(prompt):
    resp = await llm.ainvoke(prompt)

    if hasattr(resp, "content"):
        txt = resp.content.strip()
    else:
        raise TypeError("Model format error")

    if txt not in ["1", "2", "3", "4", "5"]:
        raise ValueError(f"Invalid score from model: {txt}")

    return normalize(int(txt))


# =============================
# Calculate Scores for a Single Record
# =============================
async def evaluate_single(narrative, cot, cause, answer):
    # Initialize results dictionary, default all to None
    results = {
        "faithfulness": None,
        "logicality": None,
        "support": None,
        "completeness": None,
        "ntsb_style": None,
        "causal_accuracy": None,
        "causal_completeness": None,
        "causal_precision": None,
        "cause_alignment": None,
        "error": None
    }

    # Prepare Prompt dictionary
    prompts = {}

    # A. Answer-related metrics (always calculate if answer exists)
    prompts.update({
        "causal_accuracy": CAUSAL_ACCURACY_PROMPT.format(narrative=narrative, answer=answer),
        "causal_completeness": CAUSAL_COMPLETENESS_PROMPT.format(narrative=narrative, answer=answer),
        "causal_precision": CAUSAL_PRECISION_PROMPT.format(narrative=narrative, answer=answer),
        "cause_alignment": CAUSE_ALIGNMENT_PROMPT.format(narrative=narrative, cause=cause, answer=answer),
    })

    # B. CoT-related metrics (only if cot is not empty)
    if cot and cot.strip():
        prompts.update({
            "faithfulness": FAITHFULNESS_PROMPT.format(narrative=narrative, cot=cot),
            "logicality":   LOGICALITY_PROMPT.format(narrative=narrative, cot=cot),
            "support":      SUPPORT_PROMPT.format(narrative=narrative, cot=cot, cause=cause),
            "completeness": COMPLETENESS_PROMPT.format(narrative=narrative, cot=cot),
            "ntsb_style":   NTSB_STYLE_PROMPT.format(cot=cot),
        })
    else:
        # Optional: print a log message for debugging if Cot is empty
        print(" CoT is empty, skipping CoT-related metrics")
        pass

    # Loop through the prompts and call the model
    for key, p in prompts.items():
        try:
            results[key] = await ask_score(p)
        except Exception as e:
            results[key] = None
            # Log the error without overwriting previous ones
            current_error = results.get("error")
            results["error"] = f"{current_error}; {key}:{str(e)}" if current_error else f"{key}:{str(e)}"

    print(results)
    return results


# =============================
# Main Process (Merge + Score)
# =============================
async def main():
    file_name = "Qwen3-8B"
    cot_path = f"./evaluation/contrast_eva/process_results/{file_name}.json"  # Contains ev_id, Aircraft_Key, answer, chain_of_thought
    raw_path = "./evaluation/contrast_eva/contrast_sample.json"             # Contains ev_id, Aircraft_Key, narr_accp, narr_cause
    
    output_path = f"./evaluation/contrast_eva/eva_results/{file_name}_scores.json"
    fail_path   = f"./evaluation/contrast_eva/eva_results/{file_name}_fail.json"

    print("Loading files...")

    async with aiofiles.open(cot_path, "r", encoding="utf-8") as f:
        cot_data = json.loads(await f.read())

    async with aiofiles.open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(await f.read())

    # Create a dictionary with (ev_id, Aircraft_Key) as the composite key
    raw_dict = {}
    for item in raw_data:
        key = (str(item["ev_id"]), str(item["Aircraft_Key"]))
        raw_dict[key] = item

    print(f"COT entries: {len(cot_data)}, Raw data entries: {len(raw_data)}")
    print("Starting matching by (ev_id + Aircraft_Key) and scoring...")

    results = []
    failures = []

    semaphore = asyncio.Semaphore(20)  # Limit concurrency to avoid API rate limits

    async def process(cot_item):
        ev_id = str(cot_item.get("ev_id"))
        ac_key = str(cot_item.get("Aircraft_Key"))
        
        cot_text = cot_item.get("chain_of_thought", "")
        answer_text = cot_item.get("answer", "")
        if not answer_text:
            answer_text = cot_item.get("model_output", "")

        raw = raw_dict.get((ev_id, ac_key))

        if raw is None:
            failures.append({
                "ev_id": ev_id, 
                "Aircraft_Key": ac_key, 
                "error": "No matching (ev_id, Aircraft_Key) found in raw data"
            })
            return

        narrative = (raw.get("narr_accp", "") + "\n" + raw.get("narr_accf", "")).strip()
        cause = raw.get("narr_cause", "")

        async with semaphore:
            try:
                scores = await evaluate_single(narrative, cot_text, cause, answer_text)
                print(f"Scored: {ev_id} | {ac_key}")

                return {
                    "ev_id": ev_id,
                    "Aircraft_Key": ac_key,
                    "scores": scores
                }

            except Exception as e:
                print(f"Error: {ev_id} - {e}")
                failures.append({"ev_id": ev_id, "Aircraft_Key": ac_key, "error": str(e)})
                return

    tasks = [process(item) for item in cot_data]
    all_res = await asyncio.gather(*tasks)
    
    results = [r for r in all_res if r is not None]

    print("Saving results...")

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=4, ensure_ascii=False))

    async with aiofiles.open(fail_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(failures, indent=4, ensure_ascii=False))

    print("All tasks complete!")

if __name__ == "__main__":
    asyncio.run(main())