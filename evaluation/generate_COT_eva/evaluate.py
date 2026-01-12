"""
Batch Evaluate Chain-of-Thought (COT) Scoring Script (Automatically Match Original Data + Output 0-1 Score System)
"""

import json
import asyncio
import aiofiles
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

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

SUPPORT_PROMPT = """
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

COMPLETENESS_PROMPT = """
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

NTSB_STYLE_PROMPT = """
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

# =============================
# Convert 1–5 to 0–1
# =============================
def normalize(score):
    return round((score - 1) / 4, 4)

# =============================
# Call the model
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
# Calculate the five scores for a single record
# =============================
async def evaluate_single(narrative, cot, cause):
    prompts = {
        "faithfulness": FAITHFULNESS_PROMPT.format(narrative=narrative, cot=cot),
        "logicality":   LOGICALITY_PROMPT.format(narrative=narrative, cot=cot),
        "support":      SUPPORT_PROMPT.format(narrative=narrative, cot=cot, cause=cause),
        "completeness": COMPLETENESS_PROMPT.format(narrative=narrative, cot=cot),
        "ntsb_style":   NTSB_STYLE_PROMPT.format(cot=cot),
    }

    results = {}
    for key, p in prompts.items():
        try:
            results[key] = await ask_score(p)
        except Exception as e:
            results[key] = None
            results["error"] = str(e)

    print(results)
    return results

# =============================
# Main process (Merge + Score)
# =============================
async def main():

    # A: COT Result File (Generated by you)
    cot_path = "./evaluation/generate_COT/results/DeepSeek-V3.2_cot.json"

    # B: Raw NTSB Data File (Contains narrative + cause)
    raw_path = "./evaluation/generate_COT_eva/sample.json"

    output_path = "./evaluation/generate_COT_eva/eva_results/DeepSeek-V3.2_scores.json"
    fail_path   = "./evaluation/generate_COT_eva/eva_results/DeepSeek-V3.2_scores_fail.json"

    print(" Loading files...")

    # -------- Load COT File --------
    async with aiofiles.open(cot_path, "r", encoding="utf-8") as f:
        cot_data = json.loads(await f.read())

    # -------- Load Raw Data File --------
    async with aiofiles.open(raw_path, "r", encoding="utf-8") as f:
        raw_data = json.loads(await f.read())

    # -------- Convert raw data to dict — to quickly find by ev_id --------
    raw_dict = {item["ev_id"]: item for item in raw_data}

    print(f"COT entries: {len(cot_data)}, Raw data entries: {len(raw_data)}")
    print(" Starting to match by ev_id and score...")

    results = []
    failures = []

    semaphore = asyncio.Semaphore(100)

    async def process(cot_item):
        ev_id = cot_item.get("ev_id")
        cot   = cot_item.get("chain_of_thought", "")

        # -------- Find the narrative and cause --------
        raw = raw_dict.get(ev_id)

        if raw is None:
            print(f"Original narrative not found: {ev_id}")
            failures.append({"ev_id": ev_id, "error": "Missing original data"})
            return

        narrative = (raw.get("narr_accp", "") + "\n" + raw.get("narr_accf", "")).strip()
        cause = raw.get("narr_cause", "")

        async with semaphore:
            try:
                scores = await evaluate_single(narrative, cot, cause)
                print(f"Scoring completed: {ev_id}")

                return {
                    "ev_id": ev_id,
                    "scores": scores
                }

            except Exception as e:
                print(f" Scoring failed: {ev_id} - {e}")
                failures.append({"ev_id": ev_id, "error": str(e)})
                return

    # -------- Process each record --------
    for item in cot_data:
        res = await process(item)
        if res:
            results.append(res)

    # -------- Save --------
    print(" Saving results...")

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=4, ensure_ascii=False))

    async with aiofiles.open(fail_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(failures, indent=4, ensure_ascii=False))

    print(" All completed!")
    print(f"Result file: {output_path}")
    print(f"Failure file: {fail_path}")


if __name__ == "__main__":
    asyncio.run(main())
