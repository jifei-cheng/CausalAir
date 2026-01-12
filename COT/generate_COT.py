"""
Generate Chain-of-Thought
"""

import json
import asyncio
import aiofiles
import traceback
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError


llm = ChatOpenAI(
    model="",
    base_url="",
    api_key="",
    temperature=0.3,
    timeout=120,
)


PROMPT_TEMPLATE_EN  = """
You are a professional aviation accident investigator, familiar with the standard analytical style used by the NTSB (National Transportation Safety Board).

[Task Description]
I will provide two sections of content:

1. **[Accident Narrative]** – a description of the accident sequence;
2. **[Official Conclusion]** – the already determined probable cause(s) and contributing factors of the accident;

Your task is:
Only generate the **“intermediate reasoning process (Chain-of-Thought)”** that step-by-step connects the **[Accident Narrative]** to the types of causes represented in the **[Official Conclusion]**.

[Writing and Reasoning Rules]

1. Output only the body of the reasoning chain. Do **not** output any titles, introductions, or concluding phrases (such as “In summary” or “In conclusion”).
2. The reasoning chain must be **step-numbered** (1, 2, 3, …), with a clear structure, and each step should express only **one** key reasoning point.
3. The reasoning content must:

   * Come directly from information explicitly mentioned in the **[Accident Narrative]**, or
   * Be a direct and reasonable inference based on that information (such as time sequence or causal relationships).
     You must **not** introduce any facts that are completely absent from the narrative.
4. The reasoning should reflect professional aviation accident analysis logic, including but not limited to:

   * Clearly identifying the **phase of flight** (e.g., taxi, takeoff, climb, cruise, descent, approach, landing, go-around, etc.);
   * Identifying key abnormal events (e.g., power loss, control anomalies, stall indications, runway excursion, etc.);
   * Analyzing the pilot’s actions and their possible effects (manipulation of controls, decision-making, crew/resource management, etc.);
   * Using evidence-based reasoning to **retain or rule out** common factors (such as weather, fuel, mechanical failure, operational error, etc.),
     and relying as much as possible on **evidence-based exclusion/retention** rather than subjective speculation;
   * Making the causal chain explicit (initial event → change in aircraft/operational state or environment → pilot response → subsequent event/loss of control → final outcome).
5. You must **not fabricate** the following information:

   * Specific meteorological conditions (e.g., exact visibility, cloud base, wind direction/speed values, etc.);
   * Specific instrument readings (e.g., precise airspeed, altitude, rpm, fuel quantity values, etc.);
   * Specific maintenance history, company policies, or modification status of the aircraft type;
   * Any dialogue, checklist items, cabin conditions, etc. that are not present in the narrative.
6. The reasoning chain should **naturally lead** toward the categories of causes stated in the **[Official Conclusion]**:

   * You may use expressions such as “this suggests that…”, “it can be inferred that…”, “this is consistent with events of the … type”, etc.,
   * But you must **not** directly repeat or provide an equivalent rephrasing of the original wording of the **[Official Conclusion]**.
7. The tone must be professional, objective, and evidence-based:

   * Avoid emotional or accusatory language;
   * Avoid absolute statements such as “it must be” or “it definitely is”;
   * Prefer expressions like “it is more likely that…”, “this is consistent with…”, “there is insufficient evidence to support the hypothesis that…”, etc.

[Output Format]
Strictly follow the format below, and do not add any extra explanations:

1. …
2. …
3. …

—— Content to be analyzed ——

[Accident Narrative]
{narrative}

[Official Conclusion]
{official_cause}

Please produce the reasoning chain strictly according to the above rules:
"""


# =============================
# Define asynchronous calls + retry logic
# =============================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_cot(record):
    prompt = PROMPT_TEMPLATE_EN.format(
        narrative=record.get("narr_accp", "") + "\n\n" + record.get("narr_accf", ""),
        official_cause=record.get("narr_cause", ""),
    )

    response = await llm.ainvoke(prompt)

    if hasattr(response, "content"):
        content = response.content
    else:
        raise TypeError(f"Unexpected response type: {type(response)}")

    return {
        "ev_id": record.get("ev_id"),
        "chain_of_thought": content.strip(),
    }

# =============================
# Main process: Save every N successes + save failed records separately
# =============================
async def main():
    input_path = "./evaluation/generate_COT_eva/sample.json"
    output_path = "./evaluation/generate_COT_eva/results/DeepSeek-V3.2_cot.json"
    fail_path = "./evaluation/generate_COT_eva/results/DeepSeek-V3.2_cot_fail.json"

    SAVE_EVERY_N = 10  # ✔ Save after every N successful records

    async with aiofiles.open(input_path, "r", encoding="utf-8") as f:
        data = json.loads(await f.read())

    print(f"Read {len(data)} accident records, starting to generate the Chain-of-Thought...")

    semaphore = asyncio.Semaphore(50)

    results = []         # All results (successes + failures)
    failed_records = []  # All failed records (saved separately)

    success_count = 0

    async def process_record(record):
        nonlocal success_count

        async with semaphore:
            ev_id = record.get("ev_id")

            try:
                result = await generate_cot(record)
                print(f"Successfully generated: {ev_id}")
                success_count += 1
                return result

            except Exception as e:
                # Parsing RetryError
                if isinstance(e, RetryError):
                    original = e.last_attempt.exception()
                    error_msg = f"{type(original).__name__}: {original}"
                else:
                    error_msg = f"{type(e).__name__}: {e}"

                print(f"{ev_id} generation failed: {error_msg}")

                fail_obj = {
                    "ev_id": ev_id,
                    "error": error_msg
                }
                failed_records.append(fail_obj)

                return fail_obj

    # =============================
    # Process each record, save every N successes
    # =============================
    for r in data:
        result = await process_record(r)
        results.append(result)

        # ---- Save automatically after every N successes ----
        if success_count > 0 and success_count % SAVE_EVERY_N == 0:
            print(f"Reached {SAVE_EVERY_N} successful records, automatically saving...")
            async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(results, indent=4, ensure_ascii=False))
            async with aiofiles.open(fail_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(failed_records, indent=4, ensure_ascii=False))

    # =============================
    # Final save (complete results + failed records)
    # =============================
    print("All processing complete, saving final results...")

    async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(results, indent=4, ensure_ascii=False))

    async with aiofiles.open(fail_path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(failed_records, indent=4, ensure_ascii=False))

    print(f"All results saved:\n- Success + Failure: {output_path}\n- Failure List: {fail_path}")


if __name__ == "__main__":
    asyncio.run(main())
