import os
import sys
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from typing import List
import re

# =====================================================
# ✅ Global OpenAI client (reuse for all calls)
# =====================================================
client = OpenAI()
MODEL_NAME = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
BATCH_SIZE = 20
MAX_WORKERS = 10

# =====================================================
# ⚙️ Batch LLM Judge
# =====================================================
def LLM_judge_batch(gen_outputs: List[str], ground_truths: List[str], prompt: str) -> List[int]:
    """
    Batched LLM-as-judge. Model must output a bracketed list like: [1,0,1,...].
    YES -> 1, NO -> 0.
    Returns list[int] with size == len(gen_outputs).
    """

    system_prompt = """You are an evaluation model.
For each question, judge whether the candidate answer correctly conveys the essential information in the reference answer.
Output ONLY a single Python-style list of 0/1 integers, where:
- 1 = "YES", the candidate matches the reference.
- 0 = "NO", it does not match.
Example output: [1,0,1,1,0]
No explanations, no extra text.
"""

    batched_text = "\n\n".join([
        f"Item {i+1}:\nQuestion: {prompt}\nReference: {gt}\nCandidate: {gen}"
        for i, (gen, gt) in enumerate(zip(gen_outputs, ground_truths))
    ])

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": batched_text},
            ],
            temperature=0,
            max_output_tokens=100,
            prompt_cache_key="LLM_SCORE_LIST_V2",
        )

        raw_text = getattr(response, "output_text", "").strip()
        if not raw_text:
            raw_text = "".join(
                (c.text if hasattr(c, "text") else str(c))
                for o in getattr(response, "output", [])
                for c in getattr(o, "content", [])
            ).strip()

        m = re.search(r"\[([^\]]+)\]", raw_text)
        numbers = []
        if m:
            numbers = [int(x) for x in re.findall(r"[01]", m.group(1))]

        batch_n = len(gen_outputs)
        if not numbers:
            return [0] * batch_n

        if len(numbers) < batch_n:
            numbers.extend([0] * (batch_n - len(numbers)))
        return numbers[:batch_n]

    except Exception as e:
        print("[LLM_judge_batch] ERROR:", e)
        return [0] * len(gen_outputs)


# =====================================================
# ⚙️ Wrapper with parallel batches
# =====================================================
def LLM_judge(gen_outputs: List[str], ground_truths: List[str], prompt: str) -> List[int]:
    """
    Parallel batched LLM evaluation for large numbers of generations.
    """
    batches = [
        (gen_outputs[i:i+BATCH_SIZE], ground_truths[i:i+BATCH_SIZE])
        for i in range(0, len(gen_outputs), BATCH_SIZE)
    ]

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(LLM_judge_batch, g, gt, prompt): (g, gt) for g, gt in batches}
        for f in tqdm(as_completed(futures), total=len(futures), desc="LLM Judge"):
            results.extend(f.result())

    return results


# =====================================================
# ⚙️ Query Processing
# =====================================================
def process_query_responses(query_data):
    """
    Processes multiple generated responses for a single query,
    adds LLM_judge results.
    """
    ground_truth = query_data.get("ground_truth", "")
    responses = query_data.get("responses", [])
    prompt = query_data.get("prompt", "")

    prefix = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant.user\n\n"
    if prompt.startswith(prefix):
        prompt = prompt[len(prefix):]

    gen_outputs = [res.get("response", "") for res in responses]
    ground_truths = [ground_truth] * len(gen_outputs)

    judge_results = LLM_judge(gen_outputs, ground_truths, prompt)

    for i, res in enumerate(responses):
        res["LLM_judge"] = judge_results[i] if i < len(judge_results) else 0

    query_data["responses"] = responses
    return query_data


# =====================================================
# ⚙️ File Processing
# =====================================================
def main_processing_script(input_file_path: str, output_file_path: str):
    """
    Loads, evaluates, and writes JSONL file with LLM_judge results.
    """
    processed_results = []

    with open(input_file_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    for line in tqdm(lines, desc=f"Processing {os.path.basename(input_file_path)}"):
        try:
            query_data = json.loads(line)
            processed_query_data = process_query_responses(query_data)
            processed_results.append(processed_query_data)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line: {e}")
        except Exception as e:
            print(f"Error processing query: {e}")

    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for data in processed_results:
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')


# =====================================================
# ⚙️ Directory Traversal
# =====================================================
def process_all_generation_jsons(root_dirs: list):
    """
    Traverse root dirs, find generation files, evaluate them with LLM_judge.
    """
    for root_dir in root_dirs:
        if not os.path.isdir(root_dir):
            print(f"⚠️ Warning: directory '{root_dir}' not found.")
            continue

        print(f"\n📂 Searching in: {root_dir}")

        # Process the temperature=0.8 top_p=1.0 generations file
        path = os.path.join(root_dir, "forget", "temperature=0.8top_p=1.0", "generations_n200.json")
        if os.path.isfile(path):
            print(f"▶️ Processing file: {path}")
            output_path = path.replace(".json", "_llm_judge.json")
            try:
                main_processing_script(path, output_path)
                print(f"✅ Successfully saved: {output_path}")
            except Exception as e:
                print(f"❌ Error processing {path}: {e}")
        else:
            print(f"⚠️ File not found: {path}")

        print("-" * 50)
    print("✅ All directories processed.")


# =====================================================
# 🏁 Entry Point
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Judge Evaluation")
    parser.add_argument("--input", type=str, help="Input JSONL file path")
    parser.add_argument("--output", type=str, help="Output JSONL file path (default: input_llm_judge.json)")
    parser.add_argument("--dir", type=str, nargs="+", help="Directory(es) to process")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    # Set model
    MODEL_NAME = args.model
    os.environ["OPENAI_MODEL"] = MODEL_NAME
    print(f"Using OpenAI model: {MODEL_NAME}")

    import time
    start_time = time.time()

    if args.input:
        # Single file mode
        output_path = args.output or args.input.replace(".json", "_llm_judge.json")
        print(f"Processing single file: {args.input}")
        main_processing_script(args.input, output_path)
    elif args.dir:
        # Directory mode
        print(f"Processing directories: {args.dir}")
        process_all_generation_jsons(args.dir)
    else:
        print("❌ Error: Please provide --input or --dir argument")
        print("Examples:")
        print("  python LLM_judge.py --input saves/eval/Model1/forget/temperature=0.8top_p=1.0/generations_n200.json")
        print("  python LLM_judge.py --dir saves/eval/Model1 saves/eval/Model2")
        sys.exit(1)

    end_time = time.time()
    print(f"\nDuration: {end_time - start_time:.2f} seconds")