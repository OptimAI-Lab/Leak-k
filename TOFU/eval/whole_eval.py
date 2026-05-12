import os
import sys
import json
import time
import subprocess
import argparse
import shutil
import math
import csv
import numpy as np
from tqdm import tqdm

# ==========================================
#        CONFIGURATION
# ==========================================
OUTPUT_CSV_FILE = "model_evaluation_metrics.csv"
K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
TOTAL_ENTRIES_FOR_LEAK = 400


# ==========================================
#        PHASE 1: GENERATION
# ==========================================
def delete_model_from_disk(model_path):
    """Deletes Hugging Face models from cache to save space."""
    if os.path.exists(model_path):
        print(f"🛡️ [DISK CLEANUP] SKIPPING local directory: {model_path} (Safe)")
        return

    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    formatted_name = "models--" + model_path.replace("/", "--")
    target_path = os.path.join(hf_cache_dir, formatted_name)

    if os.path.exists(target_path):
        print(f"🗑️ [DISK CLEANUP] Deleting Hugging Face cache for: {model_path}")
        shutil.rmtree(target_path, ignore_errors=True)
    else:
        print(f"ℹ️ [DISK CLEANUP] Model cache not found: {target_path}")


def run_generation_phase(models, gpu_id=0):
    print(f"\n=======================================================")
    print(f"🚀 PHASE 1: GENERATION")
    print(f"=======================================================\n")

    for method_label, config in models.items():
        model_path = config["model_path"]
        eval_dir = config["eval_dir"]
        log_file = f"eval_{method_label}.log"

        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} python src/eval.py "
            f"--config-name=eval.yaml "
            f"experiment=eval/tofu/default "
            f"model=Llama-3.2-1B-Instruct "
            f"model.model_args.pretrained_model_name_or_path={model_path} "
            f"retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json "
            f"task_name={method_label} "
            f"> {log_file} 2>&1"
        )

        print(f"📡 Running: {method_label}...")
        start_time = time.time()

        try:
            subprocess.run(command, shell=True, check=True)
            duration = (time.time() - start_time) / 60
            print(f"✅ [DONE] {method_label} generated output successfully. ({duration:.2f} mins)")
            delete_model_from_disk(model_path)

        except subprocess.CalledProcessError:
            print(f"❌ [FAIL] {method_label} failed. Check {log_file} for details.")

        time.sleep(3)


# ==========================================
#        PHASE 2: SEMANTIC EVALUATION (CS & ES)
# ==========================================
def eval_cosine_similarity_with_model(gen_outputs, ground_truths, model):
    import torch
    scores = []
    with torch.no_grad():
        for gen, gt in zip(gen_outputs, ground_truths):
            gen_str, gt_str = str(gen) if gen else "", str(gt) if gt else ""
            if not gen_str or not gt_str:
                scores.append(0.0)
                continue
            gen_emb = model.encode(gen_str, show_progress_bar=False, convert_to_tensor=True)
            gt_emb = model.encode(gt_str, show_progress_bar=False, convert_to_tensor=True)
            cosine_sim = torch.nn.functional.cosine_similarity(gen_emb, gt_emb, dim=0).item()
            scores.append(float(max(0, cosine_sim)))
    return {'cosine_similarity': scores}


def get_entailment_results_with_pipeline(pipe, gen_outputs, ground_truths, eval_task, rouge_scores, bs=30, tofu=True):
    results = []
    if len(gen_outputs) != len(ground_truths):
        ground_truths = [ground_truths[0]] * len(gen_outputs)

    for i in range(0, len(gen_outputs), bs):
        targets_batch = ground_truths[i:i + bs]
        outputs_batch = gen_outputs[i:i + bs]
        rouge_scores_batch = rouge_scores[i:i + bs]

        data_list = []
        for j in range(len(targets_batch)):
            out_txt, tgt_txt = str(outputs_batch[j] or ""), str(targets_batch[j] or "")
            if not out_txt or not tgt_txt:
                results.append({'label': 'none', 'score': 0.0})
                continue

            if not tofu or 'forget' in eval_task:
                data_list.append({'text': out_txt, 'text_pair': tgt_txt})
            else:
                data_list.append({'text': tgt_txt, 'text_pair': out_txt})

        if data_list:
            batch_results = pipe(data_list)
            filtered = []
            valid_idx = 0
            for j in range(len(targets_batch)):
                if not (str(outputs_batch[j]) and str(targets_batch[j])):
                    continue
                if rouge_scores_batch[valid_idx] < 0.1:
                    filtered.append({'label': 'none', 'score': 0.0})
                else:
                    filtered.append(batch_results[valid_idx])
                valid_idx += 1
            results.extend(filtered)
        else:
            results.extend([{'label': 'none', 'score': 0.0} for _ in outputs_batch if not (str(_) and str(_))])

    return {'entailment_labels': [r['label'] for r in results]}


def process_query_responses(query_data, nli_pipeline, st_model, eval_task="forget", tofu=True):
    ground_truth = query_data.get("ground_truth", "")
    responses = query_data.get("responses", [])

    gen_outputs = [res.get("response", "") for res in responses]
    rougeL_recalls = [res.get("rougeL_recall", 0.0) for res in responses]

    cs_scores = eval_cosine_similarity_with_model(gen_outputs, [ground_truth] * len(gen_outputs), st_model)['cosine_similarity']
    es_labels = get_entailment_results_with_pipeline(nli_pipeline, gen_outputs, [ground_truth] * len(gen_outputs), eval_task, rougeL_recalls, bs=30, tofu=tofu)['entailment_labels']
    es_scores = [1.0 if l == 'entailment' else 0.0 for l in es_labels]

    for i, res in enumerate(responses):
        res["CS"] = cs_scores[i]
        res["ES"] = es_scores[i]

    query_data["Best CS"] = max(cs_scores) if cs_scores else None
    query_data["Avg CS"] = np.mean(cs_scores).item() if cs_scores else None
    query_data["Best ES"] = max(es_scores) if es_scores else None
    query_data["Avg ES"] = np.mean(es_scores).item() if es_scores else None
    query_data["responses"] = responses
    return query_data


def run_semantic_eval_phase(models, gpu_id=0):
    print(f"\n=======================================================")
    print(f"🧠 PHASE 2: SEMANTIC EVALUATION (CS & ES)")
    print(f"=======================================================\n")

    import torch
    from transformers import pipeline
    from sentence_transformers import SentenceTransformer

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"⏳ Loading Eval Models onto GPU {gpu_id}...")
    st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)
    nli_pipe = pipeline("text-classification", model="sileod/deberta-v3-base-tasksource-nli", device=device)

    for method_label, config in models.items():
        eval_dir = config["eval_dir"]
        if not os.path.exists(eval_dir):
            print(f"⚠️ Directory not found, skipping: {eval_dir}")
            continue

        for dirpath, _, filenames in os.walk(eval_dir):
            if "generations_n200.json" in filenames:
                input_file = os.path.join(dirpath, "generations_n200.json")
                output_file = os.path.join(dirpath, "generations_n200_evaluated.json")

                if os.path.exists(output_file):
                    print(f"⏭️ Skipping (Already evaluated): {input_file}")
                    continue

                print(f"\n⚙️ Evaluating: {input_file}")

                processed_results = []
                with open(input_file, 'r', encoding='utf-8') as infile:
                    for line in tqdm(infile, leave=False, desc="Evaluating"):
                        try:
                            data = json.loads(line)
                            processed = process_query_responses(data, nli_pipe, st_model, "forget", True)
                            processed_results.append(processed)
                        except Exception as e:
                            print(f"Error processing line: {e}")
                            continue

                if processed_results:
                    all_avg_cs = [res.get("Avg CS", 0) for res in processed_results if res.get("Avg CS") is not None]
                    all_avg_es = [res.get("Avg ES", 0) for res in processed_results if res.get("Avg ES") is not None]

                    overall_avg_cs = np.mean(all_avg_cs) if all_avg_cs else 0.0
                    overall_avg_es = np.mean(all_avg_es) if all_avg_es else 0.0

                    with open(output_file, 'w', encoding='utf-8') as outfile:
                        for data in processed_results:
                            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

                    print(f"✅ Saved results to: {output_file}")
                    print(f"📊 SUMMARY -> Overall Avg CS: {overall_avg_cs:.4f} | Overall Avg ES: {overall_avg_es:.4f}")


# ==========================================
#        PHASE 3: LEAK@K AGGREGATION
# ==========================================
def calculate_pass_at_k(n, c, k):
    if k > n: return 0.0
    total_combinations = math.comb(n, k)
    fail_combinations = 0 if (n - c) < k else math.comb(n - c, k)
    return 1.0 - (fail_combinations / total_combinations)


def find_file_in_dir(directory, filename):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def process_single_model_aggregation(model_name, model_dir):
    sum_pass_at_k = {k: 0.0 for k in K_VALUES}

    input_filename = find_file_in_dir(model_dir, "generations_n200_evaluated.json")
    if not input_filename:
        print(f"⚠️ Warning: Evaluated file not found for {model_name} in {model_dir}")
        return None

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            valid_entries = 0
            for i, line in enumerate(f):
                if valid_entries >= TOTAL_ENTRIES_FOR_LEAK: break
                data = json.loads(line)
                if 'responses' not in data: continue

                responses = data['responses']
                n = len(responses)
                c = sum(1 for response in responses if response.get('ES') == 1)

                for k in K_VALUES:
                    sum_pass_at_k[k] += calculate_pass_at_k(n, c, k)
                valid_entries += 1

        if valid_entries == 0: return None

        results = {'Model': model_name}
        for k, score in sum_pass_at_k.items():
            results[f'Leak@{k}'] = score / valid_entries
        return results

    except Exception as e:
        print(f"❌ Error reading {input_filename}: {e}")
        return None


def run_aggregation_phase(models):
    print(f"\n=======================================================")
    print(f"📊 PHASE 3: AGGREGATING LEAK@K")
    print(f"=======================================================\n")

    all_results = []

    for method_label, config in models.items():
        print(f"⚙️ Processing Model: {method_label}...")
        model_scores = process_single_model_aggregation(method_label, config["eval_dir"])

        if model_scores:
            all_results.append(model_scores)
            print(f"✅ {method_label} Results | " +
                  " | ".join([f"L@{k}: {model_scores[f'Leak@{k}']:.4f}" for k in [1, 128]]))
        else:
            print(f"❌ Failed to process {method_label} (Missing data).")
            missing_row = {'Model': method_label}
            missing_row.update({f'Leak@{k}': "Missing" for k in K_VALUES})
            all_results.append(missing_row)

        print("-" * 60)

    print(f"\n💾 Writing results to CSV...")
    fieldnames = ['Model'] + [f'Leak@{k}' for k in K_VALUES]
    try:
        with open(OUTPUT_CSV_FILE, mode='w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)
        print(f"🎉 Done! Results successfully saved to:\n📂 {os.path.abspath(OUTPUT_CSV_FILE)}")
    except Exception as e:
        print(f"❌ Error writing to CSV file: {e}")


# ==========================================
#                MAIN EXECUTION
# ==========================================
def load_models_from_config(config_path):
    """Load models from a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leak@K Evaluation Pipeline")
    parser.add_argument("--config", type=str, help="Path to JSON config file with models")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--skip_gen", action="store_true", help="Skip Generation Phase")
    parser.add_argument("--skip_eval", action="store_true", help="Skip Semantic Eval Phase")
    parser.add_argument("--skip_agg", action="store_true", help="Skip Aggregation Phase")
    args = parser.parse_args()

    if args.config:
        MODELS = load_models_from_config(args.config)
    else:
        print("❌ Error: Please provide --config argument with a JSON config file.")
        print("Example: python whole_eval.py --config models_config.json --gpu_id 0")
        sys.exit(1)

    print(f"📋 Loaded {len(MODELS)} models from config:")
    for name, cfg in MODELS.items():
        print(f"   - {name}: {cfg['model_path']}")

    if not args.skip_gen: run_generation_phase(MODELS, gpu_id=args.gpu_id)
    if not args.skip_eval: run_semantic_eval_phase(MODELS, gpu_id=args.gpu_id)
    if not args.skip_agg: run_aggregation_phase(MODELS)

    print("\n🏁 Pipeline execution complete!")