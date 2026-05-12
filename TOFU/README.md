# Leak@K - TOFU Evaluation Framework

## Quick Start

### 1. Configure Models

Edit `models_config.json`:

```json
{
    "YourModel": {
        "model_path": "huggingface/model-path",
        "eval_dir": "saves/eval/YourModel"
    }
}
```

### 2. Run Evaluation

```bash
sbatch whole_eval.sh
```

Or directly:

```bash
python whole_eval.py --config models_config.json --gpu_id 0
```

**Optional flags:**
- `--skip_gen` - Skip generation phase
- `--skip_eval` - Skip semantic evaluation
- `--skip_agg` - Skip aggregation

---

## LLM Judge

Requires OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run:

```bash
# Single file
python LLM_judge.py --input saves/eval/Model/forget/temperature=1.0top_p=1.0/generations_n200.json

# Multiple directories
python LLM_judge.py --dir saves/eval/Model1 saves/eval/Model2

# Use different model
python LLM_judge.py --dir saves/eval/Model --model gpt-4o
```

---

## Directory Structure

```
TOFU/
├── src/                    # Core evaluation code
├── whole_eval.py           # Main pipeline
├── whole_eval.sh           # SLURM script
├── LLM_judge.py            # LLM-as-judge
├── models_config.json      # Model config
└── saves/eval/             # Output (generated)
    └── {Model}/
        └── forget/
            └── temperature=1.0top_p=1.0/
                ├── generations_n200.json
                ├── generations_n200_evaluated.json
                └── generations_n200_llm_judge.json
```

Output: `model_evaluation_metrics.csv`