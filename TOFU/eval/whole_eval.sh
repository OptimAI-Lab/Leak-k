#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=64gb
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.out
#SBATCH --job-name=Leak_at_k_eval
#SBATCH --requeue
#SBATCH --gres=gpu:a100:1

# ==========================================
#        CONFIGURATION
# ==========================================
# GPU ID for evaluation
GPU_ID=0

# Path to models config JSON file
# Edit models_config.json to add/remove models
CONFIG_FILE="models_config.json"

# Optional flags ( uncomment to skip phases)
# SKIP_GEN="--skip_gen"
# SKIP_EVAL="--skip_eval"
# SKIP_AGG="--skip_agg"

# ==========================================
#        ENVIRONMENT SETUP
# ==========================================
module load cuda/12.1.1

echo "=========================================="
echo "Leak@K Evaluation Pipeline"
echo "=========================================="
echo ""
echo "GPU availability:"
nvidia-smi
echo ""
echo "Python executable:"
which python3
echo ""
echo "Config file: ${CONFIG_FILE}"
echo "Starting at: $(date)"
echo "Job is running on $(hostname)"
echo ""

# Set up HuggingFace (uncomment and edit if needed)
# export 'HF_TOKEN=your_hf_token'
# export HF_HOME="your_hf_home_path"
# huggingface-cli login --token $HF_TOKEN

# ==========================================
#        RUN EVALUATION
# ==========================================
python whole_eval.py \
    --config "${CONFIG_FILE}" \
    --gpu_id "${GPU_ID}" \
    ${SKIP_GEN:-} \
    ${SKIP_EVAL:-} \
    ${SKIP_AGG:-}

echo ""
echo "Finished at: $(date)"