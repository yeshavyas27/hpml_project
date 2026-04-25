#!/bin/bash
#SBATCH --job-name=mmmu_baseline
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=mmmu_%j.out
#SBATCH --error=mmmu_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ykv209@nyu.edu

# ── Paths ──────────────────────────────────────────────────────────────────────
export HF_HOME=$SCRATCH/hf_cache
export RESULTS_DIR=$SCRATCH/results
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

mkdir -p $RESULTS_DIR
mkdir -p $HF_HOME

# ── Prevent tokenizer parallelism warnings ─────────────────────────────────────
export TOKENIZERS_PARALLELISM=false

# ── Force Python stdout/stderr to flush immediately (crucial for SLURM logs) ──
export PYTHONUNBUFFERED=1

# ── Offline mode: set to 1 if model/dataset already cached in $HF_HOME ────────
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# ── Log job context ────────────────────────────────────────────────────────────
echo "======================================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Partition    : $SLURM_JOB_PARTITION"
echo "CPUs         : $SLURM_CPUS_PER_TASK"
echo "Memory       : $SLURM_MEM_PER_NODE MB"
echo "Start time   : $(date)"
echo "SCRATCH      : $SCRATCH"
echo "HF_HOME      : $HF_HOME"
echo "Results dir  : $RESULTS_DIR"
echo "======================================================"

singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        source /ext3/env.sh 2>/dev/null || true

        echo '[env] Python: '$(which python3)
        echo '[env] PyTorch: '$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')
        echo '[env] CUDA available: '$(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')
        echo '[env] GPU: '$(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'unknown')

        pip install --quiet --upgrade calflops pynvml 2>/dev/null || \
            echo '[warn] calflops/pynvml install failed – metrics will degrade gracefully'

        cd $SCRATCH && python3 evaluate_mmmu.py \
            --split validation \
            --max_new_tokens 32 \
            --output $RESULTS_DIR/mmmu_results_${SLURM_JOB_ID}.json \
            --metrics_output $RESULTS_DIR/inference_metrics_${SLURM_JOB_ID}.json
    "

EXIT_CODE=$?

echo "======================================================"
echo "End time     : $(date)"
echo "Exit code    : $EXIT_CODE"
echo "======================================================"

exit $EXIT_CODE
