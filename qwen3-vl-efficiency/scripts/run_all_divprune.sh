#!/bin/bash
#SBATCH --job-name=divprune_all
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=divprune_all_%j.out
#SBATCH --error=divprune_all_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ykv209@nyu.edu

export HF_HOME=$SCRATCH/hf_cache

# Use packages installed in the overlay by install_overlay.sh.
# PYTHONNOUSERSITE=1 prevents ~/.local from being added to sys.path,
# eliminating all CUDA library version conflicts from the host environment.
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH=/ext3/packages
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_HF_TOKEN=$HF_TOKEN

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        set -e
        cd $SCRATCH/hpml_project/qwen3-vl-efficiency

        for RATIO in 0.5 0.3 0.2; do
            echo \"=== Ratio \$RATIO ===\"
            echo \"  [1/4] MMMU\"
            python3 -m eval.eval_mmmu_divprune --subset_ratio \$RATIO || echo \"  MMMU failed for ratio \$RATIO\"
            echo \"  [2/4] DocVQA\"
            python3 -m eval.eval_docvqa_divprune --subset_ratio \$RATIO || echo \"  DocVQA failed for ratio \$RATIO\"
            echo \"  [3/4] MathVista\"
            python3 -m eval.eval_mathvista_divprune --subset_ratio \$RATIO || echo \"  MathVista failed for ratio \$RATIO\"
            echo \"  [4/4] RealWorldQA\"
            python3 -m eval.eval_realworldqa_divprune --subset_ratio \$RATIO || echo \"  RealWorldQA failed for ratio \$RATIO\"
            echo \"  Done ratio \$RATIO\"
        done

        echo '=== All done. Results in results/divprune/ ==='
    "
