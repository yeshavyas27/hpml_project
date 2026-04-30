#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=eval_all_%j.out
#SBATCH --error=eval_all_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ykv209@nyu.edu

export HF_HOME=$SCRATCH/hf_cache
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONPATH=/ext3/packages
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_HF_TOKEN=$HF_TOKEN

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        set -e
        export LD_LIBRARY_PATH=\$(ls -d ~/.local/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':'):\$LD_LIBRARY_PATH
        cd $SCRATCH/hpml_project/qwen3-vl-efficiency

        echo '========================================'
        echo 'PHASE 1: Baseline evals (no DivPrune)'
        echo '========================================'

        echo '[1/4] MMMU baseline'
        python3 -m eval.eval_mmmu_divprune --no_divprune || echo 'MMMU baseline failed'

        echo '[2/4] DocVQA baseline'
        python3 -m eval.eval_docvqa_divprune --no_divprune || echo 'DocVQA baseline failed'

        echo '[3/4] MathVista baseline'
        python3 -m eval.eval_mathvista_divprune --no_divprune || echo 'MathVista baseline failed'

        echo '[4/4] RealWorldQA baseline'
        python3 -m eval.eval_realworldqa_divprune --no_divprune || echo 'RealWorldQA baseline failed'

        echo '========================================'
        echo 'PHASE 2: DivPrune sweep (ratios 0.5 0.3 0.2)'
        echo '========================================'

        for RATIO in 0.5 0.3 0.2; do
            echo \"--- Ratio \$RATIO ---\"

            echo \"  [1/4] MMMU\"
            python3 -m eval.eval_mmmu_divprune --subset_ratio \$RATIO || echo \"  MMMU failed for ratio \$RATIO\"

            echo \"  [2/4] DocVQA\"
            python3 -m eval.eval_docvqa_divprune --subset_ratio \$RATIO || echo \"  DocVQA failed for ratio \$RATIO\"

            echo \"  [3/4] MathVista\"
            python3 -m eval.eval_mathvista_divprune --subset_ratio \$RATIO || echo \"  MathVista failed for ratio \$RATIO\"

            echo \"  [4/4] RealWorldQA\"
            python3 -m eval.eval_realworldqa_divprune --subset_ratio \$RATIO || echo \"  RealWorldQA failed for ratio \$RATIO\"
        done

        echo '========================================'
        echo 'All done. Results in results/divprune/'
        echo '========================================'
    "
