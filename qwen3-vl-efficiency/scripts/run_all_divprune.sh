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

# Set LD_PRELOAD via SINGULARITYENV_ *before* singularity exec so it is injected
# into the container by singularity itself — not inside bash -c where some singularity
# versions reset or ignore it. Use the pip-installed nvidia cupti (ABI-matched to pip
# torch) to beat the host cupti --nv injects.
PIP_CUPTI="$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
if [ -f "$PIP_CUPTI" ]; then
    export SINGULARITYENV_LD_PRELOAD="$PIP_CUPTI"
fi

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        # Belt-and-suspenders: prepend all pip nvidia lib dirs to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=\$(ls -d \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':'):\$LD_LIBRARY_PATH &&

        echo \"[debug] LD_PRELOAD=\$LD_PRELOAD\" &&

        cd $SCRATCH/hpml_project/qwen3-vl-efficiency &&

        echo '=== [1/4] MMMU ===' &&
        python3 -m eval.eval_mmmu_divprune &&

        echo '=== [2/4] DocVQA ===' &&
        python3 -m eval.eval_docvqa_divprune &&

        echo '=== [3/4] MathVista ===' &&
        python3 -m eval.eval_mathvista_divprune &&

        echo '=== [4/4] RealWorldQA ===' &&
        python3 -m eval.eval_realworldqa_divprune &&

        echo '=== All done. Results in results/divprune/ ==='
    "
