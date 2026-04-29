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

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        # LD_PRELOAD the pip-installed nvidia cupti — guaranteed ABI match with pip torch.
        # This beats RPATH (which LD_LIBRARY_PATH cannot) and avoids the host cupti
        # injected by --nv. We use the pip path specifically, NOT the container's cupti,
        # to avoid the nccl secondary ABI break seen with container cupti preloading.
        PIP_CUPTI=\$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12
        if [ -f \"\$PIP_CUPTI\" ]; then
            export LD_PRELOAD=\"\$PIP_CUPTI\${LD_PRELOAD:+:\$LD_PRELOAD}\"
        fi &&

        # Belt-and-suspenders: also prepend all pip nvidia lib dirs to LD_LIBRARY_PATH
        export LD_LIBRARY_PATH=\$(ls -d \$HOME/.local/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':'):\$LD_LIBRARY_PATH &&

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
