#!/bin/bash
#SBATCH --job-name=divprune_docvqa
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=divprune_docvqa_%j.out
#SBATCH --error=divprune_docvqa_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ykv209@nyu.edu

export HF_HOME=$SCRATCH/hf_cache

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        export LD_LIBRARY_PATH=\$(ls -d ~/.local/lib/python3.10/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':'):\$LD_LIBRARY_PATH &&
        cd $SCRATCH/hpml_project/qwen3-vl-efficiency &&
        python3 -m eval.eval_docvqa_divprune
    "
