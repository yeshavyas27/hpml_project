#!/bin/bash
#SBATCH --job-name=divprune_mmmu
#SBATCH --account=ece_gy_9143-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=divprune_mmmu_%j.out
#SBATCH --error=divprune_mmmu_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=ykv209@nyu.edu

export HF_HOME=$SCRATCH/hf_cache

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.2.2-cudnn8.9.4-devel-ubuntu22.04.3.sif \
    /bin/bash -c "
        pip install -r $SCRATCH/hpml_project/divprune/requirements.txt &&
        pip install 'git+https://github.com/huggingface/transformers' &&
        cd $SCRATCH/qwen3-vl-efficiency &&
        python -m eval.eval_mmmu_divprune --subset_ratio 0.5
    "
