#!/bin/bash
# Install divprune dependencies inside the Singularity overlay.
# Run this once from the login node before submitting jobs.

singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 &&
        pip3 install accelerate &&
        pip3 install -r $SCRATCH/hpml_project/divprune/requirements.txt --no-deps &&
        pip3 install datasets pillow requests tqdm numpy &&
        pip3 install 'git+https://github.com/huggingface/transformers' &&
        echo '=== divprune install complete ==='
    "
