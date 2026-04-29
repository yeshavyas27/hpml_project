#!/bin/bash
# Install divprune dependencies inside the Singularity overlay.
# Run this once from the login node before submitting jobs.

singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps &&
        pip3 install -r $SCRATCH/hpml_project/divprune/requirements.txt --no-deps &&
        pip3 install datasets pillow requests tqdm numpy &&
        echo '=== divprune install complete ==='
    "
