#!/bin/bash
# Install divprune dependencies inside the Singularity overlay.
# Run this once from the login node before submitting jobs.
#
# Uses python3 -m pip (not pip3) to target the container's Python 3.10,
# and installs torch WITH its nvidia-* wheel deps so NCCL/CUPTI/cudart
# are supplied by the Python packages rather than the host driver.

singularity exec --nv \
    --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall &&
        python3 -m pip install -r $SCRATCH/hpml_project/divprune/requirements.txt --no-deps &&
        python3 -m pip install datasets pillow requests tqdm numpy &&
        echo '=== divprune install complete ==='
    "
