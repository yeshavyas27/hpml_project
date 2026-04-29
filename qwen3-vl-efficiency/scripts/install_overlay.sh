#!/bin/bash
# Run this ONCE to install all packages into the singularity overlay at /ext3/packages.
# After this, run_all_divprune.sh uses PYTHONNOUSERSITE=1 + PYTHONPATH=/ext3/packages
# so ~/.local is never touched and there are no CUDA library conflicts.

singularity exec --nv --overlay $SCRATCH/overlay.ext3 \
    /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        set -e

        # torch cu126 matches the container's CUDA 12.6 exactly — no cupti ABI mismatch
        pip3 install --target /ext3/packages \
            torch torchvision \
            --index-url https://download.pytorch.org/whl/cu126

        pip3 install --target /ext3/packages \
            'transformers>=4.57.0' \
            accelerate \
            datasets \
            pillow \
            requests \
            tqdm \
            numpy \
            pandas \
            matplotlib \
            kvpress \
            qwen-vl-utils

        echo 'Done. Packages installed to /ext3/packages.'
        python3 -c 'import sys; sys.path.insert(0, \"/ext3/packages\"); import torch; print(\"torch\", torch.__version__, \"cuda:\", torch.cuda.is_available())'
    "
