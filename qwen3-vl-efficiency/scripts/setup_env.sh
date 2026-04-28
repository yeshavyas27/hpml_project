#!/bin/bash
# Run once to install all dependencies.
# Creates a marker file so subsequent runs skip the install.

MARKER="$HOME/.divprune_env_installed_v2"

if [ -f "$MARKER" ]; then
    echo "=== Dependencies already installed, skipping ==="
else
    echo "=== Installing dependencies ==="
    pip3 install "torch>=2.4.0" torchvision --index-url https://download.pytorch.org/whl/cu121 &&
    pip3 install -r $SCRATCH/hpml_project/divprune/requirements.txt --no-deps &&
    pip3 install accelerate datasets pillow requests tqdm numpy &&
    pip3 install 'git+https://github.com/huggingface/transformers' &&
    touch "$MARKER" &&
    echo "=== Install complete ==="
fi
