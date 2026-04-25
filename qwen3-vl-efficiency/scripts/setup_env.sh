#!/bin/bash
# Run once to install all dependencies.
# Creates a marker file so subsequent runs skip the install.

MARKER="$HOME/.divprune_env_installed"

if [ -f "$MARKER" ]; then
    echo "=== Dependencies already installed, skipping ==="
else
    echo "=== Installing dependencies ==="
    pip install -r $SCRATCH/hpml_project/divprune/requirements.txt &&
    pip install 'git+https://github.com/huggingface/transformers' &&
    touch "$MARKER" &&
    echo "=== Install complete ==="
fi
