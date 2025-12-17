#!/usr/bin/env bash
set -e

CONDA_ENV_NAME="scillm"

echo "Setting up conda environment '$CONDA_ENV_NAME' and variables..."
conda create -n $CONDA_ENV_NAME
conda env config vars set CONDA_ENV_NAME="$CONDA_ENV_NAME" -n $CONDA_ENV_NAME
conda env config vars set SCRATCH_DIR="/scratch/$USER/aimi-scillm" -n $CONDA_ENV_NAME
conda activate $CONDA_ENV_NAME

echo "Installing python dependencies..."
conda install python=3.13
pip install uv

uv sync --all-groups
make install-hooks
