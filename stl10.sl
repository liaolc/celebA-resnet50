#!/bin/bash
#SBATCH --job-name=stl10_radial_masking             # Job name
#SBATCH --output=stl10.out                       # Standard output
#SBATCH --error=stl10.err                        # Standard error
#SBATCH --nodes=1                                   # Number of nodes
#SBATCH --ntasks=1                                  # Number of tasks
#SBATCH --cpus-per-task=4                           # Number of CPU cores per task
#SBATCH --gres=gpu:1                                # Request 1 GPU
#SBATCH --mem=16G                                   # Total memory
#SBATCH --time=12:00:00                             # Wall time (hh:mm:ss)
#SBATCH --mail-type=BEGIN,END,FAIL                  # Email on start, end, fail
#SBATCH --mail-user=zhangcoj@bc.edu                 # Your email

# Path to your shared virtual environment
ENV_DIR="$HOME/env_interpretability"

# If the virtual environment doesn't exist, create and set it up
if [ ! -d "$ENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $ENV_DIR"
    python3 -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"

    echo "[INFO] Upgrading pip..."
    pip install --upgrade pip

    echo "[INFO] Installing required Python packages..."
    pip install torch torchvision tqdm matplotlib numpy pandas
else
    echo "[INFO] Virtual environment already exists at $ENV_DIR"
fi

# Activate the environment
source "$ENV_DIR/bin/activate"

# Run your Python script
echo "[INFO] Starting Python script..."
python stl10.py

echo "[INFO] Job completed."
