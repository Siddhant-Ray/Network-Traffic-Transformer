#!/bin/bash
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
set -o errexit  # Exit on errors

# Activate the correct venv.
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate venv

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL srun python encoder_delay.py
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python encoder_delay_varmask_chooseagglevel.py
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python encoder_delay_varmask_chooseencodelem.py
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python encoder_delay_varmask_chooseencodelem_multi.py

# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python finetune_encoder.py
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python finetune_encoder_multi.py

# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python finetune_mct.py
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python finetune_mct_multi.py

# NCCL_DEBUG=INFO; NCCL_SEBUG_SUBSYS=ALL python lstm.py 
# NCCL_DEBUG=INFO; NCCL_DEBUG_SUBSYS=ALL python arima.py --run true 

