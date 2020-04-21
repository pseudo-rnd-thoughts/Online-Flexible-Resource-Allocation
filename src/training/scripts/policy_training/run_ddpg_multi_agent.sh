#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=12:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running DDPG agent'
python -m training.scripts.policy_training.ddpg_multi_agent
