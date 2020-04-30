#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=24:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running DDPG agent'
python -m training.policy_training.ddpg_multi_agent
