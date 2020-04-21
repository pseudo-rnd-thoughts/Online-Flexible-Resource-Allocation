#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=12:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running Double DQN agent'
python -m training.scripts.policy_training.ddqn_multi_agent
