#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running Distributional DQN agent'
python -m  train_agents.training.policy_testing.distributional_multi_agent
