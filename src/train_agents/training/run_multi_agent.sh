#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=04:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running DQN multi-agent'
python -m  train_agents.training.multi_agent_train

# sbatch run_multi_agent.sh