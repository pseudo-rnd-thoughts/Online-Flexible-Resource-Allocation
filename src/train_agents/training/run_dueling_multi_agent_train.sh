#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=12:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running dueling DQN multi-agent'
python -m  train_agents.training.dueling_multi_agent_train

# sbatch run_dueling_multi_agent.sh