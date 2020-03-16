#!/bin/bash

#SBATCH --time=04:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running DQN single-agent'
python -m  train_agents.training.single_agent_train

# sbatch -p lyceum run_single_agent.sh