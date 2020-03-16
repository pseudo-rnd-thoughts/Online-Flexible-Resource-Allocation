#!/bin/bash

#SBATCH --time=04:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running GRU network DQN multi-agent'
python -m  train_agents.training.grU_multi_agent_train

# sbatch -p lyceum run_gru_multi_agent.sh