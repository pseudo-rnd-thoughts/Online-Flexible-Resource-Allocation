#!/bin bash

#SBATCH --time=04:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
conda activate py37env

echo 'Running multi-agent train'
python -m  train_agents.training.multi_agent_train