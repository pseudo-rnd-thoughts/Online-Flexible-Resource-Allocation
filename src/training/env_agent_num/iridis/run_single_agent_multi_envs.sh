#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running single agent with multiple training environment'
python -m training.env_agent_num.single_agent_mutli_envs