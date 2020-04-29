#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running standard multi-agent training'
python -m training.scripts.env_training.multi_agent_single_env
