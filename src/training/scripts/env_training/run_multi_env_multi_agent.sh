#!/bin/bash

#SBATCH --partition=lyceum
#SBATCH --time=16:00:00

cd ~/Online-Flexible-Resource-Allocation/src/

module load conda
source activate py37env

echo 'Running standard multi agent with multi environments'
python -m training.scripts.env_training.multi_env_multi_agent
